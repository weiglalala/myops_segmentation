from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.constants import EDEMA_CLASS_INDEX, MYOCARDIUM_CLASS_INDICES, SCAR_CLASS_INDEX
from src.config import Settings
from src.data import (
    MyoPSSliceDataset,
    build_sample_weights,
    restore_cropped_array,
    split_case_dirs_kfold,
    sorted_case_dirs,
    split_case_dirs,
    summarize_sample_weights,
)
from src.losses import CombinedSegmentationLoss, derive_class_weights
from src.metrics import evaluate_case
from src.models import build_model
from src.utils import AverageMeter, ensure_directories, save_json, seed_everything, setup_logger
from src.visualization import save_case_visualizations, save_training_curves

EARLY_STOPPING_MA_WINDOW = 5


def resolve_train_experiment_name(settings: Settings) -> str:
    if settings.fold_index is None:
        return settings.experiment_name
    suffix = f"_fold{settings.fold_index}"
    if settings.experiment_name.endswith(suffix):
        return settings.experiment_name
    return f"{settings.experiment_name}{suffix}"


def split_model_outputs(outputs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 3:
            return outputs[0], outputs[1], outputs[2]
        if len(outputs) == 2:
            return outputs[0], outputs[1], None
        raise ValueError(f"Expected 2 or 3 outputs for auxiliary-head model, got {len(outputs)}")
    return outputs, None, None


def keep_largest_connected_component(prediction: np.ndarray) -> np.ndarray:
    prediction_array = prediction
    non_background = prediction_array > 0
    if not non_background.any():
        return prediction_array
    labeled, num_components = ndimage.label(non_background)
    if num_components <= 1:
        return prediction_array
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest_label = int(component_sizes.argmax())
    filtered = prediction_array.copy()
    filtered[labeled != largest_label] = 0
    return filtered


def constrain_edema_to_myocardium(prediction: np.ndarray) -> np.ndarray:
    constrained = prediction.copy()
    for slice_index in range(constrained.shape[2]):
        prediction_slice = constrained[:, :, slice_index]
        myocardium_mask = np.isin(prediction_slice, MYOCARDIUM_CLASS_INDICES)
        edema_mask = prediction_slice == EDEMA_CLASS_INDEX
        if not edema_mask.any():
            continue
        expanded_myocardium = ndimage.binary_dilation(myocardium_mask, iterations=3)
        prediction_slice[edema_mask & ~expanded_myocardium] = 0
        constrained[:, :, slice_index] = prediction_slice
    return constrained


def constrain_pathology_to_myocardium(prediction: np.ndarray, dilation_iterations: int = 3) -> np.ndarray:
    constrained = prediction.copy()
    for slice_index in range(constrained.shape[2]):
        s = constrained[:, :, slice_index]
        myo = np.isin(s, MYOCARDIUM_CLASS_INDICES)
        expanded = ndimage.binary_dilation(myo, iterations=dilation_iterations)
        pathology = (s == EDEMA_CLASS_INDEX) | (s == SCAR_CLASS_INDEX)
        s[pathology & ~expanded] = 0
        constrained[:, :, slice_index] = s
    return constrained


def constrain_predictions_to_union_mask(prediction: np.ndarray, union_mask: np.ndarray) -> np.ndarray:
    constrained = prediction.copy()
    pathology_outside = ((constrained == 4) | (constrained == 5)) & ~union_mask.astype(bool)
    constrained[pathology_outside] = 1
    return constrained


def build_dataloaders(settings: Settings) -> tuple[MyoPSSliceDataset, MyoPSSliceDataset, DataLoader, DataLoader]:
    case_dirs = sorted_case_dirs(settings.train_dir)
    if settings.fold_index is not None:
        train_dirs, val_dirs = split_case_dirs_kfold(
            case_dirs,
            num_folds=settings.num_folds,
            fold_index=settings.fold_index,
            seed=settings.seed,
        )
    else:
        train_dirs, val_dirs = split_case_dirs(
            case_dirs,
            val_ratio=settings.val_ratio,
            seed=settings.seed,
            fixed_val_cases=settings.fixed_val_cases,
        )

    train_dataset = MyoPSSliceDataset(
        train_dirs,
        crop_size=settings.crop_size,
        input_mode=settings.input_mode,
        augment=True,
        rotation_degrees=settings.rotation_degrees,
        enable_elastic=settings.enable_elastic,
        enable_advanced_augment=settings.enable_advanced_augment,
        use_t2_clahe=settings.use_t2_clahe,
        t2_clahe_clip_limit=settings.t2_clahe_clip_limit,
        t2_clahe_tile_grid_size=settings.t2_clahe_tile_grid_size,
        roi_crop=settings.use_roi_crop,
        roi_padding=settings.roi_padding,
        zero_bssfp=settings.zero_bssfp,
    )
    val_dataset = MyoPSSliceDataset(
        val_dirs,
        crop_size=settings.crop_size,
        input_mode=settings.input_mode,
        augment=False,
        rotation_degrees=settings.rotation_degrees,
        enable_elastic=False,
        use_t2_clahe=settings.use_t2_clahe,
        t2_clahe_clip_limit=settings.t2_clahe_clip_limit,
        t2_clahe_tile_grid_size=settings.t2_clahe_tile_grid_size,
        roi_crop=settings.use_roi_crop,
        roi_padding=settings.roi_padding,
        zero_bssfp=settings.zero_bssfp,
    )

    weights = build_sample_weights(
        train_dataset,
        pathology_oversample_factor=settings.pathology_oversample_factor,
        sampling_strategy=settings.sampling_strategy,
    )
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=settings.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=settings.device.startswith("cuda"),
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_test_loader(settings: Settings) -> tuple[MyoPSSliceDataset, DataLoader]:
    case_dirs = sorted_case_dirs(settings.test_dir)
    test_dataset = MyoPSSliceDataset(
        case_dirs,
        crop_size=settings.crop_size,
        input_mode=settings.input_mode,
        augment=False,
        rotation_degrees=settings.rotation_degrees,
        enable_elastic=False,
        use_t2_clahe=settings.use_t2_clahe,
        t2_clahe_clip_limit=settings.t2_clahe_clip_limit,
        t2_clahe_tile_grid_size=settings.t2_clahe_tile_grid_size,
        roi_crop=settings.use_roi_crop,
        roi_padding=settings.roi_padding,
        zero_bssfp=settings.zero_bssfp,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=settings.device.startswith("cuda"),
    )
    return test_dataset, test_loader


def predict_with_tta(model: torch.nn.Module, images: torch.Tensor, use_tta: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    transforms = [
        ("identity", lambda x: x),
        ("hflip", lambda x: torch.flip(x, dims=(-1,))),
        ("vflip", lambda x: torch.flip(x, dims=(-2,))),
        ("hvflip", lambda x: torch.flip(x, dims=(-2, -1))),
    ]
    if not use_tta:
        transforms = transforms[:1]

    probabilities = []
    union_probabilities = []
    has_union = False
    for name, transform in transforms:
        augmented = transform(images)
        logits, _, union_logits = split_model_outputs(model(augmented))
        if name == "identity":
            restored = logits
            restored_union = union_logits
        elif name == "hflip":
            restored = torch.flip(logits, dims=(-1,))
            restored_union = torch.flip(union_logits, dims=(-1,)) if union_logits is not None else None
        elif name == "vflip":
            restored = torch.flip(logits, dims=(-2,))
            restored_union = torch.flip(union_logits, dims=(-2,)) if union_logits is not None else None
        else:
            restored = torch.flip(logits, dims=(-2, -1))
            restored_union = torch.flip(union_logits, dims=(-2, -1)) if union_logits is not None else None
        probabilities.append(torch.softmax(restored, dim=1))
        if restored_union is not None:
            has_union = True
            union_probabilities.append(torch.sigmoid(restored_union.squeeze(1)))
    main_probs = torch.stack(probabilities, dim=0).mean(dim=0)
    union_probs = torch.stack(union_probabilities, dim=0).mean(dim=0) if has_union else None
    return main_probs, union_probs


def predict_with_ensemble(
    models: list[torch.nn.Module],
    images: torch.Tensor,
    use_tta: bool,
    weights: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    results = [predict_with_tta(model, images, use_tta=use_tta) for model in models]
    ensemble_probabilities = [r[0] for r in results]
    ensemble_union = [r[1] for r in results]
    has_union = any(u is not None for u in ensemble_union)

    if weights is None:
        main_probs = torch.stack(ensemble_probabilities, dim=0).mean(dim=0)
        if has_union:
            valid_union = [u for u in ensemble_union if u is not None]
            union_probs = torch.stack(valid_union, dim=0).mean(dim=0) if valid_union else None
        else:
            union_probs = None
        return main_probs, union_probs

    if len(weights) != len(ensemble_probabilities):
        raise ValueError("The number of ensemble weights must match the number of ensemble checkpoints.")
    weight_tensor = torch.tensor(weights, dtype=ensemble_probabilities[0].dtype, device=ensemble_probabilities[0].device)
    if torch.any(weight_tensor < 0):
        raise ValueError("Ensemble weights must be non-negative.")
    weight_sum = weight_tensor.sum()
    if float(weight_sum.item()) <= 0:
        raise ValueError("The sum of ensemble weights must be positive.")
    normalized_weights = weight_tensor / weight_sum
    weighted_probabilities = [
        probability * normalized_weights[index]
        for index, probability in enumerate(ensemble_probabilities)
    ]
    main_probs = torch.stack(weighted_probabilities, dim=0).sum(dim=0)
    if has_union:
        valid_union_weighted = [
            u * normalized_weights[i]
            for i, u in enumerate(ensemble_union) if u is not None
        ]
        union_probs = torch.stack(valid_union_weighted, dim=0).sum(dim=0) if valid_union_weighted else None
    else:
        union_probs = None
    return main_probs, union_probs


def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedSegmentationLoss,
    optimizer: Adam,
    device: torch.device,
    max_batches: int | None,
    epoch: int = 0,
    total_epochs: int = 300,
) -> tuple[float, dict[str, float]]:
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    ce_meter = AverageMeter()
    focal_meter = AverageMeter()
    pathology_dice_meter = AverageMeter()
    myocardium_prior_meter = AverageMeter()
    edema_dice_meter = AverageMeter()
    surface_meter = AverageMeter()
    union_meter = AverageMeter()
    inclusiveness_meter = AverageMeter()
    progress = tqdm(loader, desc="Train", leave=False)

    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        dist_maps = batch["dist_map"].to(device) if "dist_map" in batch else None
        optimizer.zero_grad(set_to_none=True)
        logits, myocardium_logits, union_logits = split_model_outputs(model(images))
        loss, components = criterion(
            logits, masks, myocardium_logits=myocardium_logits,
            union_logits=union_logits,
            dist_maps=dist_maps, epoch=epoch, total_epochs=total_epochs,
        )
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        loss_meter.update(components["total_loss"], batch_size)
        dice_meter.update(components["dice_loss"], batch_size)
        ce_meter.update(components["ce_loss"], batch_size)
        focal_meter.update(components["focal_loss"], batch_size)
        pathology_dice_meter.update(components["pathology_dice_loss"], batch_size)
        myocardium_prior_meter.update(components["myocardium_prior_loss"], batch_size)
        edema_dice_meter.update(components["edema_dice_loss"], batch_size)
        surface_meter.update(components["surface_loss"], batch_size)
        union_meter.update(components["union_loss"], batch_size)
        inclusiveness_meter.update(components["inclusiveness_loss"], batch_size)
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

    return loss_meter.avg, {
        "dice_loss": dice_meter.avg,
        "ce_loss": ce_meter.avg,
        "focal_loss": focal_meter.avg,
        "pathology_dice_loss": pathology_dice_meter.avg,
        "myocardium_prior_loss": myocardium_prior_meter.avg,
        "edema_dice_loss": edema_dice_meter.avg,
        "surface_loss": surface_meter.avg,
        "union_loss": union_meter.avg,
        "inclusiveness_loss": inclusiveness_meter.avg,
    }


def evaluate_loader(
    model: torch.nn.Module,
    dataset: MyoPSSliceDataset,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
    output_dir: Path | None,
    vis_modality: str,
    save_visualizations: bool,
    visualize_all_slices: bool,
    use_connected_component: bool = False,
    use_edema_constraint: bool = False,
    use_union_constraint: bool = False,
    max_batches: int | None = None,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    case_predictions: dict[str, torch.Tensor] = {}
    case_union_probs: dict[str, list[tuple[int, np.ndarray]]] = {}
    case_seen = set()

    with torch.no_grad():
        progress = tqdm(loader, desc="Eval", leave=False)
        for batch_index, batch in enumerate(progress):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device)
            probabilities, union_probs = predict_with_tta(model, images, use_tta=use_tta)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
            union_np = union_probs.cpu().numpy() if union_probs is not None else None

            case_ids = batch["case_id"]
            slice_indices = batch["slice_idx"].tolist()
            for sample_index, case_id in enumerate(case_ids):
                if max_cases is not None and case_id not in case_seen and len(case_seen) >= max_cases:
                    continue
                case_seen.add(case_id)
                case_store = dataset.case_data[case_id]
                if case_id not in case_predictions:
                    case_predictions[case_id] = torch.zeros_like(torch.from_numpy(case_store["mask_full"]))
                crop_meta = case_store["crop_metas"][slice_indices[sample_index]]
                restored_slice = restore_cropped_array(predictions[sample_index], crop_meta)
                case_predictions[case_id][:, :, slice_indices[sample_index]] = torch.from_numpy(restored_slice)
                if union_np is not None:
                    if case_id not in case_union_probs:
                        case_union_probs[case_id] = []
                    restored_union = restore_cropped_array(union_np[sample_index], crop_meta).astype(np.float32)
                    case_union_probs[case_id].append((slice_indices[sample_index], restored_union))

    rows = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for case_id, prediction_tensor in sorted(case_predictions.items()):
        case_store = dataset.case_data[case_id]
        prediction = prediction_tensor.numpy().astype(case_store["mask_full"].dtype)
        if use_connected_component:
            prediction = keep_largest_connected_component(prediction)
        if use_edema_constraint:
            prediction = constrain_edema_to_myocardium(prediction)
        if use_union_constraint and case_id in case_union_probs:
            union_volume = np.zeros_like(prediction, dtype=np.float32)
            for si, uslice in case_union_probs[case_id]:
                union_volume[:, :, si] = uslice
            union_mask = (union_volume >= 0.5)
            prediction = constrain_predictions_to_union_mask(prediction, union_mask)
        target = case_store["mask_full"]
        metrics = evaluate_case(prediction, target, spacing=case_store["spacing"])
        rows.append({"case_id": case_id, **metrics})
        if output_dir is not None and save_visualizations:
            save_case_visualizations(
                case_id=case_id,
                base_volume=case_store["raw_modalities"][vis_modality],
                gt_volume=target,
                pred_volume=prediction,
                output_dir=output_dir / "visualizations",
                visualize_all_slices=visualize_all_slices,
            )
    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        raise RuntimeError("No cases were evaluated. Check max_cases/max_batches settings.")
    summary = _compute_summary(metrics_df)
    return metrics_df, summary


def save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def resolve_checkpoint_path(settings: Settings) -> Path:
    if settings.checkpoint_path is not None:
        return settings.checkpoint_path
    candidate = settings.experiment_ckpt_dir / "best_model.pt"
    if not candidate.exists():
        raise FileNotFoundError("No checkpoint path provided and best_model.pt not found.")
    return candidate


def resolve_ensemble_checkpoint_paths(settings: Settings) -> list[Path]:
    if settings.ensemble_checkpoint_paths:
        missing_paths = [path for path in settings.ensemble_checkpoint_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(f"Ensemble checkpoints not found: {missing_paths}")
        return list(settings.ensemble_checkpoint_paths)
    single_checkpoint = resolve_checkpoint_path(settings)
    return [single_checkpoint]


def resolve_ensemble_weights(settings: Settings, num_models: int) -> list[float] | None:
    if not settings.ensemble_weights:
        return None
    if len(settings.ensemble_weights) != num_models:
        raise ValueError("The number of ensemble weights must match the number of ensemble checkpoints.")
    return list(settings.ensemble_weights)


def load_model_checkpoint(
    checkpoint_path: Path,
    model_variant: str,
    in_channels: int,
    device: torch.device,
    num_classes: int = 6,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(model_variant, in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def train_pipeline(settings: Settings) -> None:
    experiment_name = resolve_train_experiment_name(settings)
    settings = replace(settings, experiment_name=experiment_name)
    ensure_directories(
        [
            settings.experiment_ckpt_dir,
            settings.experiment_log_dir,
            settings.experiment_result_dir,
        ]
    )
    logger = setup_logger(settings.experiment_log_dir / "train.log")
    save_json(settings.to_dict(), settings.experiment_log_dir / "config.json")
    seed_everything(settings.seed)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(settings)
    sample_weight_summary = summarize_sample_weights(train_dataset, train_loader.sampler.weights.tolist())
    device = torch.device(settings.device)
    model = build_model(settings.model_variant, in_channels=settings.input_channels).to(device)
    class_weights = derive_class_weights(scar_weight=settings.scar_class_weight).to(device)
    criterion = CombinedSegmentationLoss(
        class_weights=class_weights,
        dice_weight=settings.dice_loss_weight,
        ce_weight=settings.ce_loss_weight,
        focal_weight=settings.focal_loss_weight,
        focal_gamma=settings.focal_gamma,
        tversky_weight=settings.tversky_loss_weight,
        tversky_alpha=settings.tversky_alpha,
        tversky_beta=settings.tversky_beta,
        myocardium_prior_weight=settings.myocardium_prior_weight,
        edema_dice_weight=settings.edema_dice_weight,
        surface_loss_weight=settings.surface_loss_weight,
        union_loss_weight=settings.union_loss_weight,
        inclusiveness_loss_weight=settings.inclusiveness_loss_weight,
    )
    if (
        settings.encoder_lr > 0
        and hasattr(model, "unet")
        and hasattr(model.unet, "encoder")
    ):
        encoder_params = list(model.unet.encoder.parameters())
        encoder_ids = {id(param) for param in encoder_params}
        other_params = [param for param in model.parameters() if id(param) not in encoder_ids]
        optimizer = Adam(
            [
                {"params": encoder_params, "lr": settings.encoder_lr},
                {"params": other_params, "lr": settings.learning_rate},
            ],
            weight_decay=settings.weight_decay,
        )
    else:
        optimizer = Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    if settings.scheduler_type == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=settings.cosine_t0,
            T_mult=1,
            eta_min=1e-6,
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=settings.scheduler_factor,
            patience=settings.scheduler_patience,
        )

    start_epoch = 1
    best_score = float("-inf")
    best_edema_score = float("-inf")
    best_moving_average_score = float("-inf")
    history: list[dict] = []

    resume_path = settings.experiment_ckpt_dir / "last_model.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", float("-inf"))
        best_edema_score = ckpt.get("best_edema_score", float("-inf"))
        history_csv = settings.experiment_log_dir / "history.csv"
        if history_csv.exists():
            history = pd.read_csv(history_csv).to_dict("records")
            recent = history[-EARLY_STOPPING_MA_WINDOW:]
            best_moving_average_score = max(
                best_moving_average_score,
                sum(r["val_mean_dice"] for r in recent) / len(recent),
            )
        logger.info("Resumed from epoch %d (best_score=%.4f)", ckpt["epoch"], best_score)

    logger.info("Train cases: %s", sorted(train_dataset.case_data.keys()))
    logger.info("Val cases: %s", sorted(val_dataset.case_data.keys()))
    if settings.fold_index is not None:
        logger.info(
            "K-Fold split enabled: fold=%d/%d",
            settings.fold_index,
            settings.num_folds,
        )
    logger.info(
        "Sampling strategy: %s | oversample_factor=%.2f | weight_stats=%s",
        settings.sampling_strategy,
        settings.pathology_oversample_factor,
        {key: round(value, 4) for key, value in sample_weight_summary.items()},
    )
    logger.info("Class weights: %s", class_weights.detach().cpu().numpy().round(4).tolist())

    best_state = None
    epochs_without_improvement = 0

    for epoch in range(start_epoch, settings.epochs + 1):
        train_loss, train_components = run_train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=settings.max_train_batches,
            epoch=epoch,
            total_epochs=settings.epochs,
        )
        metrics_df, summary = evaluate_loader(
            model=model,
            dataset=val_dataset,
            loader=val_loader,
            device=device,
            use_tta=False,
            output_dir=settings.experiment_result_dir / "val",
            vis_modality=settings.vis_modality,
            save_visualizations=settings.save_visualizations,
            visualize_all_slices=settings.visualize_all_slices,
            use_connected_component=False,
            max_batches=settings.max_val_batches,
        )
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(summary["mean_dice"])
        else:
            scheduler.step(epoch)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice_loss": train_components["dice_loss"],
            "train_ce_loss": train_components["ce_loss"],
            "train_focal_loss": train_components["focal_loss"],
            "train_pathology_dice_loss": train_components["pathology_dice_loss"],
            "train_myocardium_prior_loss": train_components["myocardium_prior_loss"],
            "train_edema_dice_loss": train_components["edema_dice_loss"],
            "train_surface_loss": train_components["surface_loss"],
            "train_union_loss": train_components["union_loss"],
            "train_inclusiveness_loss": train_components["inclusiveness_loss"],
            "val_edema_dice": summary["edema_dice"],
            "val_scar_dice": summary["scar_dice"],
            "val_mean_dice": summary["mean_dice"],
            "val_edema_hd95": summary["edema_hd95"],
            "val_scar_hd95": summary["scar_hd95"],
            "val_mean_hd95": summary["mean_hd95"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(history_row)
        pd.DataFrame(history).to_csv(settings.experiment_log_dir / "history.csv", index=False)
        metrics_df.to_csv(settings.experiment_result_dir / "val" / f"epoch_{epoch:03d}_metrics.csv", index=False)

        recent_rows = history[-EARLY_STOPPING_MA_WINDOW:]
        moving_average_mean_dice = sum(row["val_mean_dice"] for row in recent_rows) / len(recent_rows)

        logger.info(
            "Epoch %03d | train_loss=%.4f | val_mean_dice=%.4f | ma%d_mean_dice=%.4f | edema_dice=%.4f | scar_dice=%.4f | mean_hd95=%.4f",
            epoch,
            train_loss,
            summary["mean_dice"],
            EARLY_STOPPING_MA_WINDOW,
            moving_average_mean_dice,
            summary["edema_dice"],
            summary["scar_dice"],
            summary["mean_hd95"],
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "settings": settings.to_dict(),
            "best_score": best_score,
            "best_edema_score": best_edema_score,
        }
        save_checkpoint(checkpoint_payload, settings.experiment_ckpt_dir / "last_model.pt")

        if summary["mean_dice"] > best_score:
            best_score = summary["mean_dice"]
            best_state = copy.deepcopy(model.state_dict())
            checkpoint_payload["best_score"] = best_score
            save_checkpoint(checkpoint_payload, settings.experiment_ckpt_dir / "best_model.pt")

        if summary["edema_dice"] > best_edema_score:
            best_edema_score = summary["edema_dice"]
            checkpoint_payload["best_edema_score"] = best_edema_score
            save_checkpoint(checkpoint_payload, settings.experiment_ckpt_dir / "best_edema_model.pt")

        if (
            settings.early_abort_epoch > 0
            and epoch == settings.early_abort_epoch
            and best_score < settings.early_abort_min_dice
        ):
            logger.info(
                "Early abort: best val_mean_dice=%.4f < threshold %.4f at epoch %d",
                best_score,
                settings.early_abort_min_dice,
                epoch,
            )
            break

        if moving_average_mean_dice > best_moving_average_score:
            best_moving_average_score = moving_average_mean_dice
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= settings.early_stopping_patience:
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_training_curves(history, settings.experiment_log_dir)
    logger.info("Training finished. Best validation mean Dice: %.4f", best_score)


NO_SCAR_CASES = {"Case3032"}


def _compute_summary(metrics_df: pd.DataFrame) -> dict[str, float]:
    has_scar = metrics_df[~metrics_df["case_id"].isin(NO_SCAR_CASES)]
    scar_dice = float(has_scar["scar_dice"].mean()) if not has_scar.empty else 0.0
    edema_dice = float(metrics_df["edema_dice"].mean())
    scar_hd95 = float(has_scar["scar_hd95"].mean()) if not has_scar.empty else 0.0
    edema_hd95 = float(metrics_df["edema_hd95"].mean())

    union_dice = float(metrics_df["union_dice"].mean()) if "union_dice" in metrics_df.columns else None
    union_hd95 = float(metrics_df["union_hd95"].mean()) if "union_hd95" in metrics_df.columns else None

    result = {
        "edema_dice": edema_dice,
        "scar_dice": scar_dice,
        "mean_dice": (edema_dice + scar_dice) / 2.0,
        "edema_hd95": edema_hd95,
        "scar_hd95": scar_hd95,
        "mean_hd95": (edema_hd95 + scar_hd95) / 2.0,
        "n_cases": len(metrics_df),
        "n_scar_cases": len(has_scar),
    }
    if union_dice is not None:
        result["union_dice"] = union_dice
    if union_hd95 is not None:
        result["union_hd95"] = union_hd95
    return result


def _log_scar_breakdown(metrics_df: pd.DataFrame, logger: logging.Logger) -> None:
    has_scar = metrics_df[~metrics_df["case_id"].isin(NO_SCAR_CASES)]
    no_scar = metrics_df[metrics_df["case_id"].isin(NO_SCAR_CASES)]
    if not has_scar.empty:
        logger.info(
            "Scar dice (excl. no-scar cases, n=%d): %.4f +/- %.4f",
            len(has_scar), has_scar["scar_dice"].mean(), has_scar["scar_dice"].std(),
        )
    if not no_scar.empty:
        for _, row in no_scar.iterrows():
            logger.info(
                "No-scar case %s: scar_dice=%.4f, edema_dice=%.4f",
                row["case_id"], row["scar_dice"], row["edema_dice"],
            )
    logger.info(
        "Scar dice (all cases, n=%d): %.4f +/- %.4f",
        len(metrics_df), metrics_df["scar_dice"].mean(), metrics_df["scar_dice"].std(),
    )


def test_pipeline(settings: Settings) -> None:
    ensure_directories(
        [
            settings.experiment_ckpt_dir,
            settings.experiment_log_dir,
            settings.experiment_result_dir,
        ]
    )
    logger = setup_logger(settings.experiment_log_dir / "test.log")
    save_json(settings.to_dict(), settings.experiment_log_dir / "test_config.json")
    seed_everything(settings.seed)

    dataset, loader = build_test_loader(settings)
    checkpoint_path = resolve_checkpoint_path(settings)
    checkpoint = torch.load(checkpoint_path, map_location=settings.device, weights_only=False)
    device = torch.device(settings.device)
    model = build_model(settings.model_variant, in_channels=settings.input_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    metrics_df, summary = evaluate_loader(
        model=model,
        dataset=dataset,
        loader=loader,
        device=device,
        use_tta=settings.enable_tta,
        output_dir=settings.experiment_result_dir / "test",
        vis_modality=settings.vis_modality,
        save_visualizations=settings.save_visualizations,
        visualize_all_slices=settings.visualize_all_slices,
        use_connected_component=settings.use_connected_component,
        use_edema_constraint=settings.use_edema_constraint,
        use_union_constraint=settings.use_union_constraint,
        max_cases=settings.max_test_cases,
    )
    metrics_path = settings.experiment_result_dir / "test" / "test_metrics.csv"
    summary_path = settings.experiment_result_dir / "test" / "test_summary.json"
    metrics_df.to_csv(metrics_path, index=False)
    save_json(summary, summary_path)
    logger.info("Test summary: %s", summary)
    _log_scar_breakdown(metrics_df, logger)


def ensemble_test_pipeline(settings: Settings) -> None:
    ensure_directories(
        [
            settings.experiment_ckpt_dir,
            settings.experiment_log_dir,
            settings.experiment_result_dir,
        ]
    )
    logger = setup_logger(settings.experiment_log_dir / "ensemble_test.log")
    save_json(settings.to_dict(), settings.experiment_log_dir / "ensemble_test_config.json")
    seed_everything(settings.seed)

    dataset, loader = build_test_loader(settings)
    checkpoint_paths = resolve_ensemble_checkpoint_paths(settings)
    ensemble_weights = resolve_ensemble_weights(settings, num_models=len(checkpoint_paths))
    model_variants = list(settings.ensemble_model_variants) if settings.ensemble_model_variants else [settings.model_variant] * len(checkpoint_paths)
    if len(model_variants) != len(checkpoint_paths):
        raise ValueError("The number of ensemble model variants must match the number of ensemble checkpoints.")
    device = torch.device(settings.device)
    models = [
        load_model_checkpoint(
            checkpoint_path=checkpoint_path,
            model_variant=variant,
            in_channels=settings.input_channels,
            device=device,
        )
        for checkpoint_path, variant in zip(checkpoint_paths, model_variants)
    ]
    logger.info("Loaded %d ensemble checkpoints.", len(models))
    for checkpoint_path in checkpoint_paths:
        logger.info("Ensemble member: %s", checkpoint_path)
    if ensemble_weights is not None:
        logger.info("Ensemble weights: %s", [round(weight, 4) for weight in ensemble_weights])

    case_predictions: dict[str, torch.Tensor] = {}
    case_union_probs: dict[str, list[tuple[int, np.ndarray]]] = {}
    case_seen = set()
    with torch.no_grad():
        progress = tqdm(loader, desc="Ensemble Eval", leave=False)
        for batch in progress:
            images = batch["image"].to(device)
            probabilities, union_probs = predict_with_ensemble(
                models,
                images,
                use_tta=settings.enable_tta,
                weights=ensemble_weights,
            )
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
            union_np = union_probs.cpu().numpy() if union_probs is not None else None

            case_ids = batch["case_id"]
            slice_indices = batch["slice_idx"].tolist()
            for sample_index, case_id in enumerate(case_ids):
                if settings.max_test_cases is not None and case_id not in case_seen and len(case_seen) >= settings.max_test_cases:
                    continue
                case_seen.add(case_id)
                case_store = dataset.case_data[case_id]
                if case_id not in case_predictions:
                    case_predictions[case_id] = torch.zeros_like(torch.from_numpy(case_store["mask_full"]))
                crop_meta = case_store["crop_metas"][slice_indices[sample_index]]
                restored_slice = restore_cropped_array(predictions[sample_index], crop_meta)
                case_predictions[case_id][:, :, slice_indices[sample_index]] = torch.from_numpy(restored_slice)
                if union_np is not None:
                    if case_id not in case_union_probs:
                        case_union_probs[case_id] = []
                    restored_union = restore_cropped_array(union_np[sample_index], crop_meta).astype(np.float32)
                    case_union_probs[case_id].append((slice_indices[sample_index], restored_union))

    rows = []
    output_dir = settings.experiment_result_dir / "test_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_id, prediction_tensor in sorted(case_predictions.items()):
        case_store = dataset.case_data[case_id]
        prediction = prediction_tensor.numpy().astype(case_store["mask_full"].dtype)
        if settings.use_connected_component:
            prediction = keep_largest_connected_component(prediction)
        if settings.use_edema_constraint:
            prediction = constrain_edema_to_myocardium(prediction)
        if settings.use_union_constraint and case_id in case_union_probs:
            union_volume = np.zeros_like(prediction, dtype=np.float32)
            for si, uslice in case_union_probs[case_id]:
                union_volume[:, :, si] = uslice
            union_mask = (union_volume >= 0.5)
            prediction = constrain_predictions_to_union_mask(prediction, union_mask)
        target = case_store["mask_full"]
        metrics = evaluate_case(prediction, target, spacing=case_store["spacing"])
        rows.append({"case_id": case_id, **metrics})
        if settings.save_visualizations:
            save_case_visualizations(
                case_id=case_id,
                base_volume=case_store["raw_modalities"][settings.vis_modality],
                gt_volume=target,
                pred_volume=prediction,
                output_dir=output_dir / "visualizations",
                visualize_all_slices=settings.visualize_all_slices,
            )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        raise RuntimeError("No cases were evaluated in ensemble test. Check max_test_cases settings.")
    summary = _compute_summary(metrics_df)
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    save_json(summary, output_dir / "test_summary.json")
    logger.info("Ensemble test summary: %s", summary)
    _log_scar_breakdown(metrics_df, logger)


def run_3d_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedSegmentationLoss,
    optimizer: Adam,
    device: torch.device,
    epoch: int = 0,
    total_epochs: int = 300,
) -> tuple[float, dict[str, float]]:
    model.train()
    loss_meter = AverageMeter()
    progress = tqdm(loader, desc="Train3D", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        b, c, d, h, w = logits.shape
        logits_2d = logits.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        masks_2d = masks.reshape(b * d, h, w)
        loss, components = criterion(logits_2d, masks_2d, epoch=epoch, total_epochs=total_epochs)
        loss.backward()
        optimizer.step()
        loss_meter.update(components["total_loss"], b)
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}")
    return loss_meter.avg, {"total_loss": loss_meter.avg}


def evaluate_3d_loader(
    model: torch.nn.Module,
    dataset,
    loader: DataLoader,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val3D", leave=False):
            images = batch["image"].to(device)
            masks_np = batch["mask"].numpy()
            case_ids = batch["case_id"]
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            for i in range(len(case_ids)):
                pred_vol = preds[i]
                gt_vol = masks_np[i]
                metrics = evaluate_case(pred_vol, gt_vol, spacing=(1.0, 1.0, 1.0))
                rows.append({"case_id": case_ids[i], **metrics})
    df = pd.DataFrame(rows)
    summary = _compute_summary(df)
    return df, summary
