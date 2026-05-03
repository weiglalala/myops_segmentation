from __future__ import annotations

import os
from pathlib import Path

import matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg", force=True)
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Settings
from src.constants import DEFAULT_MODALITIES, EDEMA_CLASS_INDEX, MYOCARDIUM_CLASS_INDICES, SCAR_CLASS_INDEX
from src.engine import load_model_checkpoint, predict_with_ensemble, predict_with_tta
from src.metrics import evaluate_case
from src.models import build_model
from src.roi import restore_roi_crop
from src.two_stage import (
    Stage1BinaryDataset,
    Stage2ROISliceDataset,
    _build_stage1_test_loader,
    _evaluate_stage1_loader,
)
from src.utils import ensure_directories, save_json, seed_everything, setup_logger
from src.visualization import save_case_visualizations


def _collect_stage2_predictions(
    models: list[torch.nn.Module],
    dataset: Stage2ROISliceDataset,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
    ensemble_weights: list[float] | None = None,
    num_classes: int = 6,
) -> dict[str, np.ndarray]:
    for m in models:
        m.eval()
    case_predictions: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        progress = tqdm(loader, desc="Stage2 Predict", leave=False)
        for batch in progress:
            images = batch["image"].to(device)
            if len(models) > 1:
                probabilities, _ = predict_with_ensemble(models, images, use_tta=use_tta, weights=ensemble_weights)
            else:
                probabilities, _ = predict_with_tta(models[0], images, use_tta=use_tta)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

            case_ids = batch["case_id"]
            slice_indices = batch["slice_idx"].tolist()
            for i, case_id in enumerate(case_ids):
                case_store = dataset.case_data[case_id]
                if case_id not in case_predictions:
                    case_predictions[case_id] = torch.zeros_like(torch.from_numpy(case_store["mask_full"]))
                roi_meta = case_store["roi_metas"][slice_indices[i]]
                restored = restore_roi_crop(predictions[i], roi_meta)
                if num_classes == 2:
                    remap = np.zeros_like(restored)
                    remap[restored == 1] = 5
                    restored = remap
                elif num_classes == 3:
                    remap = np.zeros_like(restored)
                    remap[restored == 1] = 4
                    remap[restored == 2] = 5
                    restored = remap
                case_predictions[case_id][:, :, slice_indices[i]] = torch.from_numpy(restored)

    return {cid: t.numpy().astype(np.int64) for cid, t in case_predictions.items()}


def _constrain_edema_to_myocardium(
    prediction: np.ndarray,
    anatomy_mask: np.ndarray | None = None,
) -> np.ndarray:
    constrained = prediction.copy()
    for s in range(constrained.shape[2]):
        sl = constrained[:, :, s]
        edema_mask = sl == EDEMA_CLASS_INDEX
        if not edema_mask.any():
            continue
        if anatomy_mask is not None:
            myo_mask = np.isin(anatomy_mask[:, :, s], MYOCARDIUM_CLASS_INDICES)
        else:
            myo_mask = np.isin(sl, MYOCARDIUM_CLASS_INDICES)
        expanded = ndimage.binary_dilation(myo_mask, iterations=3)
        sl[edema_mask & ~expanded] = 0
        constrained[:, :, s] = sl
    return constrained


def _remove_small_edema_components(prediction: np.ndarray, min_size: int = 50) -> np.ndarray:
    cleaned = prediction.copy()
    edema_mask = (cleaned == EDEMA_CLASS_INDEX)
    if not edema_mask.any():
        return cleaned
    labeled, num = ndimage.label(edema_mask)
    if num <= 0:
        return cleaned
    sizes = np.bincount(labeled.ravel())
    for comp_id in range(1, num + 1):
        if sizes[comp_id] < min_size:
            cleaned[labeled == comp_id] = 0
    return cleaned


def _fuse_predictions(
    scar_pred: np.ndarray,
    edema_pred: np.ndarray,
    apply_edema_constraint: bool = True,
    edema_min_size: int = 50,
    anatomy_mask: np.ndarray | None = None,
) -> np.ndarray:
    fused = scar_pred.copy()
    fused[edema_pred == EDEMA_CLASS_INDEX] = EDEMA_CLASS_INDEX
    fused[scar_pred == SCAR_CLASS_INDEX] = SCAR_CLASS_INDEX
    if apply_edema_constraint:
        fused = _constrain_edema_to_myocardium(fused, anatomy_mask=anatomy_mask)
    if edema_min_size > 0:
        fused = _remove_small_edema_components(fused, min_size=edema_min_size)
    return fused


def _build_stage2_test_loader_for_fusion(
    settings: Settings,
    roi_masks: dict[str, np.ndarray],
    use_coarse_input: bool,
    num_classes: int = 6,
) -> tuple[Stage2ROISliceDataset, DataLoader]:
    from src.data import sorted_case_dirs
    case_dirs = sorted_case_dirs(settings.test_dir)
    dataset = Stage2ROISliceDataset(
        case_dirs,
        crop_size=settings.crop_size,
        input_mode=settings.stage2_input_mode,
        roi_padding=settings.roi_padding,
        roi_mode=settings.stage2_roi_mode,
        target_type=settings.stage1_target_type,
        roi_mask_source="pred",
        roi_masks=roi_masks,
        use_t2_clahe=settings.use_t2_clahe,
        t2_clahe_clip_limit=settings.t2_clahe_clip_limit,
        t2_clahe_tile_grid_size=settings.t2_clahe_tile_grid_size,
        augment=False,
        rotation_degrees=settings.rotation_degrees,
        enable_elastic=False,
        use_coarse_input=use_coarse_input,
        num_classes=num_classes,
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=settings.device.startswith("cuda"),
    )
    return dataset, loader


def fusion_test_pipeline(settings: Settings) -> None:
    output_dir = settings.experiment_result_dir / "fusion_test"
    log_dir = settings.experiment_log_dir / "fusion_test"
    ensure_directories([output_dir, log_dir])
    logger = setup_logger(log_dir / "fusion_test.log")
    save_json(settings.to_dict(), log_dir / "fusion_config.json")
    seed_everything(settings.seed)
    device = torch.device(settings.device)

    stage1_ckpt_path = settings.stage1_checkpoint_path
    if stage1_ckpt_path is None:
        raise FileNotFoundError("Provide --stage1-checkpoint-path for fusion pipeline.")
    stage1_dataset, stage1_loader = _build_stage1_test_loader(settings)
    stage1_ckpt = torch.load(stage1_ckpt_path, map_location=settings.device, weights_only=False)
    stage1_model = build_model(
        settings.stage1_model_variant,
        in_channels=settings.stage1_input_channels,
        num_classes=settings.stage1_num_classes,
    ).to(device)
    stage1_model.load_state_dict(stage1_ckpt["model_state_dict"])
    logger.info("Loaded Stage1: %s", stage1_ckpt_path)

    _, stage1_summary, stage1_predictions = _evaluate_stage1_loader(
        model=stage1_model,
        dataset=stage1_dataset,
        loader=stage1_loader,
        device=device,
        output_dir=output_dir / "stage1",
        num_classes=settings.stage1_num_classes,
    )
    logger.info("Stage1 summary: %s", stage1_summary)

    if not settings.scar_checkpoint_paths:
        raise FileNotFoundError("Provide --scar-checkpoint-paths for fusion pipeline.")
    scar_variants = list(settings.scar_model_variants) or [settings.model_variant] * len(settings.scar_checkpoint_paths)
    scar_in_ch = settings.stage2_input_channels
    scar_models = [
        load_model_checkpoint(p, v, scar_in_ch, device, settings.scar_num_classes)
        for p, v in zip(settings.scar_checkpoint_paths, scar_variants)
    ]
    logger.info("Loaded %d scar models (num_classes=%d)", len(scar_models), settings.scar_num_classes)

    scar_dataset, scar_loader = _build_stage2_test_loader_for_fusion(
        settings, stage1_predictions, use_coarse_input=settings.use_coarse_input,
        num_classes=settings.scar_num_classes,
    )
    scar_preds = _collect_stage2_predictions(
        scar_models, scar_dataset, scar_loader, device,
        use_tta=settings.enable_tta,
        num_classes=settings.scar_num_classes,
    )
    logger.info("Scar predictions collected for %d cases", len(scar_preds))

    if not settings.edema_checkpoint_paths:
        raise FileNotFoundError("Provide --edema-checkpoint-paths for fusion pipeline.")
    edema_variants = list(settings.edema_model_variants) or [settings.model_variant] * len(settings.edema_checkpoint_paths)
    edema_in_ch = settings.stage2_input_channels
    edema_models = [
        load_model_checkpoint(p, v, edema_in_ch, device, settings.edema_num_classes)
        for p, v in zip(settings.edema_checkpoint_paths, edema_variants)
    ]
    logger.info("Loaded %d edema models (num_classes=%d)", len(edema_models), settings.edema_num_classes)

    edema_dataset, edema_loader = _build_stage2_test_loader_for_fusion(
        settings, stage1_predictions, use_coarse_input=settings.use_coarse_input,
        num_classes=settings.edema_num_classes,
    )
    edema_preds = _collect_stage2_predictions(
        edema_models, edema_dataset, edema_loader, device,
        use_tta=settings.enable_tta,
        num_classes=settings.edema_num_classes,
    )
    logger.info("Edema predictions collected for %d cases", len(edema_preds))

    rows = []
    vis_dir = output_dir / "visualizations"
    for case_id in sorted(scar_preds.keys()):
        scar_pred = scar_preds[case_id]
        edema_pred = edema_preds[case_id]
        fused = _fuse_predictions(
            scar_pred, edema_pred,
            apply_edema_constraint=settings.fusion_edema_constraint,
            edema_min_size=settings.fusion_edema_min_size,
            anatomy_mask=stage1_predictions.get(case_id),
        )
        case_store = scar_dataset.case_data[case_id]
        target = case_store["mask_full"]
        metrics = evaluate_case(fused, target, spacing=case_store["spacing"])
        rows.append({"case_id": case_id, **metrics})

        scar_only_metrics = evaluate_case(scar_pred, target, spacing=case_store["spacing"])
        edema_only_metrics = evaluate_case(edema_pred, target, spacing=case_store["spacing"])
        logger.info(
            "%s | fused: edema=%.3f scar=%.3f | scar_model: edema=%.3f scar=%.3f | edema_model: edema=%.3f scar=%.3f",
            case_id,
            metrics["edema_dice"], metrics["scar_dice"],
            scar_only_metrics["edema_dice"], scar_only_metrics["scar_dice"],
            edema_only_metrics["edema_dice"], edema_only_metrics["scar_dice"],
        )

        if settings.save_visualizations:
            save_case_visualizations(
                case_id=case_id,
                base_volume=case_store["raw_modalities"][settings.vis_modality],
                gt_volume=target,
                pred_volume=fused,
                output_dir=vis_dir,
                visualize_all_slices=settings.visualize_all_slices,
            )

    metrics_df = pd.DataFrame(rows)
    summary = {
        "edema_dice": float(metrics_df["edema_dice"].mean()),
        "scar_dice": float(metrics_df["scar_dice"].mean()),
        "mean_dice": float(metrics_df["mean_dice"].mean()),
        "edema_hd95": float(metrics_df["edema_hd95"].mean()),
        "scar_hd95": float(metrics_df["scar_hd95"].mean()),
        "mean_hd95": float(metrics_df["mean_hd95"].mean()),
    }
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    save_json(summary, output_dir / "test_summary.json")
    logger.info("Fusion test summary: %s", summary)
