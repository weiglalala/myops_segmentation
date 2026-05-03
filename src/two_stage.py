from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg", force=True)
import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import ndimage
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.config import Settings
from src.constants import DEFAULT_MODALITIES
from src.data import (
    SliceAugmenter,
    _apply_t2_clahe,
    build_sample_weights,
    center_crop_array,
    compute_center_crop_meta,
    map_labels,
    normalize_nonzero_percentile,
    normalize_roi_percentile,
    restore_center_crop,
    sorted_case_dirs,
    split_case_dirs,
    split_case_dirs_kfold,
    summarize_sample_weights,
)
from src.engine import EARLY_STOPPING_MA_WINDOW, _log_scar_breakdown, load_model_checkpoint, predict_with_ensemble, predict_with_tta, run_train_epoch, save_checkpoint
from src.losses import CombinedSegmentationLoss, derive_class_weights, derive_edema_expert_class_weights
from src.metrics import dice_binary, evaluate_case, hd95_binary
from src.models import build_model
from src.roi import compute_roi_crop_meta, compute_roi_crop_resize_meta, crop_with_roi_meta, restore_roi_crop
from src.utils import ensure_directories, save_json, seed_everything, setup_logger
from src.visualization import save_case_visualizations, save_training_curves


MYOCARDIUM_TARGET_CLASSES = (1, 4, 5)


def _extract_roi_target(mask_full: np.ndarray, target_type: str) -> np.ndarray:
    if target_type == "foreground":
        return (mask_full > 0).astype(np.uint8)
    if target_type == "myocardium":
        return np.isin(mask_full, MYOCARDIUM_TARGET_CLASSES).astype(np.uint8)
    if target_type == "anatomy":
        result = np.zeros_like(mask_full, dtype=np.uint8)
        result[np.isin(mask_full, MYOCARDIUM_TARGET_CLASSES)] = 1
        result[mask_full == 2] = 2
        result[mask_full == 3] = 3
        return result
    raise ValueError(f"Unsupported target_type: {target_type}")


def _resize_hwc(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    if image.ndim == 3 and image.shape[2] == 1:
        resized = cv2.resize(image[:, :, 0], (width, height), interpolation=cv2.INTER_LINEAR)
        return resized[:, :, np.newaxis]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _resize_hw(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)


def _keep_largest_component_3d(mask_volume: np.ndarray) -> np.ndarray:
    binary = mask_volume.astype(bool)
    if not np.any(binary):
        return binary.astype(np.uint8)
    labeled, num_components = ndimage.label(binary)
    if num_components <= 1:
        return binary.astype(np.uint8)
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest_label = int(component_sizes.argmax())
    return (labeled == largest_label).astype(np.uint8)


def _fill_holes_per_slice(mask_volume: np.ndarray) -> np.ndarray:
    filled = np.zeros_like(mask_volume, dtype=np.uint8)
    for slice_idx in range(mask_volume.shape[2]):
        filled[:, :, slice_idx] = ndimage.binary_fill_holes(mask_volume[:, :, slice_idx].astype(bool)).astype(np.uint8)
    return filled


def _load_case_bundle(
    case_dir: Path,
    use_t2_clahe: bool = False,
    t2_clahe_clip_limit: float = 2.0,
    t2_clahe_tile_grid_size: int = 8,
) -> dict:
    case_id = case_dir.name
    modalities = {}
    spacing = None
    shape = None
    for modality in DEFAULT_MODALITIES:
        path = next(case_dir.glob(f"*_{modality}.nii.gz"))
        image = nib.load(str(path))
        data = image.get_fdata().astype(np.float32)
        modalities[modality] = data
        spacing = tuple(float(x) for x in image.header.get_zooms())
        shape = data.shape

    gd_path = next(case_dir.glob("*_gd.nii.gz"))
    gd_img = nib.load(str(gd_path))
    gd = gd_img.get_fdata().astype(np.int32)
    if shape != gd.shape:
        raise ValueError(f"Shape mismatch in case {case_id}: {shape} vs {gd.shape}")

    preprocessed_modalities = {key: value.astype(np.float32) for key, value in modalities.items()}
    if use_t2_clahe:
        preprocessed_modalities["T2"] = _apply_t2_clahe(
            preprocessed_modalities["T2"],
            clip_limit=t2_clahe_clip_limit,
            tile_grid_size=t2_clahe_tile_grid_size,
        )
    normalized_modalities = {
        key: normalize_nonzero_percentile(value)
        for key, value in preprocessed_modalities.items()
    }
    mask_full = map_labels(gd).astype(np.int64)
    return {
        "case_id": case_id,
        "normalized_modalities": normalized_modalities,
        "raw_modalities": {key: value.astype(np.float32) for key, value in modalities.items()},
        "mask_full": mask_full,
        "spacing": spacing,
        "orig_shape": shape,
    }


def _compute_stage1_class_weights(dataset: "Stage1BinaryDataset", num_classes: int) -> torch.Tensor:
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for case in dataset.case_data.values():
        mask_stack = case["mask_stack"]
        for c in range(num_classes):
            class_counts[c] += int((mask_stack == c).sum())
    total = max(int(class_counts.sum()), 1)
    frequencies = class_counts.astype(np.float32) / total
    weights = 1.0 / np.sqrt(frequencies + 1e-8)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _save_binary_training_curves(history: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_foreground_dice = [row["val_foreground_dice"] for row in history]

    figure = Figure(figsize=(12, 4.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    axes[0].plot(epochs, train_loss, label="train_loss", color="tab:red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Stage1 Training Loss")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(epochs, val_foreground_dice, label="val_foreground_dice", color="tab:blue")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Stage1 Validation Dice")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    figure.clear()


class Stage1BinaryDataset(Dataset):
    def __init__(
        self,
        case_dirs: list[Path],
        crop_size: tuple[int, int],
        input_modality: str,
        target_type: str,
        preprocess_mode: str,
        augment: bool = False,
        rotation_degrees: float = 15.0,
        enable_elastic: bool = True,
    ) -> None:
        self.case_dirs = case_dirs
        self.crop_size = crop_size
        self.input_modality = input_modality
        self.target_type = target_type
        if preprocess_mode not in {"center_crop", "resize_full"}:
            raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")
        self.preprocess_mode = preprocess_mode
        self.augment = augment
        self.augmenter = SliceAugmenter(rotation_degrees=rotation_degrees, enable_elastic=enable_elastic)
        self.case_data: dict[str, dict] = {}
        self.samples: list[dict] = []
        self._load_cases()

    def _load_cases(self) -> None:
        crop_h, crop_w = self.crop_size
        for case_dir in self.case_dirs:
            bundle = _load_case_bundle(case_dir)
            case_id = bundle["case_id"]
            shape = bundle["orig_shape"]
            binary_mask_full = _extract_roi_target(bundle["mask_full"], self.target_type).astype(np.int64)

            if self.input_modality == "all":
                volume_stack = np.stack(
                    [bundle["normalized_modalities"][modality] for modality in DEFAULT_MODALITIES],
                    axis=0,
                ).transpose(3, 0, 1, 2)
            else:
                modality_volume = bundle["normalized_modalities"][self.input_modality]
                volume_stack = modality_volume[np.newaxis, ...].transpose(3, 0, 1, 2)

            if self.preprocess_mode == "center_crop":
                crop_meta = compute_center_crop_meta(shape[0], shape[1], crop_h, crop_w)
                image_slices = [
                    center_crop_array(np.transpose(volume_stack[slice_idx], (1, 2, 0)), crop_meta, fill_value=0.0)
                    for slice_idx in range(volume_stack.shape[0])
                ]
                image_stack = np.stack([np.transpose(image_slice, (2, 0, 1)) for image_slice in image_slices], axis=0)
                mask_stack = center_crop_array(binary_mask_full, crop_meta, fill_value=0).transpose(2, 0, 1).astype(np.int64)
            else:
                image_stack = np.stack(
                    [
                        np.transpose(
                            _resize_hwc(np.transpose(volume_stack[slice_idx], (1, 2, 0)), (crop_h, crop_w)),
                            (2, 0, 1),
                        )
                        for slice_idx in range(volume_stack.shape[0])
                    ],
                    axis=0,
                ).astype(np.float32)
                mask_stack = np.stack(
                    [
                        _resize_hw(binary_mask_full[:, :, slice_idx], (crop_h, crop_w))
                        for slice_idx in range(binary_mask_full.shape[2])
                    ],
                    axis=0,
                ).astype(np.int64)
                crop_meta = None

            self.case_data[case_id] = {
                "case_id": case_id,
                "image_stack": image_stack.astype(np.float32),
                "mask_stack": mask_stack.astype(np.int64),
                "mask_full": binary_mask_full.astype(np.int64),
                "crop_meta": crop_meta,
                "preprocess_mode": self.preprocess_mode,
                "orig_shape": shape,
                "spacing": bundle["spacing"],
                "raw_modalities": bundle["raw_modalities"],
            }
            for slice_idx in range(image_stack.shape[0]):
                self.samples.append({"case_id": case_id, "slice_idx": slice_idx})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        case = self.case_data[sample["case_id"]]
        image = case["image_stack"][sample["slice_idx"]].copy()
        mask = case["mask_stack"][sample["slice_idx"]].copy()
        if self.augment:
            image, mask = self.augmenter(image, mask)
        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "case_id": sample["case_id"],
            "slice_idx": sample["slice_idx"],
        }


class Stage2ROISliceDataset(Dataset):
    def __init__(
        self,
        case_dirs: list[Path],
        crop_size: tuple[int, int],
        input_mode: str,
        roi_padding: int,
        roi_mode: str,
        target_type: str,
        roi_mask_source: str,
        roi_masks: dict[str, np.ndarray] | None = None,
        coarse_masks: dict[str, np.ndarray] | None = None,
        use_t2_clahe: bool = False,
        t2_clahe_clip_limit: float = 2.0,
        t2_clahe_tile_grid_size: int = 8,
        augment: bool = False,
        rotation_degrees: float = 15.0,
        enable_elastic: bool = True,
        enable_advanced_augment: bool = False,
        enable_copy_paste: bool = False,
        use_coarse_input: bool = False,
        use_t2_roi_norm: bool = False,
        num_classes: int = 6,
    ) -> None:
        self.case_dirs = case_dirs
        self.crop_size = crop_size
        if input_mode not in {"2d", "2p5d"}:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
        self.input_mode = input_mode
        if roi_mask_source not in {"gt", "pred"}:
            raise ValueError(f"Unsupported roi_mask_source: {roi_mask_source}")
        if roi_mode not in {"slice", "case"}:
            raise ValueError(f"Unsupported roi_mode: {roi_mode}")
        self.roi_mode = roi_mode
        self.target_type = target_type
        self.roi_mask_source = roi_mask_source
        self.roi_masks = roi_masks or {}
        self.coarse_masks = coarse_masks or {}
        self.roi_padding = roi_padding
        self.use_t2_clahe = use_t2_clahe
        self.t2_clahe_clip_limit = t2_clahe_clip_limit
        self.t2_clahe_tile_grid_size = t2_clahe_tile_grid_size
        self.use_coarse_input = use_coarse_input
        self.use_t2_roi_norm = use_t2_roi_norm
        self.num_classes = num_classes
        self.augment = augment
        self.enable_copy_paste = enable_copy_paste and augment
        self.augmenter = SliceAugmenter(
            rotation_degrees=rotation_degrees,
            enable_elastic=enable_elastic,
            enable_advanced_augment=enable_advanced_augment,
        )
        self.case_data: dict[str, dict] = {}
        self.samples: list[dict] = []
        self._load_cases()

    def _load_cases(self) -> None:
        for case_dir in self.case_dirs:
            bundle = _load_case_bundle(
                case_dir,
                use_t2_clahe=self.use_t2_clahe,
                t2_clahe_clip_limit=self.t2_clahe_clip_limit,
                t2_clahe_tile_grid_size=self.t2_clahe_tile_grid_size,
            )
            case_id = bundle["case_id"]
            image_stack_full = np.stack(
                [bundle["normalized_modalities"][modality] for modality in DEFAULT_MODALITIES],
                axis=0,
            ).transpose(3, 0, 1, 2)
            if self.use_t2_roi_norm:
                coarse_vol = self.coarse_masks.get(case_id)
                if coarse_vol is None and self.roi_mask_source == "gt":
                    coarse_vol = _extract_roi_target(bundle["mask_full"], self.target_type)
                if coarse_vol is not None:
                    myo_roi = (coarse_vol == 1)
                    t2_normed = normalize_roi_percentile(bundle["raw_modalities"]["T2"], myo_roi)
                    image_stack_full[:, 1, :, :] = t2_normed.transpose(2, 0, 1)
            mask_stack_full = bundle["mask_full"].transpose(2, 0, 1)
            if self.roi_mask_source == "gt":
                roi_volume = _extract_roi_target(bundle["mask_full"], self.target_type).astype(np.uint8)
                roi_binary = (roi_volume > 0).astype(np.uint8)
            else:
                if case_id not in self.roi_masks:
                    raise ValueError(f"Missing predicted ROI mask for case {case_id}")
                roi_volume = self.roi_masks[case_id].astype(np.uint8)
                if roi_volume.shape != bundle["mask_full"].shape:
                    raise ValueError(f"ROI mask shape mismatch for case {case_id}: {roi_volume.shape} vs {bundle['mask_full'].shape}")
                roi_binary = (roi_volume > 0).astype(np.uint8)
                roi_binary = _fill_holes_per_slice(_keep_largest_component_3d(roi_binary))

            if self.roi_mode == "case":
                case_mask = np.any(roi_binary > 0, axis=2).astype(np.uint8)
                case_roi_meta = compute_roi_crop_resize_meta(
                    mask=case_mask,
                    target_size=self.crop_size,
                    padding=self.roi_padding,
                    fallback_shape=bundle["mask_full"].shape[:2],
                )
                roi_metas = [case_roi_meta for _ in range(bundle["mask_full"].shape[2])]
            else:
                roi_metas = [
                    compute_roi_crop_resize_meta(
                        mask=roi_binary[:, :, slice_idx],
                        target_size=self.crop_size,
                        padding=self.roi_padding,
                        fallback_shape=bundle["mask_full"].shape[:2],
                    )
                    for slice_idx in range(bundle["mask_full"].shape[2])
                ]

            self.case_data[case_id] = {
                "case_id": case_id,
                "image_stack_full": image_stack_full.astype(np.float32),
                "mask_stack_full": mask_stack_full.astype(np.int64),
                "mask_full": bundle["mask_full"].astype(np.int64),
                "raw_modalities": bundle["raw_modalities"],
                "spacing": bundle["spacing"],
                "roi_metas": roi_metas,
                "orig_shape": bundle["orig_shape"],
            }
            if self.use_coarse_input:
                if case_id in self.coarse_masks:
                    coarse_vol = self.coarse_masks[case_id].astype(np.uint8)
                else:
                    coarse_vol = roi_volume.astype(np.uint8)
                self.case_data[case_id]["coarse_mask_stack"] = coarse_vol.transpose(2, 0, 1)
            case_edema_pixels = int((mask_stack_full == 4).sum())

            for slice_idx in range(image_stack_full.shape[0]):
                mask_slice = mask_stack_full[slice_idx]
                edema_pixels = int((mask_slice == 4).sum())
                self.samples.append(
                    {
                        "case_id": case_id,
                        "slice_idx": slice_idx,
                        "edema_pixels": edema_pixels,
                        "case_edema_pixels": case_edema_pixels,
                        "has_edema": edema_pixels > 0,
                        "has_pathology": bool(np.isin(mask_slice, (4, 5)).any()),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _build_image_input(self, image_stack_full: np.ndarray, slice_idx: int) -> np.ndarray:
        if self.input_mode == "2d":
            return image_stack_full[slice_idx].copy()
        max_idx = image_stack_full.shape[0] - 1
        neighbor_indices = [max(0, slice_idx - 1), slice_idx, min(max_idx, slice_idx + 1)]
        return np.concatenate([image_stack_full[idx] for idx in neighbor_indices], axis=0).astype(np.float32)

    def _copy_paste_pathology(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pathology_samples = [s for s in self.samples if s["has_pathology"] and s["case_id"] != ""]
        if not pathology_samples:
            return image, mask
        donor_sample = random.choice(pathology_samples)
        donor_case = self.case_data[donor_sample["case_id"]]
        donor_roi_meta = donor_case["roi_metas"][donor_sample["slice_idx"]]
        donor_img = self._build_image_input(donor_case["image_stack_full"], donor_sample["slice_idx"])
        donor_img_hwc = np.transpose(donor_img, (1, 2, 0))
        donor_img_crop = crop_with_roi_meta(donor_img_hwc, donor_roi_meta, fill_value=0.0)
        donor_img_crop = np.transpose(donor_img_crop, (2, 0, 1)).astype(np.float32)
        if self.use_coarse_input and "coarse_mask_stack" in donor_case:
            donor_coarse = donor_case["coarse_mask_stack"][donor_sample["slice_idx"]]
            donor_coarse_crop = crop_with_roi_meta(
                donor_coarse.astype(np.float32), donor_roi_meta, fill_value=0.0, is_mask=True,
            )
            donor_coarse_crop = np.round(donor_coarse_crop).astype(np.int64)
            donor_onehot = np.stack([
                (donor_coarse_crop == 1).astype(np.float32),
                (donor_coarse_crop == 2).astype(np.float32),
                (donor_coarse_crop == 3).astype(np.float32),
            ], axis=0)
            donor_img_crop = np.concatenate([donor_img_crop, donor_onehot], axis=0)
        donor_mask_crop = crop_with_roi_meta(
            donor_case["mask_stack_full"][donor_sample["slice_idx"]],
            donor_roi_meta, fill_value=0, is_mask=True,
        ).astype(np.int64)
        if self.num_classes == 2:
            dm = np.zeros_like(donor_mask_crop)
            dm[donor_mask_crop == 5] = 1
            donor_mask_crop = dm
        elif self.num_classes == 3:
            dm = np.zeros_like(donor_mask_crop)
            dm[donor_mask_crop == 4] = 1
            dm[donor_mask_crop == 5] = 2
            donor_mask_crop = dm
        pathology_region = donor_mask_crop > 0
        if not pathology_region.any():
            return image, mask
        image[:, pathology_region] = donor_img_crop[:, pathology_region]
        mask[pathology_region] = donor_mask_crop[pathology_region]
        return image, mask

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        case = self.case_data[sample["case_id"]]
        roi_meta = case["roi_metas"][sample["slice_idx"]]
        image = self._build_image_input(case["image_stack_full"], sample["slice_idx"])
        image_hwc = np.transpose(image, (1, 2, 0))
        image_crop = crop_with_roi_meta(image_hwc, roi_meta, fill_value=0.0)
        image = np.transpose(image_crop, (2, 0, 1)).astype(np.float32)
        if self.use_coarse_input:
            coarse_slice = case["coarse_mask_stack"][sample["slice_idx"]]
            coarse_crop = crop_with_roi_meta(coarse_slice.astype(np.float32), roi_meta, fill_value=0.0, is_mask=True)
            coarse_crop = np.round(coarse_crop).astype(np.int64)
            onehot_channels = np.stack([
                (coarse_crop == 1).astype(np.float32),
                (coarse_crop == 2).astype(np.float32),
                (coarse_crop == 3).astype(np.float32),
            ], axis=0)
            image = np.concatenate([image, onehot_channels], axis=0)
        mask = crop_with_roi_meta(case["mask_stack_full"][sample["slice_idx"]], roi_meta, fill_value=0, is_mask=True).astype(np.int64)
        if self.num_classes == 2:
            remapped = np.zeros_like(mask)
            remapped[mask == 5] = 1
            mask = remapped
        elif self.num_classes == 3:
            remapped = np.zeros_like(mask)
            remapped[mask == 4] = 1
            remapped[mask == 5] = 2
            mask = remapped
        if self.enable_copy_paste and random.random() < 0.3:
            image, mask = self._copy_paste_pathology(image, mask)
        if self.augment:
            image, mask = self.augmenter(image, mask)
        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "case_id": sample["case_id"],
            "slice_idx": sample["slice_idx"],
            "has_pathology": sample["has_pathology"],
            "has_edema": sample["has_edema"],
            "edema_pixels": sample["edema_pixels"],
        }


def _two_stage_dirs(settings: Settings) -> dict[str, Path]:
    return {
        "stage1_ckpt": settings.experiment_ckpt_dir / "stage1",
        "stage1_log": settings.experiment_log_dir / "stage1",
        "stage1_result": settings.experiment_result_dir / "stage1",
        "stage2_ckpt": settings.experiment_ckpt_dir / "stage2",
        "stage2_log": settings.experiment_log_dir / "stage2",
        "stage2_result": settings.experiment_result_dir / "stage2",
        "two_stage_log": settings.experiment_log_dir / "two_stage_test",
        "two_stage_result": settings.experiment_result_dir / "two_stage_test",
    }


def _build_stage1_dataloaders(settings: Settings) -> tuple[Stage1BinaryDataset, Stage1BinaryDataset, DataLoader, DataLoader]:
    case_dirs = sorted_case_dirs(settings.train_dir)
    train_dirs, val_dirs = split_case_dirs(
        case_dirs,
        val_ratio=settings.val_ratio,
        seed=settings.seed,
        fixed_val_cases=settings.fixed_val_cases,
    )
    train_dataset = Stage1BinaryDataset(
        train_dirs,
        crop_size=settings.crop_size,
        input_modality=settings.stage1_modality,
        target_type=settings.stage1_target_type,
        preprocess_mode=settings.stage1_preprocess_mode,
        augment=True,
        rotation_degrees=settings.rotation_degrees,
        enable_elastic=settings.enable_elastic,
    )
    val_dataset = Stage1BinaryDataset(
        val_dirs,
        crop_size=settings.crop_size,
        input_modality=settings.stage1_modality,
        target_type=settings.stage1_target_type,
        preprocess_mode=settings.stage1_preprocess_mode,
        augment=False,
        rotation_degrees=settings.rotation_degrees,
        enable_elastic=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
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


def _build_stage2_dataloaders(settings: Settings) -> tuple[Stage2ROISliceDataset, Stage2ROISliceDataset, DataLoader, DataLoader]:
    case_dirs = sorted_case_dirs(settings.train_dir)
    if settings.fold_index is not None:
        train_dirs, val_dirs = split_case_dirs_kfold(
            case_dirs, num_folds=settings.num_folds,
            fold_index=settings.fold_index, seed=settings.seed,
        )
    else:
        train_dirs, val_dirs = split_case_dirs(
            case_dirs,
            val_ratio=settings.val_ratio,
            seed=settings.seed,
            fixed_val_cases=settings.fixed_val_cases,
        )
    coarse_masks: dict[str, np.ndarray] | None = None
    if settings.use_coarse_input or settings.use_t2_roi_norm:
        if settings.coarse_oof_dir is not None:
            oof_dir = settings.coarse_oof_dir
        else:
            dirs = _two_stage_dirs(settings)
            oof_dir = dirs["stage1_result"] / "oof_masks"
        all_dirs = list(train_dirs) + list(val_dirs)
        coarse_masks = _load_oof_coarse_masks(oof_dir, all_dirs)
    train_dataset = Stage2ROISliceDataset(
        train_dirs,
        crop_size=settings.crop_size,
        input_mode=settings.stage2_input_mode,
        roi_padding=settings.roi_padding,
        roi_mode=settings.stage2_roi_mode,
        target_type=settings.stage1_target_type,
        roi_mask_source="gt",
        coarse_masks=coarse_masks,
        use_t2_clahe=settings.use_t2_clahe,
        t2_clahe_clip_limit=settings.t2_clahe_clip_limit,
        t2_clahe_tile_grid_size=settings.t2_clahe_tile_grid_size,
        augment=True,
        rotation_degrees=settings.rotation_degrees,
        enable_elastic=settings.enable_elastic,
        enable_advanced_augment=settings.enable_advanced_augment,
        enable_copy_paste=True,
        use_coarse_input=settings.use_coarse_input,
        use_t2_roi_norm=settings.use_t2_roi_norm,
        num_classes=settings.stage2_num_classes,
    )
    val_dataset = Stage2ROISliceDataset(
        val_dirs,
        crop_size=settings.crop_size,
        input_mode=settings.stage2_input_mode,
        roi_padding=settings.roi_padding,
        roi_mode=settings.stage2_roi_mode,
        target_type=settings.stage1_target_type,
        roi_mask_source="gt",
        coarse_masks=coarse_masks,
        use_t2_clahe=settings.use_t2_clahe,
        t2_clahe_clip_limit=settings.t2_clahe_clip_limit,
        t2_clahe_tile_grid_size=settings.t2_clahe_tile_grid_size,
        augment=False,
        rotation_degrees=settings.rotation_degrees,
        enable_elastic=False,
        use_coarse_input=settings.use_coarse_input,
        use_t2_roi_norm=settings.use_t2_roi_norm,
        num_classes=settings.stage2_num_classes,
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


def _build_stage1_test_loader(settings: Settings) -> tuple[Stage1BinaryDataset, DataLoader]:
    case_dirs = sorted_case_dirs(settings.test_dir)
    dataset = Stage1BinaryDataset(
        case_dirs,
        crop_size=settings.crop_size,
        input_modality=settings.stage1_modality,
        target_type=settings.stage1_target_type,
        preprocess_mode=settings.stage1_preprocess_mode,
        augment=False,
        rotation_degrees=settings.rotation_degrees,
        enable_elastic=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=settings.device.startswith("cuda"),
    )
    return dataset, loader


def _build_stage2_test_loader(
    settings: Settings,
    roi_masks: dict[str, np.ndarray],
) -> tuple[Stage2ROISliceDataset, DataLoader]:
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
        coarse_masks=roi_masks,
        use_t2_clahe=settings.use_t2_clahe,
        t2_clahe_clip_limit=settings.t2_clahe_clip_limit,
        t2_clahe_tile_grid_size=settings.t2_clahe_tile_grid_size,
        augment=False,
        rotation_degrees=settings.rotation_degrees,
        enable_elastic=False,
        use_coarse_input=settings.use_coarse_input,
        use_t2_roi_norm=settings.use_t2_roi_norm,
        num_classes=settings.stage2_num_classes,
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=settings.device.startswith("cuda"),
    )
    return dataset, loader


def _evaluate_stage1_loader(
    model: torch.nn.Module,
    dataset: Stage1BinaryDataset,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path | None,
    max_batches: int | None = None,
    num_classes: int = 2,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    case_predictions: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        progress = tqdm(loader, desc="Stage1 Eval", leave=False)
        for batch_index, batch in enumerate(progress):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device)
            probabilities, _ = predict_with_tta(model, images, use_tta=False)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy().astype(np.int64)

            case_ids = batch["case_id"]
            slice_indices = batch["slice_idx"].tolist()
            for sample_index, case_id in enumerate(case_ids):
                case_store = dataset.case_data[case_id]
                if case_id not in case_predictions:
                    case_predictions[case_id] = torch.zeros_like(torch.from_numpy(case_store["mask_full"]))
                if case_store["preprocess_mode"] == "center_crop":
                    restored_slice = restore_center_crop(predictions[sample_index], case_store["crop_meta"])
                else:
                    restored_slice = _resize_hw(predictions[sample_index], case_store["orig_shape"][:2]).astype(np.int64)
                case_predictions[case_id][:, :, slice_indices[sample_index]] = torch.from_numpy(restored_slice)

    rows = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    prediction_volumes: dict[str, np.ndarray] = {}
    for case_id, prediction_tensor in sorted(case_predictions.items()):
        case_store = dataset.case_data[case_id]
        prediction = prediction_tensor.numpy().astype(np.int64)
        target = case_store["mask_full"]
        prediction_volumes[case_id] = prediction
        if num_classes > 2:
            per_class = [dice_binary((prediction == c), (target == c)) for c in range(1, num_classes)]
            row = {"case_id": case_id, "foreground_dice": float(np.mean(per_class))}
            anatomy_names = ["myo", "lv", "rv"]
            for idx, name in enumerate(anatomy_names[:num_classes - 1]):
                row[f"{name}_dice"] = per_class[idx]
            rows.append(row)
        else:
            rows.append(
                {
                    "case_id": case_id,
                    "foreground_dice": dice_binary(prediction, target),
                    "foreground_hd95": hd95_binary(prediction, target, spacing=case_store["spacing"]),
                }
            )
    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        raise RuntimeError("No cases were evaluated in Stage1.")
    summary = {"foreground_dice": float(metrics_df["foreground_dice"].mean())}
    if "foreground_hd95" in metrics_df.columns:
        summary["foreground_hd95"] = float(metrics_df["foreground_hd95"].mean())
    return metrics_df, summary, prediction_volumes


def _evaluate_stage2_loader(
    model: torch.nn.Module,
    dataset: Stage2ROISliceDataset,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
    output_dir: Path | None,
    vis_modality: str,
    save_visualizations: bool,
    visualize_all_slices: bool,
    max_batches: int | None = None,
    max_cases: int | None = None,
    ensemble_models: list[torch.nn.Module] | None = None,
    ensemble_weights: list[float] | None = None,
    num_classes: int = 6,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    if ensemble_models:
        for m in ensemble_models:
            m.eval()
    case_predictions: dict[str, torch.Tensor] = {}
    case_seen = set()

    with torch.no_grad():
        progress = tqdm(loader, desc="Stage2 Eval", leave=False)
        for batch_index, batch in enumerate(progress):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device)
            if ensemble_models:
                probabilities, _ = predict_with_ensemble(ensemble_models, images, use_tta=use_tta, weights=ensemble_weights)
            else:
                probabilities, _ = predict_with_tta(model, images, use_tta=use_tta)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

            case_ids = batch["case_id"]
            slice_indices = batch["slice_idx"].tolist()
            for sample_index, case_id in enumerate(case_ids):
                if max_cases is not None and case_id not in case_seen and len(case_seen) >= max_cases:
                    continue
                case_seen.add(case_id)
                case_store = dataset.case_data[case_id]
                if case_id not in case_predictions:
                    case_predictions[case_id] = torch.zeros_like(torch.from_numpy(case_store["mask_full"]))
                roi_meta = case_store["roi_metas"][slice_indices[sample_index]]
                restored_slice = restore_roi_crop(predictions[sample_index], roi_meta)
                if num_classes == 2:
                    remap = np.zeros_like(restored_slice)
                    remap[restored_slice == 1] = 5
                    restored_slice = remap
                elif num_classes == 3:
                    remap = np.zeros_like(restored_slice)
                    remap[restored_slice == 1] = 4
                    remap[restored_slice == 2] = 5
                    restored_slice = remap
                case_predictions[case_id][:, :, slice_indices[sample_index]] = torch.from_numpy(restored_slice)

    rows = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for case_id, prediction_tensor in sorted(case_predictions.items()):
        case_store = dataset.case_data[case_id]
        prediction = prediction_tensor.numpy().astype(case_store["mask_full"].dtype)
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
        raise RuntimeError("No cases were evaluated in Stage2.")
    summary = {
        "edema_dice": float(metrics_df["edema_dice"].mean()),
        "scar_dice": float(metrics_df["scar_dice"].mean()),
        "mean_dice": float(metrics_df["mean_dice"].mean()),
        "edema_hd95": float(metrics_df["edema_hd95"].mean()),
        "scar_hd95": float(metrics_df["scar_hd95"].mean()),
        "mean_hd95": float(metrics_df["mean_hd95"].mean()),
    }
    return metrics_df, summary


def _resolve_stage1_checkpoint(settings: Settings) -> Path:
    if settings.stage1_checkpoint_path is not None:
        return settings.stage1_checkpoint_path
    candidate = _two_stage_dirs(settings)["stage1_ckpt"] / "best_model.pt"
    if not candidate.exists():
        raise FileNotFoundError("Stage1 checkpoint not found. Provide --stage1-checkpoint-path or train Stage1 first.")
    return candidate


def _resolve_stage2_checkpoint(settings: Settings) -> Path:
    if settings.stage2_checkpoint_path is not None:
        return settings.stage2_checkpoint_path
    candidate = _two_stage_dirs(settings)["stage2_ckpt"] / "best_model.pt"
    if not candidate.exists():
        raise FileNotFoundError("Stage2 checkpoint not found. Provide --stage2-checkpoint-path or train Stage2 first.")
    return candidate


def stage1_train_pipeline(settings: Settings) -> None:
    dirs = _two_stage_dirs(settings)
    ensure_directories([dirs["stage1_ckpt"], dirs["stage1_log"], dirs["stage1_result"]])
    logger = setup_logger(dirs["stage1_log"] / "train.log")
    save_json(settings.to_dict(), dirs["stage1_log"] / "config.json")
    seed_everything(settings.seed)

    train_dataset, val_dataset, train_loader, val_loader = _build_stage1_dataloaders(settings)
    device = torch.device(settings.device)
    model = build_model(
        settings.stage1_model_variant,
        in_channels=settings.stage1_input_channels,
        num_classes=settings.stage1_num_classes,
    ).to(device)
    class_weights = _compute_stage1_class_weights(train_dataset, settings.stage1_num_classes).to(device)
    criterion = CombinedSegmentationLoss(
        class_weights=class_weights,
        dice_weight=settings.dice_loss_weight,
        ce_weight=settings.ce_loss_weight,
        focal_weight=settings.focal_loss_weight,
        focal_gamma=settings.focal_gamma,
        surface_loss_weight=settings.surface_loss_weight,
        union_loss_weight=settings.union_loss_weight,
        inclusiveness_loss_weight=settings.inclusiveness_loss_weight,
    )
    optimizer = Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    if settings.scheduler_type == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=settings.cosine_t0, T_mult=1, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=settings.scheduler_factor,
            patience=settings.scheduler_patience,
        )

    logger.info("Train cases: %s", sorted(train_dataset.case_data.keys()))
    logger.info("Val cases: %s", sorted(val_dataset.case_data.keys()))
    logger.info(
        "Stage1 setup: modality=%s | target=%s | preprocess=%s",
        settings.stage1_modality,
        settings.stage1_target_type,
        settings.stage1_preprocess_mode,
    )
    logger.info("Stage1 class weights: %s", class_weights.detach().cpu().numpy().round(4).tolist())

    history: list[dict] = []
    best_score = float("-inf")
    best_moving_average_score = float("-inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, settings.epochs + 1):
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
        metrics_df, summary, _ = _evaluate_stage1_loader(
            model=model,
            dataset=val_dataset,
            loader=val_loader,
            device=device,
            output_dir=dirs["stage1_result"] / "val",
            max_batches=settings.max_val_batches,
            num_classes=settings.stage1_num_classes,
        )
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(summary["foreground_dice"])
        else:
            scheduler.step(epoch)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice_loss": train_components["dice_loss"],
            "train_focal_loss": train_components["focal_loss"],
            "train_pathology_dice_loss": train_components["pathology_dice_loss"],
            "val_foreground_dice": summary["foreground_dice"],
            "val_foreground_hd95": summary["foreground_hd95"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(history_row)
        pd.DataFrame(history).to_csv(dirs["stage1_log"] / "history.csv", index=False)
        metrics_df.to_csv(dirs["stage1_result"] / "val" / f"epoch_{epoch:03d}_metrics.csv", index=False)

        recent_rows = history[-EARLY_STOPPING_MA_WINDOW:]
        moving_average_score = sum(row["val_foreground_dice"] for row in recent_rows) / len(recent_rows)

        logger.info(
            "Epoch %03d | train_loss=%.4f | val_foreground_dice=%.4f | ma%d_foreground_dice=%.4f | foreground_hd95=%.4f",
            epoch,
            train_loss,
            summary["foreground_dice"],
            EARLY_STOPPING_MA_WINDOW,
            moving_average_score,
            summary["foreground_hd95"],
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "settings": settings.to_dict(),
            "best_score": best_score,
        }
        save_checkpoint(checkpoint_payload, dirs["stage1_ckpt"] / "last_model.pt")

        if summary["foreground_dice"] > best_score:
            best_score = summary["foreground_dice"]
            best_state = copy.deepcopy(model.state_dict())
            checkpoint_payload["best_score"] = best_score
            save_checkpoint(checkpoint_payload, dirs["stage1_ckpt"] / "best_model.pt")

        if moving_average_score > best_moving_average_score:
            best_moving_average_score = moving_average_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= settings.early_stopping_patience:
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    _save_binary_training_curves(history, dirs["stage1_log"])
    logger.info("Stage1 training finished. Best validation foreground Dice: %.4f", best_score)


def stage1_oof_pipeline(settings: Settings) -> None:
    """Train Stage1 with K-fold CV and generate out-of-fold predictions for all training cases."""
    dirs = _two_stage_dirs(settings)
    oof_dir = dirs["stage1_result"] / "oof_masks"
    oof_dir.mkdir(parents=True, exist_ok=True)
    base_logger = setup_logger(dirs["stage1_log"] / "oof.log")
    save_json(settings.to_dict(), dirs["stage1_log"] / "oof_config.json")

    case_dirs = sorted_case_dirs(settings.train_dir)
    num_folds = settings.num_folds
    device = torch.device(settings.device)
    all_oof_dices: list[float] = []

    for fold_idx in range(num_folds):
        seed_everything(settings.seed)
        train_dirs, val_dirs = split_case_dirs_kfold(
            case_dirs, num_folds=num_folds, fold_index=fold_idx, seed=settings.seed,
        )
        fold_ckpt_dir = dirs["stage1_ckpt"] / f"fold{fold_idx}"
        fold_log_dir = dirs["stage1_log"] / f"fold{fold_idx}"
        ensure_directories([fold_ckpt_dir, fold_log_dir])
        fold_logger = setup_logger(fold_log_dir / "train.log")

        train_dataset = Stage1BinaryDataset(
            train_dirs, crop_size=settings.crop_size,
            input_modality=settings.stage1_modality, target_type=settings.stage1_target_type,
            preprocess_mode=settings.stage1_preprocess_mode,
            augment=True, rotation_degrees=settings.rotation_degrees, enable_elastic=settings.enable_elastic,
        )
        val_dataset = Stage1BinaryDataset(
            val_dirs, crop_size=settings.crop_size,
            input_modality=settings.stage1_modality, target_type=settings.stage1_target_type,
            preprocess_mode=settings.stage1_preprocess_mode,
            augment=False, rotation_degrees=settings.rotation_degrees, enable_elastic=False,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=settings.batch_size, shuffle=True,
            num_workers=settings.num_workers, pin_memory=settings.device.startswith("cuda"),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=settings.batch_size, shuffle=False,
            num_workers=settings.num_workers, pin_memory=settings.device.startswith("cuda"),
        )

        model = build_model(
            settings.stage1_model_variant, in_channels=settings.stage1_input_channels,
            num_classes=settings.stage1_num_classes,
        ).to(device)
        class_weights = _compute_stage1_class_weights(train_dataset, settings.stage1_num_classes).to(device)
        criterion = CombinedSegmentationLoss(
            class_weights=class_weights, dice_weight=settings.dice_loss_weight,
            ce_weight=settings.ce_loss_weight,
            focal_weight=settings.focal_loss_weight, focal_gamma=settings.focal_gamma,
            surface_loss_weight=settings.surface_loss_weight,
            union_loss_weight=settings.union_loss_weight,
            inclusiveness_loss_weight=settings.inclusiveness_loss_weight,
        )
        optimizer = Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
        if settings.scheduler_type == "cosine":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=settings.cosine_t0, T_mult=1, eta_min=1e-6)
        else:
            scheduler = ReduceLROnPlateau(
                optimizer, mode="max", factor=settings.scheduler_factor, patience=settings.scheduler_patience,
            )

        fold_logger.info("Fold %d: train=%s, val=%s", fold_idx,
                         sorted(train_dataset.case_data.keys()), sorted(val_dataset.case_data.keys()))

        best_score = float("-inf")
        best_moving_average_score = float("-inf")
        best_state = None
        epochs_without_improvement = 0
        history: list[dict] = []

        for epoch in range(1, settings.epochs + 1):
            train_loss, train_components = run_train_epoch(
                model=model, loader=train_loader, criterion=criterion,
                optimizer=optimizer, device=device, max_batches=settings.max_train_batches,
                epoch=epoch, total_epochs=settings.epochs,
            )
            _, summary, _ = _evaluate_stage1_loader(
                model=model, dataset=val_dataset, loader=val_loader,
                device=device, output_dir=None, max_batches=settings.max_val_batches,
                num_classes=settings.stage1_num_classes,
            )
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(summary["foreground_dice"])
            else:
                scheduler.step(epoch)

            history.append({
                "epoch": epoch, "train_loss": train_loss,
                "val_foreground_dice": summary["foreground_dice"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

            checkpoint_payload = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "settings": settings.to_dict(), "best_score": best_score,
            }
            save_checkpoint(checkpoint_payload, fold_ckpt_dir / "last_model.pt")

            if summary["foreground_dice"] > best_score:
                best_score = summary["foreground_dice"]
                best_state = copy.deepcopy(model.state_dict())
                checkpoint_payload["best_score"] = best_score
                save_checkpoint(checkpoint_payload, fold_ckpt_dir / "best_model.pt")

            recent_rows = history[-EARLY_STOPPING_MA_WINDOW:]
            moving_average_score = sum(r["val_foreground_dice"] for r in recent_rows) / len(recent_rows)
            if moving_average_score > best_moving_average_score:
                best_moving_average_score = moving_average_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            fold_logger.info(
                "Fold %d Epoch %03d | loss=%.4f | val_dice=%.4f | ma_dice=%.4f",
                fold_idx, epoch, train_loss, summary["foreground_dice"], moving_average_score,
            )

            if epochs_without_improvement >= settings.early_stopping_patience:
                fold_logger.info("Fold %d early stopping at epoch %d.", fold_idx, epoch)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        pd.DataFrame(history).to_csv(fold_log_dir / "history.csv", index=False)

        _, val_summary, prediction_volumes = _evaluate_stage1_loader(
            model=model, dataset=val_dataset, loader=val_loader,
            device=device, output_dir=None,
            num_classes=settings.stage1_num_classes,
        )
        fold_logger.info("Fold %d best val dice: %.4f", fold_idx, val_summary["foreground_dice"])
        all_oof_dices.append(val_summary["foreground_dice"])

        for case_id, pred_vol in prediction_volumes.items():
            np.save(oof_dir / f"{case_id}.npy", pred_vol.astype(np.uint8))
            fold_logger.info("Saved OOF mask for %s", case_id)

    base_logger.info(
        "Stage1 OOF complete. Per-fold dice: %s, mean: %.4f",
        [round(d, 4) for d in all_oof_dices],
        sum(all_oof_dices) / len(all_oof_dices),
    )
    base_logger.info("OOF masks saved to %s", oof_dir)


def _load_oof_coarse_masks(oof_dir: Path, case_dirs: list[Path]) -> dict[str, np.ndarray]:
    """Load pre-computed out-of-fold coarse masks from disk."""
    masks: dict[str, np.ndarray] = {}
    for case_dir in case_dirs:
        case_id = case_dir.name
        mask_path = oof_dir / f"{case_id}.npy"
        if not mask_path.exists():
            parent_id = case_id.split("_aug")[0] if "_aug" in case_id else None
            if parent_id:
                mask_path = oof_dir / f"{parent_id}.npy"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"OOF mask not found for {case_id} at {mask_path}. "
                "Run --two-stage-step stage1_oof first."
            )
        masks[case_id] = np.load(mask_path).astype(np.uint8)
    return masks


def stage2_train_pipeline(settings: Settings) -> None:
    dirs = _two_stage_dirs(settings)
    if settings.fold_index is not None:
        fold_suffix = f"_fold{settings.fold_index}"
        dirs["stage2_ckpt"] = dirs["stage2_ckpt"].parent / (dirs["stage2_ckpt"].name + fold_suffix)
        dirs["stage2_log"] = dirs["stage2_log"].parent / (dirs["stage2_log"].name + fold_suffix)
        dirs["stage2_result"] = dirs["stage2_result"].parent / (dirs["stage2_result"].name + fold_suffix)
    ensure_directories([dirs["stage2_ckpt"], dirs["stage2_log"], dirs["stage2_result"]])
    logger = setup_logger(dirs["stage2_log"] / "train.log")
    save_json(settings.to_dict(), dirs["stage2_log"] / "config.json")
    seed_everything(settings.seed)

    train_dataset, val_dataset, train_loader, val_loader = _build_stage2_dataloaders(settings)
    sample_weight_summary = summarize_sample_weights(train_dataset, train_loader.sampler.weights.tolist())
    device = torch.device(settings.device)
    model = build_model(settings.model_variant, in_channels=settings.stage2_input_channels, num_classes=settings.stage2_num_classes).to(device)
    if settings.stage2_num_classes == 2:
        class_weights = torch.tensor([0.1, 5.0], dtype=torch.float32).to(device)
        criterion = CombinedSegmentationLoss(
            class_weights=class_weights,
            dice_weight=settings.dice_loss_weight,
            ce_weight=settings.ce_loss_weight,
            focal_weight=settings.focal_loss_weight if settings.focal_loss_weight > 0 else 0.5,
            focal_gamma=settings.focal_gamma,
            pathology_dice_weight=0.0,
            myocardium_prior_weight=0.0,
            edema_dice_weight=0.0,
            surface_loss_weight=0.0,
            union_loss_weight=0.0,
            inclusiveness_loss_weight=0.0,
        )
    elif settings.stage2_num_classes == 3:
        class_weights = torch.tensor([0.1, 3.0, 2.0], dtype=torch.float32).to(device)
        criterion = CombinedSegmentationLoss(
            class_weights=class_weights,
            dice_weight=settings.dice_loss_weight,
            ce_weight=settings.ce_loss_weight,
            focal_weight=settings.focal_loss_weight if settings.focal_loss_weight > 0 else 0.5,
            focal_gamma=settings.focal_gamma,
            pathology_dice_weight=0.0,
            myocardium_prior_weight=0.0,
            edema_dice_weight=0.0,
            surface_loss_weight=0.0,
            union_loss_weight=0.0,
            inclusiveness_loss_weight=0.0,
        )
    else:
        if settings.use_edema_expert_weights:
            class_weights = derive_edema_expert_class_weights().to(device)
        else:
            class_weights = derive_class_weights().to(device)
        criterion = CombinedSegmentationLoss(
            class_weights=class_weights,
            dice_weight=settings.dice_loss_weight,
            ce_weight=settings.ce_loss_weight,
            focal_weight=settings.focal_loss_weight,
            focal_gamma=settings.focal_gamma,
            tversky_weight=settings.tversky_loss_weight,
            tversky_alpha=settings.tversky_alpha,
            tversky_beta=settings.tversky_beta,
            edema_dice_weight=settings.edema_dice_weight,
            myocardium_prior_weight=settings.myocardium_prior_weight,
            surface_loss_weight=settings.surface_loss_weight,
            union_loss_weight=settings.union_loss_weight,
            inclusiveness_loss_weight=settings.inclusiveness_loss_weight,
        )
    optimizer = Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    if settings.scheduler_type == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=settings.cosine_t0, T_mult=1, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=settings.scheduler_factor,
            patience=settings.scheduler_patience,
        )

    logger.info("Train cases: %s", sorted(train_dataset.case_data.keys()))
    logger.info("Val cases: %s", sorted(val_dataset.case_data.keys()))
    logger.info(
        "Sampling strategy: %s | oversample_factor=%.2f | weight_stats=%s",
        settings.sampling_strategy,
        settings.pathology_oversample_factor,
        {key: round(value, 4) for key, value in sample_weight_summary.items()},
    )
    logger.info(
        "Stage2 ROI setup: target=%s | roi_mode=%s | padding=%d | t2_clahe=%s",
        settings.stage1_target_type,
        settings.stage2_roi_mode,
        settings.roi_padding,
        settings.use_t2_clahe,
    )
    logger.info("Class weights: %s", class_weights.detach().cpu().numpy().round(4).tolist())

    history: list[dict] = []
    best_score = float("-inf")
    best_edema_score = float("-inf")
    best_moving_average_score = float("-inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, settings.epochs + 1):
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
        metrics_df, summary = _evaluate_stage2_loader(
            model=model,
            dataset=val_dataset,
            loader=val_loader,
            device=device,
            use_tta=False,
            output_dir=dirs["stage2_result"] / "val",
            vis_modality=settings.vis_modality,
            save_visualizations=settings.save_visualizations,
            visualize_all_slices=settings.visualize_all_slices,
            max_batches=settings.max_val_batches,
            num_classes=settings.stage2_num_classes,
        )
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(summary["mean_dice"])
        else:
            scheduler.step(epoch)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice_loss": train_components["dice_loss"],
            "train_focal_loss": train_components["focal_loss"],
            "train_pathology_dice_loss": train_components["pathology_dice_loss"],
            "val_edema_dice": summary["edema_dice"],
            "val_scar_dice": summary["scar_dice"],
            "val_mean_dice": summary["mean_dice"],
            "val_edema_hd95": summary["edema_hd95"],
            "val_scar_hd95": summary["scar_hd95"],
            "val_mean_hd95": summary["mean_hd95"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(history_row)
        pd.DataFrame(history).to_csv(dirs["stage2_log"] / "history.csv", index=False)
        metrics_df.to_csv(dirs["stage2_result"] / "val" / f"epoch_{epoch:03d}_metrics.csv", index=False)

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
        save_checkpoint(checkpoint_payload, dirs["stage2_ckpt"] / "last_model.pt")

        if summary["mean_dice"] > best_score:
            best_score = summary["mean_dice"]
            best_state = copy.deepcopy(model.state_dict())
            checkpoint_payload["best_score"] = best_score
            save_checkpoint(checkpoint_payload, dirs["stage2_ckpt"] / "best_model.pt")

        if summary["edema_dice"] > best_edema_score:
            best_edema_score = summary["edema_dice"]
            checkpoint_payload["best_edema_score"] = best_edema_score
            save_checkpoint(checkpoint_payload, dirs["stage2_ckpt"] / "best_edema_model.pt")

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
    save_training_curves(history, dirs["stage2_log"])
    logger.info("Stage2 training finished. Best validation mean Dice: %.4f", best_score)


def two_stage_test_pipeline(settings: Settings) -> None:
    dirs = _two_stage_dirs(settings)
    ensure_directories([dirs["two_stage_log"], dirs["two_stage_result"]])
    logger = setup_logger(dirs["two_stage_log"] / "test.log")
    save_json(settings.to_dict(), dirs["two_stage_log"] / "test_config.json")
    seed_everything(settings.seed)

    stage1_dataset, stage1_loader = _build_stage1_test_loader(settings)
    stage1_checkpoint_path = _resolve_stage1_checkpoint(settings)

    device = torch.device(settings.device)

    stage1_checkpoint = torch.load(stage1_checkpoint_path, map_location=settings.device, weights_only=False)
    stage1_model = build_model(
        settings.stage1_model_variant,
        in_channels=settings.stage1_input_channels,
        num_classes=settings.stage1_num_classes,
    ).to(device)
    stage1_model.load_state_dict(stage1_checkpoint["model_state_dict"])
    logger.info("Loaded Stage1 checkpoint: %s", stage1_checkpoint_path)
    logger.info(
        "Two-stage setup: stage1_modality=%s | stage1_target=%s | stage1_preprocess=%s | roi_mode=%s | t2_clahe=%s",
        settings.stage1_modality,
        settings.stage1_target_type,
        settings.stage1_preprocess_mode,
        settings.stage2_roi_mode,
        settings.use_t2_clahe,
    )
    _, stage1_summary, stage1_predictions = _evaluate_stage1_loader(
        model=stage1_model,
        dataset=stage1_dataset,
        loader=stage1_loader,
        device=device,
        output_dir=dirs["two_stage_result"] / "stage1",
        num_classes=settings.stage1_num_classes,
    )
    pd.DataFrame([stage1_summary]).to_csv(dirs["two_stage_result"] / "stage1" / "stage1_summary.csv", index=False)
    logger.info("Stage1 test summary: %s", stage1_summary)

    stage2_dataset, stage2_loader = _build_stage2_test_loader(settings, roi_masks=stage1_predictions)

    use_ensemble = bool(settings.ensemble_checkpoint_paths)
    if use_ensemble:
        models = []
        for ckpt_path, variant in zip(settings.ensemble_checkpoint_paths, settings.ensemble_model_variants):
            m = load_model_checkpoint(
                checkpoint_path=ckpt_path,
                model_variant=variant,
                in_channels=settings.stage2_input_channels,
                device=device,
                num_classes=settings.stage2_num_classes,
            )
            models.append(m)
        weights = list(settings.ensemble_weights) if settings.ensemble_weights else None
        logger.info(
            "Ensemble Stage2: %d models, variants=%s, weights=%s",
            len(models), settings.ensemble_model_variants, weights,
        )
        metrics_df, summary = _evaluate_stage2_loader(
            model=models[0],
            dataset=stage2_dataset,
            loader=stage2_loader,
            device=device,
            use_tta=settings.enable_tta,
            output_dir=dirs["two_stage_result"],
            vis_modality=settings.vis_modality,
            save_visualizations=settings.save_visualizations,
            visualize_all_slices=settings.visualize_all_slices,
            max_cases=settings.max_test_cases,
            ensemble_models=models,
            ensemble_weights=weights,
            num_classes=settings.stage2_num_classes,
        )
    else:
        stage2_checkpoint_path = _resolve_stage2_checkpoint(settings)
        stage2_checkpoint = torch.load(stage2_checkpoint_path, map_location=settings.device, weights_only=False)
        stage2_model = build_model(
            settings.model_variant,
            in_channels=settings.stage2_input_channels,
            num_classes=settings.stage2_num_classes,
        ).to(device)
        stage2_model.load_state_dict(stage2_checkpoint["model_state_dict"])
        logger.info("Loaded Stage2 checkpoint: %s", stage2_checkpoint_path)

        metrics_df, summary = _evaluate_stage2_loader(
            model=stage2_model,
            dataset=stage2_dataset,
            loader=stage2_loader,
            device=device,
            use_tta=settings.enable_tta,
            output_dir=dirs["two_stage_result"],
            vis_modality=settings.vis_modality,
            save_visualizations=settings.save_visualizations,
            visualize_all_slices=settings.visualize_all_slices,
            max_cases=settings.max_test_cases,
            num_classes=settings.stage2_num_classes,
        )
    metrics_path = dirs["two_stage_result"] / "test_metrics.csv"
    summary_path = dirs["two_stage_result"] / "test_summary.json"
    metrics_df.to_csv(metrics_path, index=False)
    save_json(summary, summary_path)
    logger.info("Two-stage test summary: %s", summary)
    _log_scar_breakdown(metrics_df, logger)


def run_two_stage(settings: Settings) -> None:
    if settings.mode == "train" and settings.two_stage_step == "stage1_train":
        stage1_train_pipeline(settings)
        return
    if settings.mode == "train" and settings.two_stage_step == "stage1_oof":
        stage1_oof_pipeline(settings)
        return
    if settings.mode == "train" and settings.two_stage_step == "stage2_train":
        stage2_train_pipeline(settings)
        return
    if settings.mode == "test" and settings.two_stage_step == "two_stage_test":
        two_stage_test_pipeline(settings)
        return
    raise ValueError(
        f"Unsupported two-stage combination: mode={settings.mode}, two_stage_step={settings.two_stage_step}"
    )
