from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates, rotate
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from src.constants import DEFAULT_MODALITIES, LABEL_TO_INDEX, PATHOLOGY_CLASS_INDICES
from src.losses import one_hot2dist
from src.roi import ROICropMeta, compute_roi_crop_meta, compute_roi_crop_resize_meta, crop_with_roi_meta, restore_roi_crop

SMALL_EDEMA_PIXEL_THRESHOLD = 400
MAX_TARGETED_SAMPLE_WEIGHT = 8.0
SCALING_RANGE = (0.85, 1.15)
BRIGHTNESS_RANGE = (-0.1, 0.1)
CONTRAST_RANGE = (-0.15, 0.15)
GAUSSIAN_BLUR_SIGMA_RANGE = (0.1, 1.0)
COARSE_DROPOUT_SIZE_RANGE = (16, 32)


@dataclass
class CropMeta:
    orig_h: int
    orig_w: int
    crop_h: int
    crop_w: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    top: int
    left: int


CropRestoreMeta = CropMeta | ROICropMeta


def sorted_case_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir()])


def split_case_dirs(
    case_dirs: list[Path],
    val_ratio: float,
    seed: int,
    fixed_val_cases: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    if fixed_val_cases:
        fixed_val_set = set(fixed_val_cases)
        available_cases = {path.name for path in case_dirs}
        missing_cases = sorted(fixed_val_set - available_cases)
        if missing_cases:
            raise ValueError(f"Fixed validation cases not found: {missing_cases}")
        val_dirs = sorted([path for path in case_dirs if path.name in fixed_val_set])
        train_dirs = sorted([path for path in case_dirs if path.name not in fixed_val_set])
        if not val_dirs:
            raise ValueError("Fixed validation case list is empty after filtering.")
        return train_dirs, val_dirs

    rng = random.Random(seed)
    shuffled = case_dirs[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_dirs = sorted(shuffled[:val_count])
    train_dirs = sorted(shuffled[val_count:])
    return train_dirs, val_dirs


def split_case_dirs_kfold(
    case_dirs: list[Path],
    num_folds: int,
    fold_index: int,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if num_folds < 2:
        raise ValueError(f"num_folds must be at least 2 for K-Fold splitting, got {num_folds}")
    if not 0 <= fold_index < num_folds:
        raise ValueError(f"fold_index must be in [0, {num_folds - 1}], got {fold_index}")
    if len(case_dirs) < num_folds:
        raise ValueError(f"Number of cases ({len(case_dirs)}) must be >= num_folds ({num_folds})")

    sorted_dirs = sorted(case_dirs, key=lambda path: path.name)

    groups: dict[str, list[Path]] = {}
    for d in sorted_dirs:
        parent_id = d.name.split("_aug")[0] if "_aug" in d.name else d.name
        groups.setdefault(parent_id, []).append(d)
    unique_parents = sorted(groups.keys())

    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(unique_parents))
    train_indices, val_indices = splits[fold_index]

    train_dirs = []
    for idx in train_indices:
        train_dirs.extend(groups[unique_parents[idx]])
    val_dirs = []
    for idx in val_indices:
        for d in groups[unique_parents[idx]]:
            if "_aug" not in d.name:
                val_dirs.append(d)
    train_dirs.sort(key=lambda p: p.name)
    val_dirs.sort(key=lambda p: p.name)

    if not train_dirs or not val_dirs:
        raise ValueError(
            f"K-Fold split produced an empty dataset for fold {fold_index}: "
            f"train={len(train_dirs)}, val={len(val_dirs)}"
        )
    return train_dirs, val_dirs


def compute_center_crop_meta(orig_h: int, orig_w: int, crop_h: int, crop_w: int) -> CropMeta:
    pad_h = max(0, crop_h - orig_h)
    pad_w = max(0, crop_w - orig_w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded_h = orig_h + pad_top + pad_bottom
    padded_w = orig_w + pad_left + pad_right
    top = max(0, (padded_h - crop_h) // 2)
    left = max(0, (padded_w - crop_w) // 2)
    return CropMeta(
        orig_h=orig_h,
        orig_w=orig_w,
        crop_h=crop_h,
        crop_w=crop_w,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        top=top,
        left=left,
    )


def center_crop_array(array: np.ndarray, meta: CropMeta, fill_value: float = 0.0) -> np.ndarray:
    if array.ndim == 2:
        pad_width = ((meta.pad_top, meta.pad_bottom), (meta.pad_left, meta.pad_right))
    elif array.ndim == 3:
        pad_width = ((meta.pad_top, meta.pad_bottom), (meta.pad_left, meta.pad_right), (0, 0))
    else:
        raise ValueError(f"Unsupported ndim for center crop: {array.ndim}")
    padded = np.pad(array, pad_width=pad_width, mode="constant", constant_values=fill_value)
    return padded[meta.top : meta.top + meta.crop_h, meta.left : meta.left + meta.crop_w, ...]


def restore_center_crop(array: np.ndarray, meta: CropMeta) -> np.ndarray:
    if array.ndim == 2:
        padded = np.zeros(
            (meta.orig_h + meta.pad_top + meta.pad_bottom, meta.orig_w + meta.pad_left + meta.pad_right),
            dtype=array.dtype,
        )
        padded[meta.top : meta.top + meta.crop_h, meta.left : meta.left + meta.crop_w] = array
        return padded[meta.pad_top : meta.pad_top + meta.orig_h, meta.pad_left : meta.pad_left + meta.orig_w]
    if array.ndim == 3:
        padded = np.zeros(
            (
                meta.orig_h + meta.pad_top + meta.pad_bottom,
                meta.orig_w + meta.pad_left + meta.pad_right,
                array.shape[2],
            ),
            dtype=array.dtype,
        )
        padded[meta.top : meta.top + meta.crop_h, meta.left : meta.left + meta.crop_w, :] = array
        return padded[
            meta.pad_top : meta.pad_top + meta.orig_h,
            meta.pad_left : meta.pad_left + meta.orig_w,
            :,
        ]
    raise ValueError(f"Unsupported ndim for restore crop: {array.ndim}")


def restore_cropped_array(array: np.ndarray, meta: CropRestoreMeta) -> np.ndarray:
    if isinstance(meta, CropMeta):
        return restore_center_crop(array, meta)
    if isinstance(meta, ROICropMeta):
        return restore_roi_crop(array, meta)
    raise TypeError(f"Unsupported crop meta type: {type(meta)!r}")


def compute_heart_roi_crop_meta(mask: np.ndarray, crop_size: tuple[int, int], roi_padding: int) -> ROICropMeta:
    if mask.ndim != 2:
        raise ValueError(f"compute_heart_roi_crop_meta expects a 2D mask, got ndim={mask.ndim}")
    return compute_roi_crop_meta(
        mask=(mask > 0).astype(np.uint8),
        crop_size=crop_size,
        padding=roi_padding,
        fallback_shape=mask.shape,
    )


def compute_case_level_roi_crop_meta(
    volume: np.ndarray, crop_size: tuple[int, int], roi_padding: int,
) -> ROICropMeta:
    h, w = volume.shape[0], volume.shape[1]
    global_ymin, global_ymax = h, 0
    global_xmin, global_xmax = w, 0
    found = False
    for s in range(volume.shape[2]):
        coords = np.argwhere(volume[:, :, s] > 0)
        if coords.size == 0:
            continue
        found = True
        global_ymin = min(global_ymin, int(coords[:, 0].min()))
        global_ymax = max(global_ymax, int(coords[:, 0].max()))
        global_xmin = min(global_xmin, int(coords[:, 1].min()))
        global_xmax = max(global_xmax, int(coords[:, 1].max()))
    if not found:
        union_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        union_mask = np.zeros((h, w), dtype=np.uint8)
        union_mask[global_ymin:global_ymax + 1, global_xmin:global_xmax + 1] = 1
    return compute_roi_crop_resize_meta(
        mask=union_mask,
        target_size=crop_size,
        padding=roi_padding,
        fallback_shape=(h, w),
    )


def normalize_nonzero_percentile(volume: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    volume = volume.astype(np.float32)
    mask = volume > 0
    normalized = np.zeros_like(volume, dtype=np.float32)
    if not np.any(mask):
        return normalized
    valid = volume[mask]
    low = np.percentile(valid, lower)
    high = np.percentile(valid, upper)
    clipped = np.clip(volume, low, high)
    stats_values = clipped[mask]
    mean = float(stats_values.mean())
    std = float(stats_values.std())
    normalized[mask] = (clipped[mask] - mean) / max(std, 1e-6)
    return normalized


def normalize_roi_percentile(
    volume: np.ndarray,
    roi_mask: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
    min_roi_pixels: int = 100,
) -> np.ndarray:
    """Normalize using statistics from ROI region only.

    Falls back to nonzero-pixel normalization if ROI is too small.
    """
    volume = volume.astype(np.float32)
    normalized = np.zeros_like(volume, dtype=np.float32)

    roi_pixels = volume[roi_mask]
    if roi_pixels.size < min_roi_pixels:
        return normalize_nonzero_percentile(volume, lower, upper)

    low = np.percentile(roi_pixels, lower)
    high = np.percentile(roi_pixels, upper)
    clipped = np.clip(volume, low, high)

    roi_clipped = clipped[roi_mask]
    mean = float(roi_clipped.mean())
    std = float(roi_clipped.std())

    nonzero = volume > 0
    normalized[nonzero] = (clipped[nonzero] - mean) / max(std, 1e-6)
    return normalized


def map_labels(mask: np.ndarray) -> np.ndarray:
    mapped = np.zeros_like(mask, dtype=np.int64)
    unique_labels = np.unique(mask)
    for label in unique_labels:
        key = int(label)
        if key not in LABEL_TO_INDEX:
            raise ValueError(f"Unexpected label value: {label}")
        mapped[mask == label] = LABEL_TO_INDEX[key]
    return mapped


def pathology_flags(mask_slice: np.ndarray) -> dict[str, bool]:
    unique_values = set(np.unique(mask_slice).tolist())
    has_edema = PATHOLOGY_CLASS_INDICES[0] in unique_values
    has_scar = PATHOLOGY_CLASS_INDICES[1] in unique_values
    return {
        "has_edema": has_edema,
        "has_scar": has_scar,
        "has_pathology": has_edema or has_scar,
    }


def _apply_clahe_to_slice(slice_2d: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    output = np.zeros_like(slice_2d, dtype=np.float32)
    nonzero_mask = slice_2d > 0
    if not np.any(nonzero_mask):
        return output
    valid = slice_2d[nonzero_mask].astype(np.float32)
    low = float(np.percentile(valid, 1))
    high = float(np.percentile(valid, 99))
    if high <= low:
        high = low + 1.0
    scaled = np.zeros_like(slice_2d, dtype=np.uint8)
    scaled_values = np.clip((slice_2d[nonzero_mask] - low) / (high - low), 0.0, 1.0)
    scaled[nonzero_mask] = np.round(scaled_values * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    enhanced = clahe.apply(scaled)
    output[nonzero_mask] = enhanced[nonzero_mask].astype(np.float32) / 255.0
    return output


def _apply_t2_clahe(volume: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    enhanced = np.zeros_like(volume, dtype=np.float32)
    for slice_idx in range(volume.shape[2]):
        enhanced[:, :, slice_idx] = _apply_clahe_to_slice(
            volume[:, :, slice_idx],
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )
    return enhanced


def elastic_deform(image: np.ndarray, mask: np.ndarray, alpha: float = 10.0, sigma: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = image.shape
    random_state = np.random.RandomState(None)
    dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma=sigma, mode="reflect") * alpha
    dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma=sigma, mode="reflect") * alpha
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))

    warped_image = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[0]):
        warped_image[channel] = map_coordinates(
            image[channel],
            indices,
            order=1,
            mode="reflect",
        ).reshape(height, width)
    warped_mask = map_coordinates(mask, indices, order=0, mode="nearest").reshape(height, width)
    return warped_image, warped_mask.astype(np.int64)


def normalize_image_channels(image: np.ndarray) -> np.ndarray:
    normalized_channels = [normalize_nonzero_percentile(channel) for channel in image]
    return np.stack(normalized_channels, axis=0).astype(np.float32)


def resize_to_canvas(array: np.ndarray, target_h: int, target_w: int, interpolation: int) -> np.ndarray:
    resized_h, resized_w = array.shape[:2]
    if resized_h == target_h and resized_w == target_w:
        return array.copy()

    if resized_h >= target_h:
        top = (resized_h - target_h) // 2
        bottom = top + target_h
    else:
        top = -(target_h - resized_h) // 2
        bottom = top + target_h

    if resized_w >= target_w:
        left = (resized_w - target_w) // 2
        right = left + target_w
    else:
        left = -(target_w - resized_w) // 2
        right = left + target_w

    if array.ndim == 2:
        canvas = np.zeros((target_h, target_w), dtype=array.dtype)
    else:
        canvas = np.zeros((target_h, target_w, array.shape[2]), dtype=array.dtype)

    src_top = max(0, top)
    src_left = max(0, left)
    src_bottom = min(resized_h, bottom)
    src_right = min(resized_w, right)
    dst_top = max(0, -top)
    dst_left = max(0, -left)
    dst_bottom = dst_top + (src_bottom - src_top)
    dst_right = dst_left + (src_right - src_left)
    canvas[dst_top:dst_bottom, dst_left:dst_right, ...] = array[src_top:src_bottom, src_left:src_right, ...]
    return canvas


def apply_random_scaling(image: np.ndarray, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = image.shape
    scaled_h = max(1, int(round(height * scale)))
    scaled_w = max(1, int(round(width * scale)))

    scaled_image = np.stack(
        [
            cv2.resize(channel, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            for channel in image
        ],
        axis=0,
    ).transpose(1, 2, 0)
    scaled_mask = cv2.resize(mask.astype(np.uint8), (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

    restored_image = resize_to_canvas(scaled_image, height, width, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    restored_mask = resize_to_canvas(scaled_mask, height, width, interpolation=cv2.INTER_NEAREST)
    return restored_image.astype(np.float32), restored_mask.astype(np.int64)


def apply_random_brightness_contrast(image: np.ndarray) -> np.ndarray:
    adjusted = image.copy().astype(np.float32)
    for channel_index in range(adjusted.shape[0]):
        channel = adjusted[channel_index]
        nonzero_mask = channel != 0
        if not np.any(nonzero_mask):
            continue
        valid = channel[nonzero_mask]
        low = float(np.percentile(valid, 1))
        high = float(np.percentile(valid, 99))
        span = max(high - low, 1e-6)
        contrast = 1.0 + random.uniform(*CONTRAST_RANGE)
        brightness = random.uniform(*BRIGHTNESS_RANGE) * span
        channel = (channel - low) * contrast + low + brightness
        adjusted[channel_index] = channel
    return adjusted


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    blurred = np.empty_like(image, dtype=np.float32)
    for channel_index in range(image.shape[0]):
        blurred[channel_index] = cv2.GaussianBlur(image[channel_index], (3, 3), sigmaX=sigma, sigmaY=sigma)
    return blurred


def apply_coarse_dropout(image: np.ndarray) -> np.ndarray:
    dropped = image.copy()
    _, height, width = dropped.shape
    num_regions = random.randint(1, 3)
    for _ in range(num_regions):
        rect_h = random.randint(COARSE_DROPOUT_SIZE_RANGE[0], min(COARSE_DROPOUT_SIZE_RANGE[1], height))
        rect_w = random.randint(COARSE_DROPOUT_SIZE_RANGE[0], min(COARSE_DROPOUT_SIZE_RANGE[1], width))
        top = random.randint(0, max(0, height - rect_h))
        left = random.randint(0, max(0, width - rect_w))
        dropped[:, top : top + rect_h, left : left + rect_w] = 0.0
    return dropped


class SliceAugmenter:
    def __init__(
        self,
        rotation_degrees: float = 15.0,
        enable_elastic: bool = True,
        enable_advanced_augment: bool = False,
    ) -> None:
        self.rotation_degrees = rotation_degrees
        self.enable_elastic = enable_elastic
        self.enable_advanced_augment = enable_advanced_augment

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = np.stack(
                [rotate(channel, angle=angle, reshape=False, order=1, mode="reflect") for channel in image],
                axis=0,
            )
            mask = rotate(mask, angle=angle, reshape=False, order=0, mode="nearest")

        if self.enable_elastic and random.random() < 0.25:
            image, mask = elastic_deform(image, mask)

        if self.enable_advanced_augment and random.random() < 0.3:
            image, mask = apply_random_scaling(image, mask, scale=random.uniform(*SCALING_RANGE))

        if self.enable_advanced_augment and random.random() < 0.3:
            image = apply_random_brightness_contrast(image)

        if self.enable_advanced_augment and random.random() < 0.2:
            image = apply_gaussian_blur(image, sigma=random.uniform(*GAUSSIAN_BLUR_SIGMA_RANGE))

        if self.enable_advanced_augment and random.random() < 0.2:
            image = apply_coarse_dropout(image)

        return image.astype(np.float32), mask.astype(np.int64)


class MyoPSSliceDataset(Dataset):
    def __init__(
        self,
        case_dirs: list[Path],
        crop_size: tuple[int, int],
        input_mode: str = "2d",
        augment: bool = False,
        rotation_degrees: float = 15.0,
        enable_elastic: bool = True,
        enable_advanced_augment: bool = False,
        use_t2_clahe: bool = False,
        t2_clahe_clip_limit: float = 2.0,
        t2_clahe_tile_grid_size: int = 8,
        roi_crop: bool = False,
        roi_masks: dict[str, np.ndarray] | None = None,
        roi_padding: int = 20,
        zero_bssfp: bool = False,
    ) -> None:
        self.case_dirs = case_dirs
        self.crop_size = crop_size
        if input_mode not in {"2d", "2p5d"}:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
        self.input_mode = input_mode
        self.augment = augment
        self.augmenter = SliceAugmenter(
            rotation_degrees=rotation_degrees,
            enable_elastic=enable_elastic,
            enable_advanced_augment=enable_advanced_augment,
        )
        self.use_t2_clahe = use_t2_clahe
        self.t2_clahe_clip_limit = t2_clahe_clip_limit
        self.t2_clahe_tile_grid_size = t2_clahe_tile_grid_size
        self.roi_crop = roi_crop
        self.roi_masks = roi_masks or {}
        self.roi_padding = roi_padding
        self.zero_bssfp = zero_bssfp
        self.case_data: dict[str, dict] = {}
        self.samples: list[dict] = []
        self._load_cases()

    def _load_cases(self) -> None:
        crop_h, crop_w = self.crop_size
        for case_dir in self.case_dirs:
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

            raw_modalities = {key: value.astype(np.float32) for key, value in modalities.items()}
            if self.use_t2_clahe:
                modalities["T2"] = _apply_t2_clahe(
                    modalities["T2"], self.t2_clahe_clip_limit, self.t2_clahe_tile_grid_size
                )
            mask_full = map_labels(gd)
            normalized_modalities_full = {
                key: normalize_nonzero_percentile(value)
                for key, value in modalities.items()
            }

            if self.roi_crop:
                if self.roi_masks:
                    if case_id not in self.roi_masks:
                        raise ValueError(f"Missing roi_masks entry for case {case_id}")
                    roi_volume = self.roi_masks[case_id]
                    if roi_volume.shape != gd.shape:
                        raise ValueError(f"ROI mask shape mismatch for case {case_id}: {roi_volume.shape} vs {gd.shape}")
                else:
                    roi_volume = gd

                case_crop_meta = compute_case_level_roi_crop_meta(
                    roi_volume, crop_size=(crop_h, crop_w), roi_padding=self.roi_padding,
                )
                crop_metas: list[CropRestoreMeta] = [case_crop_meta for _ in range(shape[2])]
                raw_slices = []
                normalized_slices = []
                mask_slices = []
                for slice_idx in range(shape[2]):
                    raw_slices.append(
                        np.stack(
                            [
                                crop_with_roi_meta(raw_modalities[modality][:, :, slice_idx], case_crop_meta, fill_value=0.0)
                                for modality in DEFAULT_MODALITIES
                            ],
                            axis=0,
                        ).astype(np.float32)
                    )
                    normalized_slices.append(
                        np.stack(
                            [
                                crop_with_roi_meta(
                                    normalized_modalities_full[modality][:, :, slice_idx],
                                    case_crop_meta,
                                    fill_value=0.0,
                                )
                                for modality in DEFAULT_MODALITIES
                            ],
                            axis=0,
                        ).astype(np.float32)
                    )
                    mask_slices.append(
                        crop_with_roi_meta(mask_full[:, :, slice_idx], case_crop_meta, fill_value=0, is_mask=True).astype(np.int64)
                    )
                image_stack = np.stack(normalized_slices, axis=0).astype(np.float32)
                raw_image_stack = np.stack(raw_slices, axis=0).astype(np.float32)
                mask_stack = np.stack(mask_slices, axis=0).astype(np.int64)
                crop_meta = None
            else:
                crop_meta = compute_center_crop_meta(shape[0], shape[1], crop_h, crop_w)
                crop_metas = [crop_meta for _ in range(shape[2])]
                cropped_modalities = {
                    key: center_crop_array(value, crop_meta, fill_value=0.0).astype(np.float32)
                    for key, value in modalities.items()
                }
                normalized_modalities = {
                    key: center_crop_array(normalized_modalities_full[key], crop_meta, fill_value=0.0)
                    for key in DEFAULT_MODALITIES
                }
                mask_crop = center_crop_array(mask_full, crop_meta, fill_value=0).astype(np.int64)
                image_stack = np.stack(
                    [normalized_modalities[modality] for modality in DEFAULT_MODALITIES],
                    axis=0,
                ).transpose(3, 0, 1, 2)
                raw_image_stack = np.stack(
                    [cropped_modalities[modality] for modality in DEFAULT_MODALITIES],
                    axis=0,
                ).transpose(3, 0, 1, 2)
                mask_stack = mask_crop.transpose(2, 0, 1)
            case_edema_pixels = int((mask_stack == PATHOLOGY_CLASS_INDICES[0]).sum())

            self.case_data[case_id] = {
                "case_id": case_id,
                "raw_image_stack": raw_image_stack.astype(np.float32),
                "image_stack": image_stack.astype(np.float32),
                "mask_stack": mask_stack.astype(np.int64),
                "mask_full": mask_full.astype(np.int64),
                "raw_modalities": raw_modalities,
                "spacing": spacing,
                "crop_meta": crop_meta,
                "crop_metas": crop_metas,
                "roi_crop": self.roi_crop,
                "orig_shape": shape,
                "case_edema_pixels": case_edema_pixels,
            }

            for slice_idx in range(image_stack.shape[0]):
                mask_slice = mask_stack[slice_idx]
                flags = pathology_flags(mask_slice)
                edema_pixels = int((mask_slice == PATHOLOGY_CLASS_INDICES[0]).sum())
                self.samples.append(
                    {
                        "case_id": case_id,
                        "slice_idx": slice_idx,
                        "edema_pixels": edema_pixels,
                        "case_edema_pixels": case_edema_pixels,
                        "is_small_edema": edema_pixels > 0 and edema_pixels < SMALL_EDEMA_PIXEL_THRESHOLD,
                        **flags,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _build_image_input(self, image_stack: np.ndarray, slice_idx: int) -> np.ndarray:
        if self.input_mode == "2d":
            return image_stack[slice_idx].copy()

        max_idx = image_stack.shape[0] - 1
        neighbor_indices = [
            max(0, slice_idx - 1),
            slice_idx,
            min(max_idx, slice_idx + 1),
        ]
        return np.concatenate([image_stack[idx] for idx in neighbor_indices], axis=0).astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        case = self.case_data[sample["case_id"]]
        image = self._build_image_input(case["image_stack"], sample["slice_idx"])
        mask = case["mask_stack"][sample["slice_idx"]].copy()
        if self.augment:
            image, mask = self.augmenter(image, mask)
        if self.zero_bssfp:
            if self.input_mode == "2p5d":
                image[0, :, :] = 0.0
                image[3, :, :] = 0.0
                image[6, :, :] = 0.0
            else:
                image[0, :, :] = 0.0
        result = {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "case_id": sample["case_id"],
            "slice_idx": sample["slice_idx"],
            "has_pathology": sample["has_pathology"],
            "has_edema": sample["has_edema"],
            "edema_pixels": sample["edema_pixels"],
        }
        if self.augment:
            result["dist_map"] = torch.from_numpy(one_hot2dist(mask, num_classes=6))
        return result


def _baseline_edema_weight(edema_pixels: int, pathology_oversample_factor: float) -> float:
    edema_weight = max(1.0, float(pathology_oversample_factor))
    small_edema_weight = edema_weight + 1.0
    if edema_pixels == 0:
        return 1.0
    if edema_pixels < SMALL_EDEMA_PIXEL_THRESHOLD:
        return small_edema_weight
    return edema_weight


def _targeted_edema_weight(edema_pixels: int, case_edema_pixels: int) -> float:
    if edema_pixels == 0:
        weight = 1.0
    elif edema_pixels <= 80:
        weight = 7.0
    elif edema_pixels <= 200:
        weight = 5.5
    elif edema_pixels <= 400:
        weight = 4.0
    elif edema_pixels <= 800:
        weight = 3.0
    else:
        weight = 2.0

    if edema_pixels > 0:
        if case_edema_pixels < 400:
            weight *= 1.5
        elif case_edema_pixels < 1000:
            weight *= 1.2

    return min(weight, MAX_TARGETED_SAMPLE_WEIGHT)


def build_sample_weights(
    dataset: MyoPSSliceDataset,
    pathology_oversample_factor: float,
    sampling_strategy: str = "baseline_edema",
) -> list[float]:
    weights = []
    for sample in dataset.samples:
        edema_pixels = int(sample["edema_pixels"])
        case_edema_pixels = int(sample["case_edema_pixels"])
        if sampling_strategy == "baseline_edema":
            weight = _baseline_edema_weight(edema_pixels, pathology_oversample_factor)
        elif sampling_strategy == "targeted_edema_bins":
            weight = _targeted_edema_weight(edema_pixels, case_edema_pixels)
        else:
            raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")
        weights.append(weight)
    return weights


def summarize_sample_weights(dataset: MyoPSSliceDataset, weights: list[float]) -> dict[str, float]:
    weight_array = np.asarray(weights, dtype=np.float32)
    edema_positive_mask = np.asarray([sample["edema_pixels"] > 0 for sample in dataset.samples], dtype=bool)
    summary = {
        "num_samples": float(len(dataset.samples)),
        "edema_positive_slices": float(edema_positive_mask.sum()),
        "weight_min": float(weight_array.min()),
        "weight_max": float(weight_array.max()),
        "weight_mean": float(weight_array.mean()),
        "weight_sum": float(weight_array.sum()),
    }
    if edema_positive_mask.any():
        summary["edema_positive_weight_mean"] = float(weight_array[edema_positive_mask].mean())
    else:
        summary["edema_positive_weight_mean"] = 0.0
    if (~edema_positive_mask).any():
        summary["edema_negative_weight_mean"] = float(weight_array[~edema_positive_mask].mean())
    else:
        summary["edema_negative_weight_mean"] = 0.0
    return summary


class MyoPS3DDataset(Dataset):
    def __init__(
        self,
        case_dirs: list[Path],
        crop_size: tuple[int, int] = (320, 320),
        augment: bool = False,
        use_t2_clahe: bool = True,
        t2_clahe_clip_limit: float = 2.0,
        t2_clahe_tile_grid_size: int = 8,
        roi_crop: bool = False,
        roi_masks: dict[str, np.ndarray] | None = None,
        roi_padding: int = 20,
    ) -> None:
        self.augment = augment
        self.cases: list[dict] = []
        for case_dir in case_dirs:
            case_id = case_dir.name
            modalities = {}
            shape = None
            for modality in DEFAULT_MODALITIES:
                path = next(case_dir.glob(f"*_{modality}.nii.gz"))
                image = nib.load(str(path))
                data = image.get_fdata().astype(np.float32)
                modalities[modality] = data
                shape = data.shape
            gd_path = next(case_dir.glob("*_gd.nii.gz"))
            gd = nib.load(str(gd_path)).get_fdata().astype(np.int32)
            if use_t2_clahe:
                modalities["T2"] = _apply_t2_clahe(
                    modalities["T2"], t2_clahe_clip_limit, t2_clahe_tile_grid_size,
                )
            normalized = {
                key: normalize_nonzero_percentile(value)
                for key, value in modalities.items()
            }
            mask_full = map_labels(gd)
            crop_h, crop_w = crop_size
            if roi_crop:
                roi_masks_dict = roi_masks or {}
                roi_vol = roi_masks_dict.get(case_id, gd)
                meta = compute_case_level_roi_crop_meta(
                    roi_vol, crop_size=(crop_h, crop_w), roi_padding=roi_padding,
                )
                vol = np.stack([
                    np.stack([
                        crop_with_roi_meta(normalized[m][:, :, z], meta, fill_value=0.0)
                        for m in DEFAULT_MODALITIES
                    ], axis=0)
                    for z in range(shape[2])
                ], axis=0)
                mask = np.stack([
                    crop_with_roi_meta(mask_full[:, :, z], meta, fill_value=0, is_mask=True)
                    for z in range(shape[2])
                ], axis=0)
            else:
                meta = compute_center_crop_meta(shape[0], shape[1], crop_h, crop_w)
                cropped = {
                    key: center_crop_array(normalized[key], meta, fill_value=0.0)
                    for key in DEFAULT_MODALITIES
                }
                vol = np.stack(
                    [cropped[m] for m in DEFAULT_MODALITIES], axis=0,
                ).transpose(3, 0, 1, 2)
                mask = center_crop_array(mask_full, meta, fill_value=0).transpose(2, 0, 1)
            self.cases.append({
                "case_id": case_id,
                "volume": vol.astype(np.float32),
                "mask": mask.astype(np.int64),
                "crop_meta": meta,
                "orig_shape": shape,
            })

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict:
        case = self.cases[index]
        vol = case["volume"].copy()
        mask = case["mask"].copy()
        if self.augment:
            if np.random.random() < 0.5:
                vol = np.flip(vol, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.random() < 0.5:
                vol = np.flip(vol, axis=3).copy()
                mask = np.flip(mask, axis=2).copy()
        vol_t = torch.from_numpy(vol).permute(1, 0, 2, 3)
        return {
            "image": vol_t,
            "mask": torch.from_numpy(mask),
            "case_id": case["case_id"],
        }
