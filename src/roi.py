from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ROICropMeta:
    orig_h: int
    orig_w: int
    crop_h: int
    crop_w: int
    src_top: int
    src_bottom: int
    src_left: int
    src_right: int
    dst_top: int
    dst_bottom: int
    dst_left: int
    dst_right: int
    center_y: float
    center_x: float
    resize_target_h: int = 0
    resize_target_w: int = 0
    roi_h: int = 0
    roi_w: int = 0


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.ndim != 2:
        raise ValueError(f"bbox_from_mask expects a 2D mask, got ndim={mask.ndim}")
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    ymin = int(coords[:, 0].min())
    ymax = int(coords[:, 0].max())
    xmin = int(coords[:, 1].min())
    xmax = int(coords[:, 1].max())
    return xmin, xmax, ymin, ymax


def compute_roi_crop_meta(
    mask: np.ndarray,
    crop_size: tuple[int, int],
    padding: int,
    fallback_shape: tuple[int, int],
) -> ROICropMeta:
    crop_h, crop_w = crop_size
    orig_h, orig_w = fallback_shape
    bbox = bbox_from_mask(mask)

    if bbox is None:
        center_x = (orig_w - 1) / 2.0
        center_y = (orig_h - 1) / 2.0
    else:
        xmin, xmax, ymin, ymax = bbox
        xmin = max(0, xmin - padding)
        xmax = min(orig_w - 1, xmax + padding)
        ymin = max(0, ymin - padding)
        ymax = min(orig_h - 1, ymax + padding)
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0

    top = int(round(center_y - crop_h / 2.0))
    left = int(round(center_x - crop_w / 2.0))
    bottom = top + crop_h
    right = left + crop_w

    src_top = max(0, top)
    src_bottom = min(orig_h, bottom)
    src_left = max(0, left)
    src_right = min(orig_w, right)

    dst_top = src_top - top
    dst_left = src_left - left
    dst_bottom = dst_top + max(0, src_bottom - src_top)
    dst_right = dst_left + max(0, src_right - src_left)

    return ROICropMeta(
        orig_h=orig_h,
        orig_w=orig_w,
        crop_h=crop_h,
        crop_w=crop_w,
        src_top=src_top,
        src_bottom=src_bottom,
        src_left=src_left,
        src_right=src_right,
        dst_top=dst_top,
        dst_bottom=dst_bottom,
        dst_left=dst_left,
        dst_right=dst_right,
        center_y=center_y,
        center_x=center_x,
    )


def compute_roi_crop_resize_meta(
    mask: np.ndarray,
    target_size: tuple[int, int],
    padding: int,
    fallback_shape: tuple[int, int],
) -> ROICropMeta:
    """Crop tightly around the ROI bbox+padding, then resize to target_size."""
    target_h, target_w = target_size
    orig_h, orig_w = fallback_shape
    bbox = bbox_from_mask(mask)

    if bbox is None:
        center_x = (orig_w - 1) / 2.0
        center_y = (orig_h - 1) / 2.0
        roi_h, roi_w = target_h, target_w
    else:
        xmin, xmax, ymin, ymax = bbox
        xmin = max(0, xmin - padding)
        xmax = min(orig_w - 1, xmax + padding)
        ymin = max(0, ymin - padding)
        ymax = min(orig_h - 1, ymax + padding)
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        roi_h = ymax - ymin + 1
        roi_w = xmax - xmin + 1

    # Make the crop square using the larger dimension
    side = max(roi_h, roi_w)
    top = int(round(center_y - side / 2.0))
    left = int(round(center_x - side / 2.0))
    bottom = top + side
    right = left + side

    src_top = max(0, top)
    src_bottom = min(orig_h, bottom)
    src_left = max(0, left)
    src_right = min(orig_w, right)

    dst_top = src_top - top
    dst_left = src_left - left
    dst_bottom = dst_top + (src_bottom - src_top)
    dst_right = dst_left + (src_right - src_left)

    return ROICropMeta(
        orig_h=orig_h,
        orig_w=orig_w,
        crop_h=side,
        crop_w=side,
        src_top=src_top,
        src_bottom=src_bottom,
        src_left=src_left,
        src_right=src_right,
        dst_top=dst_top,
        dst_bottom=dst_bottom,
        dst_left=dst_left,
        dst_right=dst_right,
        center_y=center_y,
        center_x=center_x,
        resize_target_h=target_h,
        resize_target_w=target_w,
        roi_h=roi_h,
        roi_w=roi_w,
    )


def crop_with_roi_meta(
    array: np.ndarray, meta: ROICropMeta, fill_value: float = 0.0, is_mask: bool = False,
) -> np.ndarray:
    if array.ndim == 2:
        output = np.full((meta.crop_h, meta.crop_w), fill_value, dtype=array.dtype)
        output[meta.dst_top : meta.dst_bottom, meta.dst_left : meta.dst_right] = array[
            meta.src_top : meta.src_bottom,
            meta.src_left : meta.src_right,
        ]
    elif array.ndim == 3:
        output = np.full((meta.crop_h, meta.crop_w, array.shape[2]), fill_value, dtype=array.dtype)
        output[meta.dst_top : meta.dst_bottom, meta.dst_left : meta.dst_right, :] = array[
            meta.src_top : meta.src_bottom,
            meta.src_left : meta.src_right,
            :,
        ]
    else:
        raise ValueError(f"Unsupported ndim for ROI crop: {array.ndim}")

    if meta.resize_target_h > 0 and meta.resize_target_w > 0:
        target = (meta.resize_target_w, meta.resize_target_h)
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        output = cv2.resize(output, target, interpolation=interp)

    return output


def restore_roi_crop(array: np.ndarray, meta: ROICropMeta) -> np.ndarray:
    if meta.resize_target_h > 0 and meta.resize_target_w > 0:
        interp = cv2.INTER_NEAREST
        array = cv2.resize(
            array, (meta.crop_w, meta.crop_h), interpolation=interp,
        )

    if array.ndim == 2:
        output = np.zeros((meta.orig_h, meta.orig_w), dtype=array.dtype)
        output[meta.src_top : meta.src_bottom, meta.src_left : meta.src_right] = array[
            meta.dst_top : meta.dst_bottom,
            meta.dst_left : meta.dst_right,
        ]
        return output

    if array.ndim == 3:
        output = np.zeros((meta.orig_h, meta.orig_w, array.shape[2]), dtype=array.dtype)
        output[meta.src_top : meta.src_bottom, meta.src_left : meta.src_right, :] = array[
            meta.dst_top : meta.dst_bottom,
            meta.dst_left : meta.dst_right,
            :,
        ]
        return output

    raise ValueError(f"Unsupported ndim for ROI restore: {array.ndim}")
