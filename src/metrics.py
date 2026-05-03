from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

from src.constants import PATHOLOGY_CLASS_INDICES


def dice_binary(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    pred_sum = prediction.sum()
    target_sum = target.sum()
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0
    intersection = np.logical_and(prediction, target).sum()
    return float(2.0 * intersection / (pred_sum + target_sum))


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    structure = generate_binary_structure(mask_a.ndim, 1)
    surface_a = np.logical_xor(mask_a, binary_erosion(mask_a, structure=structure, border_value=0))
    surface_b = np.logical_xor(mask_b, binary_erosion(mask_b, structure=structure, border_value=0))
    if not np.any(surface_a):
        surface_a = mask_a
    if not np.any(surface_b):
        surface_b = mask_b
    dt_b = distance_transform_edt(~surface_b, sampling=spacing)
    dt_a = distance_transform_edt(~surface_a, sampling=spacing)
    distances_a = dt_b[surface_a]
    distances_b = dt_a[surface_b]
    return np.concatenate([distances_a, distances_b]).astype(np.float32)


def hd95_binary(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    pred_sum = int(prediction.sum())
    target_sum = int(target.sum())
    if pred_sum == 0 and target_sum == 0:
        return 0.0
    if pred_sum == 0 or target_sum == 0:
        shape = np.array(prediction.shape, dtype=np.float32)
        spacing_array = np.array(spacing, dtype=np.float32)
        return float(np.sqrt(np.sum((shape * spacing_array) ** 2)))
    distances = _surface_distances(prediction, target, spacing=spacing)
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def evaluate_case(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for class_index, class_name in zip(PATHOLOGY_CLASS_INDICES, ("edema", "scar")):
        pred_binary = prediction == class_index
        target_binary = target == class_index
        metrics[f"{class_name}_dice"] = dice_binary(pred_binary, target_binary)
        metrics[f"{class_name}_hd95"] = hd95_binary(pred_binary, target_binary, spacing=spacing)
        metrics[f"{class_name}_inter"] = int(np.logical_and(pred_binary, target_binary).sum())
        metrics[f"{class_name}_pred_sum"] = int(pred_binary.sum())
        metrics[f"{class_name}_gt_sum"] = int(target_binary.sum())
    pred_union = np.isin(prediction, PATHOLOGY_CLASS_INDICES)
    target_union = np.isin(target, PATHOLOGY_CLASS_INDICES)
    metrics["union_dice"] = dice_binary(pred_union, target_union)
    metrics["union_hd95"] = hd95_binary(pred_union, target_union, spacing=spacing)
    metrics["mean_dice"] = (metrics["edema_dice"] + metrics["scar_dice"]) / 2.0
    metrics["mean_hd95"] = (metrics["edema_hd95"] + metrics["scar_hd95"]) / 2.0
    return metrics
