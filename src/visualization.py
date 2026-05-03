from __future__ import annotations

import os
from pathlib import Path

import matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg", force=True)
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.constants import CLASS_COLORS, CLASS_NAMES, PATHOLOGY_CLASS_INDICES


def to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mask = image > 0
    if np.any(mask):
        low = np.percentile(image[mask], 1)
        high = np.percentile(image[mask], 99)
    else:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        high = low + 1.0
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def overlay_segmentation(base_image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    gray = to_grayscale_uint8(base_image)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    output = rgb.copy()
    for class_index, color in CLASS_COLORS.items():
        if class_index == 0:
            continue
        region = mask == class_index
        if not np.any(region):
            continue
        color_array = np.array(color, dtype=np.float32)
        output[region] = (1.0 - alpha) * output[region] + alpha * color_array
    return np.clip(output, 0, 255).astype(np.uint8)


def select_visualization_slices(gt_volume: np.ndarray, pred_volume: np.ndarray, visualize_all_slices: bool) -> list[int]:
    depth = gt_volume.shape[2]
    if visualize_all_slices:
        return list(range(depth))
    chosen = [
        index
        for index in range(depth)
        if np.isin(gt_volume[:, :, index], PATHOLOGY_CLASS_INDICES).any()
        or np.isin(pred_volume[:, :, index], PATHOLOGY_CLASS_INDICES).any()
    ]
    if chosen:
        return chosen
    return [depth // 2]


def save_case_visualizations(
    case_id: str,
    base_volume: np.ndarray,
    gt_volume: np.ndarray,
    pred_volume: np.ndarray,
    output_dir: Path,
    visualize_all_slices: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    legend_handles = [
        Patch(facecolor=np.array(CLASS_COLORS[idx]) / 255.0, edgecolor="black", label=CLASS_NAMES[idx])
        for idx in range(1, len(CLASS_NAMES))
    ]
    for slice_idx in select_visualization_slices(gt_volume, pred_volume, visualize_all_slices=visualize_all_slices):
        base = base_volume[:, :, slice_idx]
        gt = gt_volume[:, :, slice_idx]
        pred = pred_volume[:, :, slice_idx]

        figure = Figure(figsize=(15, 5))
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3)
        axes[0].imshow(to_grayscale_uint8(base), cmap="gray")
        axes[0].set_title("Input")
        axes[1].imshow(overlay_segmentation(base, gt))
        axes[1].set_title("Ground Truth")
        axes[2].imshow(overlay_segmentation(base, pred))
        axes[2].set_title("Prediction")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(f"{case_id} | slice {slice_idx}")
        figure.legend(handles=legend_handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
        figure.tight_layout()
        save_path = output_dir / f"{case_id}_slice_{slice_idx:02d}.png"
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        figure.clear()


def save_training_curves(history: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_mean_dice = [row["val_mean_dice"] for row in history]
    val_edema_dice = [row["val_edema_dice"] for row in history]
    val_scar_dice = [row["val_scar_dice"] for row in history]

    figure = Figure(figsize=(12, 4.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    axes[0].plot(epochs, train_loss, label="train_loss", color="tab:red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(epochs, val_mean_dice, label="val_mean_dice", color="tab:blue")
    axes[1].plot(epochs, val_edema_dice, label="val_edema_dice", color="tab:orange")
    axes[1].plot(epochs, val_scar_dice, label="val_scar_dice", color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Validation Dice")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    figure.clear()
