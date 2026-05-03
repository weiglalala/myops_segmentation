"""Evaluate fixed mapping strategy: edema from baseline, scar from augmented model.
Produces merged predictions and computes full metrics including union dice."""
from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import build_parser, make_settings
from src.constants import EDEMA_CLASS_INDEX, SCAR_CLASS_INDEX
from src.engine import (
    build_test_loader,
    load_model_checkpoint,
    predict_with_ensemble,
    resolve_ensemble_weights,
    restore_cropped_array,
    NO_SCAR_CASES,
)
from src.metrics import evaluate_case

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASELINE_CHECKPOINTS = [f"checkpoints/exp_baseline_5fold_fold{i}/best_model.pt" for i in range(5)]
AUG_CHECKPOINTS = [f"checkpoints/exp_aug_5fold_fold{i}/best_model.pt" for i in range(5)]
BASELINE_CROP = (224, 224)
AUG_CROP = (320, 320)


def run_inference(settings, checkpoint_paths, device):
    dataset, loader = build_test_loader(settings)
    models = [
        load_model_checkpoint(Path(cp), settings.model_variant, settings.input_channels, device)
        for cp in checkpoint_paths
    ]
    weights = resolve_ensemble_weights(settings, num_models=len(models))

    case_preds: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            images = batch["image"].to(device)
            probs, _ = predict_with_ensemble(models, images, use_tta=False, weights=weights)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            for i, cid in enumerate(batch["case_id"]):
                store = dataset.case_data[cid]
                if cid not in case_preds:
                    case_preds[cid] = np.zeros_like(store["mask_full"])
                si = batch["slice_idx"][i].item()
                meta = store["crop_metas"][si]
                case_preds[cid][:, :, si] = restore_cropped_array(preds[i], meta)
    return dataset, case_preds


def merge_fixed_mapping(baseline_pred, aug_pred):
    merged = baseline_pred.copy()
    merged[baseline_pred == SCAR_CLASS_INDEX] = 0
    merged[aug_pred == SCAR_CLASS_INDEX] = SCAR_CLASS_INDEX
    return merged


def main():
    parser = build_parser()
    args = parser.parse_args()
    settings = make_settings(args=args, project_root=PROJECT_ROOT)
    device = torch.device(settings.device)

    print("Running baseline inference (edema source)...")
    s_base = replace(settings, crop_height=BASELINE_CROP[0], crop_width=BASELINE_CROP[1])
    base_paths = [str(PROJECT_ROOT / cp) for cp in BASELINE_CHECKPOINTS]
    dataset, base_preds = run_inference(s_base, base_paths, device)

    print("Running aug inference (scar source)...")
    s_aug = replace(settings, crop_height=AUG_CROP[0], crop_width=AUG_CROP[1])
    aug_paths = [str(PROJECT_ROOT / cp) for cp in AUG_CHECKPOINTS]
    _, aug_preds = run_inference(s_aug, aug_paths, device)

    output_dir = PROJECT_ROOT / "results" / "fixed_mapping" / "test_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cid in sorted(base_preds.keys()):
        merged = merge_fixed_mapping(base_preds[cid], aug_preds[cid])
        gt = dataset.case_data[cid]["mask_full"]
        spacing = dataset.case_data[cid].get("spacing") or (1.0, 1.0, 1.0)
        metrics = evaluate_case(merged, gt, spacing)
        rows.append({"case_id": cid, **metrics})

    csv_path = output_dir / "test_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scar_dices = [r["scar_dice"] for r in rows if r["case_id"] not in NO_SCAR_CASES]
    edema_dices = [r["edema_dice"] for r in rows]
    union_dices = [r["union_dice"] for r in rows]

    avg_e = sum(edema_dices) / len(edema_dices)
    avg_s = sum(scar_dices) / len(scar_dices)
    avg_u = sum(union_dices) / len(union_dices)

    print(f"\nFixed Mapping Results (edema=baseline, scar=aug):")
    print(f"  Edema Dice:  {avg_e:.4f}")
    print(f"  Scar Dice:   {avg_s:.4f}")
    print(f"  Mean Dice:   {(avg_e + avg_s) / 2:.4f}")
    print(f"  Union Dice:  {avg_u:.4f}")


if __name__ == "__main__":
    main()
