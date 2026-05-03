"""Run ensemble testing for baseline, augmented, or 10-model configurations."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

CONFIGS = {
    "baseline": {
        "checkpoints": [f"checkpoints/exp_baseline_5fold_fold{i}/best_model.pt" for i in range(5)],
        "crop": (224, 224),
        "output": "ensemble_baseline_5fold",
    },
    "aug": {
        "checkpoints": [f"checkpoints/exp_aug_5fold_fold{i}/best_model.pt" for i in range(5)],
        "crop": (320, 320),
        "output": "ensemble_aug_5fold",
    },
    "10model": {
        "checkpoints": (
            [f"checkpoints/exp_baseline_5fold_fold{i}/best_model.pt" for i in range(5)]
            + [f"checkpoints/exp_aug_5fold_fold{i}/best_model.pt" for i in range(5)]
        ),
        "crop": (224, 224),
        "output": "ensemble_10model",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(CONFIGS), required=True)
    args = parser.parse_args()

    cfg = CONFIGS[args.experiment]
    ckpts = [str(PROJECT_ROOT / cp) for cp in cfg["checkpoints"]]

    cmd = [
        PYTHON, str(PROJECT_ROOT / "main.py"),
        "--mode", "test",
        "--model-variant", "smp_unet",
        "--input-mode", "2p5d",
        "--experiment-name", cfg["output"],
        "--crop-height", str(cfg["crop"][0]),
        "--crop-width", str(cfg["crop"][1]),
        "--ensemble-checkpoint-paths", *ckpts,
    ]

    print(f"Running ensemble test: {args.experiment}")
    print(f"  Checkpoints: {len(ckpts)}")
    print(f"  Crop: {cfg['crop']}")
    print(f"  Output: {cfg['output']}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[FAILED] exit code {result.returncode}")
    else:
        print("[DONE]")


if __name__ == "__main__":
    main()
