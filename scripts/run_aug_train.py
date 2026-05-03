"""Train augmented 5-fold model: aug data, lr=5e-4, crop=320."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

BASE_ARGS = [
    "--mode", "train",
    "--model-variant", "smp_unet",
    "--input-mode", "2p5d",
    "--batch-size", "8",
    "--num-folds", "5",
    "--epochs", "300",
    "--scheduler-patience", "15",
    "--early-stopping-patience", "60",
    "--focal-loss-weight", "1.0",
    "--rotation-degrees", "15",
    "--seed", "3407",
    "--sampling-strategy", "baseline_edema",
    "--pathology-oversample-factor", "2.0",
    "--myocardium-prior-weight", "0.25",
    "--disable-t2-clahe",
    "--train-dir", str(PROJECT_ROOT / "data" / "train_augmented"),
    "--learning-rate", "5e-4",
    "--crop-height", "320", "--crop-width", "320",
]

EXPERIMENT_NAME = "exp_aug_5fold"


def main() -> None:
    print(f"Training augmented 5-fold: {EXPERIMENT_NAME}")
    print(f"Config: aug data, lr=5e-4, crop=320\n")

    for fold in range(5):
        print(f"\n{'='*60}")
        print(f"Starting fold {fold}/4")
        print(f"{'='*60}\n")

        cmd = [
            PYTHON, str(PROJECT_ROOT / "main.py"),
            "--experiment-name", EXPERIMENT_NAME,
            "--fold-index", str(fold),
            *BASE_ARGS,
        ]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"[FAILED] fold {fold} exited with code {result.returncode}")
        else:
            print(f"[DONE] fold {fold}")

    print("\n" + "="*60)
    print("ALL 5 FOLDS COMPLETE.")
    print("="*60)


if __name__ == "__main__":
    main()
