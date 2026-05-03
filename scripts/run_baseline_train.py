"""5-fold cross-validation training with baseline configuration."""
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
    "--learning-rate", "1e-3",
    "--crop-height", "224", "--crop-width", "224",
]

EXPERIMENT_NAME = "exp_baseline_5fold"


def main() -> None:
    print(f"Training baseline 5-fold: {EXPERIMENT_NAME}")
    print(f"Config: original data, lr=1e-3, crop=224\n")

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

    print("\nAll 5 folds complete.")


if __name__ == "__main__":
    main()
