"""Run ensemble test on trained 5-fold checkpoints."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["baseline", "aug", "10model"], default="baseline")
    args = parser.parse_args()

    if args.experiment == "baseline":
        name = "ensemble_baseline_5fold"
        checkpoints = [f"checkpoints/exp_baseline_5fold_fold{i}/best_model.pt" for i in range(5)]
        crop = ("224", "224")
    elif args.experiment == "aug":
        name = "ensemble_aug_5fold"
        checkpoints = [f"checkpoints/exp_aug_5fold_fold{i}/best_model.pt" for i in range(5)]
        crop = ("320", "320")
    else:
        name = "ensemble_10model"
        checkpoints = (
            [f"checkpoints/exp_baseline_5fold_fold{i}/best_model.pt" for i in range(5)]
            + [f"checkpoints/exp_aug_5fold_fold{i}/best_model.pt" for i in range(5)]
        )
        crop = ("320", "320")

    checkpoint_args = []
    for cp in checkpoints:
        checkpoint_args.extend(["--ensemble-checkpoint-paths", str(PROJECT_ROOT / cp)])

    cmd = [
        PYTHON, str(PROJECT_ROOT / "main.py"),
        "--mode", "test",
        "--model-variant", "smp_unet",
        "--input-mode", "2p5d",
        "--experiment-name", name,
        "--crop-height", crop[0], "--crop-width", crop[1],
        *checkpoint_args,
    ]

    print(f"Running ensemble test: {name}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[FAILED] exit code {result.returncode}")
    else:
        print(f"[DONE] Results saved to results/{name}/")


if __name__ == "__main__":
    main()
