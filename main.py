from __future__ import annotations

from pathlib import Path

from src.config import build_parser, make_settings
from src.engine import ensemble_test_pipeline, test_pipeline, train_pipeline
from src.fusion_inference import fusion_test_pipeline
from src.two_stage import run_two_stage


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    settings = make_settings(args=args, project_root=project_root)

    if settings.pipeline_mode == "fusion":
        fusion_test_pipeline(settings)
        return

    if settings.pipeline_mode == "two_stage":
        run_two_stage(settings)
        return

    if settings.mode == "train":
        train_pipeline(settings)
        return
    if settings.mode == "test":
        if settings.ensemble_checkpoint_paths:
            ensemble_test_pipeline(settings)
            return
        test_pipeline(settings)
        return
    raise ValueError(f"Unsupported mode: {settings.mode}")


if __name__ == "__main__":
    main()
