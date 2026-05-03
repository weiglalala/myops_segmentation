from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.constants import DEFAULT_CROP_SIZE, DEFAULT_VIS_MODALITY


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


@dataclass
class Settings:
    project_root: Path
    mode: str = "train"
    pipeline_mode: str = "single_stage"
    two_stage_step: str = "stage1_train"
    experiment_name: str = "mainline_attention_se_myo"
    model_variant: str = "attention_se_myo"
    stage1_model_variant: str = "unet"
    input_mode: str = "2p5d"
    stage1_input_mode: str = "2d"
    stage2_input_mode: str = "2p5d"
    stage1_modality: str = "C0"
    stage1_target_type: str = "myocardium"
    stage1_preprocess_mode: str = "resize_full"
    stage2_roi_mode: str = "case"
    stage2_num_classes: int = 6
    seed: int = 3407
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    data_root: Path = field(default_factory=Path)
    train_dir: Path = field(default_factory=Path)
    test_dir: Path = field(default_factory=Path)
    checkpoint_root: Path = field(default_factory=Path)
    log_root: Path = field(default_factory=Path)
    result_root: Path = field(default_factory=Path)
    checkpoint_path: Path | None = None
    ensemble_checkpoint_paths: tuple[Path, ...] = ()
    ensemble_model_variants: tuple[str, ...] = ()
    ensemble_weights: tuple[float, ...] = ()
    stage1_checkpoint_path: Path | None = None
    stage2_checkpoint_path: Path | None = None

    crop_height: int = DEFAULT_CROP_SIZE[0]
    crop_width: int = DEFAULT_CROP_SIZE[1]
    roi_padding: int = 20
    use_roi_crop: bool = False
    batch_size: int = 4
    num_workers: int = 0
    epochs: int = 300
    learning_rate: float = 1e-3
    encoder_lr: float = 0.0
    weight_decay: float = 1e-4
    scheduler_type: str = "plateau"
    cosine_t0: int = 50
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    early_stopping_patience: int = 50
    early_abort_epoch: int = 0
    early_abort_min_dice: float = 0.0
    val_ratio: float = 0.2
    fixed_val_cases: tuple[str, ...] = ("Case3011", "Case3006", "Case3025", "Case3010", "Case3004")
    num_folds: int = 5
    fold_index: int | None = None
    sampling_strategy: str = "baseline_edema"
    pathology_oversample_factor: float = 2.0
    myocardium_prior_weight: float = 0.25
    use_t2_clahe: bool = True
    t2_clahe_clip_limit: float = 2.0
    t2_clahe_tile_grid_size: int = 8

    dice_loss_weight: float = 1.0
    ce_loss_weight: float = 1.0
    focal_loss_weight: float = 0.0
    focal_gamma: float = 2.0
    tversky_loss_weight: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    edema_dice_weight: float = 0.0
    surface_loss_weight: float = 0.0
    union_loss_weight: float = 0.0
    inclusiveness_loss_weight: float = 0.0
    scar_class_weight: float = 1.10

    rotation_degrees: float = 30.0
    enable_elastic: bool = True
    enable_advanced_augment: bool = False
    enable_tta: bool = True
    use_connected_component: bool = False
    use_edema_constraint: bool = False
    use_union_constraint: bool = False
    use_coarse_input: bool = False
    use_t2_roi_norm: bool = False
    use_edema_expert_weights: bool = False
    coarse_oof_dir: Path | None = None
    scar_experiment: str = ""
    edema_experiment: str = ""
    scar_checkpoint_paths: tuple[Path, ...] = ()
    scar_model_variants: tuple[str, ...] = ()
    edema_checkpoint_paths: tuple[Path, ...] = ()
    edema_model_variants: tuple[str, ...] = ()
    fusion_edema_constraint: bool = True
    fusion_edema_min_size: int = 50
    scar_num_classes: int = 2
    edema_num_classes: int = 3
    vis_modality: str = DEFAULT_VIS_MODALITY
    save_visualizations: bool = True
    visualize_all_slices: bool = False
    zero_bssfp: bool = False

    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_test_cases: int | None = None

    @property
    def crop_size(self) -> tuple[int, int]:
        return self.crop_height, self.crop_width

    @property
    def input_channels(self) -> int:
        return 9 if self.input_mode == "2p5d" else 3

    @property
    def stage1_input_channels(self) -> int:
        return 3 if self.stage1_modality == "all" else 1

    @property
    def stage1_num_classes(self) -> int:
        if self.stage1_target_type == "anatomy":
            return 4
        return 2

    @property
    def stage2_input_channels(self) -> int:
        base = 9 if self.stage2_input_mode == "2p5d" else 3
        return base + (3 if self.use_coarse_input else 0)

    @property
    def experiment_ckpt_dir(self) -> Path:
        return self.checkpoint_root / self.experiment_name

    @property
    def experiment_log_dir(self) -> Path:
        return self.log_root / self.experiment_name

    @property
    def experiment_result_dir(self) -> Path:
        return self.result_root / self.experiment_name

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: _serialize_value(value) for key, value in payload.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MyoPS multimodal segmentation")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--pipeline-mode", choices=("single_stage", "two_stage", "fusion"), default="single_stage")
    parser.add_argument(
        "--two-stage-step",
        choices=("stage1_train", "stage1_oof", "stage2_train", "two_stage_test"),
        default="stage1_train",
    )
    parser.add_argument("--experiment-name", default="mainline_attention_se_myo")
    parser.add_argument(
        "--model-variant",
        choices=(
            "unet",
            "attention_unet",
            "attention_se",
            "attention_se_myo",
            "attention_se_myo_guided",
            "smp_unet",
            "smp_unetpp",
            "smp_deeplabv3p",
            "smp_unet_effb4",
            "smp_unet_3ch",
            "unet_3d",
        ),
        default="attention_se_myo",
    )
    parser.add_argument(
        "--stage1-model-variant",
        choices=("unet", "attention_unet", "attention_se"),
        default="unet",
    )
    parser.add_argument("--input-mode", choices=("2d", "2p5d"), default="2p5d")
    parser.add_argument("--stage1-input-mode", choices=("2d",), default="2d")
    parser.add_argument("--stage2-input-mode", choices=("2d", "2p5d"), default="2p5d")
    parser.add_argument("--stage1-modality", choices=("C0", "T2", "LGE", "all"), default="C0")
    parser.add_argument("--stage1-target-type", choices=("foreground", "myocardium", "anatomy"), default="myocardium")
    parser.add_argument("--stage1-preprocess-mode", choices=("center_crop", "resize_full"), default="resize_full")
    parser.add_argument("--stage2-roi-mode", choices=("slice", "case"), default="case")
    parser.add_argument("--stage2-num-classes", type=int, choices=(2, 3, 6), default=6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--ensemble-checkpoint-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--ensemble-model-variants", nargs="*", default=None)
    parser.add_argument("--ensemble-weights", type=float, nargs="*", default=None)
    parser.add_argument("--stage1-checkpoint-path", type=Path, default=None)
    parser.add_argument("--stage2-checkpoint-path", type=Path, default=None)
    parser.add_argument("--crop-height", type=int, default=224)
    parser.add_argument("--crop-width", type=int, default=224)
    parser.add_argument("--roi-padding", type=int, default=20)
    parser.add_argument("--use-roi-crop", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-type", choices=("plateau", "cosine"), default="plateau")
    parser.add_argument("--cosine-t0", type=int, default=50)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--early-abort-epoch", type=int, default=0)
    parser.add_argument("--early-abort-min-dice", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--fixed-val-cases",
        nargs="*",
        default=["Case3011", "Case3006", "Case3025", "Case3010", "Case3004"],
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=None)
    parser.add_argument(
        "--sampling-strategy",
        choices=("baseline_edema", "targeted_edema_bins"),
        default="baseline_edema",
    )
    parser.add_argument("--pathology-oversample-factor", type=float, default=2.0)
    parser.add_argument("--myocardium-prior-weight", type=float, default=0.25)
    parser.add_argument("--disable-t2-clahe", action="store_true")
    parser.add_argument("--t2-clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--t2-clahe-tile-grid-size", type=int, default=8)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--ce-loss-weight", type=float, default=1.0)
    parser.add_argument("--focal-loss-weight", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--tversky-loss-weight", type=float, default=0.0)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--edema-dice-weight", type=float, default=0.0)
    parser.add_argument("--surface-loss-weight", type=float, default=0.0)
    parser.add_argument("--union-loss-weight", type=float, default=0.0)
    parser.add_argument("--inclusiveness-loss-weight", type=float, default=0.0)
    parser.add_argument("--scar-class-weight", type=float, default=1.10)
    parser.add_argument("--rotation-degrees", type=float, default=30.0)
    parser.add_argument("--disable-elastic", action="store_true")
    parser.add_argument("--enable-advanced-augment", action="store_true")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--use-connected-component", action="store_true")
    parser.add_argument("--use-edema-constraint", action="store_true")
    parser.add_argument("--use-union-constraint", action="store_true")
    parser.add_argument("--use-coarse-input", action="store_true")
    parser.add_argument("--use-t2-roi-norm", action="store_true")
    parser.add_argument("--use-edema-expert-weights", action="store_true")
    parser.add_argument("--coarse-oof-dir", type=Path, default=None)
    parser.add_argument("--scar-experiment", default="")
    parser.add_argument("--edema-experiment", default="")
    parser.add_argument("--scar-checkpoint-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--scar-model-variants", nargs="*", default=None)
    parser.add_argument("--edema-checkpoint-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--edema-model-variants", nargs="*", default=None)
    parser.add_argument("--scar-num-classes", type=int, choices=(2, 3, 6), default=2)
    parser.add_argument("--edema-num-classes", type=int, choices=(2, 3, 6), default=3)
    parser.add_argument("--fusion-edema-constraint", action="store_true", default=True)
    parser.add_argument("--no-fusion-edema-constraint", dest="fusion_edema_constraint", action="store_false")
    parser.add_argument("--fusion-edema-min-size", type=int, default=50)
    parser.add_argument("--vis-modality", choices=("C0", "T2", "LGE"), default=DEFAULT_VIS_MODALITY)
    parser.add_argument("--disable-visualizations", action="store_true")
    parser.add_argument("--visualize-all-slices", action="store_true")
    parser.add_argument("--zero-bssfp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-cases", type=int, default=None)
    return parser


def make_settings(args: argparse.Namespace, project_root: Path) -> Settings:
    data_root = (project_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    train_dir = args.train_dir or data_root / "train"
    test_dir = args.test_dir or data_root / "test"
    checkpoint_root = (
        (project_root / args.checkpoint_root).resolve()
        if not args.checkpoint_root.is_absolute()
        else args.checkpoint_root
    )
    log_root = (project_root / args.log_root).resolve() if not args.log_root.is_absolute() else args.log_root
    result_root = (
        (project_root / args.result_root).resolve()
        if not args.result_root.is_absolute()
        else args.result_root
    )
    checkpoint_path = None
    if args.checkpoint_path is not None:
        checkpoint_path = (
            (project_root / args.checkpoint_path).resolve()
            if not args.checkpoint_path.is_absolute()
            else args.checkpoint_path
        )
    ensemble_checkpoint_paths: tuple[Path, ...] = ()
    if args.ensemble_checkpoint_paths:
        ensemble_checkpoint_paths = tuple(
            (
                (project_root / checkpoint_path_arg).resolve()
                if not checkpoint_path_arg.is_absolute()
                else checkpoint_path_arg
            )
            for checkpoint_path_arg in args.ensemble_checkpoint_paths
        )
    ensemble_weights = tuple(args.ensemble_weights) if args.ensemble_weights else ()
    ensemble_model_variants = tuple(args.ensemble_model_variants) if args.ensemble_model_variants else ()
    stage1_checkpoint_path = None
    if args.stage1_checkpoint_path is not None:
        stage1_checkpoint_path = (
            (project_root / args.stage1_checkpoint_path).resolve()
            if not args.stage1_checkpoint_path.is_absolute()
            else args.stage1_checkpoint_path
        )
    stage2_checkpoint_path = None
    if args.stage2_checkpoint_path is not None:
        stage2_checkpoint_path = (
            (project_root / args.stage2_checkpoint_path).resolve()
            if not args.stage2_checkpoint_path.is_absolute()
            else args.stage2_checkpoint_path
        )
    return Settings(
        project_root=project_root,
        mode=args.mode,
        pipeline_mode=args.pipeline_mode,
        two_stage_step=args.two_stage_step,
        experiment_name=args.experiment_name,
        model_variant=args.model_variant,
        stage1_model_variant=args.stage1_model_variant,
        input_mode=args.input_mode,
        stage1_input_mode=args.stage1_input_mode,
        stage2_input_mode=args.stage2_input_mode,
        stage1_modality=args.stage1_modality,
        stage1_target_type=args.stage1_target_type,
        stage1_preprocess_mode=args.stage1_preprocess_mode,
        stage2_roi_mode=args.stage2_roi_mode,
        stage2_num_classes=args.stage2_num_classes,
        seed=args.seed,
        device=args.device,
        data_root=data_root,
        train_dir=train_dir,
        test_dir=test_dir,
        checkpoint_root=checkpoint_root,
        log_root=log_root,
        result_root=result_root,
        checkpoint_path=checkpoint_path,
        ensemble_checkpoint_paths=ensemble_checkpoint_paths,
        ensemble_model_variants=ensemble_model_variants,
        ensemble_weights=ensemble_weights,
        stage1_checkpoint_path=stage1_checkpoint_path,
        stage2_checkpoint_path=stage2_checkpoint_path,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        roi_padding=args.roi_padding,
        use_roi_crop=args.use_roi_crop,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        cosine_t0=args.cosine_t0,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        early_stopping_patience=args.early_stopping_patience,
        early_abort_epoch=args.early_abort_epoch,
        early_abort_min_dice=args.early_abort_min_dice,
        val_ratio=args.val_ratio,
        fixed_val_cases=tuple(args.fixed_val_cases),
        num_folds=args.num_folds,
        fold_index=args.fold_index,
        sampling_strategy=args.sampling_strategy,
        pathology_oversample_factor=args.pathology_oversample_factor,
        myocardium_prior_weight=args.myocardium_prior_weight,
        use_t2_clahe=not args.disable_t2_clahe,
        t2_clahe_clip_limit=args.t2_clahe_clip_limit,
        t2_clahe_tile_grid_size=args.t2_clahe_tile_grid_size,
        dice_loss_weight=args.dice_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        focal_loss_weight=args.focal_loss_weight,
        focal_gamma=args.focal_gamma,
        tversky_loss_weight=args.tversky_loss_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        edema_dice_weight=args.edema_dice_weight,
        surface_loss_weight=args.surface_loss_weight,
        union_loss_weight=args.union_loss_weight,
        inclusiveness_loss_weight=args.inclusiveness_loss_weight,
        scar_class_weight=args.scar_class_weight,
        rotation_degrees=args.rotation_degrees,
        enable_elastic=not args.disable_elastic,
        enable_advanced_augment=args.enable_advanced_augment,
        enable_tta=not args.disable_tta,
        use_connected_component=args.use_connected_component,
        use_edema_constraint=args.use_edema_constraint,
        use_union_constraint=args.use_union_constraint,
        use_coarse_input=args.use_coarse_input,
        use_t2_roi_norm=args.use_t2_roi_norm,
        use_edema_expert_weights=args.use_edema_expert_weights,
        coarse_oof_dir=(
            (project_root / args.coarse_oof_dir).resolve()
            if args.coarse_oof_dir is not None and not args.coarse_oof_dir.is_absolute()
            else args.coarse_oof_dir
        ),
        scar_experiment=args.scar_experiment,
        edema_experiment=args.edema_experiment,
        scar_checkpoint_paths=tuple(
            (project_root / p).resolve() if not p.is_absolute() else p
            for p in (args.scar_checkpoint_paths or [])
        ),
        scar_model_variants=tuple(args.scar_model_variants or []),
        edema_checkpoint_paths=tuple(
            (project_root / p).resolve() if not p.is_absolute() else p
            for p in (args.edema_checkpoint_paths or [])
        ),
        edema_model_variants=tuple(args.edema_model_variants or []),
        scar_num_classes=args.scar_num_classes,
        edema_num_classes=args.edema_num_classes,
        fusion_edema_constraint=args.fusion_edema_constraint,
        fusion_edema_min_size=args.fusion_edema_min_size,
        vis_modality=args.vis_modality,
        save_visualizations=not args.disable_visualizations,
        visualize_all_slices=args.visualize_all_slices,
        zero_bssfp=args.zero_bssfp,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_test_cases=args.max_test_cases,
    )
