# MyoPS Segmentation

Automated segmentation of myocardial pathology (edema and scar) from multi-sequence cardiac MRI, built for the [MyoPS 2020 Challenge]([MyoPS 2020](https://zmiclab.github.io/zxh/0/myops20/index.html)).

## Motivation

The core challenge here is the extremely small dataset: only 25 labeled training cases, with pathology regions occupying less than 0.6% of total voxels. That rules out most heavy architectures and fancy training tricks — they just overfit immediately.

I went with a 2.5D approach: stack 3 adjacent slices across 3 MRI modalities (C0, T2, LGE) into a 9-channel 2D input, then feed it into a standard UNet with a ResNet34 encoder. Simple, but it keeps the parameter count low enough to actually learn from 25 cases. 5-fold cross-validation with softmax averaging gives a solid ensemble without needing multiple architectures.

The most interesting finding was around data augmentation. I generated offline augmented data (elastic deformation, intensity shifts, spatial transforms) to expand the training set from 25 to 125 cases. This significantly improved scar segmentation (+0.040 Dice) but actually hurt edema (-0.029 Dice). Edema boundaries are inherently fuzzy and irregular, so the augmented distribution shifts made things worse for that class.

This led to the **fixed mapping strategy**: use the baseline model for edema predictions and the augmented model for scar predictions. It's a simple idea, but it gave the best overall Mean Dice (0.532) by combining each model's strength.

I also spent time on directions that didn't work out — task decomposition into separate expert models, scar class weight tuning, and constraining predictions to the myocardium region. All failed, and the reasons are documented below. The myocardium constraint was particularly surprising: even using perfect ground-truth myocardium masks, the Dice score dropped dramatically, because the dataset annotations themselves place scar and edema outside the myocardium boundary.

**Final results**: Scar Dice **0.699** (close to the challenge champion's 0.708), Mean Dice **0.532**, Union Dice **0.695**. The main gap to the champion (UESTC, Mean Dice 0.720) is in edema segmentation — that remains the hard problem.

## Method Overview

This project implements a 2.5D multi-modal segmentation pipeline:

- **Architecture**: UNet (via [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)) with ResNet34 encoder
- **Input**: 2.5D slices — 3 adjacent slices × 3 MRI modalities (C0, T2, LGE) = 9 channels
- **Training**: 5-fold cross-validation with combined Focal + Dice loss
- **Inference**: Ensemble of 5-fold models with softmax averaging
- **Data Augmentation**: Offline augmentation (elastic deformation, intensity shifts, spatial transforms) expanding 25 cases to 125

### Ensemble Strategies

| Strategy | Edema Dice | Scar Dice | Mean Dice | Union Dice |
|---|---|---|---|---|
| Baseline 5-fold | 0.375 | 0.659 | 0.517 | 0.695 |
| Augmented 5-fold | 0.346 | 0.699 | 0.523 | 0.674 |
| 10-model ensemble | 0.373 | 0.686 | 0.530 | 0.690 |
| Fixed mapping | 0.365 | 0.698 | 0.532 | 0.681 |

**Best results**: Scar Dice **0.699** (augmented model), Union Dice **0.695** (baseline), Mean Dice **0.532** (fixed mapping).

*Union Dice* measures the Dice score over the combined scar ∪ edema region.

*Fixed mapping* takes edema predictions from the baseline model and scar predictions from the augmented model.

## Dataset

This project uses the **MyoPS 2020 Challenge** dataset (25 train / 20 test cases). See [data/README.md](data/README.md) for download instructions and expected directory structure.

**6-class segmentation**: background, normal myocardium, LV pool, RV pool, edema, scar.

## Installation

```bash
git clone https://github.com/weiglalala/myops_segmentation.git
cd myops_segmentation
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU.

## Usage

### Data Augmentation (Optional)

Generate augmented training data (25 → 125 cases):

```bash
python augment_offline.py
```

### Training

Train baseline 5-fold model:

```bash
python scripts/run_baseline_train.py
```

Train with augmented data:

```bash
python scripts/run_aug_train.py
```

Or train a single fold manually:

```bash
python main.py \
    --mode train \
    --model-variant smp_unet \
    --input-mode 2p5d \
    --experiment-name my_experiment \
    --fold-index 0 \
    --num-folds 5 \
    --epochs 300 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --crop-height 224 --crop-width 224
```

### Ensemble Testing

```bash
python scripts/run_ensemble_test.py --experiment baseline
python scripts/run_ensemble_test.py --experiment aug
python scripts/run_ensemble_test.py --experiment 10model
```

### Fixed Mapping Evaluation

```bash
python scripts/evaluate_mapping.py --mode test --model-variant smp_unet --input-mode 2p5d
```

## Project Structure

```
myops_segmentation/
├── main.py                  # Main entry point (train / test / ensemble)
├── augment_offline.py       # Offline data augmentation
├── src/
│   ├── config.py            # CLI arguments and settings
│   ├── constants.py         # Label mappings and class definitions
│   ├── data.py              # Dataset, data loading, preprocessing
│   ├── engine.py            # Training loop, evaluation, ensemble inference
│   ├── losses.py            # Combined Focal + Dice loss
│   ├── metrics.py           # Dice, HD95, union dice
│   ├── models.py            # 2D UNet model definitions
│   ├── models_3d.py         # 3D model variants
│   ├── roi.py               # ROI extraction utilities
│   ├── two_stage.py         # Two-stage coarse-to-fine pipeline
│   ├── fusion_inference.py  # Multi-model fusion inference
│   ├── utils.py             # General utilities
│   └── visualization.py     # Plotting and visualization
├── scripts/
│   ├── run_baseline_train.py
│   ├── run_aug_train.py
│   ├── run_ensemble_test.py
│   └── evaluate_mapping.py
├── data/                    # Dataset directory (see data/README.md)
├── checkpoints/             # Saved model weights (generated)
├── logs/                    # Training logs (generated)
└── results/                 # Test results and metrics (generated)
```

## Key Design Decisions

- **2.5D over 3D**: With only 25 training volumes, 2.5D provides spatial context while keeping the parameter count manageable.
- **Offline augmentation**: Pre-computed augmented cases ensure consistent augmentation across epochs and allow augmented data to be used only in training splits during cross-validation.
- **Fixed mapping strategy**: Edema and scar respond differently to augmentation. The baseline model is better at edema while the augmented model excels at scar. Fixed mapping combines the best of both.
- **Union Dice metric**: Since edema and scar often co-occur spatially, evaluating the combined pathology region gives a more clinically relevant measure.

## License

MIT License. See [LICENSE](LICENSE) for details.
