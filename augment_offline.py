#!/usr/bin/env python
"""Offline 3D geometric augmentation for MyoPS training data.

Generates warped copies of each training case (all modalities + mask)
using random affine + elastic deformation applied consistently across volumes.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


MODALITIES = ("C0", "T2", "LGE")
MASK_SUFFIX = "gd"


def random_affine_matrix(
    shape: tuple[int, int, int],
    rotation_range: float = 15.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    angles = np.deg2rad(rng.uniform(-rotation_range, rotation_range, size=3))
    cx, cy, cz = [s / 2.0 for s in shape]

    rx = np.array([
        [1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])],
    ])
    ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])],
    ])
    rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1],
    ])
    rot = rz @ ry @ rx

    scale = rng.uniform(scale_range[0], scale_range[1], size=3)
    scale_mat = np.diag(scale)

    transform = rot @ scale_mat
    center = np.array([cx, cy, cz])
    offset = center - transform @ center
    return transform, offset


def elastic_deformation_field(
    shape: tuple[int, int, int],
    alpha: float = 500.0,
    sigma: float = 12.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    dx = gaussian_filter(rng.standard_normal(shape), sigma) * alpha
    dy = gaussian_filter(rng.standard_normal(shape), sigma) * alpha
    dz = gaussian_filter(rng.standard_normal(shape), sigma) * alpha
    return dx, dy, dz


def build_coordinate_grid(
    shape: tuple[int, int, int],
    affine_matrix: np.ndarray | None = None,
    affine_offset: np.ndarray | None = None,
    elastic_fields: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float64)
    x, y, z = coords[0], coords[1], coords[2]

    if affine_matrix is not None:
        flat = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)
        transformed = affine_matrix @ flat
        if affine_offset is not None:
            transformed += affine_offset[:, None]
        x = transformed[0].reshape(shape)
        y = transformed[1].reshape(shape)
        z = transformed[2].reshape(shape)

    if elastic_fields is not None:
        dx, dy, dz = elastic_fields
        x = x + dx
        y = y + dy
        z = z + dz

    return x, y, z


def warp_volume(volume: np.ndarray, coords: tuple[np.ndarray, ...], order: int = 1) -> np.ndarray:
    return map_coordinates(volume, coords, order=order, mode="nearest").astype(volume.dtype)


def augment_case(
    case_dir: Path,
    output_dir: Path,
    aug_index: int,
    rng: np.random.Generator,
    rotation_range: float = 15.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
    elastic_alpha: float = 500.0,
    elastic_sigma: float = 12.0,
) -> Path:
    case_id = case_dir.name
    aug_id = f"{case_id}_aug{aug_index}"
    aug_dir = output_dir / aug_id
    aug_dir.mkdir(parents=True, exist_ok=True)

    ref_nii = nib.load(str(case_dir / f"{case_id}_C0.nii.gz"))
    shape = ref_nii.shape

    affine_mat, affine_off = random_affine_matrix(shape, rotation_range, scale_range, rng)
    elastic = elastic_deformation_field(shape, elastic_alpha, elastic_sigma, rng)
    coords = build_coordinate_grid(shape, affine_mat, affine_off, elastic)

    for mod in MODALITIES:
        nii = nib.load(str(case_dir / f"{case_id}_{mod}.nii.gz"))
        warped = warp_volume(nii.get_fdata(dtype=np.float32), coords, order=1)
        out_nii = nib.Nifti1Image(warped, nii.affine, nii.header)
        nib.save(out_nii, str(aug_dir / f"{aug_id}_{mod}.nii.gz"))

    mask_nii = nib.load(str(case_dir / f"{case_id}_{MASK_SUFFIX}.nii.gz"))
    warped_mask = warp_volume(mask_nii.get_fdata(dtype=np.float32).astype(np.int32), coords, order=0)
    out_mask = nib.Nifti1Image(warped_mask, mask_nii.affine, mask_nii.header)
    nib.save(out_mask, str(aug_dir / f"{aug_id}_{MASK_SUFFIX}.nii.gz"))

    return aug_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline 3D geometric augmentation")
    parser.add_argument("--data-root", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/train_augmented"))
    parser.add_argument("--num-augments", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotation-range", type=float, default=15.0)
    parser.add_argument("--scale-min", type=float, default=0.85)
    parser.add_argument("--scale-max", type=float, default=1.15)
    parser.add_argument("--elastic-alpha", type=float, default=500.0)
    parser.add_argument("--elastic-sigma", type=float, default=12.0)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    print(f"Found {len(case_dirs)} cases in {data_root}")

    for case_dir in case_dirs:
        dst = output_dir / case_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(case_dir, dst)
    print(f"Copied {len(case_dirs)} original cases to {output_dir}")

    rng = np.random.default_rng(args.seed)
    total_aug = 0
    for case_dir in case_dirs:
        for k in range(args.num_augments):
            aug_dir = augment_case(
                case_dir, output_dir, k,
                rng=rng,
                rotation_range=args.rotation_range,
                scale_range=(args.scale_min, args.scale_max),
                elastic_alpha=args.elastic_alpha,
                elastic_sigma=args.elastic_sigma,
            )
            total_aug += 1
            print(f"  [{total_aug}/{len(case_dirs) * args.num_augments}] {aug_dir.name}")

    total = len(case_dirs) + total_aug
    print(f"\nDone: {len(case_dirs)} original + {total_aug} augmented = {total} cases")


if __name__ == "__main__":
    main()