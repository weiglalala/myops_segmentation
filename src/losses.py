from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from src.constants import DEFAULT_CLASS_FREQUENCIES


def one_hot2dist(seg: np.ndarray, num_classes: int = 6) -> np.ndarray:
    res = np.zeros((num_classes, *seg.shape), dtype=np.float32)
    for c in range(num_classes):
        posmask = (seg == c).astype(bool)
        if not posmask.any():
            continue
        negmask = ~posmask
        res[c] = distance_transform_edt(negmask) * negmask.astype(np.float32) \
            - (distance_transform_edt(posmask) - 1) * posmask.astype(np.float32)
    return res


def derive_class_weights(scar_weight: float = 1.10) -> torch.Tensor:
    frequencies = DEFAULT_CLASS_FREQUENCIES.copy()
    weights = 1.0 / np.sqrt(frequencies + 1e-8)
    weights[0] *= 0.05
    weights[4] *= 1.80
    weights[5] *= scar_weight
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def derive_edema_expert_class_weights() -> torch.Tensor:
    frequencies = DEFAULT_CLASS_FREQUENCIES.copy()
    weights = 1.0 / np.sqrt(frequencies + 1e-8)
    weights[0] *= 0.05
    weights[4] *= 2.50
    weights[5] *= 0.80
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class DiceLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, smooth: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        denominator = torch.sum(probs + target_one_hot, dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        weights = self.class_weights / self.class_weights.sum()
        loss = (1.0 - dice_score) * weights
        return loss.sum()


class TverskyLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        true_positive = torch.sum(probs * target_one_hot, dim=dims)
        false_positive = torch.sum(probs * (1.0 - target_one_hot), dim=dims)
        false_negative = torch.sum((1.0 - probs) * target_one_hot, dim=dims)
        tversky_score = (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        loss = 1.0 - tversky_score
        if self.class_weights is not None:
            weights = self.class_weights / self.class_weights.sum()
            return (loss * weights).sum()
        return loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.class_weights)


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.class_weights)
        probs = torch.softmax(logits, dim=1)
        target_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_weight = (1.0 - target_probs) ** self.gamma
        return (focal_weight * ce_loss).mean()


class PathologyDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        pathology_probs = probs[:, 4:6].sum(dim=1)
        pathology_targets = ((targets == 4) | (targets == 5)).float()
        intersection = torch.sum(pathology_probs * pathology_targets, dim=(0, 1, 2))
        denominator = torch.sum(pathology_probs + pathology_targets, dim=(0, 1, 2))
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_score


class EdemaSpecificDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        edema_probs = probs[:, 4]  # (B, H, W)
        edema_targets = (targets == 4).float()
        intersection = torch.sum(edema_probs * edema_targets, dim=(1, 2))
        denominator = torch.sum(edema_probs + edema_targets, dim=(1, 2))
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice_score).mean()


class SurfaceLoss(nn.Module):
    def __init__(self, idc: tuple[int, ...] = (4, 5)) -> None:
        super().__init__()
        self.idc = list(idc)

    def forward(self, logits: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        pc = probs[:, self.idc, ...].float()
        dc = dist_maps[:, self.idc, ...].float()
        return (pc * dc).mean()


class UnionDiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, union_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        union_gt = ((targets == 4) | (targets == 5)).float()
        union_logits_flat = union_logits.squeeze(1)
        bce = F.binary_cross_entropy_with_logits(union_logits_flat, union_gt)
        union_probs = torch.sigmoid(union_logits_flat)
        intersection = (union_probs * union_gt).sum(dim=(1, 2))
        denominator = (union_probs + union_gt).sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = (1.0 - dice).mean()
        return (bce + dice_loss) / 2.0


class InclusivenessLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        p_edema = probs[:, 4]
        p_scar = probs[:, 5]
        violation = F.relu(p_scar - p_edema)
        has_scar = (targets == 5).flatten(1).any(dim=1)
        if not has_scar.any():
            return logits.new_tensor(0.0)
        return violation[has_scar].mean()


class MyocardiumPriorLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        foreground_targets = ((targets == 1) | (targets == 4) | (targets == 5)).float()
        foreground_logits = logits[:, 0, :, :]
        bce = F.binary_cross_entropy_with_logits(foreground_logits, foreground_targets)
        probs = torch.sigmoid(foreground_logits)
        intersection = torch.sum(probs * foreground_targets, dim=(0, 1, 2))
        denominator = torch.sum(probs + foreground_targets, dim=(0, 1, 2))
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return bce + (1.0 - dice_score)


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        tversky_weight: float = 0.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        pathology_dice_weight: float = 0.4,
        myocardium_prior_weight: float = 0.25,
        edema_dice_weight: float = 0.0,
        surface_loss_weight: float = 0.0,
        union_loss_weight: float = 0.0,
        inclusiveness_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.pathology_dice_weight = pathology_dice_weight
        self.myocardium_prior_weight = myocardium_prior_weight
        self.edema_dice_weight = edema_dice_weight
        self.surface_loss_weight = surface_loss_weight
        self.union_loss_weight = union_loss_weight
        self.inclusiveness_loss_weight = inclusiveness_loss_weight
        self.dice_loss = DiceLoss(class_weights=class_weights)
        self.ce_loss = WeightedCrossEntropyLoss(class_weights=class_weights)
        self.focal_loss = FocalCrossEntropyLoss(class_weights=class_weights, gamma=focal_gamma)
        self.tversky_loss = TverskyLoss(
            class_weights=class_weights,
            alpha=tversky_alpha,
            beta=tversky_beta,
        )
        self.pathology_dice_loss = PathologyDiceLoss()
        self.myocardium_prior_loss = MyocardiumPriorLoss()
        self.edema_dice_loss = EdemaSpecificDiceLoss()
        self.surface_loss = SurfaceLoss(idc=(4, 5))
        self.union_loss = UnionDiceBCELoss()
        self.inclusiveness_loss = InclusivenessLoss()

    @staticmethod
    def compute_alpha(epoch: int, total_epochs: int) -> float:
        alpha_begin = 1.0
        alpha_end = 0.01
        decay = (alpha_begin - alpha_end) / max(total_epochs, 1)
        return max(alpha_end, alpha_begin - decay * epoch)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        myocardium_logits: torch.Tensor | None = None,
        union_logits: torch.Tensor | None = None,
        dist_maps: torch.Tensor | None = None,
        epoch: int = 0,
        total_epochs: int = 300,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        dice_value = self.dice_loss(logits, targets)
        ce_value = (
            self.ce_loss(logits, targets)
            if self.ce_weight > 0
            else logits.new_tensor(0.0)
        )
        focal_value = (
            self.focal_loss(logits, targets)
            if self.focal_weight > 0
            else logits.new_tensor(0.0)
        )
        tversky_value = (
            self.tversky_loss(logits, targets)
            if self.tversky_weight > 0
            else logits.new_tensor(0.0)
        )
        pathology_dice_value = (
            self.pathology_dice_loss(logits, targets)
            if self.pathology_dice_weight > 0
            else logits.new_tensor(0.0)
        )
        myocardium_prior_value = (
            self.myocardium_prior_loss(myocardium_logits, targets)
            if myocardium_logits is not None and self.myocardium_prior_weight > 0
            else logits.new_tensor(0.0)
        )
        edema_dice_value = (
            self.edema_dice_loss(logits, targets)
            if self.edema_dice_weight > 0
            else logits.new_tensor(0.0)
        )
        surface_value = (
            self.surface_loss(logits, dist_maps)
            if self.surface_loss_weight > 0 and dist_maps is not None
            else logits.new_tensor(0.0)
        )
        union_value = (
            self.union_loss(union_logits, targets)
            if union_logits is not None and self.union_loss_weight > 0
            else logits.new_tensor(0.0)
        )
        inclusiveness_value = (
            self.inclusiveness_loss(logits, targets)
            if self.inclusiveness_loss_weight > 0
            else logits.new_tensor(0.0)
        )
        region_loss = (
            self.dice_weight * dice_value
            + self.ce_weight * ce_value
            + self.focal_weight * focal_value
            + self.tversky_weight * tversky_value
            + self.pathology_dice_weight * pathology_dice_value
            + self.myocardium_prior_weight * myocardium_prior_value
            + self.edema_dice_weight * edema_dice_value
            + self.union_loss_weight * union_value
            + self.inclusiveness_loss_weight * inclusiveness_value
        )
        if self.surface_loss_weight > 0 and dist_maps is not None:
            alpha = self.compute_alpha(epoch, total_epochs)
            total = alpha * region_loss + (1.0 - alpha) * self.surface_loss_weight * surface_value
        else:
            total = region_loss
        components = {
            "dice_loss": float(dice_value.detach().item()),
            "ce_loss": float(ce_value.detach().item()),
            "focal_loss": float(focal_value.detach().item()),
            "tversky_loss": float(tversky_value.detach().item()),
            "pathology_dice_loss": float(pathology_dice_value.detach().item()),
            "myocardium_prior_loss": float(myocardium_prior_value.detach().item()),
            "edema_dice_loss": float(edema_dice_value.detach().item()),
            "surface_loss": float(surface_value.detach().item()),
            "union_loss": float(union_value.detach().item()),
            "inclusiveness_loss": float(inclusiveness_value.detach().item()),
            "total_loss": float(total.detach().item()),
        }
        return total, components
