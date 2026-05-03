from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock3D(in_channels, c)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock3D(c, c * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock3D(c * 2, c * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock3D(c * 4, c * 8)
        self.dropout = nn.Dropout3d(0.15)

        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(c * 2, c)

        self.head = nn.Conv3d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.dropout(self.bottleneck(self.pool3(e3)))

        d3 = self._upsample_cat(self.up3, b, e3)
        d3 = self.dec3(d3)
        d2 = self._upsample_cat(self.up2, d3, e2)
        d2 = self.dec2(d2)
        d1 = self._upsample_cat(self.up1, d2, e1)
        d1 = self.dec1(d1)

        return self.head(d1)

    @staticmethod
    def _upsample_cat(
        up: nn.ConvTranspose3d, x: torch.Tensor, skip: torch.Tensor,
    ) -> torch.Tensor:
        x = up(x)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)
