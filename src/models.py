from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(self.pool(x))
        return x * weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_se: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(x))


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = self.psi(self.gate(gate) + self.skip(skip))
        return skip * attention


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = False,
        use_se: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = (
            AttentionGate(out_channels, skip_channels, inter_channels=max(out_channels // 2, 8))
            if use_attention
            else nn.Identity()
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, use_se=use_se, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if isinstance(self.attention, AttentionGate):
            skip = self.attention(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetVariant(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_channels: int = 32,
        use_attention: bool = False,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.enc1 = ConvBlock(in_channels, channels[0], use_se=use_se)
        self.enc2 = ConvBlock(channels[0], channels[1], use_se=use_se)
        self.enc3 = ConvBlock(channels[1], channels[2], use_se=use_se)
        self.enc4 = ConvBlock(channels[2], channels[3], use_se=use_se, dropout=0.1)
        self.bottleneck = ConvBlock(channels[3], channels[4], use_se=use_se, dropout=0.15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dec4 = UpBlock(channels[4], channels[3], channels[3], use_attention=use_attention, use_se=use_se, dropout=0.1)
        self.dec3 = UpBlock(channels[3], channels[2], channels[2], use_attention=use_attention, use_se=use_se, dropout=0.05)
        self.dec2 = UpBlock(channels[2], channels[1], channels[1], use_attention=use_attention, use_se=use_se)
        self.dec1 = UpBlock(channels[1], channels[0], channels[0], use_attention=use_attention, use_se=use_se)
        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        bottleneck = self.bottleneck(self.pool(s4))
        x = self.dec4(bottleneck, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)


class AttentionSEMyoVariant(UNetVariant):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_channels: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            use_attention=True,
            use_se=True,
        )
        self.myo_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.union_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        bottleneck = self.bottleneck(self.pool(s4))
        x = self.dec4(bottleneck, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x), self.myo_head(x), self.union_head(x)


class AttentionSEMyoGuidedVariant(UNetVariant):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_channels: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            use_attention=True,
            use_se=True,
        )
        self.myo_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.union_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.guidance_alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        bottleneck = self.bottleneck(self.pool(s4))
        x = self.dec4(bottleneck, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        myocardium_logits = self.myo_head(x)
        myocardium_attention = torch.sigmoid(myocardium_logits)
        guided_features = x * (1.0 + self.guidance_alpha * myocardium_attention)
        return self.head(guided_features), myocardium_logits, self.union_head(x)


class SMPUnetVariant(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 6,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_use_batchnorm=True,
            decoder_attention_type="scse",
        )
        self.myo_head = nn.Conv2d(16, 1, kernel_size=1)
        self.union_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(features)
        masks = self.unet.segmentation_head(decoder_output)
        myo_logits = self.myo_head(decoder_output)
        union_logits = self.union_head(decoder_output)
        return masks, myo_logits, union_logits


class SMPUnetPPVariant(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 6,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.unet = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_use_batchnorm=True,
            decoder_attention_type="scse",
        )
        self.myo_head = nn.Conv2d(16, 1, kernel_size=1)
        self.union_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(features)
        masks = self.unet.segmentation_head(decoder_output)
        myo_logits = self.myo_head(decoder_output)
        union_logits = self.union_head(decoder_output)
        return masks, myo_logits, union_logits


class SMPDeepLabVariant(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 6,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.unet = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_use_batchnorm=True,
        )
        self.myo_head = nn.Conv2d(256, 1, kernel_size=1)
        self.union_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(features)
        masks = self.unet.segmentation_head(decoder_output)
        myo_logits = nn.functional.interpolate(
            self.myo_head(decoder_output), size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        union_logits = nn.functional.interpolate(
            self.union_head(decoder_output), size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return masks, myo_logits, union_logits


class SMPUnetProjectionVariant(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 6,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.projection = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.projection.weight)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            decoder_use_batchnorm=True,
            decoder_attention_type="scse",
        )
        self.myo_head = nn.Conv2d(16, 1, kernel_size=1)
        self.union_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.projection(x)
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(features)
        masks = self.unet.segmentation_head(decoder_output)
        myo_logits = self.myo_head(decoder_output)
        union_logits = self.union_head(decoder_output)
        return masks, myo_logits, union_logits


def build_model(model_variant: str, in_channels: int = 3, num_classes: int = 6) -> nn.Module:
    if model_variant == "unet":
        return UNetVariant(in_channels=in_channels, num_classes=num_classes, use_attention=False, use_se=False)
    if model_variant == "attention_unet":
        return UNetVariant(in_channels=in_channels, num_classes=num_classes, use_attention=True, use_se=False)
    if model_variant == "attention_se":
        return UNetVariant(in_channels=in_channels, num_classes=num_classes, use_attention=True, use_se=True)
    if model_variant == "attention_se_myo":
        return AttentionSEMyoVariant(in_channels=in_channels, num_classes=num_classes)
    if model_variant == "attention_se_myo_guided":
        return AttentionSEMyoGuidedVariant(in_channels=in_channels, num_classes=num_classes)
    if model_variant == "smp_unet":
        return SMPUnetVariant(in_channels=in_channels, num_classes=num_classes, encoder_name="resnet34")
    if model_variant == "smp_unetpp":
        return SMPUnetPPVariant(in_channels=in_channels, num_classes=num_classes, encoder_name="resnet34")
    if model_variant == "smp_deeplabv3p":
        return SMPDeepLabVariant(in_channels=in_channels, num_classes=num_classes, encoder_name="resnet34")
    if model_variant == "smp_unet_effb4":
        return SMPUnetVariant(in_channels=in_channels, num_classes=num_classes, encoder_name="efficientnet-b4")
    if model_variant == "smp_unet_3ch":
        return SMPUnetProjectionVariant(in_channels=in_channels, num_classes=num_classes, encoder_name="resnet34")
    if model_variant == "unet_3d":
        from src.models_3d import UNet3D
        return UNet3D(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unsupported model variant: {model_variant}")
