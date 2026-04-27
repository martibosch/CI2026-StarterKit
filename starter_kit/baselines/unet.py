#!/bin/env python
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging
from typing import Dict, Any, List

# External modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Internal modules
from starter_kit.model import BaseModel
from starter_kit.layers import InputNormalisation
from starter_kit.baselines.mlp import (
    _normalisation_mean,
    _normalisation_std,
)


main_logger = logging.getLogger(__name__)


class SphereConv2d(nn.Module):
    r"""
    2D convolution with sphere-aware padding: circular along the
    longitude axis (W) and replicate along the latitude axis (H).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pad
        x = F.pad(x, (p, p, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, p, p), mode="replicate")
        return self.conv(x)


class ConvBlock(nn.Module):
    r"""
    Two SphereConv2d + GroupNorm + SiLU, the standard U-Net block.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            SphereConv2d(in_ch, out_ch, 3),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            SphereConv2d(out_ch, out_ch, 3),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetwork(nn.Module):
    r"""
    Small U-Net for global cloud-cover prediction.

    The pressure-level fields are flattened along the level axis into
    channels and concatenated with the first two auxiliary fields
    (land-sea mask and orography), matching the MLP baseline so that
    its precomputed normalisation statistics can be reused.

    Parameters
    ----------
    input_dim : int, default = 30
        Number of input channels (4 vars x 7 levels + 2 aux).
    base_channels : int, default = 32
        Channel count of the first U-Net stage; doubled at each
        downsampling level.
    depth : int, default = 3
        Number of downsampling stages. With ``depth=3`` and a 64x64
        input, the bottleneck is 8x8.
    """

    def __init__(
        self,
        input_dim: int = 30,
        base_channels: int = 32,
        depth: int = 3,
    ) -> None:
        super().__init__()
        mean = torch.tensor(_normalisation_mean).reshape(1, -1, 1, 1)
        std = torch.tensor(_normalisation_std).reshape(1, -1, 1, 1)
        self.normalisation = InputNormalisation(mean=mean, std=std)

        channels: List[int] = [base_channels * (2**i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        prev = input_dim
        for ch in channels[:-1]:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch
        self.pool = nn.AvgPool2d(2)

        self.bottleneck = ConvBlock(channels[-2], channels[-1])

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose2d(channels[i], channels[i - 1], 2, stride=2)
            )
            self.decoders.append(ConvBlock(channels[i], channels[i - 1]))

        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.normal_(self.head.weight, std=1e-6)
        nn.init.constant_(self.head.bias, 0.5)

    def forward(
        self,
        input_level: torch.Tensor,
        input_auxiliary: torch.Tensor,
    ) -> torch.Tensor:
        flattened_level = input_level.reshape(
            input_level.shape[0], -1, *input_level.shape[-2:]
        )
        sliced_aux = input_auxiliary[:, :2]
        x = torch.cat([flattened_level, sliced_aux], dim=1)
        x = self.normalisation(x)

        skips: List[torch.Tensor] = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)


class UNetModel(BaseModel):
    r"""
    Trainer wrapper for the U-Net network. Uses latitude-weighted MAE
    as the training loss, identical in spirit to the MLP baseline.
    """

    def estimate_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        prediction = self.network(
            input_level=batch["input_level"],
            input_auxiliary=batch["input_auxiliary"],
        )
        prediction = prediction.clamp(0.0, 1.0)
        loss = (prediction - batch["target"]).abs()
        loss = (loss * self.lat_weights).mean()
        return {"loss": loss, "prediction": prediction}

    def estimate_auxiliary_loss(
        self, batch: Dict[str, torch.Tensor], outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        mse = (outputs["prediction"] - batch["target"]).pow(2)
        mse = (mse * self.lat_weights).mean()
        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = (prediction_bool == target_bool).float()
        accuracy = (accuracy * self.lat_weights).mean()
        return {"mse": mse, "accuracy": accuracy}
