#!/bin/env python
#
# Built for the CI 2026 hackathon starter kit

# System modules
import json
import logging
from typing import Any, Dict, List, Optional

# External modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Internal modules
from starter_kit.baselines.mlp import _normalisation_mean, _normalisation_std
from starter_kit.baselines.utils import estimate_relative_humidity
from starter_kit.layers import InputNormalisation
from starter_kit.model import BaseModel

main_logger = logging.getLogger(__name__)


class SphereConv2d(nn.Module):
    r"""
    2D convolution with replicate padding on both axes.

    Both latitude (H) and longitude (W) axes use replicate padding because
    the tiles cover a regional sub-domain, not the full globe — circular
    longitude wrapping would connect spatially discontinuous edges.
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
        x = F.pad(x, (p, p, p, p), mode="replicate")
        return self.conv(x)


class ConvBlock(nn.Module):
    r"""Two SphereConv2d + GroupNorm + SiLU."""

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


class ConvNeXtBlock(nn.Module):
    r"""
    ConvNeXt residual block adapted for regional atmospheric tiles.

    Depthwise 7×7 conv (replicate-padded) → LayerNorm → inverted bottleneck
    (Linear 1→4×) → GELU → Linear 4→1× → residual add.
    """

    def __init__(self, dim: int, expansion: int = 4, kernel_size: int = 7) -> None:
        super().__init__()
        self._pad = kernel_size // 2
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.pw_expand = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pw_contract = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self._pad,) * 4, mode="replicate")
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)  # BCHW → BHWC for LayerNorm
        x = self.norm(x)
        x = self.pw_contract(self.act(self.pw_expand(x)))
        x = x.permute(0, 3, 1, 2)  # BHWC → BCHW
        return x + residual


class ConvNeXtStage(nn.Module):
    r"""
    1×1 channel projection followed by ``blocks`` ConvNeXt residual blocks.

    Used as a drop-in replacement for ConvBlock inside UNetwork when
    ``use_convnext=True``.
    """

    def __init__(self, in_ch: int, out_ch: int, blocks: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_ch) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class UNetwork(nn.Module):
    r"""
    U-Net for regional cloud-cover prediction.

    Supports two block types (``use_convnext``):
    - ``False`` (default): original ConvBlock (GroupNorm + SiLU) — backward compatible.
    - ``True``: ConvNeXtStage (LayerNorm + depthwise 7×7 + inverted bottleneck).

    Both use replicate padding on all sides. The longitude axis no longer uses
    circular padding because tiles are regional sub-domains.

    Parameters
    ----------
    input_dim : int
        Number of input channels after normalisation (before any derived features).
    base_channels : int
        Channel count at the first encoder stage; doubled at each level.
    depth : int
        Number of downsampling stages (bottleneck spatial size = 64 / 2**depth).
    use_convnext : bool
        Switch to ConvNeXt blocks. Set ``blocks_per_stage`` to control depth.
    blocks_per_stage : int
        Number of ConvNeXt residual blocks per stage (ignored when use_convnext=False).
    """

    def __init__(
        self,
        input_dim: int = 30,
        base_channels: int = 32,
        depth: int = 3,
        use_rh: bool = False,
        n_auxiliary_fields: int = 2,
        normalisation_path: Optional[str] = None,
        use_convnext: bool = False,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        self.use_rh = use_rh
        self.n_auxiliary_fields = n_auxiliary_fields

        if normalisation_path is not None:
            with open(normalisation_path) as f:
                stats = json.load(f)
            mean_list = stats["mean"]
            std_list = stats["std"]
            if use_rh:
                if not stats.get("use_rh", False):
                    raise ValueError(
                        "use_rh=True but normalisation_path was computed without"
                        " --use_rh"
                    )
                pressure_levels_pa = stats["pressure_levels_pa"]
        elif use_rh or n_auxiliary_fields != 2:
            raise ValueError(
                "Hardcoded normalisation only supports use_rh=False and "
                "n_auxiliary_fields=2; pass a normalisation_path computed "
                "via scripts/compute_normalization.py."
            )
        else:
            mean_list = _normalisation_mean
            std_list = _normalisation_std

        if use_rh:
            self.register_buffer(
                "pressure_levels",
                torch.tensor(pressure_levels_pa, dtype=torch.float32).reshape(-1, 1, 1),
            )

        if len(mean_list) != input_dim:
            raise ValueError(
                f"final stats length is {len(mean_list)} but input_dim={input_dim}"
            )

        mean = torch.tensor(mean_list).reshape(1, -1, 1, 1)
        std = torch.tensor(std_list).reshape(1, -1, 1, 1)
        self.normalisation = InputNormalisation(mean=mean, std=std)

        def _make_stage(in_ch: int, out_ch: int) -> nn.Module:
            if use_convnext:
                return ConvNeXtStage(in_ch, out_ch, blocks=blocks_per_stage)
            return ConvBlock(in_ch, out_ch)

        channels: List[int] = [base_channels * (2**i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        prev = input_dim
        for ch in channels[:-1]:
            self.encoders.append(_make_stage(prev, ch))
            prev = ch
        self.pool = nn.AvgPool2d(2)

        self.bottleneck = _make_stage(channels[-2], channels[-1])

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose2d(channels[i], channels[i - 1], 2, stride=2)
            )
            self.decoders.append(_make_stage(channels[i], channels[i - 1]))

        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.normal_(self.head.weight, std=1e-6)
        nn.init.constant_(self.head.bias, 0.5)

    def forward(
        self,
        input_level: torch.Tensor,
        input_auxiliary: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_rh:
            rh = estimate_relative_humidity(
                temperature=input_level[:, 0:1],
                specific_humidity=input_level[:, 1:2],
                pressure=self.pressure_levels,
            )
            input_level = torch.cat([input_level, rh], dim=1)

        flattened_level = input_level.reshape(
            input_level.shape[0], -1, *input_level.shape[-2:]
        )
        sliced_aux = input_auxiliary[:, : self.n_auxiliary_fields]
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
