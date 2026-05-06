#!/bin/env python
#
# Built for the CI 2026 hackathon starter kit.
#
# Translation-equivariant U-Net for total cloud cover prediction.
# Designed to generalise to unseen geographic regions:
#   - inputs are *only* the 28 pressure-level meteorological fields
#     (temperature, specific humidity, u, v at 7 levels). No static
#     geography (land_sea_mask, orography, vegetation, ...) and no
#     absolute lat/lon — the model has to infer cloud cover from the
#     atmospheric state alone, which is the same physics in every region.
#   - replicate padding on both axes (each region is only a ~96°-wide
#     longitude window, not a full global wrap)
#   - random longitude rolls and rectangular input-masking during training,
#     with extra loss weight on the masked pixels so the network must learn
#     to infer cloud cover from spatial context.

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from starter_kit.baselines.utils import estimate_relative_humidity
from starter_kit.layers import InputNormalisation
from starter_kit.model import BaseModel

main_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry-aware padding + convolution
# ---------------------------------------------------------------------------


def _geo_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Replicate-pad on both axes."""
    if pad == 0:
        return x
    return F.pad(x, (pad, pad, pad, pad), mode="replicate")


class GeoConv2d(nn.Module):
    """Conv2d with replicate padding on both axes."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_geo_pad(x, self.pad))


class LayerNorm2d(nn.Module):
    """LayerNorm over channel dim of (B,C,H,W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise 7x7 (geo-pad) -> LN -> 1x1 -> GELU -> 1x1 + resid."""

    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.dw = GeoConv2d(dim, dim, kernel_size=7, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * mlp_ratio, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * mlp_ratio, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.norm(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.pw2(h)
        h = h * self.gamma[None, :, None, None]
        return x + h


# ---------------------------------------------------------------------------
# Stats loading
# ---------------------------------------------------------------------------


def _load_stats(path: str) -> Dict[str, Any]:
    """Load precomputed normalisation stats from the existing format."""
    if not path:
        raise ValueError("normalisation_path is required for GeoUNet")
    if not path.endswith(".json"):
        path += ".json"
    if not path.startswith("/"):
        import os

        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Normalisation stats not found at {path}. "
            f"Run scripts/compute_normalization.py first."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------


class GeoUNet(nn.Module):
    """
    Translation-equivariant U-Net for cloud-cover prediction.

    Inputs
    ------
    input_level : (B, 4, 7, H, W) — pressure-level fields. The only
        physically meaningful input the model sees.
    input_auxiliary : ignored, accepted only for compatibility with the
        existing forecast script.
    input_mask : (B, 1, H, W) — optional binary indicator marking masked
        spatial cells (1 = masked input). When ``None``, no mask is applied.
    """

    def __init__(
        self,
        base_dim: int = 96,
        depth_mult: Tuple[int, int, int, int] = (2, 2, 6, 2),
        channel_mult: Tuple[int, int, int, int] = (1, 2, 3, 4),
        n_level_vars: int = 4,
        n_levels: int = 7,
        normalisation_path: str = "",
        dropout: float = 0.0,
        use_shear: bool = False,
        use_rh: bool = False,
        use_rh_gradient: bool = False,
        use_theta: bool = False,
        training_noise_std: float = 0.0,
    ) -> None:
        super().__init__()

        stats = _load_stats(normalisation_path)
        n_lv_ch = n_level_vars * n_levels
        assert len(stats["mean"]) >= n_lv_ch, (
            f"Expected at least {n_lv_ch} channels in stats, got {len(stats['mean'])}"
        )

        mean = torch.tensor(stats["mean"][:n_lv_ch]).float()
        std = torch.tensor(stats["std"][:n_lv_ch]).float()
        self.normalisation = InputNormalisation(mean=mean, std=std)

        # learnable token: replaces level-field values at masked positions
        self.mask_token = nn.Parameter(torch.zeros(n_lv_ch))

        self.use_shear = use_shear
        self.use_rh = use_rh
        self.use_rh_gradient = use_rh_gradient
        self.use_theta = use_theta
        self.training_noise_std = training_noise_std
        self._n_levels = n_levels
        # channel indices for each variable (T,q,u,v ordering)
        self._t_start = 0
        self._q_start = n_levels
        self._u_start = 2 * n_levels
        self._v_start = 3 * n_levels

        if use_rh or use_rh_gradient or use_theta:
            pressure_levels_pa = stats["pressure_levels_pa"]
            self.register_buffer(
                "pressure_levels",
                torch.tensor(pressure_levels_pa, dtype=torch.float32).reshape(
                    1, n_levels, 1, 1
                ),
            )

        channel_names = stats.get("channel_names", [])
        stats_mean = stats.get("mean", [])
        stats_std = stats.get("std", [])

        def _derived_stats(
            names: List[str],
            fallback_mean: float,
            fallback_std: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            means = []
            stds = []
            for name in names:
                try:
                    idx = channel_names.index(name)
                except ValueError:
                    means.append(fallback_mean)
                    stds.append(fallback_std)
                else:
                    means.append(float(stats_mean[idx]))
                    stds.append(float(stats_std[idx]))
            return torch.tensor(means).float(), torch.tensor(stds).float()

        rh_mean = None
        rh_std = None
        if use_rh or use_rh_gradient:
            rh_mean, rh_std = _derived_stats(
                [f"rh@L{i}" for i in range(n_levels)],
                fallback_mean=0.5,
                fallback_std=0.25,
            )
        if use_rh:
            self.register_buffer("rh_mean", rh_mean.reshape(1, n_levels, 1, 1))
            self.register_buffer("rh_std", rh_std.reshape(1, n_levels, 1, 1))
        if use_rh_gradient:
            drh_names = [f"drh@L{i}-L{i + 1}" for i in range(n_levels - 1)]
            if all(name in channel_names for name in drh_names):
                drh_mean, drh_std = _derived_stats(
                    drh_names,
                    fallback_mean=0.0,
                    fallback_std=0.25,
                )
            else:
                logp = torch.log(
                    torch.tensor(stats["pressure_levels_pa"], dtype=torch.float32)
                )
                dlogp = (logp[1:] - logp[:-1]).abs().clamp_min(1e-6)
                drh_mean = (rh_mean[1:] - rh_mean[:-1]) / dlogp
                drh_std = torch.sqrt(rh_std[1:].pow(2) + rh_std[:-1].pow(2)) / dlogp
            self.register_buffer("drh_mean", drh_mean.reshape(1, n_levels - 1, 1, 1))
            self.register_buffer("drh_std", drh_std.reshape(1, n_levels - 1, 1, 1))
        if use_theta:
            theta_names = [f"theta@L{i}" for i in range(n_levels)]
            if all(name in channel_names for name in theta_names):
                theta_mean, theta_std = _derived_stats(
                    theta_names,
                    fallback_mean=300.0,
                    fallback_std=20.0,
                )
            else:
                pressure = torch.tensor(
                    stats["pressure_levels_pa"], dtype=torch.float32
                ).reshape(n_levels)
                theta_factor = (100000.0 / pressure) ** 0.2854
                theta_mean = (
                    mean[self._t_start : self._t_start + n_levels] * theta_factor
                )
                theta_std = std[self._t_start : self._t_start + n_levels] * theta_factor
            self.register_buffer("theta_mean", theta_mean.reshape(1, n_levels, 1, 1))
            self.register_buffer("theta_std", theta_std.reshape(1, n_levels, 1, 1))

        # input channels: normalised level fields + derived features + mask flag
        n_derived = (
            (n_levels if use_rh else 0)
            + (n_levels - 1 if use_rh_gradient else 0)
            + (n_levels if use_theta else 0)
            + (n_levels - 1 if use_shear else 0)
        )
        in_ch = n_lv_ch + n_derived + 1

        dims = [base_dim * m for m in channel_mult]
        self.dims = dims

        self.stem = GeoConv2d(in_ch, dims[0], kernel_size=3)
        self.stem_norm = LayerNorm2d(dims[0])

        # encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, n_blocks in enumerate(depth_mult):
            blocks = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(n_blocks)])
            self.enc_blocks.append(blocks)
            if i < len(depth_mult) - 1:
                self.downs.append(
                    nn.Sequential(
                        LayerNorm2d(dims[i]),
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                    )
                )

        # decoder: upsample + concat skip + project + blocks
        self.ups = nn.ModuleList()
        self.dec_proj = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(depth_mult) - 1, 0, -1):
            self.ups.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
            self.dec_proj.append(
                nn.Sequential(
                    GeoConv2d(dims[i] + dims[i - 1], dims[i - 1], kernel_size=3),
                    LayerNorm2d(dims[i - 1]),
                )
            )
            self.dec_blocks.append(
                nn.Sequential(*[ConvNeXtBlock(dims[i - 1]) for _ in range(2)])
            )

        self.head_norm = LayerNorm2d(dims[0])
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(dims[0], 1, kernel_size=1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.constant_(self.head.bias, 0.5)

    def forward(
        self,
        input_level: torch.Tensor,
        input_auxiliary: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del input_auxiliary

        b = input_level.shape[0]
        h, w = input_level.shape[-2:]

        x_level = input_level.reshape(b, -1, h, w)
        x_raw = x_level

        if input_mask is not None:
            tok = self.mask_token.view(1, -1, 1, 1)
            x_raw = torch.where(input_mask.bool(), tok, x_raw)

        # normalise (channel-last)
        x = self.normalisation(x_raw.movedim(1, -1)).movedim(-1, 1)

        if self.training and self.training_noise_std > 0.0:
            x = x + torch.randn_like(x) * self.training_noise_std

        extra: list = []

        rh = None
        if self.use_rh or self.use_rh_gradient:
            T_raw = x_level[:, self._t_start : self._t_start + self._n_levels]
            q_raw = x_level[:, self._q_start : self._q_start + self._n_levels]
            # RH uses exponentials, so keep it in fp32 under AMP and avoid
            # invalid meteorological states from producing non-finite values.
            rh = estimate_relative_humidity(
                T_raw.float().clamp(150.0, 350.0),
                q_raw.float().clamp(0.0, 0.1),
                self.pressure_levels.float(),
            )
            rh = torch.nan_to_num(rh, nan=0.0, posinf=1.0, neginf=0.0)

        if self.use_rh:
            rh_feat = ((rh - self.rh_mean.float()) / (self.rh_std.float() + 1e-6)).to(
                x.dtype
            )
            if input_mask is not None:
                rh_feat = torch.where(
                    input_mask.bool(), torch.zeros_like(rh_feat), rh_feat
                )
            extra.append(rh_feat)

        if self.use_rh_gradient:
            dlogp = (
                torch.log(self.pressure_levels.float()[:, 1:])
                - torch.log(self.pressure_levels.float()[:, :-1])
            ).abs()
            drh = (rh[:, 1:] - rh[:, :-1]) / dlogp.clamp_min(1e-6)
            drh = ((drh - self.drh_mean.float()) / (self.drh_std.float() + 1e-6)).to(
                x.dtype
            )
            if input_mask is not None:
                drh = torch.where(input_mask.bool(), torch.zeros_like(drh), drh)
            extra.append(drh)

        if self.use_theta:
            T_raw = x_level[:, self._t_start : self._t_start + self._n_levels]
            theta = (
                T_raw.float().clamp(150.0, 350.0)
                * (100000.0 / self.pressure_levels.float()) ** 0.2854
            )
            theta = torch.nan_to_num(theta, nan=300.0, posinf=300.0, neginf=300.0)
            theta = (
                (theta - self.theta_mean.float()) / (self.theta_std.float() + 1e-6)
            ).to(x.dtype)
            if input_mask is not None:
                theta = torch.where(input_mask.bool(), torch.zeros_like(theta), theta)
            extra.append(theta)

        if self.use_shear:
            u = x[:, self._u_start : self._u_start + self._n_levels]
            v = x[:, self._v_start : self._v_start + self._n_levels]
            shear = torch.sqrt(
                (u[:, 1:] - u[:, :-1]).pow(2) + (v[:, 1:] - v[:, :-1]).pow(2) + 1e-8
            )
            extra.append(shear)

        if input_mask is None:
            mask_flag = torch.zeros(b, 1, h, w, device=x.device, dtype=x.dtype)
        else:
            mask_flag = input_mask.to(dtype=x.dtype)

        x = torch.cat([x, *extra, mask_flag], dim=1)

        # encoder
        skips: List[torch.Tensor] = []
        x = self.stem(x)
        x = self.stem_norm(x)
        for i, blocks in enumerate(self.enc_blocks):
            x = blocks(x)
            if i < len(self.downs):
                skips.append(x)
                x = self.downs[i](x)

        # decoder
        for j, (up, proj, blocks) in enumerate(
            zip(self.ups, self.dec_proj, self.dec_blocks)
        ):
            x = up(x)
            skip = skips[-1 - j]
            x = torch.cat([x, skip], dim=1)
            x = proj(x)
            x = blocks(x)

        x = self.head_norm(x)
        x = self.dropout(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Batch augmentations
# ---------------------------------------------------------------------------


def _random_lon_roll(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Roll all spatial tensors by the same random shift along the longitude axis."""
    w = batch["input_level"].shape[-1]
    shift = int(torch.randint(0, w, (1,)).item())
    out = {}
    for k, v in batch.items():
        if v.dim() >= 2 and v.shape[-1] == w:
            out[k] = torch.roll(v, shifts=shift, dims=-1)
        else:
            out[k] = v
    return out


def _make_random_block_mask(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    n_blocks: Tuple[int, int] = (1, 3),
    block_frac: Tuple[float, float] = (0.10, 0.35),
    p_apply: float = 0.5,
) -> torch.Tensor:
    """
    Build a (B, 1, H, W) binary mask. With probability ``p_apply``, each sample
    in the batch gets ``n_blocks`` random rectangular regions masked out;
    otherwise it gets an all-zero mask.

    Each block size is sampled in [block_frac_lo, block_frac_hi] of (H, W).
    """
    mask = torch.zeros(b, 1, h, w, device=device)
    for i in range(b):
        if torch.rand(1, device=device).item() > p_apply:
            continue
        n_b = int(torch.randint(n_blocks[0], n_blocks[1] + 1, (1,)).item())
        for _ in range(n_b):
            fh = float(torch.empty(1).uniform_(block_frac[0], block_frac[1]).item())
            fw = float(torch.empty(1).uniform_(block_frac[0], block_frac[1]).item())
            bh = max(2, int(round(h * fh)))
            bw = max(2, int(round(w * fw)))
            top = int(torch.randint(0, h - bh + 1, (1,)).item())
            left = int(torch.randint(0, w - bw + 1, (1,)).item())
            mask[i, 0, top : top + bh, left : left + bw] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Trainer model
# ---------------------------------------------------------------------------


class GeoUNetModel(BaseModel):
    """
    BaseModel subclass with bf16 autocast, masked-region training, and a
    cosine LR schedule. Loss is lat-weighted L1 (matches the ERA5 MAE metric);
    pixels in the masked region get extra weight.
    """

    def __init__(
        self,
        *args,
        mask_loss_weight: float = 2.0,
        mask_p_apply: float = 0.5,
        mask_n_blocks_max: int = 3,
        mask_block_frac_max: float = 0.35,
        lon_roll_p: float = 1.0,
        crop_size: int = 0,
        crop_p_apply: float = 0.0,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        warmup_epochs: int = 1,
        scheduler: str = "cosine",
        restart_period_epochs: int = 20,
        eta_min_ratio: float = 0.0,
        compile_model: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if compile_model:
            try:
                self.network = torch.compile(self.network)
            except Exception as exc:
                main_logger.warning("torch.compile unavailable: %s", exc)
        self.mask_loss_weight = mask_loss_weight
        self.mask_p_apply = mask_p_apply
        self.mask_n_blocks_max = int(mask_n_blocks_max)
        self.mask_block_frac_max = float(mask_block_frac_max)
        self.lon_roll_p = lon_roll_p
        self.crop_size = int(crop_size)
        self.crop_p_apply = float(crop_p_apply)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.scheduler_kind = scheduler
        self.restart_period_epochs = int(restart_period_epochs)
        self.eta_min_ratio = float(eta_min_ratio)
        self._scheduler = self._build_scheduler()
        self._global_step = 0

    def _setup_optimizer(self) -> None:
        """Create AdamW only; skip ReduceLROnPlateau (handled by _build_scheduler)."""
        self._optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        try:
            steps_per_epoch = max(1, len(self.train_loader))
        except TypeError:
            steps_per_epoch = 1
        warmup_steps = self.warmup_epochs * steps_per_epoch
        total_steps = max(self.n_epochs, 1) * steps_per_epoch
        eta_min = self.eta_min_ratio

        if self.scheduler_kind == "cosine":

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                progress = min(max(progress, 0.0), 1.0)
                return eta_min + (1.0 - eta_min) * 0.5 * (
                    1.0 + math.cos(progress * math.pi)
                )

            return torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda)

        if self.scheduler_kind == "cosine_restarts":
            period_steps = max(1, self.restart_period_epochs * steps_per_epoch)

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / max(warmup_steps, 1)
                t = (step - warmup_steps) % period_steps
                progress = t / period_steps
                return eta_min + (1.0 - eta_min) * 0.5 * (
                    1.0 + math.cos(progress * math.pi)
                )

            return torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda)

        if self.scheduler_kind == "constant":

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / max(warmup_steps, 1)
                return 1.0

            return torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda)

        raise ValueError(f"unknown scheduler: {self.scheduler_kind}")

    def _lat_weights_for(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        lat_w = self.lat_weights.reshape(1, 1, -1, 1)
        if "_lat_top" in batch and "_lat_height" in batch:
            top = int(batch["_lat_top"].item())
            height = int(batch["_lat_height"].item())
            lat_w = lat_w[:, :, top : top + height, :]
        return lat_w

    def estimate_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        target = batch["target"]
        if target.dim() == 3:
            target = target.unsqueeze(1)

        mask = batch.get("input_mask")
        prediction = self.network(
            input_level=batch["input_level"],
            input_auxiliary=batch["input_auxiliary"],
            input_mask=mask,
        )
        clamped = prediction.clamp(0.0, 1.0)

        diff = (prediction - target).abs()
        lat_w = self._lat_weights_for(batch)
        if mask is not None:
            pix_w = (1.0 + (self.mask_loss_weight - 1.0) * mask) * lat_w
            loss = (diff * pix_w).sum() / pix_w.sum()
        else:
            loss = (diff * lat_w).mean()

        return {"loss": loss, "prediction": clamped}

    def estimate_auxiliary_loss(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        target = batch["target"]
        if target.dim() == 3:
            target = target.unsqueeze(1)
        pred = outputs["prediction"]
        lat_w = self._lat_weights_for(batch)
        mae = ((pred - target).abs() * lat_w).mean()
        mse = ((pred - target).pow(2) * lat_w).mean()
        return {"mae": mae, "mse": mse}

    def _augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # random longitude roll
        if torch.rand(1).item() < self.lon_roll_p:
            batch = _random_lon_roll(batch)
        # random sub-patch crop
        H, W = batch["input_level"].shape[-2:]
        if (
            self.crop_p_apply > 0.0
            and self.crop_size > 0
            and self.crop_size < H
            and torch.rand(1).item() < self.crop_p_apply
        ):
            ch = self.crop_size
            cw = self.crop_size
            top = int(torch.randint(0, H - ch + 1, (1,)).item())
            left = int(torch.randint(0, W - cw + 1, (1,)).item())
            for k in list(batch.keys()):
                v = batch[k]
                if v.dim() >= 2 and v.shape[-2] == H and v.shape[-1] == W:
                    batch[k] = v[..., top : top + ch, left : left + cw]
            batch["_lat_top"] = torch.tensor(top, device=batch["input_level"].device)
            batch["_lat_height"] = torch.tensor(ch, device=batch["input_level"].device)
        # random rectangular input-masking
        b = batch["input_level"].shape[0]
        h, w = batch["input_level"].shape[-2:]
        batch["input_mask"] = _make_random_block_mask(
            b,
            h,
            w,
            device=batch["input_level"].device,
            p_apply=self.mask_p_apply,
            n_blocks=(1, self.mask_n_blocks_max),
            block_frac=(0.10, self.mask_block_frac_max),
        )
        return batch

    def _train_epoch(self) -> float:
        from tqdm.autonotebook import tqdm

        amp_dtype = torch.bfloat16

        _n_samples = 0
        _acc_loss = 0.0
        self.network.train()
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            batch = self._move_to_device(batch)
            batch = self._augment_batch(batch)

            self._optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = self.estimate_loss(batch)
                loss = out["loss"]
            else:
                out = self.estimate_loss(batch)
                loss = out["loss"]

            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.grad_clip
                )
            self._optimizer.step()
            self._scheduler.step()

            cur = loss.item()
            curr_samples = batch["input_level"].shape[0]
            _n_samples += curr_samples
            _acc_loss += cur * curr_samples
            self._global_step += 1
            self.log(
                {
                    "train/loss": cur,
                    "lr": self._optimizer.param_groups[0]["lr"],
                },
                flush=False,
            )
            pbar.set_postfix(loss=cur)
        return _acc_loss / max(_n_samples, 1)

    def _val_epoch(self) -> Tuple[float, Dict[str, float]]:
        from tqdm.autonotebook import tqdm

        amp_dtype = torch.bfloat16

        _n_samples = 0
        _losses_list: List[Dict[str, float]] = []
        self.network.eval()
        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        for batch in pbar:
            batch = self._move_to_device(batch)
            with torch.no_grad():
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out = self.estimate_loss(batch)
                else:
                    out = self.estimate_loss(batch)
            aux = self.estimate_auxiliary_loss(batch, out)
            cur_samples = batch["input_level"].shape[0]
            _n_samples += cur_samples
            row = {k: v.item() * cur_samples for k, v in aux.items()}
            row["loss"] = out["loss"].item() * cur_samples
            _losses_list.append(row)
            pbar.set_postfix(loss=out["loss"].item())

        val_loss = sum(_loss["loss"] for _loss in _losses_list) / max(_n_samples, 1)
        aux_losses = {
            k: sum(_loss[k] for _loss in _losses_list) / max(_n_samples, 1)
            for k in _losses_list[0]
            if k != "loss"
        }
        return val_loss, aux_losses

    def train(self) -> torch.nn.Module:
        """Simplified epoch loop: no ReduceLROnPlateau step, no early stopping."""
        from tqdm.autonotebook import tqdm

        epoch_pbar = tqdm(
            range(1, self.n_epochs + 1),
            desc="Epochs",
            smoothing=0.1,
        )
        for idx_epoch in epoch_pbar:
            train_loss = self._train_epoch()
            val_loss, aux_losses = self._val_epoch()
            epoch_pbar.set_postfix(
                train_loss=train_loss,
                val_loss=val_loss,
            )
            improved = self._check_save_checkpoint(val_loss)
            if improved:
                self._epochs_since_best = 0
            else:
                self._epochs_since_best += 1
            self.log(
                {
                    "epoch": idx_epoch,
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "lr": self._optimizer.param_groups[0]["lr"],
                    **{f"val/{k}": v for k, v in aux_losses.items()},
                },
                flush=True,
                step=idx_epoch,
            )
            if (
                self.early_stop_patience is not None
                and self._epochs_since_best >= self.early_stop_patience
            ):
                main_logger.info(
                    "Early stop at epoch %d: val/epoch_loss has not improved "
                    "for %d consecutive epochs (patience=%d).",
                    idx_epoch,
                    self._epochs_since_best,
                    self.early_stop_patience,
                )
                break
        if os.path.exists(self.best_model_path):
            self._load_best_checkpoint()
        else:
            main_logger.warning(
                "No checkpoint was saved during training. "
                "Returning model with final weights."
            )
        return self.network
