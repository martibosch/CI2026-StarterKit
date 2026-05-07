#!/bin/env python
#
# SegFormer-based cloud-cover predictor for the CI 2026 hackathon.
#
# Same geography-agnostic philosophy as GeoUNet: only pressure-level
# meteorological fields are used (no static geography, no lat/lon).
#
# The Mix Transformer (MiT) encoder is loaded with ImageNet-pretrained
# weights via HuggingFace, except the first patch-embedding conv which is
# reinitialized to accept `in_channels` inputs instead of 3.  The rest of
# the encoder fine-tunes normally.
#
# Per-level RH normalization is used when use_rh=True, consistent with the
# finding that fixed (rh - 0.5)*4 is miscentered by 0.1–0.3 at most levels.

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from starter_kit.baselines.utils import estimate_relative_humidity
from starter_kit.layers import InputNormalization
from starter_kit.model import BaseModel
from starter_kit.models.geounet import _load_stats, _random_lon_roll

main_logger = logging.getLogger(__name__)

_PRETRAINED_IDS = {
    "b0": "nvidia/mit-b0",
    "b1": "nvidia/mit-b1",
    "b2": "nvidia/mit-b2",
    "b3": "nvidia/mit-b3",
    "b4": "nvidia/mit-b4",
    "b5": "nvidia/mit-b5",
}


class GeoSegFormer(nn.Module):
    """
    SegFormer for cloud-cover regression.

    Inputs
    ------
    input_level : (B, 4, 7, H, W) — pressure-level fields only.
    input_auxiliary : ignored (geography-agnostic design).
    input_mask : ignored (no mask token in SegFormer).
    """

    def __init__(
        self,
        normalization_path: str = "",
        mit_variant: str = "b2",
        pretrained: bool = True,
        n_level_vars: int = 4,
        n_levels: int = 7,
        use_rh: bool = False,
    ) -> None:
        super().__init__()

        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "transformers is required for GeoSegFormer. "
                "Install it with: pip install transformers"
            ) from exc

        stats = _load_stats(normalization_path)
        n_lv_ch = n_level_vars * n_levels

        mean = torch.tensor(stats["mean"][:n_lv_ch]).float()
        std = torch.tensor(stats["std"][:n_lv_ch]).float()
        self.normalization = InputNormalization(mean=mean, std=std)

        self.use_rh = use_rh
        self._n_levels = n_levels
        self._t_start = 0
        self._q_start = n_levels

        if use_rh:
            self.register_buffer(
                "pressure_levels",
                torch.tensor(stats["pressure_levels_pa"], dtype=torch.float32).reshape(
                    1, n_levels, 1, 1
                ),
            )
            channel_names = stats.get("channel_names", [])
            rh_means, rh_stds = [], []
            for i in range(n_levels):
                name = f"rh@L{i}"
                try:
                    idx = channel_names.index(name)
                    rh_means.append(float(stats["mean"][idx]))
                    rh_stds.append(float(stats["std"][idx]))
                except ValueError:
                    rh_means.append(0.5)
                    rh_stds.append(0.25)
            self.register_buffer(
                "rh_mean",
                torch.tensor(rh_means).float().reshape(1, n_levels, 1, 1),
            )
            self.register_buffer(
                "rh_std",
                torch.tensor(rh_stds).float().reshape(1, n_levels, 1, 1),
            )

        in_channels = n_lv_ch + (n_levels if use_rh else 0)
        hub_id = _PRETRAINED_IDS.get(mit_variant, mit_variant)

        if pretrained:
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                hub_id,
                num_labels=1,
                num_channels=in_channels,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(hub_id)
            config.num_labels = 1
            config.num_channels = in_channels
            config.id2label = {0: "cloud_cover"}
            config.label2id = {"cloud_cover": 0}
            self.segformer = SegformerForSemanticSegmentation(config)

    def forward(
        self,
        input_level: torch.Tensor,
        input_auxiliary: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del input_auxiliary, input_mask

        b = input_level.shape[0]
        h, w = input_level.shape[-2:]
        x_level = input_level.reshape(b, -1, h, w)

        x = self.normalization(x_level.movedim(1, -1)).movedim(-1, 1)

        if self.use_rh:
            T_raw = x_level[:, self._t_start : self._t_start + self._n_levels]
            q_raw = x_level[:, self._q_start : self._q_start + self._n_levels]
            rh = estimate_relative_humidity(
                T_raw.float().clamp(150.0, 350.0),
                q_raw.float().clamp(0.0, 0.1),
                self.pressure_levels.float(),
            )
            rh = torch.nan_to_num(rh, nan=0.0, posinf=1.0, neginf=0.0)
            rh_feat = ((rh - self.rh_mean.float()) / (self.rh_std.float() + 1e-6)).to(
                x.dtype
            )
            x = torch.cat([x, rh_feat], dim=1)

        # SegFormer decoder outputs at H/4, W/4 — upsample to input resolution
        logits = self.segformer(pixel_values=x).logits
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SegFormerModel(BaseModel):
    """
    BaseModel subclass for GeoSegFormer.

    Reuses GeoUNetModel's cosine LR schedule, bf16 AMP, and longitude-roll
    augmentation. No masking augmentation (SegFormer has no mask token).
    Loss is lat-weighted L1, identical to GeoUNetModel.
    """

    def __init__(
        self,
        *args,
        lon_roll_p: float = 1.0,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        warmup_epochs: int = 1,
        scheduler: str = "cosine",
        restart_period_epochs: int = 20,
        eta_min_ratio: float = 0.01,
        compile_model: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if compile_model:
            try:
                self.network = torch.compile(self.network)
            except Exception as exc:
                main_logger.warning("torch.compile unavailable: %s", exc)
        self.lon_roll_p = lon_roll_p
        self.use_amp = use_amp and self.device.type == "cuda"
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.scheduler_kind = scheduler
        self.restart_period_epochs = int(restart_period_epochs)
        self.eta_min_ratio = float(eta_min_ratio)
        self._scheduler = self._build_scheduler()
        self._global_step = 0
        self._current_epoch = 1

    def _setup_optimizer(self) -> None:
        self._optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        steps_per_epoch = max(1, len(self.train_loader))
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

        raise ValueError(f"unknown scheduler: {self.scheduler_kind!r}")

    def _augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1).item() < self.lon_roll_p:
            batch = _random_lon_roll(batch)
        return batch

    def estimate_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        target = batch["target"]
        if target.dim() == 3:
            target = target.unsqueeze(1)
        prediction = self.network(
            input_level=batch["input_level"],
            input_auxiliary=batch["input_auxiliary"],
        )
        clamped = prediction.clamp(0.0, 1.0)
        diff = (prediction - target).abs()
        lat_w = self.lat_weights.reshape(1, 1, -1, 1)
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
        lat_w = self.lat_weights.reshape(1, 1, -1, 1)
        mae = ((pred - target).abs() * lat_w).mean()
        mse = ((pred - target).pow(2) * lat_w).mean()
        return {"mae": mae, "mse": mse}

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
            else:
                out = self.estimate_loss(batch)

            out["loss"].backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.grad_clip
                )
            self._optimizer.step()
            self._scheduler.step()

            cur = out["loss"].item()
            curr_samples = batch["input_level"].shape[0]
            _n_samples += curr_samples
            _acc_loss += cur * curr_samples
            self._global_step += 1
            self.log(
                {"train/loss": cur, "lr": self._optimizer.param_groups[0]["lr"]},
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

        val_loss = sum(r["loss"] for r in _losses_list) / max(_n_samples, 1)
        aux_losses = {
            k: sum(r[k] for r in _losses_list) / max(_n_samples, 1)
            for k in _losses_list[0]
            if k != "loss"
        }
        return val_loss, aux_losses

    def train(self) -> nn.Module:
        import os

        from tqdm.autonotebook import tqdm

        epoch_pbar = tqdm(
            range(1, self.n_epochs + 1),
            desc="Epochs",
            smoothing=0.1,
        )
        for idx_epoch in epoch_pbar:
            self._current_epoch = idx_epoch
            train_loss = self._train_epoch()
            val_loss, aux_losses = self._val_epoch()
            epoch_pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)
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
                    "Early stop at epoch %d (patience=%d).",
                    idx_epoch,
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
