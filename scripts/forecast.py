#!/usr/bin/env python
#
# Built for the CI 2026 hackathon starter kit

r"""
Forecasting script for weather/climate baseline models.

Loads a trained network checkpoint, runs the forward pass over a
test dataset (no targets), and writes predictions to a netCDF file.

Run with::

    python scripts/forecast.py

Override config values on the command line::

    python scripts/forecast.py device=cuda store_path=runs/mlp
"""

# System modules
import logging
import os
from typing import List

import hydra

# External modules
import numpy as np
import torch
import torch.nn
import xarray as xr
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

# Internal modules
from starter_kit.data import TestDataset

main_logger = logging.getLogger(__name__)


def _build_network(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    r"""
    Instantiate and load a trained network from a checkpoint.

    Parameters
    ----------
    cfg : DictConfig
        Network sub-config (``cfg.network``). Must contain
        ``_target_``.
    device : torch.device
        Device to place the network on.

    Returns
    -------
    torch.nn.Module
        Network loaded with checkpoint weights, in eval mode.
    """
    network = hydra.utils.instantiate(cfg)
    return network.to(device)


def _load_checkpoint(
    network: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    r"""
    Load state-dict from a checkpoint file into the network.

    Parameters
    ----------
    network : torch.nn.Module
        Network whose parameters will be overwritten.
    checkpoint_path : str
        Path to the ``.ckpt`` / ``.pt`` checkpoint file.
    device : torch.device
        Device to map tensors onto when loading.

    Returns
    -------
    torch.nn.Module
        Network in eval mode with loaded weights.

    Raises
    ------
    FileNotFoundError
        If ``checkpoint_path`` does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(state_dict)
    return network


def _build_loader(data_path: str, cfg: DictConfig) -> DataLoader:
    r"""
    Build a DataLoader over the test dataset.

    Parameters
    ----------
    cfg : DictConfig
        Data sub-config (``cfg.data``).
    data_path : str
        Path to the test zarr dataset.

    Returns
    -------
    DataLoader
        Non-shuffled loader over the test set.
    """
    test_ds = TestDataset(data_path=data_path)
    return DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=(cfg.pin_memory if torch.cuda.is_available() else False),
    )


def _load_coordinates(data_path: str) -> xr.Dataset:
    r"""
    Read latitude and longitude coordinates from the zarr store.

    Parameters
    ----------
    data_path : str
        Path to the test zarr dataset.

    Returns
    -------
    xr.Dataset
        Dataset containing at least ``latitude`` and ``longitude``
        coordinate arrays.
    """
    with xr.open_zarr(data_path) as ds:
        return ds[["lat", "lon"]].load()


def _forward_batch(
    network: torch.nn.Module,
    batch: dict,
    flip_h: bool = False,
    flip_w: bool = False,
) -> torch.Tensor:
    """Forward pass with optional spatial flips, undone on the output."""
    dims = []
    if flip_h:
        dims.append(-2)
    if flip_w:
        dims.append(-1)
    if dims:
        batch = {
            k: torch.flip(v, dims) if v.dim() >= 2 else v for k, v in batch.items()
        }
    pred = network(
        input_level=batch["input_level"],
        input_auxiliary=batch["input_auxiliary"],
    )
    if dims:
        pred = torch.flip(pred, dims)
    return pred.clamp(0.0, 1.0)


@torch.inference_mode()
def _run_inference(
    network: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
) -> np.ndarray:
    r"""
    Run the forward pass over all batches and collect predictions.

    Parameters
    ----------
    network : torch.nn.Module
        Trained network in eval mode.
    loader : DataLoader
        DataLoader yielding test batches without targets.
    device : torch.device
        Device for computation.
    tta : bool, optional
        If True, average predictions over 4 flip augmentations
        (identity, flip-H, flip-W, flip-both).

    Returns
    -------
    np.ndarray
        Predictions with shape ``(T, H, W)``, values in ``[0, 1]``.
    """
    _tta_flips = [(False, False), (True, False), (False, True), (True, True)]
    predictions: List[np.ndarray] = []
    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        if tta:
            preds = [_forward_batch(network, batch, fh, fw) for fh, fw in _tta_flips]
            pred = torch.stack(preds, dim=0).mean(dim=0)
        else:
            pred = _forward_batch(network, batch)
        predictions.append(pred.squeeze(1).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _save_predictions(
    predictions: np.ndarray,
    coord_ds: xr.Dataset,
    output_path: str,
) -> None:
    r"""
    Write predictions to a netCDF file with spatial coordinates.

    Parameters
    ----------
    predictions : np.ndarray
        Predictions of shape ``(T, H, W)``.
    coord_ds : xr.Dataset
        Dataset providing ``latitude`` and ``longitude`` arrays.
    output_path : str
        Destination path for the netCDF file.
    """
    sample_idx = np.arange(predictions.shape[0])
    ds = xr.Dataset(
        {
            "total_cloud_cover": (
                ["sample", "lat", "lon"],
                predictions,
                {"long_name": "Total cloud cover", "units": "1"},
            )
        },
        coords={
            "sample": sample_idx,
            "lat": coord_ds["lat"].values,
            "lon": coord_ds["lon"].values,
        },
    )
    ds.to_netcdf(output_path)
    main_logger.info("Predictions saved to %s", output_path)


def run_forecast(cfg: DictConfig) -> None:
    r"""
    Load checkpoint(s), run inference, and save predictions.

    Supports single-checkpoint and ensemble modes. In ensemble mode,
    set ``ensemble_ckpt_paths`` to a list of checkpoint paths; predictions
    are averaged across all checkpoints (and optionally TTA-augmented).

    Importable entry point for programmatic use (e.g. from submit.py).

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration tree. Must contain ``input_path``,
        ``output_path``, ``device``, ``network``, and ``data``.
        Either ``ckpt_path`` (single) or ``ensemble_ckpt_paths`` (list).
    """
    device = torch.device(cfg.device)
    tta = cfg.get("tta", False)

    loader = _build_loader(cfg.input_path, cfg.data)
    coord_ds = _load_coordinates(cfg.input_path)

    ensemble_paths = list(cfg.get("ensemble_ckpt_paths") or [])
    if ensemble_paths:
        all_preds = []
        for ckpt_path in ensemble_paths:
            network = _build_network(cfg.network, device)
            network = _load_checkpoint(network, ckpt_path, device)
            network = network.eval()
            all_preds.append(_run_inference(network, loader, device, tta=tta))
        predictions = np.mean(all_preds, axis=0)
    else:
        network = _build_network(cfg.network, device)
        if cfg.ckpt_path is not None:
            network = _load_checkpoint(network, cfg.ckpt_path, device)
        network = network.eval()
        predictions = _run_inference(network, loader, device, tta=tta)

    os.makedirs(os.path.split(cfg.output_path)[0], exist_ok=True)
    _save_predictions(predictions, coord_ds, cfg.output_path)

    main_logger.info("Forecasting complete.")


@hydra.main(config_path="../configs", config_name="forecast", version_base="1.3")
def main(cfg: DictConfig) -> None:
    r"""
    Hydra CLI entry point for forecasting.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration tree.
    """
    run_forecast(cfg)


if __name__ == "__main__":
    main()
