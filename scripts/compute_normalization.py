#!/usr/bin/env python
#
# Built for the CI 2026 hackathon starter kit

r"""
Compute lat-weighted, time-averaged mean/std for the model input channels and
write them to a JSON artifact consumed by the network at construction time.

The output channel ordering matches the model's flatten convention:

    [vars_level[0]@L0..L{L-1},
     vars_level[1]@L0..L{L-1},
     ...,
     (rh@L0..L{L-1} if --use_rh),
     vars_aux[0..n_auxiliary_fields-1]]

Run with::

    python scripts/compute_normalization.py \
        --zarr_path data/train_data/train.zarr \
        --n_auxiliary_fields 2 \
        --output_path data/stats/normalization_no_rh_aux2.json

Add ``--use_rh`` to append per-level relative-humidity stats (RH is computed
inline with the same Magnus formula as
``starter_kit.baselines.utils.estimate_relative_humidity``).
"""

import argparse
import json
import logging
import pathlib

import numpy as np
import xarray as xr

from starter_kit import lat_weights as LAT_WEIGHTS_LIST

main_logger = logging.getLogger(__name__)


# Magnus-formula constants, mirroring starter_kit.baselines.utils.
_RD = 287.0597
_RV = 461.51
_EPSILON = _RD / _RV
_A1_W, _A3_W, _A4_W = 611.21, 17.502, 32.19
_A1_I, _A3_I, _A4_I = 611.21, 22.587, -0.7
_T0 = 273.16
_TICE = 250.16


def _level_to_pa(levels: np.ndarray) -> np.ndarray:
    # Heuristic: levels expressed in hPa (max ~1000) vs Pa (max ~100000).
    return levels * 100.0 if float(levels.max()) < 2000 else levels


def _relative_humidity(
    temperature: xr.DataArray,
    specific_humidity: xr.DataArray,
    pressure: xr.DataArray,
) -> xr.DataArray:
    sat_w = _A1_W * np.exp(_A3_W * (temperature - _T0) / (temperature - _A4_W))
    sat_i = _A1_I * np.exp(_A3_I * (temperature - _T0) / (temperature - _A4_I))
    alpha = (((temperature - _TICE) / (_T0 - _TICE)) ** 2).clip(0.0, 1.0)
    saturation = alpha * sat_w + (1.0 - alpha) * sat_i
    vapor = (specific_humidity * pressure) / (
        _EPSILON + specific_humidity * (1.0 - _EPSILON)
    )
    return (vapor / (saturation + 1e-12)).clip(0.0, 1.0)


def _flatten_var_level(
    mean_da: xr.DataArray,
    std_da: xr.DataArray,
    vars_level: list[str],
    n_levels: int,
    means: list[float],
    stds: list[float],
    channel_names: list[str],
    name_fmt: str = "{v}@L{l}",
) -> None:
    for v_name in vars_level:
        for l_idx in range(n_levels):
            m = float(mean_da.sel(vars_level=v_name).isel(level=l_idx).values)
            s = float(std_da.sel(vars_level=v_name).isel(level=l_idx).values)
            means.append(m)
            stds.append(s)
            channel_names.append(name_fmt.format(v=v_name, l=l_idx))
            main_logger.info("  %s mean=%.6g std=%.6g", channel_names[-1], m, s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--use_rh", action="store_true")
    parser.add_argument("--n_auxiliary_fields", type=int, default=2)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ds = xr.open_zarr(args.zarr_path)
    vars_level = [str(v) for v in ds["vars_level"].values]
    vars_aux = [str(v) for v in ds["vars_aux"].values]
    levels = ds["level"].values
    n_levels = len(levels)
    main_logger.info("vars_level=%s vars_aux=%s", vars_level, vars_aux)

    if args.n_auxiliary_fields > len(vars_aux):
        raise ValueError(
            f"--n_auxiliary_fields={args.n_auxiliary_fields} exceeds "
            f"available vars_aux={len(vars_aux)}"
        )

    lat_weights = xr.DataArray(LAT_WEIGHTS_LIST, dims=["lat"])

    means: list[float] = []
    stds: list[float] = []
    channel_names: list[str] = []

    # --- Level fields: single chunked pass over the full (time, vars_level,
    # level, lat, lon) array. Reduces to (vars_level, level).
    main_logger.info("Reducing input_level over time/lat/lon...")
    level_block = ds["input_level"].weighted(lat_weights)
    level_mean = level_block.mean(dim=["time", "lat", "lon"]).compute()
    level_std = level_block.std(dim=["time", "lat", "lon"]).compute()
    _flatten_var_level(
        level_mean, level_std, vars_level, n_levels, means, stds, channel_names
    )

    # --- Per-level RH: build a derived DataArray, reduce in one pass.
    if args.use_rh:
        # CMIP short names: ta = temperature, hus = specific humidity.
        t_name = next((n for n in ("ta", "temperature") if n in vars_level), None)
        q_name = next(
            (n for n in ("hus", "specific_humidity") if n in vars_level), None
        )
        if t_name is None or q_name is None:
            raise ValueError(
                "--use_rh requires temperature ('ta'|'temperature') and "
                f"specific humidity ('hus'|'specific_humidity') in vars_level "
                f"(got {vars_level})"
            )
        pressures_pa = _level_to_pa(levels)
        pressure = xr.DataArray(
            pressures_pa, dims=["level"], coords={"level": ds["level"]}
        )
        rh = _relative_humidity(
            temperature=ds["input_level"].sel(vars_level=t_name),
            specific_humidity=ds["input_level"].sel(vars_level=q_name),
            pressure=pressure,
        )
        main_logger.info("Reducing rh over time/lat/lon...")
        rh_block = rh.weighted(lat_weights)
        rh_mean = rh_block.mean(dim=["time", "lat", "lon"]).compute()
        rh_std = rh_block.std(dim=["time", "lat", "lon"]).compute()
        for l_idx in range(n_levels):
            m = float(rh_mean.isel(level=l_idx).values)
            s = float(rh_std.isel(level=l_idx).values)
            means.append(m)
            stds.append(s)
            channel_names.append(f"rh@L{l_idx}")
            main_logger.info("  rh@L%d mean=%.6g std=%.6g", l_idx, m, s)

    # --- Aux fields: single pass over the selected slice.
    main_logger.info("Reducing input_auxiliary...")
    aux_sel = ds["input_auxiliary"].sel(vars_aux=vars_aux[: args.n_auxiliary_fields])
    aux_reduce_dims = [d for d in aux_sel.dims if d != "vars_aux"]
    aux_block = aux_sel.weighted(lat_weights)
    aux_mean = aux_block.mean(dim=aux_reduce_dims).compute()
    aux_std = aux_block.std(dim=aux_reduce_dims).compute()
    for a_idx, a_name in enumerate(vars_aux[: args.n_auxiliary_fields]):
        m = float(aux_mean.isel(vars_aux=a_idx).values)
        s = float(aux_std.isel(vars_aux=a_idx).values)
        means.append(m)
        stds.append(s)
        channel_names.append(f"aux:{a_name}")
        main_logger.info("  aux:%s mean=%.6g std=%.6g", a_name, m, s)

    out = {
        "channel_names": channel_names,
        "mean": means,
        "std": stds,
        "use_rh": bool(args.use_rh),
        "n_auxiliary_fields": int(args.n_auxiliary_fields),
        "vars_level": vars_level,
        "vars_aux": vars_aux[: args.n_auxiliary_fields],
        "levels": [float(x) for x in levels],
        "pressure_levels_pa": [float(x) for x in _level_to_pa(levels)],
    }
    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(out, f, indent=2)
    main_logger.info("Wrote %d-channel stats to %s", len(means), output_path)


if __name__ == "__main__":
    main()
