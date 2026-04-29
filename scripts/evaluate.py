#!/bin/env python
#
# Built for the CI 2026 hackathon starter kit

r"""
Validation utilities for the CI 2026 starter kit.

This module provides error and probabilistic score estimation helpers for
model output validation, including mean absolute error and CRPS for ensemble
forecasts.
"""

# System modules
import argparse
import json
import logging
import os
from typing import Dict

import numpy as np

# External modules
import xarray as xr

# Internal modules


main_logger = logging.getLogger(__name__)


argument_parser = argparse.ArgumentParser(
    description="Climate Informatics 2026 Hackathon Validator"
)
argument_parser.add_argument(
    "--prediction_dir",
    type=str,
    required=True,
    help="Directory containing the prediction files.",
)
argument_parser.add_argument(
    "--reference_dir",
    type=str,
    default="data/train_data",
    required=False,
    help="Directory containing the reference target files.",
)
argument_parser.add_argument(
    "--prefix",
    type=str,
    default="val",
    required=False,
    help="Prefix for the prediction and target file names (e.g., 'val' or 'test').",
)
argument_parser.add_argument(
    "--output_path",
    type=str,
    default="validation_scores.json",
    required=False,
    help="Path to save the validation scores as a JSON file.",
)
argument_parser.add_argument(
    "--team_name",
    type=str,
    default="my_team",
    required=False,
    help="Name of the team, stored within the json for logging purpose.",
)
argument_parser.add_argument(
    "--to_json",
    action="store_true",
    help="Whether to save the validation scores to a JSON file specified "
    "by --output_path.",
)


lat_weights = [
    1.02223544,
    1.03733447,
    1.05172255,
    1.06538984,
    1.07832696,
    1.09052506,
    1.10197576,
    1.11267122,
    1.12260411,
    1.13176763,
    1.14015549,
    1.14776194,
    1.15458177,
    1.16061031,
    1.16584343,
    1.17027754,
    1.17390959,
    1.17673711,
    1.17875815,
    1.17997133,
    1.18037582,
    1.17997133,
    1.17875815,
    1.17673711,
    1.17390959,
    1.17027754,
    1.16584343,
    1.16061031,
    1.15458177,
    1.14776194,
    1.14015549,
    1.13176763,
    1.12260411,
    1.11267122,
    1.10197576,
    1.09052506,
    1.07832696,
    1.06538984,
    1.05172255,
    1.03733447,
    1.02223544,
    1.00643583,
    0.98994646,
    0.97277862,
    0.95494409,
    0.9364551,
    0.9173243,
    0.89756481,
    0.87719018,
    0.85621436,
    0.83465174,
    0.81251709,
    0.78982559,
    0.76659277,
    0.74283457,
    0.71856727,
    0.6938075,
    0.66857222,
    0.64287875,
    0.61674467,
    0.59018791,
    0.56322666,
    0.53587941,
    0.50816489,
]


def estimate_mean_abs_error(
    predictions: xr.DataArray,
    targets: xr.DataArray,
) -> xr.DataArray:
    """
    Compute mean absolute error between predictions and targets.

    Parameters
    ----------
    predictions: xr.DataArray
        Predicted values.
    targets: xr.DataArray
        Target observations.

    Returns:
        Mean absolute error as xarray.DataArray
    """
    mae = np.abs(predictions - targets)
    return mae


def estimate_crps_ens(
    ens: xr.DataArray,
    target: xr.DataArray,
) -> xr.DataArray:
    """
    Compute CRPS for ensemble forecasts using xarray.DataArray.
    Based on Hersbach, 2000, Eq. 10, Eq. 11, and Eq. 20. Uses
    the fair CRPS estimator to avoid bias for small ensemble sizes.

    Parameters
    ----------
    ens: xr.DataArray
        Ensemble forecast data (must include the ensemble dimension).
    target: xr.DataArray
        Target observations (must have same dimensions as ens except
        "ensemble").

    Returns:
    xr.DataArray
        Estimate CRPS score with the same shape as the target data.
    """
    # Sort ensemble members along ensemble dimension
    ens_sorted = xr.apply_ufunc(
        np.sort,
        ens,
        input_core_dims=[["ensemble"]],
        output_core_dims=[["ensemble"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ens.dtype],
    )

    # Compute differences between consecutive members
    diff = ens_sorted.diff(dim="ensemble")

    # Create weights array
    N = ens_sorted.sizes["ensemble"]
    i = xr.DataArray(np.arange(1, N), dims=["ensemble"])
    weight = i * (N - i)

    # Compute CRPS components
    abs_diff_mean = np.abs(ens_sorted - target).mean(dim="ensemble")
    weighted_diff_sum = (weight * diff).sum(dim="ensemble")
    weighted_diff_sum = weighted_diff_sum / N / (N - 1)

    # Final CRPS calculation
    crps = abs_diff_mean - weighted_diff_sum
    return crps


class Validator:
    _DEFAULT_BASELINE_SCORE = {
        "ERA5_1": 0.16069515545145657,
        "ERA5_2": 0.15156865092228178,
        "AIMIP_1": 0.15072763610422082,
        "AIMIP_2": 0.1509060015480829,
    }
    _SCORE_FUNCS = {
        "ERA5_1": estimate_mean_abs_error,
        "ERA5_2": estimate_mean_abs_error,
        "AIMIP_1": estimate_crps_ens,
        "AIMIP_2": estimate_crps_ens,
    }
    _LOSS_NAMES = {
        "ERA5_1": "mae_era5_region1",
        "ERA5_2": "mae_era5_region2",
        "AIMIP_1": "crps_aimip_region1",
        "AIMIP_2": "crps_aimip_region2",
    }

    def __init__(self):
        r"""
        Initialize the validator with latitude weights and baseline scores.
        """
        self.lat_weights = xr.DataArray(lat_weights, dims=["lat"])

    def __call__(
        self,
        predictions: Dict[str, xr.DataArray],
        targets: Dict[str, xr.DataArray],
    ) -> Dict[str, float]:
        r"""
        Compute weighted validation scores for prediction targets.

        Parameters
        ----------
        predictions : Dict of str to xr.DataArray
            Predicted variables. Each element may be deterministic or an
            ensemble forecast.
        targets : Dict of str to xr.DataArray
            Corresponding target observations.

        Returns
        -------
        Dict of str to float
            Weighted score followed by individual scores for each target.
        """
        scores = {}
        for name, prediction in predictions.items():
            score = self._SCORE_FUNCS[name](prediction, targets[name])
            score = (score * self.lat_weights).mean()
            scores[name] = score.item()
        output_dict = {self._LOSS_NAMES[name]: score for name, score in scores.items()}
        output_dict["score"] = (
            sum(
                1 - score / self._DEFAULT_BASELINE_SCORE[name]
                for name, score in scores.items()
            )
            / 4
        )
        return output_dict


def to_ensemble_pred(predictions: xr.DataArray) -> xr.DataArray:
    r"""
    Convert AIMIP prediction to ensemble format by inverting the order of the
    time dimension and splitting the time dimension into three ensemble
    members.

    Parameters
    ----------
    predictions : xr.DataArray
        Predictions ordered according to the AIMIP submission order.

    Returns
    -------
    xr.DataArray
        Predictions reordered and split into ensemble members.
    """
    n_sample = predictions.sizes["sample"]
    n_ens = 3
    sample_per_ens = n_sample // n_ens
    predictions = xr.concat(
        [
            predictions.isel(
                sample=slice(i * sample_per_ens, (i + 1) * sample_per_ens)
            ).drop_vars("sample")
            for i in range(n_ens)
        ],
        dim="ensemble",
        join="outer",
    )
    predictions = predictions.transpose("ensemble", "sample", "lat", "lon")
    return predictions


def evaluate_dir(
    prediction_dir: str,
    reference_dir: str,
    prefix: str,
    output_path: str,
    team_name: str = "my_team",
    to_json: bool = False,
) -> None:
    r"""
    Main function to run the validator.

    Parameters
    ----------
    prediction_dir : str
        Directory containing the prediction files.
    reference_dir : str
        Directory containing the reference target files.
    prefix : str
        Prefix for the prediction and target file names
        (e.g., "val" or "test").
    output_path : str
        Path to save the validation scores as a JSON file.
    team_name : str, optional
        Name of the team, stored within the json for logging purpose. Default
        is "my_team".
    to_json : bool, optional
        Whether to save the validation scores to a JSON file specified by
        output_path. Default is False.
    """
    # Initialize the validator
    validator = Validator()

    # Prediction paths
    prediction_paths = {
        "ERA5_1": os.path.join(prediction_dir, f"{prefix:s}_era5_region1.nc"),
        "ERA5_2": os.path.join(prediction_dir, f"{prefix:s}_era5_region2.nc"),
        "AIMIP_1": os.path.join(prediction_dir, f"{prefix:s}_aimip_region1.nc"),
        "AIMIP_2": os.path.join(prediction_dir, f"{prefix:s}_aimip_region2.nc"),
    }

    # Target paths
    target_paths = {
        "ERA5_1": os.path.join(reference_dir, f"{prefix:s}_target_era5_region1.nc"),
        "ERA5_2": os.path.join(reference_dir, f"{prefix:s}_target_era5_region2.nc"),
        "AIMIP_1": os.path.join(reference_dir, f"{prefix:s}_target_aimip_region1.nc"),
        "AIMIP_2": os.path.join(reference_dir, f"{prefix:s}_target_aimip_region2.nc"),
    }

    # Load predictions and targets
    predictions = {
        name: xr.open_dataarray(path) for name, path in prediction_paths.items()
    }
    targets = {name: xr.open_dataarray(path) for name, path in target_paths.items()}
    predictions["AIMIP_1"] = to_ensemble_pred(predictions["AIMIP_1"])
    predictions["AIMIP_2"] = to_ensemble_pred(predictions["AIMIP_2"])
    scores = validator(predictions, targets)
    print("Finished evaluation with scores:", scores)

    if to_json:
        scores["team_name"] = team_name
        with open(output_path, "w") as f:
            json.dump(scores, f)


if __name__ == "__main__":
    args = argument_parser.parse_args()
    evaluate_dir(
        prediction_dir=args.prediction_dir,
        reference_dir=args.reference_dir,
        prefix=args.prefix,
        output_path=args.output_path,
        team_name=args.team_name,
        to_json=args.to_json,
    )
