#!/bin/env python
#
# Built for the CI 2026 hackathon starter kit

# System modules
import json
import logging
from typing import Any, Dict, Optional

# External modules
import torch
import torch.nn

from starter_kit.baselines.utils import estimate_relative_humidity
from starter_kit.layers import InputNormalization

# Internal modules
from starter_kit.model import BaseModel

main_logger = logging.getLogger(__name__)

r"""
The normalization mean and std values are pre-computed from the training data.
As in the MLP, all pressure levels are collapsed into the channels dimension
and only the first two auxiliary fields (land sea mask and geopotential) are
used. For each of these 30 input features we compute the mean and std across
all spatial locations, weighted by the latitude weights, and averaged across
all time steps in the training set. These values are stored in the lists below
and used to initialize the InputNormalization layer in the MLPNetwork.
"""

_normalization_mean = [
    294.531359,
    287.010605,
    278.507482,
    262.805241,
    227.580722,
    201.364517,
    209.719502,
    0.010667,
    0.006922,
    0.003784,
    0.001229,
    0.000088,
    0.000003,
    0.000003,
    -1.412110,
    -0.914917,
    0.431349,
    3.504875,
    11.699176,
    6.758849,
    -1.214763,
    0.167424,
    -0.105374,
    -0.172138,
    -0.022648,
    0.030789,
    0.281048,
    -0.094608,
    0.410844,
    2129.684371,
]
_normalization_std = [
    62.864550,
    61.180621,
    58.938862,
    56.016099,
    47.532073,
    32.281805,
    38.084321,
    0.006102,
    0.004648,
    0.003013,
    0.001266,
    0.000080,
    0.000001,
    0.000000,
    4.661358,
    6.159993,
    7.763541,
    9.877940,
    16.068963,
    11.681901,
    10.705570,
    4.119853,
    4.318767,
    4.810067,
    6.209760,
    10.585627,
    5.680168,
    2.978756,
    0.498762,
    3602.712270,
]


class MLPNetwork(torch.nn.Module):
    r"""
    Multi-layer perceptron operating on flattened pressure-
    level and auxiliary fields.

    Parameters
    ----------
    input_dim : int, optional, default = 30
        Total number of input features after concatenation.
    hidden_dim : int, optional, default = 64
        Width of each hidden layer.
    n_layers : int, optional, default = 4
        Number of hidden Linear+SiLU blocks.
    normalization : InputNormalization or None, optional
        Pre-normalization layer applied before the MLP. When
        provided it must accept the concatenated input tensor
        of shape ``(..., input_dim)``.

    Attributes
    ----------
    normalization : InputNormalization or None
        Normalization layer stored as a sub-module.
    mlp : torch.nn.Sequential
        Sequence of linear layers with SiLU activations.
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 64,
        n_layers: int = 4,
        use_rh: bool = False,
        n_auxiliary_fields: int = 2,
        normalization_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.use_rh = use_rh
        self.n_auxiliary_fields = n_auxiliary_fields

        if normalization_path is not None:
            with open(normalization_path) as f:
                stats = json.load(f)
            mean = stats["mean"]
            std = stats["std"]
            if len(mean) != input_dim:
                raise ValueError(
                    f"normalization_path has {len(mean)} channels but "
                    f"input_dim={input_dim}"
                )
            if use_rh:
                if not stats.get("use_rh", False):
                    raise ValueError(
                        "use_rh=True but normalization_path was computed "
                        "without --use_rh"
                    )
                pressure_levels_pa = stats["pressure_levels_pa"]
        elif use_rh or n_auxiliary_fields != 2:
            raise ValueError(
                "Hardcoded normalization only supports use_rh=False and "
                "n_auxiliary_fields=2; pass a normalization_path computed "
                "via scripts/compute_normalization.py."
            )
        else:
            mean = _normalization_mean
            std = _normalization_std

        if use_rh:
            self.register_buffer(
                "pressure_levels",
                torch.tensor(pressure_levels_pa, dtype=torch.float32).reshape(-1, 1, 1),
            )
        self.normalization = InputNormalization(
            mean=torch.tensor(mean), std=torch.tensor(std)
        )
        layers = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.append(torch.nn.LayerNorm(hidden_dim))
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.SiLU())
        output_layer = torch.nn.Linear(hidden_dim, 1)
        torch.nn.init.normal_(output_layer.weight, std=1e-6)
        torch.nn.init.constant_(output_layer.bias, 0.5)
        layers.append(output_layer)
        self.mlp = torch.nn.Sequential(*layers)

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Forward pass: concatenate inputs, optionally normalize,
        then apply the MLP.

        Parameters
        ----------
        input_level : torch.Tensor
            Pressure-level fields, shape ``(B, C_l, L, H, W)``.
        input_auxiliary : torch.Tensor
            Auxiliary fields, shape ``(B, C_a, H, W)``.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(B, 1, H, W)``.
        """
        if self.use_rh:
            rh = estimate_relative_humidity(
                temperature=input_level[:, 0:1],
                specific_humidity=input_level[:, 1:2],
                pressure=self.pressure_levels,
            )
            input_level = torch.cat([input_level, rh], dim=1)

        # We collapse all levels into the channel dimension
        flattened_input_level = input_level.reshape(
            input_level.shape[0], -1, *input_level.shape[-2:]
        )
        sliced_auxiliary = input_auxiliary[:, : self.n_auxiliary_fields]

        # Concatenate the level and auxiliary fields
        mlp_input = torch.cat([flattened_input_level, sliced_auxiliary], dim=1)

        # Move the feature dimension to the end for normalization and MLP
        mlp_input = mlp_input.movedim(1, -1)

        # Apply input normalization
        mlp_input = self.normalization(mlp_input)

        # Apply the MLP
        prediction = self.mlp(mlp_input)

        # Move the channel dimension to the expected position
        prediction = prediction.movedim(-1, 1)
        return prediction


class MLPModel(BaseModel):
    r"""
    Model wrapper for an MLP network with standard loss outputs.

    This class delegates forward execution to a hidden MLP network and
    computes a mean absolute error loss together with auxiliary metrics.
    """

    def estimate_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        r"""
        Compute the primary training loss and prediction output.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing ``input_level``,
            ``input_auxiliary``, and ``target`` tensors.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``loss`` and ``prediction``.
            ``loss`` is the mean absolute error and ``prediction`` is the
            model output clamped to ``[0, 1]``.
        """
        prediction = self.network(
            input_level=batch["input_level"], input_auxiliary=batch["input_auxiliary"]
        )
        prediction = prediction.clamp(0.0, 1.0)
        loss = (prediction - batch["target"]).abs()
        loss = loss * self.lat_weights
        loss = loss.mean()
        return {"loss": loss, "prediction": prediction}

    def estimate_auxiliary_loss(
        self, batch: Dict[str, torch.Tensor], outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        r"""
        Compute auxiliary regression and classification metrics.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing the ground-truth ``target`` tensor.
        outputs : Dict[str, Any]
            Model outputs from ``estimate_loss`` containing ``prediction``.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``mse`` and ``accuracy``.
            ``mse`` is the mean squared error and ``accuracy`` is the
            thresholded classification accuracy at 0.5.
        """
        mse = (outputs["prediction"] - batch["target"]).pow(2)
        mse = (mse * self.lat_weights).mean()
        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = (prediction_bool == target_bool).float()
        accuracy = (accuracy * self.lat_weights).mean()
        return {"mse": mse, "accuracy": accuracy}
