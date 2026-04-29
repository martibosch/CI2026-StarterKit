#!/bin/env python
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging

# External modules
import torch

# Internal modules
from .utils import approximate_surface_pressure, estimate_relative_humidity

main_logger = logging.getLogger(__name__)


class SundquistNetwork(torch.nn.Module):
    r"""
    Sundqvist cloud cover parametrisation.

    Computes per-level cloud cover as:

    .. math::

        C(p) = 1 - \sqrt{1 - \frac{r(p) - r_0(p)}{r_{sat} - r_0(p)}}

    clamped to [0, 1], then collapses levels with a random-overlap
    assumption.

    Parameters
    ----------
    None

    Attributes
    ----------
    logit_r0 : torch.nn.Parameter
        Per-level logit of the critical relative humidity r_0(p).
        Shape (7, 1, 1). Sigmoid maps to (0, 1).
    logit_r_sat : torch.nn.Parameter
        Scalar logit of the saturation relative humidity r_sat.
        Sigmoid maps to (0, 1); initialised near 1.
    pressure_levels : torch.Tensor
        Fixed pressure levels in Pa, shape (7, 1, 1).
    """

    _EPSILON = 1e-6

    def __init__(self) -> None:
        super().__init__()
        # r_0(p): critical RH per level; sigmoid(0) = 0.5 start
        self.register_parameter("logit_r0", torch.nn.Parameter(torch.zeros(7, 1, 1)))
        # r_sat: saturation RH; sigmoid(4) ≈ 0.98 start
        self.register_parameter(
            "logit_r_sat", torch.nn.Parameter(torch.full((1,), 4.0))
        )
        self.register_buffer(
            "pressure_levels",
            torch.tensor(
                [1000_00, 850_00, 700_00, 500_00, 250_00, 100_00, 50_00],
                dtype=torch.float32,
            ).reshape(-1, 1, 1),
        )

    def forward(
        self,
        input_level: torch.Tensor,
        input_auxiliary: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute total cloud cover from level inputs.

        Parameters
        ----------
        input_level : torch.Tensor
            Pressure-level fields, shape (B, C_level, L, H, W).
            Channel 0 is temperature (K), channel 1 is specific
            humidity (kg/kg).
        input_auxiliary : torch.Tensor
            Static fields, shape (B, C_aux, H, W).
            Channel 1 is surface geopotential (m^2/s^2).

        Returns
        -------
        torch.Tensor
            Total cloud cover in [0, 1], shape (B, 1, H, W).
        """
        r0 = torch.sigmoid(self.logit_r0)  # (7, 1, 1)
        r_sat = torch.sigmoid(self.logit_r_sat)  # (1,)

        level_rh = estimate_relative_humidity(
            temperature=input_level[:, 0:1],
            specific_humidity=input_level[:, 1:2],
            pressure=self.pressure_levels,
        )

        gap = (r_sat - r0).clamp(min=self._EPSILON)
        ratio = ((level_rh - r0) / gap).clamp(0.0, 1.0)
        # Clamp away from 0 before sqrt: gradient is 1/(2*sqrt(x)),
        # which diverges at x=0 and produces NaN in backprop.
        level_cloud_cover = 1 - torch.sqrt((1 - ratio).clamp(min=self._EPSILON))

        surface_pressure = approximate_surface_pressure(input_auxiliary[:, 1:2])
        valid_mask = self.pressure_levels < surface_pressure.unsqueeze(2)

        level_clear_sky = 1 - level_cloud_cover * valid_mask
        total_cloud_cover = 1 - torch.prod(level_clear_sky, dim=2)
        return total_cloud_cover
