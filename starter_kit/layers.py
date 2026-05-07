#!/bin/env python
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging

# External modules
import torch

# Internal modules


main_logger = logging.getLogger(__name__)


class InputNormalization(torch.nn.Module):
    r"""
    Normalizes input tensors by a predefined mean and standard
    deviation, stored as non-trainable buffers.

    Mean and standard deviation are registered as buffers so they
    are included in ``state_dict()``, transferred with ``.to()``,
    and serialized with the network checkpoint — but never updated
    by the optimizer.

    Parameters
    ----------
    mean : torch.Tensor
        Per-channel mean. Must be broadcastable with the input.
    std : torch.Tensor
        Per-channel standard deviation. Must be broadcastable
        with the input.
    eps : float, optional, default = 1e-6
        Small constant added to ``std`` to avoid division by zero.

    Attributes
    ----------
    mean : torch.Tensor
        Registered buffer holding the normalization mean.
    std : torch.Tensor
        Registered buffer holding the normalization std.
    eps : float
        Numerical stability constant.

    Examples
    --------
    >>> mean = torch.zeros(3, 1, 1)
    >>> std = torch.ones(3, 1, 1)
    >>> norm = InputNormalization(mean, std)
    >>> x = torch.randn(8, 3, 64, 64)
    >>> norm(x).shape
    torch.Size([8, 3, 64, 64])
    """

    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Normalize ``x`` to zero mean and unit variance.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, broadcastable with ``self.mean`` and
            ``self.std``.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as ``x``.
        """
        return (x - self.mean) / (self.std + self.eps)
