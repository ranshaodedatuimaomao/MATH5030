"""
Heston stochastic volatility — primary SV benchmark vs PyFENG / FFT baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfft_bsde.models.base import ModelSpec


@dataclass
class HestonParams:
    """Placeholder Heston parameters."""

    kappa: float = 0.0
    theta: float = 0.0
    xi: float = 0.0
    rho: float = 0.0
    v0: float = 0.0


class Heston(ModelSpec):
    def __init__(self, params: HestonParams | None = None) -> None:
        self.params = params or HestonParams()

    def label(self) -> str:
        return "heston"

    def driver(self, *args, **kwargs):
        raise NotImplementedError("Heston BSDE driver")
