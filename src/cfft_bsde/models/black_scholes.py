"""
Black–Scholes — controlled 1D benchmark (literature reproduction).

Use for validating the scheme against published CFFT-BSDE figures.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfft_bsde.models.base import ModelSpec


@dataclass
class BlackScholesParams:
    """Placeholder parameters (spot, vol, rate, dividend, ...)."""

    sigma: float = 0.0
    r: float = 0.0
    q: float = 0.0


class BlackScholes(ModelSpec):
    def __init__(self, params: BlackScholesParams | None = None) -> None:
        self.params = params or BlackScholesParams()

    def label(self) -> str:
        return "black_scholes"

    def driver(self, *args, **kwargs):
        raise NotImplementedError("BS BSDE driver")
