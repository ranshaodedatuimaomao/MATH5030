"""
GARCH diffusion — exploratory SV application (instructor-suggested second model).

Compare against: Barone-Adesi et al. (2005) uncorrelated approximation and
Monte Carlo (e.g. Milstein) baselines when full fast pricing is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfft_bsde.models.base import ModelSpec


@dataclass
class GarchDiffusionParams:
    """Placeholder GARCH-diffusion parameters (match course notation when implementing)."""

    # Intentionally minimal; fill from SDE specification.
    omega: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0


class GarchDiffusion(ModelSpec):
    def __init__(self, params: GarchDiffusionParams | None = None) -> None:
        self.params = params or GarchDiffusionParams()

    def label(self) -> str:
        return "garch_diffusion"

    def driver(self, *args, **kwargs):
        raise NotImplementedError("GARCH diffusion BSDE driver")
