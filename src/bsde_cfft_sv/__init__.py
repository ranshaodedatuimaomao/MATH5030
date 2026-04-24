"""Public API for the BSDE-CFFT stochastic-volatility package."""

from __future__ import annotations

from ._bsde_cfft_1d import BSDECFFT1D, bs_call_delta, bs_call_price
from ._bsde_cfft_2d import GARCHDiffusionBSDECFFT, HestonBSDECFFT
from ._core import OneDResult, TwoDResult, grid_damping_sensitivity, price_black_scholes_1d, price_garch_2d, price_heston_2d
from ._experiments import run_experiments

__all__ = [
    "__version__",
    "BSDECFFT1D",
    "HestonBSDECFFT",
    "GARCHDiffusionBSDECFFT",
    "OneDResult",
    "TwoDResult",
    "bs_call_price",
    "bs_call_delta",
    "price_black_scholes_1d",
    "price_heston_2d",
    "price_garch_2d",
    "grid_damping_sensitivity",
    "run_experiments",
]

__version__ = "0.1.0"
