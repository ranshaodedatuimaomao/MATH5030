"""cfft-bsde: core convolution-FFT algorithm for BSDEs."""

from __future__ import annotations

from . import cfft
from .cfft import CoreConfig, CoreInputs, CoreSolution, solve_core

__all__ = [
    "__version__",
    "cfft",
    "CoreConfig",
    "CoreInputs",
    "CoreSolution",
    "solve_core",
]

__version__ = "0.0.0"
