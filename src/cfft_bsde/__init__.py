"""
cfft-bsde: convolution–FFT methods for BSDEs (course project scaffold).

Sections: ``cfft`` (HO2017 / GH2025 / core algorithm), ``models``, ``validation``, ``benchmarks``,
``robustness``, ``adaptive``, ``baselines``.
"""

from __future__ import annotations

from . import adaptive, baselines, benchmarks, cfft, models, robustness, types, validation

__all__ = [
    "__version__",
    "cfft",
    "models",
    "validation",
    "benchmarks",
    "robustness",
    "adaptive",
    "baselines",
    "types",
]

__version__ = "0.0.0"
