"""
Hyndman & Oyono Ngou (2017) — CFFT-BSDE scheme (placeholder).

Implement convolution-in-frequency discretization and backward recursion here.
"""

from __future__ import annotations

from typing import Any

from cfft_bsde.types import BSDESolution, CFFTSolverConfig


def solve(
    *,
    driver: Any,
    terminal_condition: Any,
    config: CFFTSolverConfig | None = None,
) -> BSDESolution:
    """Run the CFFT-BSDE solver (not implemented)."""
    raise NotImplementedError("HO2017 CFFT-BSDE solve")
