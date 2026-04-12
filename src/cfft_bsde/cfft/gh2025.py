"""
Gao & Hyndman (2025) — boundary error control via damping / shifting (placeholder).

Wrap or extend the HO2017 stepping with GH2025 modifications.
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
    """Run CFFT-BSDE with GH2025 boundary controls (not implemented)."""
    raise NotImplementedError("GH2025 CFFT-BSDE solve")
