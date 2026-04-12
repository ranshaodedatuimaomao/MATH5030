"""
Shared types and configuration placeholders for models and solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class GridSpec:
    """Spatial / frequency grid parameters (placeholder)."""

    n_points: int = 0
    domain_truncation: float = 0.0


@dataclass
class TimeStepSpec:
    """Time discretization (placeholder)."""

    n_steps: int = 0
    maturity: float = 0.0


@dataclass
class CFFTSolverConfig:
    """Hyndman–Oyono Ngou (2017) + optional Gao–Hyndman (2025) controls."""

    grid: GridSpec | None = None
    time: TimeStepSpec | None = None
    # GH2025-style knobs (placeholders)
    damping: float | None = None
    time_dependent_shift: bool = False
    extra: Mapping[str, Any] | None = None


@dataclass
class BSDESolution:
    """Container for Y/Z paths or terminal values (placeholder)."""

    values: Any = None
    metadata: Mapping[str, Any] | None = None
