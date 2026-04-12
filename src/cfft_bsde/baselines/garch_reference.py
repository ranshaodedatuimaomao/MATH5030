"""
GARCH diffusion references:

- Barone-Adesi–Rasmussen–Ravanelli (2005) analytic approximation (uncorrelated case)
- Monte Carlo (Milstein) as in instructor PyFENG notebook workflow

Placeholders only; implement against PyFENG / paper formulas when integrating.
"""

from __future__ import annotations

from typing import Any, Mapping


def price_barone_adesi_uncorr(*, config: Mapping[str, Any] | None = None) -> float:
    """Placeholder: BARR (2005) approximation for rho = 0."""
    raise NotImplementedError("GARCH analytic approximation baseline")


def price_monte_carlo(*, config: Mapping[str, Any] | None = None) -> float:
    """Placeholder: delegate to MC path (see baselines.monte_carlo)."""
    raise NotImplementedError("GARCH MC baseline")
