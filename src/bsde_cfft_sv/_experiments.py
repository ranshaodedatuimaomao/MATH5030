"""
Experiment driver (``run_experiments``) — signature matches upstream; not implemented here.

Reference upstream: ``BSDE-CFFT-Method-For-Stochastic-Volatility-Models-main`` / ``bsde_cfft_sv._experiments``.
"""

from __future__ import annotations

PART_CHOICES = ("all", "black-scholes", "heston", "garch", "sensitivity")


def run_experiments(part: str = "all") -> None:
    """Run one section or the full experiment suite (upstream prints tables)."""
    if part not in PART_CHOICES:
        raise ValueError(f"Unsupported part: {part!r}")
    raise NotImplementedError("Stub: implement run_experiments in _experiments.py.")
