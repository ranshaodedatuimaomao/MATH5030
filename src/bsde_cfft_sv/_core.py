"""Stub high-level pricing API (matches upstream ``_core`` exports; not wired to full solvers here)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OneDResult:
    price: float
    delta: float
    analytic_price: float
    analytic_delta: float
    abs_price_error: float
    abs_delta_error: float


@dataclass(frozen=True)
class TwoDResult:
    price: float
    delta: float
    z_x: float
    z_v: float
    reference_price: float | None = None
    abs_price_error: float | None = None


def price_black_scholes_1d(*_args, **_kwargs) -> OneDResult:
    raise NotImplementedError(
        "Stub: for the working 1D Black–Scholes-style CFFT core, use "
        "``bsde_cfft_sv.implementation_version_0.cfft.solve_core``."
    )


def price_heston_2d(*_args, **_kwargs) -> TwoDResult:
    raise NotImplementedError("Stub: full Heston 2D solver not bundled in this course layout.")


def price_garch_2d(*_args, **_kwargs) -> TwoDResult:
    raise NotImplementedError("Stub: full GARCH 2D solver not bundled in this course layout.")


def grid_damping_sensitivity(*_args, **_kwargs) -> dict[str, list[dict[str, float | str | None]]]:
    raise NotImplementedError("Stub: sensitivity helpers require full 2D solvers.")
