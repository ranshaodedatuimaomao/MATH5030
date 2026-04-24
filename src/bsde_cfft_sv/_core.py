"""High-level API (signatures match BSDE-CFFT-Method-For-Stochastic-Volatility-Models-main; no implementation)."""

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


def price_black_scholes_1d(
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.01,
    sigma: float = 0.2,
    T: float = 1.0,
    *,
    L: float = 10.0,
    N: int = 4096,
    n_steps: int = 1000,
    alpha: float = -3.0,
) -> OneDResult:
    """Run the 1D BSDE-CFFT Black-Scholes validation problem."""
    raise NotImplementedError("Stub: implement price_black_scholes_1d in _core.py.")


def price_heston_2d(
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    V0: float = 0.04,
    T: float = 1.0,
    *,
    Nx: int = 96,
    Nv: int = 48,
    Lx: float = 10.0,
    Lv: float = 0.25,
    n_steps: int = 700,
    alpha_x: float = -3.0,
    alpha_v: float = 0.0,
    v_boundary: str = "neumann",
    fft_workers: int | None = None,
) -> TwoDResult:
    """Run the 2D local-kernel CFFT Heston solver at a single initial state."""
    raise NotImplementedError("Stub: implement price_heston_2d in _core.py.")


def price_garch_2d(
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.05,
    mu: float = 0.05,
    a: float = 2.0,
    b: float = 0.04,
    c: float = 0.4,
    rho: float = 0.0,
    V0: float = 0.04,
    T: float = 1.0,
    *,
    Nx: int = 96,
    Nv: int = 48,
    Lx: float = 10.0,
    Lv: float = 0.32,
    n_steps: int = 700,
    alpha_x: float = -3.0,
    alpha_v: float = 0.0,
    v_boundary: str = "neumann",
    fft_workers: int | None = None,
) -> TwoDResult:
    """Run the 2D local-kernel CFFT GARCH-diffusion solver at a single initial state."""
    raise NotImplementedError("Stub: implement price_garch_2d in _core.py.")


def grid_damping_sensitivity(
    *,
    Nx: int = 96,
    Nv: int = 48,
    heston_n_steps: int = 700,
    garch_n_steps: int = 700,
    Lx: float = 10.0,
    V0: float = 0.04,
    heston_Lv_values: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.40),
    garch_alpha_values: tuple[float, ...] = (-2.5, -3.0, -3.5),
) -> dict[str, list[dict[str, float | str | None]]]:
    """Return lightweight sensitivity tables for Heston and GARCH diagnostics."""
    raise NotImplementedError("Stub: implement grid_damping_sensitivity in _core.py.")
