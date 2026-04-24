"""High-level API wrappers for the BSDE-CFFT stochastic-volatility package."""

from __future__ import annotations

from dataclasses import dataclass

from ._benchmarks import (
    garch_diffusion_pyfeng_price,
    heston_call_price,
)
from ._bsde_cfft_1d import BSDECFFT1D, bs_call_delta, bs_call_price
from ._bsde_cfft_2d import GARCHDiffusionBSDECFFT, HestonBSDECFFT


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
    solver = BSDECFFT1D(
        r=r,
        mu=r,
        sigma=sigma,
        K=K,
        T=T,
        L=L,
        N=N,
        n_steps=n_steps,
        alpha=alpha,
    )
    price, delta = solver.price_at(S0)
    analytic_price = bs_call_price(S0, K, r, sigma, T)
    analytic_delta = bs_call_delta(S0, K, r, sigma, T)
    return OneDResult(
        price=price,
        delta=delta,
        analytic_price=analytic_price,
        analytic_delta=analytic_delta,
        abs_price_error=abs(price - analytic_price),
        abs_delta_error=abs(delta - analytic_delta),
    )


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
    solver = HestonBSDECFFT(
        r=r,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        K=K,
        T=T,
        Nx=Nx,
        Nv=Nv,
        Lx=Lx,
        Lv=Lv,
        n_steps=n_steps,
        v_center=V0,
        alpha_x=alpha_x,
        alpha_v=alpha_v,
        v_boundary=v_boundary,
        fft_workers=fft_workers,
    )
    price, delta, z_x, z_v = solver.price_delta_z_at(S0, V0)
    reference_price = heston_call_price(S0, K, r, kappa, theta, xi, rho, V0, T)
    return TwoDResult(
        price=price,
        delta=delta,
        z_x=z_x,
        z_v=z_v,
        reference_price=reference_price,
        abs_price_error=abs(price - reference_price),
    )


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
    solver = GARCHDiffusionBSDECFFT(
        r=r,
        mu=mu,
        a=a,
        b=b,
        c=c,
        rho=rho,
        K=K,
        T=T,
        Nx=Nx,
        Nv=Nv,
        Lx=Lx,
        Lv=Lv,
        n_steps=n_steps,
        v_center=V0,
        alpha_x=alpha_x,
        alpha_v=alpha_v,
        v_boundary=v_boundary,
        fft_workers=fft_workers,
    )
    price, delta, z_x, z_v = solver.price_delta_z_at(S0, V0)
    reference_price = garch_diffusion_pyfeng_price(S0, K, r, a, b, c, V0, T)
    abs_price_error = None if reference_price is None else abs(price - reference_price)
    return TwoDResult(
        price=price,
        delta=delta,
        z_x=z_x,
        z_v=z_v,
        reference_price=reference_price,
        abs_price_error=abs_price_error,
    )


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
    heston_rows: list[dict[str, float | str | None]] = []
    for Lv in heston_Lv_values:
        result = price_heston_2d(Lv=Lv, Nx=Nx, Nv=Nv, Lx=Lx, V0=V0, n_steps=heston_n_steps)
        heston_rows.append(
            {
                "Lv": Lv,
                "price": result.price,
                "reference_price": result.reference_price,
                "abs_price_error": result.abs_price_error,
            }
        )

    garch_rows: list[dict[str, float | str | None]] = []
    for alpha_x in garch_alpha_values:
        result = price_garch_2d(alpha_x=alpha_x, Nx=Nx, Nv=Nv, Lx=Lx, V0=V0, n_steps=garch_n_steps)
        garch_rows.append(
            {
                "alpha_x": alpha_x,
                "price": result.price,
                "reference_price": result.reference_price,
                "abs_price_error": result.abs_price_error,
            }
        )

    return {
        "heston_lv_sensitivity": heston_rows,
        "garch_alpha_sensitivity": garch_rows,
    }
