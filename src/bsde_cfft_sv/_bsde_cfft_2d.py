"""
2D local-kernel CFFT API (structure matches upstream; not implemented in this repo).

Reference upstream: ``BSDE-CFFT-Method-For-Stochastic-Volatility-Models-main`` / ``bsde_cfft_sv._bsde_cfft_2d``.
"""

from __future__ import annotations

import math


class TwoDimDensity:
    """Short-time 2D Gaussian transition density helper (see upstream for math)."""

    def __init__(self, mu_x, mu_v, Sigma, dt):
        self.mu_x = mu_x
        self.mu_v = mu_v
        self.Sigma = Sigma
        self.dt = dt

    def char_func_2d(self, xi_x, xi_v):
        """2D characteristic function of the short-time increment."""
        raise NotImplementedError("Stub: implement TwoDimDensity.char_func_2d in _bsde_cfft_2d.py.")


class BSDECFFT2D:
    """Generic 2D local-kernel CFFT base (Heston / GARCH inherit)."""

    def __init__(
        self,
        Nx,
        Nv,
        Lx,
        Lv,
        x0,
        v0,
        n_steps,
        T,
        alpha_x=-3.0,
        alpha_v=0.0,
        v_boundary="neumann",
        fft_workers=None,
    ):
        self.Nx = Nx
        self.Nv = Nv
        self.Lx = Lx
        self.Lv = Lv
        self.x0 = x0
        self.v0 = v0
        self.n_steps = n_steps
        self.T = T
        self.alpha_x = alpha_x
        self.alpha_v = alpha_v
        self.v_boundary = v_boundary
        self.fft_workers = fft_workers
        self.dt = T / n_steps if n_steps else 0.0

    def price_delta_z_at(self, S0, V0):
        """Return (price, delta, z_x, z_v) at the given state."""
        raise NotImplementedError("Stub: implement BSDECFFT2D.price_delta_z_at in _bsde_cfft_2d.py.")


class HestonBSDECFFT(BSDECFFT2D):
    """Heston model — 2D CFFT prototype (see upstream for SDEs)."""

    def __init__(
        self,
        r,
        kappa,
        theta,
        xi,
        rho,
        K,
        T,
        Nx,
        Nv,
        Lx,
        Lv,
        n_steps,
        v_center=None,
        alpha_x=-3.0,
        alpha_v=0.0,
        v_boundary="neumann",
        fft_workers=None,
    ):
        x0 = math.log(K) - Lx / 2
        v0 = 1e-4
        super().__init__(Nx, Nv, Lx, Lv, x0, v0, n_steps, T, alpha_x, alpha_v, v_boundary, fft_workers)
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.K = K
        self.v_bar = theta if v_center is None else v_center

    def price_at(self, S0, V0):
        """Price and delta at given (S0, V0)."""
        price, delta, _, _ = self.price_delta_z_at(S0, V0)
        return price, delta

    def price_delta_z_at(self, S0, V0):
        raise NotImplementedError("Stub: implement HestonBSDECFFT.price_delta_z_at in _bsde_cfft_2d.py.")


class GARCHDiffusionBSDECFFT(BSDECFFT2D):
    """GARCH diffusion model — 2D CFFT prototype (see upstream for SDEs)."""

    def __init__(
        self,
        r,
        mu,
        a,
        b,
        c,
        rho,
        K,
        T,
        Nx,
        Nv,
        Lx,
        Lv,
        n_steps,
        v_center=None,
        alpha_x=-3.0,
        alpha_v=0.0,
        v_boundary="neumann",
        fft_workers=None,
    ):
        x0 = math.log(K) - Lx / 2
        v0 = 1e-5
        super().__init__(Nx, Nv, Lx, Lv, x0, v0, n_steps, T, alpha_x, alpha_v, v_boundary, fft_workers)
        self.r = r
        self.mu = mu
        self.a = a
        self.b = b
        self.c = c
        self.rho = rho
        self.K = K
        self.v_bar = b if v_center is None else v_center

    def price_at(self, S0, V0):
        """Price and delta at given (S0, V0=σ₀²)."""
        price, delta, _, _ = self.price_delta_z_at(S0, V0)
        return price, delta

    def price_delta_z_at(self, S0, V0):
        raise NotImplementedError("Stub: implement GARCHDiffusionBSDECFFT.price_delta_z_at in _bsde_cfft_2d.py.")
