"""
Reference / benchmark pricers (names match upstream; not implemented in this repo).

Reference upstream: ``BSDE-CFFT-Method-For-Stochastic-Volatility-Models-main`` / ``bsde_cfft_sv._benchmarks``.
"""

from __future__ import annotations


def heston_char_func(phi, S0, V0, r, kappa, theta, xi, rho, T):
    raise NotImplementedError("Stub: implement heston_char_func in _benchmarks.py.")


def heston_call_price(S0, K, r, kappa, theta, xi, rho, V0, T):
    raise NotImplementedError("Stub: implement heston_call_price in _benchmarks.py.")


def heston_pyfeng_price(S0, K, r, kappa, theta, xi, rho, V0, T):
    raise NotImplementedError("Stub: implement heston_pyfeng_price in _benchmarks.py.")


def heston_delta_fd(S0, K, r, kappa, theta, xi, rho, V0, T, rel_bump=1e-4):
    raise NotImplementedError("Stub: implement heston_delta_fd in _benchmarks.py.")


def heston_vderiv_fd(S0, K, r, kappa, theta, xi, rho, V0, T, rel_bump=1e-4):
    raise NotImplementedError("Stub: implement heston_vderiv_fd in _benchmarks.py.")


def heston_z_fd(S0, K, r, kappa, theta, xi, rho, V0, T):
    raise NotImplementedError("Stub: implement heston_z_fd in _benchmarks.py.")


def garch_diffusion_approx_price(S0, K, r, mu, a, b, c, V0, T):
    raise NotImplementedError("Stub: implement garch_diffusion_approx_price in _benchmarks.py.")


def garch_diffusion_pyfeng_price(S0, K, r, a, b, c, V0, T):
    raise NotImplementedError("Stub: implement garch_diffusion_pyfeng_price in _benchmarks.py.")


def garch_diffusion_pyfeng_delta_fd(S0, K, r, a, b, c, V0, T, rel_bump=1e-4):
    raise NotImplementedError("Stub: implement garch_diffusion_pyfeng_delta_fd in _benchmarks.py.")


def garch_diffusion_pyfeng_vderiv_fd(S0, K, r, a, b, c, V0, T, rel_bump=1e-4):
    raise NotImplementedError("Stub: implement garch_diffusion_pyfeng_vderiv_fd in _benchmarks.py.")


def garch_diffusion_pyfeng_z_fd(S0, K, r, a, b, c, rho, V0, T):
    raise NotImplementedError("Stub: implement garch_diffusion_pyfeng_z_fd in _benchmarks.py.")


def garch_diffusion_pyfeng_mc_price(
    S0, K, r, a, b, c, rho, V0, T, n_paths=100000, n_steps=500, seed=42
):
    raise NotImplementedError("Stub: implement garch_diffusion_pyfeng_mc_price in _benchmarks.py.")


def heston_mc_milstein(
    S0, K, r, kappa, theta, xi, rho, V0, T, n_paths=50000, n_steps=252, seed=42
):
    raise NotImplementedError("Stub: implement heston_mc_milstein in _benchmarks.py.")


def garch_diffusion_mc_milstein(
    S0, K, r, mu, a, b, c, rho, V0, T, n_paths=50000, n_steps=252, seed=42
):
    raise NotImplementedError("Stub: implement garch_diffusion_mc_milstein in _benchmarks.py.")
