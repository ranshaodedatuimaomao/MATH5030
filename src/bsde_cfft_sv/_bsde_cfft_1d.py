"""
1D BSDE-CFFT solver API (structure matches upstream; not implemented in this repo).

Reference upstream: ``BSDE-CFFT-Method-For-Stochastic-Volatility-Models-main`` / ``bsde_cfft_sv._bsde_cfft_1d``.
"""

from __future__ import annotations


def bs_call_price(S0, K, r, sigma, T):
    """Black-Scholes European call price."""
    raise NotImplementedError("Stub: implement bs_call_price in _bsde_cfft_1d.py.")


def bs_call_delta(S0, K, r, sigma, T):
    """Black-Scholes delta."""
    raise NotImplementedError("Stub: implement bs_call_delta in _bsde_cfft_1d.py.")


class BSDECFFT1D:
    """1D BSDE via convolution-FFT (boundary error control) — implement per upstream paper."""

    def __init__(self, r, mu, sigma, K, T, L, N, n_steps, alpha=-3.0):
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.K = K
        self.T = T
        self.L = L
        self.N = N
        self.n_steps = n_steps
        self.alpha = alpha

    def price_at(self, S0):
        """Price and delta at a given spot price S0."""
        raise NotImplementedError("Stub: implement BSDECFFT1D.price_at in _bsde_cfft_1d.py.")


def run_bs_convergence_test():
    """Compare BSDE-CFFT prices vs Black–Scholes analytic formula (upstream helper)."""
    raise NotImplementedError("Stub: implement run_bs_convergence_test in _bsde_cfft_1d.py.")
