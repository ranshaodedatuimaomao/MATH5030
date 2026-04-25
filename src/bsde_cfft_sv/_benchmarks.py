"""
Reference pricers for model validation:

1. Heston semi-analytic formula (characteristic function inversion via scipy)
2. GARCH Diffusion mean-variance Black-Scholes approximation
3. PyFeng references for Heston FFT and GARCH Barone-Adesi order-2
4. Monte Carlo (Milstein scheme) for both models
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────
#  1. Heston semi-analytic pricer
# ─────────────────────────────────────────────────────────────────────

def heston_char_func(phi, S0, V0, r, kappa, theta, xi, rho, T):
    """
    Heston characteristic function of log(S_T) using the 'little trap' formulation
    (Albrecher et al. 2007) to avoid discontinuities.
    """
    x = np.log(S0)
    a = kappa * theta

    # Little Trap formulation
    u = -0.5
    b = kappa
    d = np.sqrt((rho * xi * phi * 1j - b)**2 - xi**2 * (2*u*phi*1j - phi**2))
    g = (b - rho * xi * phi * 1j - d) / (b - rho * xi * phi * 1j + d)

    C = r * phi * 1j * T + (a / xi**2) * (
        (b - rho * xi * phi * 1j - d) * T
        - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    D = ((b - rho * xi * phi * 1j - d) / xi**2) * (
        (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    )
    return np.exp(C + D * V0 + 1j * phi * x)


def heston_call_price(S0, K, r, kappa, theta, xi, rho, V0, T):
    """
    Heston European call price via Gil-Pelaez inversion.
    """
    def integrand_re(phi):
        cf = heston_char_func(phi - 1j, S0, V0, r, kappa, theta, xi, rho, T)
        cf0 = heston_char_func(-1j, S0, V0, r, kappa, theta, xi, rho, T)
        return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi * cf0))

    def integrand_im(phi):
        cf = heston_char_func(phi, S0, V0, r, kappa, theta, xi, rho, T)
        return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))

    # P1 and P2 (risk-neutral probabilities)
    I1, _ = quad(integrand_re, 0, 200, limit=200, epsabs=1e-10)
    P1 = 0.5 + I1 / np.pi

    I2, _ = quad(integrand_im, 0, 200, limit=200, epsabs=1e-10)
    P2 = 0.5 + I2 / np.pi

    return S0 * P1 - K * np.exp(-r * T) * P2


def heston_pyfeng_price(S0, K, r, kappa, theta, xi, rho, V0, T):
    """PyFeng Heston FFT benchmark."""
    try:
        import pyfeng as pf
    except ImportError:
        return None

    model = pf.HestonFft(V0, vov=xi, rho=rho, mr=kappa, theta=theta, intr=r)
    return float(model.price(K, S0, T, cp=1))


def heston_delta_fd(S0, K, r, kappa, theta, xi, rho, V0, T,
                    rel_bump=1e-4):
    """Central finite-difference Heston delta from the semi-analytic price."""
    h = max(abs(S0) * rel_bump, 1e-4)
    price_up = heston_call_price(S0 + h, K, r, kappa, theta, xi, rho, V0, T)
    price_dn = heston_call_price(S0 - h, K, r, kappa, theta, xi, rho, V0, T)
    return (price_up - price_dn) / (2 * h)



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
