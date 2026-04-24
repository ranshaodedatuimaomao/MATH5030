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


def heston_vderiv_fd(S0, K, r, kappa, theta, xi, rho, V0, T,
                     rel_bump=1e-4):
    """Central finite-difference derivative with respect to initial variance."""
    h = max(abs(V0) * rel_bump, 1e-6)
    h = min(h, 0.5 * V0)
    price_up = heston_call_price(S0, K, r, kappa, theta, xi, rho, V0 + h, T)
    price_dn = heston_call_price(S0, K, r, kappa, theta, xi, rho, V0 - h, T)
    return (price_up - price_dn) / (2 * h)


def heston_z_fd(S0, K, r, kappa, theta, xi, rho, V0, T):
    """
    Brownian BSDE controls inferred from finite-difference Heston Greeks.

    Independent Brownian representation:
        dX = ... + sqrt(V) dW1
        dV = ... + xi sqrt(V) (rho dW1 + sqrt(1-rho^2) dW2)
    """
    delta = heston_delta_fd(S0, K, r, kappa, theta, xi, rho, V0, T)
    u_x = S0 * delta
    u_v = heston_vderiv_fd(S0, K, r, kappa, theta, xi, rho, V0, T)
    sqrt_v = np.sqrt(V0)
    rho_bar = np.sqrt(max(1 - rho**2, 0.0))
    z_x = sqrt_v * u_x + rho * xi * sqrt_v * u_v
    z_v = rho_bar * xi * sqrt_v * u_v
    return z_x, z_v


# ─────────────────────────────────────────────────────────────────────
#  2. GARCH Diffusion mean-variance approximation and PyFeng references
# ─────────────────────────────────────────────────────────────────────

def garch_diffusion_approx_price(S0, K, r, mu, a, b, c, V0, T):
    """
    Simple mean-variance Black-Scholes approximation for rho=0.

    This is a lightweight diagnostic, not the Barone-Adesi et al. order-2
    formula. The Barone-Adesi benchmark is provided by PyFeng in
    garch_diffusion_pyfeng_price().

    V_t = σ²_t follows: dV = a(b-V)dt + c*V*dW  (GARCH diffusion)

    E[V_T] = b + (V0 - b) e^{-aT}
    E[∫₀ᵀ V_t dt] = b*T + (V0-b)/a * (1 - e^{-aT})

    For the integrated variance, we use the approximate mean:
        σ²_eff = (1/T) E[∫₀ᵀ V_t dt]
    """
    # Mean integrated variance
    if abs(a) < 1e-10:
        mean_int_var = V0 * T
    else:
        mean_int_var = b * T + (V0 - b) / a * (1 - np.exp(-a * T))

    sigma_eff = np.sqrt(mean_int_var / T)

    # Use Black-Scholes with effective volatility (approximate)
    d1 = (np.log(S0/K) + (r + 0.5*sigma_eff**2)*T) / (sigma_eff*np.sqrt(T))
    d2 = d1 - sigma_eff * np.sqrt(T)
    price_approx = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return price_approx, sigma_eff


def garch_diffusion_pyfeng_price(S0, K, r, a, b, c, V0, T):
    """
    PyFeng's Barone-Adesi et al. second-order approximation for rho=0.

    PyFeng parameter mapping:
        sigma=V0, theta=b, mr=a, vov=c
    because its GARCH diffusion class uses the variance process
        dV = mr(theta - V)dt + vov V dW.
    """
    try:
        import pyfeng as pf
    except ImportError:
        return None

    model = pf.GarchUncorrBaroneAdesi2004(V0, vov=c, mr=a, theta=b, intr=r)
    return float(model.price(K, S0, T, cp=1))


def garch_diffusion_pyfeng_delta_fd(S0, K, r, a, b, c, V0, T,
                                    rel_bump=1e-4):
    """Central finite-difference delta from PyFeng's rho=0 BA order-2 price."""
    h = max(abs(S0) * rel_bump, 1e-4)
    price_up = garch_diffusion_pyfeng_price(S0 + h, K, r, a, b, c, V0, T)
    price_dn = garch_diffusion_pyfeng_price(S0 - h, K, r, a, b, c, V0, T)
    if price_up is None or price_dn is None:
        return None
    return (price_up - price_dn) / (2 * h)


def garch_diffusion_pyfeng_vderiv_fd(S0, K, r, a, b, c, V0, T,
                                     rel_bump=1e-4):
    """Finite-difference derivative wrt initial variance in PyFeng rho=0 BA model."""
    h = max(abs(V0) * rel_bump, 1e-6)
    h = min(h, 0.5 * V0)
    price_up = garch_diffusion_pyfeng_price(S0, K, r, a, b, c, V0 + h, T)
    price_dn = garch_diffusion_pyfeng_price(S0, K, r, a, b, c, V0 - h, T)
    if price_up is None or price_dn is None:
        return None
    return (price_up - price_dn) / (2 * h)


def garch_diffusion_pyfeng_z_fd(S0, K, r, a, b, c, rho, V0, T):
    """
    Brownian controls inferred from PyFeng rho=0 BA price sensitivities.

    The price reference is only analytic/BA for rho=0, but the mapping formula
    is written for the independent Brownian representation.
    """
    delta = garch_diffusion_pyfeng_delta_fd(S0, K, r, a, b, c, V0, T)
    u_v = garch_diffusion_pyfeng_vderiv_fd(S0, K, r, a, b, c, V0, T)
    if delta is None or u_v is None:
        return None
    u_x = S0 * delta
    sqrt_v = np.sqrt(V0)
    rho_bar = np.sqrt(max(1 - rho**2, 0.0))
    z_x = sqrt_v * u_x + rho * c * V0 * u_v
    z_v = rho_bar * c * V0 * u_v
    return z_x, z_v


def garch_diffusion_pyfeng_mc_price(S0, K, r, a, b, c, rho, V0, T,
                                    n_paths=100000, n_steps=500, seed=42):
    """PyFeng conditional Monte Carlo with Milstein variance scheme."""
    try:
        import pyfeng as pf
    except ImportError:
        return None

    model = pf.GarchMcTimeDisc(V0, vov=c, rho=rho, mr=a, theta=b, intr=r)
    model.set_num_params(
        n_path=n_paths, dt=T / n_steps, rn_seed=seed, scheme=1
    )
    return float(model.price(K, S0, T, cp=1))


# ─────────────────────────────────────────────────────────────────────
#  3. Monte Carlo (Milstein scheme)
# ─────────────────────────────────────────────────────────────────────

def heston_mc_milstein(S0, K, r, kappa, theta, xi, rho, V0, T,
                       n_paths=50000, n_steps=252, seed=42):
    """
    Monte Carlo price for Heston model using Milstein scheme
    with full truncation for the variance process.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    S = np.full(n_paths, float(S0))
    V = np.full(n_paths, float(V0))

    rho_bar = np.sqrt(1 - rho**2)

    for _ in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rho * Z1 + rho_bar * rng.standard_normal(n_paths)

        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)

        # Milstein correction for variance
        dV = (kappa * (theta - V_pos) * dt
              + xi * sqrt_V * np.sqrt(dt) * Z2
              + 0.25 * xi**2 * dt * (Z2**2 - 1))
        V_new = V + dV
        V_new = np.maximum(V_new, 0.0)   # full truncation

        # Log-price update (Euler for log S)
        dX = (r - 0.5 * V_pos) * dt + sqrt_V * np.sqrt(dt) * Z1
        S = S * np.exp(dX)
        V = V_new

    payoff = np.maximum(S - K, 0.0)
    price = np.exp(-r * T) * np.mean(payoff)
    se = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_paths)
    return price, se


def garch_diffusion_mc_milstein(S0, K, r, mu, a, b, c, rho, V0, T,
                                 n_paths=50000, n_steps=252, seed=42):
    """
    Monte Carlo price for GARCH diffusion model using Milstein scheme.

    dX = (mu - 0.5 V) dt + √V dW1
    dV = a(b-V) dt + c V dW2,   Cov(dW1, dW2) = ρ dt
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    rho_bar = np.sqrt(max(1 - rho**2, 0.0))

    X = np.full(n_paths, np.log(float(S0)))
    V = np.full(n_paths, float(V0))

    for _ in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rho * Z1 + rho_bar * rng.standard_normal(n_paths)

        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)

        # Milstein for GARCH diffusion: σ(V) = c*V, σ'(V) = c
        # Milstein correction: + 0.5 * c * V * c * dt * (Z2^2 - 1) = 0.5 c^2 V dt (Z2^2-1)
        dV = (a * (b - V_pos) * dt
              + c * V_pos * np.sqrt(dt) * Z2
              + 0.5 * c**2 * V_pos * dt * (Z2**2 - 1))
        V_new = np.maximum(V + dV, 0.0)

        # Euler for log S (Milstein same as Euler for log-normal dX)
        dX = (r - 0.5 * V_pos) * dt + sqrt_V * np.sqrt(dt) * Z1
        X += dX
        V = V_new

    S_T = np.exp(X)
    payoff = np.maximum(S_T - K, 0.0)
    price = np.exp(-r * T) * np.mean(payoff)
    se = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_paths)
    return price, se
