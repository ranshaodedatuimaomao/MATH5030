"""
Run all numerical experiments and produce summary tables.

Experiments:
  Part I  – 1D Black-Scholes: validate the 1D BSDE-CFFT implementation.
  Part II – Heston benchmark: compare 2D local-kernel CFFT prices vs references.
  Part III – GARCH diffusion: compare 2D local-kernel CFFT prices vs references.
"""

import numpy as np
import time
from ._bsde_cfft_1d import BSDECFFT1D, bs_call_price, bs_call_delta
from ._bsde_cfft_2d import HestonBSDECFFT, GARCHDiffusionBSDECFFT
from ._benchmarks import (
    heston_call_price,
    heston_delta_fd,
    heston_pyfeng_price,
    heston_z_fd,
    garch_diffusion_approx_price,
    garch_diffusion_pyfeng_delta_fd,
    garch_diffusion_pyfeng_price,
    garch_diffusion_pyfeng_mc_price,
    garch_diffusion_pyfeng_z_fd,
    heston_mc_milstein,
    garch_diffusion_mc_milstein,
)


PART_CHOICES = ("all", "black-scholes", "heston", "garch", "sensitivity")


def sep(title=""):
    print()
    print("=" * 95)
    if title:
        print(f"  {title}")
        print("=" * 95)


# ─────────────────────────────────────────────────────────────────────
#  PART I — 1D Black-Scholes Validation
# ─────────────────────────────────────────────────────────────────────

def part1_black_scholes():
    sep("PART I: 1D BSDE-CFFT — Black-Scholes Validation")

    S0, K, r, sigma, T = 100.0, 100.0, 0.01, 0.2, 1.0
    L, alpha, n_steps = 10.0, -3.0, 1000

    price_bs   = bs_call_price(S0, K, r, sigma, T)
    delta_bs   = bs_call_delta(S0, K, r, sigma, T)

    print(f"\n  Black-Scholes analytic:  price={price_bs:.6f}, delta={delta_bs:.6f}")
    print(f"\n  {'N':>6}  {'Price':>12}  {'|Err price|':>14}  {'Delta':>12}  {'|Err delta|':>14}  {'Time(s)':>8}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}  {'-'*12}  {'-'*14}  {'-'*8}")

    for N_exp in [10, 11, 12, 13]:
        N = 2**N_exp
        t0 = time.time()
        solver = BSDECFFT1D(r=r, mu=r, sigma=sigma, K=K, T=T,
                            L=L, N=N, n_steps=n_steps, alpha=alpha)
        price_num, delta_num = solver.price_at(S0)
        elapsed = time.time() - t0
        print(f"  {N:>6}  {price_num:>12.6f}  {abs(price_num-price_bs):>14.2e}"
              f"  {delta_num:>12.6f}  {abs(delta_num-delta_bs):>14.2e}  {elapsed:>8.3f}")


# ─────────────────────────────────────────────────────────────────────
#  PART II — Heston Benchmark
# ─────────────────────────────────────────────────────────────────────

def part2_heston():
    sep("PART II: 2D Local-Kernel CFFT — Heston Benchmark")

    # Model parameters
    S0, K  = 100.0, 100.0
    r      = 0.05
    kappa  = 2.0      # mean reversion speed
    theta  = 0.04     # long-run variance (σ_LR = 20%)
    xi     = 0.3      # vol-of-vol
    rho    = -0.7     # correlation
    V0     = 0.04     # initial variance
    T      = 1.0

    # Reference prices
    print("\n  Computing reference prices...")
    t0 = time.time()
    price_analytic = heston_call_price(S0, K, r, kappa, theta, xi, rho, V0, T)
    delta_ref = heston_delta_fd(S0, K, r, kappa, theta, xi, rho, V0, T)
    z_x_ref, z_v_ref = heston_z_fd(S0, K, r, kappa, theta, xi, rho, V0, T)
    t_analytic = time.time() - t0

    t0 = time.time()
    price_pyfeng = heston_pyfeng_price(S0, K, r, kappa, theta, xi, rho, V0, T)
    t_pyfeng = time.time() - t0

    t0 = time.time()
    price_mc, se_mc = heston_mc_milstein(S0, K, r, kappa, theta, xi, rho, V0, T,
                                          n_paths=100000, n_steps=500)
    t_mc = time.time() - t0

    print(f"  Heston analytic:      price={price_analytic:.5f}, delta≈{delta_ref:.5f}  ({t_analytic:.3f}s)")
    print(f"  Heston FD controls:   Zx≈{z_x_ref:.5f}, Zv≈{z_v_ref:.5f}")
    if price_pyfeng is not None:
        print(f"  PyFeng Heston FFT:    price={price_pyfeng:.5f}  ({t_pyfeng:.3f}s)")
    print(f"  Monte Carlo Milstein: price={price_mc:.5f} ± {2*se_mc:.5f}  ({t_mc:.2f}s)")

    print("  Note: Zx/Zv are Brownian BSDE controls from grad(price)*diffusion.")
    print("        Deltas and Z are validated here against finite-difference references.")

    print(f"\n  {'Nx':>4} {'Nv':>4}  {'Lv':>5}  {'n_steps':>7}  {'Price':>10}  "
          f"{'|Err|':>10}  {'Delta':>9}  {'Zx':>9}  {'|eZx|':>9}  "
          f"{'Zv':>9}  {'|eZv|':>9}  {'Time(s)':>8}")
    print(f"  {'-'*4} {'-'*4}  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*9}  "
          f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}")

    configs = [
        (32, 16, 0.25, 200),
        (64, 32, 0.25, 500),
        (96, 48, 0.25, 700),
        (128, 64, 0.25, 900),
        (160, 80, 0.25, 1100),
    ]

    for Nx, Nv, Lv, n_steps in configs:
        t0 = time.time()
        solver = HestonBSDECFFT(
            r=r, kappa=kappa, theta=theta, xi=xi, rho=rho,
            K=K, T=T, Nx=Nx, Nv=Nv, Lx=10.0, Lv=Lv,
            n_steps=n_steps, v_center=V0, alpha_x=-3.0
        )
        price_num, delta_num, z_x_num, z_v_num = solver.price_delta_z_at(S0, V0)
        elapsed = time.time() - t0
        err_analytic = abs(price_num - price_analytic)
        print(f"  {Nx:>4} {Nv:>4}  {Lv:>5.2f}  {n_steps:>7}  {price_num:>10.5f}  "
              f"{err_analytic:>10.2e}  {delta_num:>9.5f}  {z_x_num:>9.5f}  "
              f"{abs(z_x_num-z_x_ref):>9.2e}  {z_v_num:>9.5f}  "
              f"{abs(z_v_num-z_v_ref):>9.2e}  {elapsed:>8.2f}")


# ─────────────────────────────────────────────────────────────────────
#  PART III — GARCH Diffusion (main application)
# ─────────────────────────────────────────────────────────────────────

def part3_garch_diffusion():
    sep("PART III: 2D Local-Kernel CFFT — GARCH Diffusion Model")

    # Model parameters
    S0, K = 100.0, 100.0
    r, mu  = 0.05, 0.05
    a, b   = 2.0, 0.04       # mean-reversion speed and long-run variance
    c      = 0.4             # vol-of-vol (GARCH: σ(V)=c*V, larger than Heston)
    V0     = 0.04            # initial variance (σ₀ = 20%)
    T      = 1.0

    # ─── Case A: ρ=0 (compare to PyFeng Barone-Adesi order-2) ─────────
    print("\n  Case A: ρ=0 (uncorrelated — compare to PyFeng Barone-Adesi)")
    rho = 0.0

    price_approx, sig_eff = garch_diffusion_approx_price(S0, K, r, mu, a, b, c, V0, T)
    price_pyfeng = garch_diffusion_pyfeng_price(S0, K, r, a, b, c, V0, T)
    delta_pyfeng = garch_diffusion_pyfeng_delta_fd(S0, K, r, a, b, c, V0, T)
    z_ref = garch_diffusion_pyfeng_z_fd(S0, K, r, a, b, c, rho, V0, T)
    print(f"  Mean-var BS approx:    price={price_approx:.5f}  (σ_eff={sig_eff:.4f})")
    if price_pyfeng is not None:
        delta_txt = f"{delta_pyfeng:.5f}" if delta_pyfeng is not None else "n/a"
        print(f"  PyFeng BA order 2:     price={price_pyfeng:.5f}, delta≈{delta_txt}")
    if z_ref is not None:
        print(f"  PyFeng FD controls:    Zx≈{z_ref[0]:.5f}, Zv≈{z_ref[1]:.5f}")

    t0 = time.time()
    price_mc, se_mc = garch_diffusion_mc_milstein(S0, K, r, mu, a, b, c, rho, V0, T,
                                                    n_paths=100000, n_steps=500)
    t_mc = time.time() - t0
    print(f"  Monte Carlo Milstein:  price={price_mc:.5f} ± {2*se_mc:.5f}  ({t_mc:.2f}s)")

    t0 = time.time()
    price_pf_mc = garch_diffusion_pyfeng_mc_price(
        S0, K, r, a, b, c, rho, V0, T, n_paths=100000, n_steps=500
    )
    t_pf_mc = time.time() - t0
    if price_pf_mc is not None:
        print(f"  PyFeng MC Milstein:    price={price_pf_mc:.5f}  ({t_pf_mc:.2f}s)")

    print("  Note: Zx/Zv are computed as Brownian controls; nonlinear Z-drivers still need a separate test.")

    print(f"\n  {'Nx':>4} {'Nv':>4}  {'Lv':>5}  {'n_steps':>7}  {'Price':>10}  "
          f"{'vs PyFeng':>10}  {'Delta':>9}  {'Zx':>9}  {'|eZx|':>9}  "
          f"{'Zv':>9}  {'|eZv|':>9}  {'Time(s)':>8}")
    print(f"  {'-'*4} {'-'*4}  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*9}  "
          f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}")

    configs = [
        (32, 16, 0.32, 200),
        (64, 32, 0.32, 500),
        (96, 48, 0.32, 700),
        (128, 64, 0.32, 900),
        (160, 80, 0.32, 1100),
    ]
    ref_pyfeng = price_pyfeng if price_pyfeng is not None else price_approx

    for Nx, Nv, Lv, n_steps in configs:
        t0 = time.time()
        solver = GARCHDiffusionBSDECFFT(
            r=r, mu=mu, a=a, b=b, c=c, rho=rho,
            K=K, T=T, Nx=Nx, Nv=Nv, Lx=10.0, Lv=Lv,
            n_steps=n_steps, v_center=V0, alpha_x=-3.0
        )
        price_num, delta_num, z_x_num, z_v_num = solver.price_delta_z_at(S0, V0)
        elapsed = time.time() - t0
        err_z_x = abs(z_x_num - z_ref[0]) if z_ref is not None else np.nan
        err_z_v = abs(z_v_num - z_ref[1]) if z_ref is not None else np.nan
        print(f"  {Nx:>4} {Nv:>4}  {Lv:>5.2f}  {n_steps:>7}  {price_num:>10.5f}  "
              f"{abs(price_num-ref_pyfeng):>10.2e}  {delta_num:>9.5f}  {z_x_num:>9.5f}  "
              f"{err_z_x:>9.2e}  {z_v_num:>9.5f}  {err_z_v:>9.2e}  {elapsed:>8.2f}")

    # ─── Case B: ρ=-0.5 (correlated, no analytic approx) ────────────
    print("\n  Case B: ρ=-0.5 (correlated — only Monte Carlo reference)")
    rho = -0.5

    t0 = time.time()
    price_mc, se_mc = garch_diffusion_mc_milstein(S0, K, r, mu, a, b, c, rho, V0, T,
                                                    n_paths=100000, n_steps=500)
    t_mc = time.time() - t0
    print(f"  Monte Carlo Milstein:  price={price_mc:.5f} ± {2*se_mc:.5f}  ({t_mc:.2f}s)")

    t0 = time.time()
    price_pf_mc = garch_diffusion_pyfeng_mc_price(
        S0, K, r, a, b, c, rho, V0, T, n_paths=100000, n_steps=500
    )
    t_pf_mc = time.time() - t0
    if price_pf_mc is not None:
        print(f"  PyFeng MC Milstein:    price={price_pf_mc:.5f}  ({t_pf_mc:.2f}s)")

    print(f"\n  {'Nx':>4} {'Nv':>4}  {'Lv':>5}  {'n_steps':>7}  {'Price':>10}  "
          f"{'vs MC':>10}  {'Zx':>9}  {'Zv':>9}  {'Time(s)':>8}")
    print(f"  {'-'*4} {'-'*4}  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*8}")

    for Nx, Nv, Lv, n_steps in configs:
        t0 = time.time()
        solver = GARCHDiffusionBSDECFFT(
            r=r, mu=mu, a=a, b=b, c=c, rho=rho,
            K=K, T=T, Nx=Nx, Nv=Nv, Lx=10.0, Lv=Lv,
            n_steps=n_steps, v_center=V0, alpha_x=-3.0
        )
        price_num, delta_num, z_x_num, z_v_num = solver.price_delta_z_at(S0, V0)
        elapsed = time.time() - t0
        print(f"  {Nx:>4} {Nv:>4}  {Lv:>5.2f}  {n_steps:>7}  {price_num:>10.5f}  "
              f"{abs(price_num-price_mc):>10.2e}  {z_x_num:>9.5f}  {z_v_num:>9.5f}  {elapsed:>8.2f}")


# ─────────────────────────────────────────────────────────────────────
#  PART IV — Sensitivity Diagnostics
# ─────────────────────────────────────────────────────────────────────

def part4_sensitivity_diagnostics():
    sep("PART IV: Grid/Damping Sensitivity Diagnostics")

    S0, K, r, V0, T = 100.0, 100.0, 0.05, 0.04, 1.0

    print("\n  Heston sensitivity to Lv (Nx=96, Nv=48, n_steps=700, alpha_x=-3, v BC=Neumann)")
    kappa, theta, xi, rho = 2.0, 0.04, 0.3, -0.7
    ref_heston = heston_call_price(S0, K, r, kappa, theta, xi, rho, V0, T)
    print(f"  {'Lv':>5}  {'Price':>10}  {'|Err|':>10}  {'Time(s)':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}")
    for Lv in [0.15, 0.20, 0.25, 0.30, 0.40]:
        t0 = time.time()
        solver = HestonBSDECFFT(
            r=r, kappa=kappa, theta=theta, xi=xi, rho=rho,
            K=K, T=T, Nx=96, Nv=48, Lx=10.0, Lv=Lv,
            n_steps=700, v_center=V0, alpha_x=-3.0
        )
        price, _ = solver.price_at(S0, V0)
        elapsed = time.time() - t0
        print(f"  {Lv:>5.2f}  {price:>10.5f}  {abs(price-ref_heston):>10.2e}  {elapsed:>8.2f}")

    print("\n  Heston variance-boundary control effect (Nx=96, Nv=48, Lv=0.20)")
    print(f"  {'v_boundary':>10}  {'Price':>10}  {'|Err|':>10}  {'Time(s)':>8}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for v_boundary in ["none", "neumann"]:
        t0 = time.time()
        solver = HestonBSDECFFT(
            r=r, kappa=kappa, theta=theta, xi=xi, rho=rho,
            K=K, T=T, Nx=96, Nv=48, Lx=10.0, Lv=0.20,
            n_steps=700, v_center=V0, alpha_x=-3.0,
            v_boundary=v_boundary
        )
        price, _ = solver.price_at(S0, V0)
        elapsed = time.time() - t0
        print(f"  {v_boundary:>10}  {price:>10.5f}  {abs(price-ref_heston):>10.2e}  {elapsed:>8.2f}")

    print("\n  GARCH rho=0 sensitivity to alpha_x (Nx=96, Nv=48, Lv=0.32, n_steps=700)")
    mu, a, b, c, rho = 0.05, 2.0, 0.04, 0.4, 0.0
    ref_garch = garch_diffusion_pyfeng_price(S0, K, r, a, b, c, V0, T)
    print(f"  {'alpha_x':>7}  {'Price':>10}  {'vs PyFeng':>10}  {'Time(s)':>8}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*8}")
    for alpha_x in [-2.5, -3.0, -3.5]:
        t0 = time.time()
        solver = GARCHDiffusionBSDECFFT(
            r=r, mu=mu, a=a, b=b, c=c, rho=rho,
            K=K, T=T, Nx=96, Nv=48, Lx=10.0, Lv=0.32,
            n_steps=700, v_center=V0, alpha_x=alpha_x
        )
        price, _ = solver.price_at(S0, V0)
        elapsed = time.time() - t0
        err = abs(price - ref_garch) if ref_garch is not None else np.nan
        print(f"  {alpha_x:>7.2f}  {price:>10.5f}  {err:>10.2e}  {elapsed:>8.2f}")


def run_experiments(part="all"):
    """Run one section or the full experiment suite."""
    if part not in PART_CHOICES:
        raise ValueError(f"Unsupported part: {part!r}")

    if part == "all":
        print("\n" + "=" * 95)
        print("  Local-kernel CFFT Y/Z extension for stochastic volatility")
        print("  Inspired by Gao & Hyndman (2025) boundary-controlled BSDE-CFFT")
        print("=" * 95)

        part1_black_scholes()
        part2_heston()
        part3_garch_diffusion()
        part4_sensitivity_diagnostics()

        print()
        sep("ALL EXPERIMENTS COMPLETE")
        return

    if part == "black-scholes":
        part1_black_scholes()
    elif part == "heston":
        part2_heston()
    elif part == "garch":
        part3_garch_diffusion()
    elif part == "sensitivity":
        part4_sensitivity_diagnostics()


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_experiments("all")