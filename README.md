# `bsde-cfft-sv`

BSDE-CFFT option pricing tools for Black-Scholes, Heston, and GARCH-diffusion models.

## What Problem It Solves

This project provides a compact Python API for pricing European call options and computing sensitivities under several diffusion models using BSDE-CFFT methods.

The package is useful when you want to:

- validate a 1D BSDE-CFFT solver against the analytic Black-Scholes formula,
- price options in 2D stochastic-volatility models such as Heston,
- study a GARCH-diffusion volatility model with the same numerical interface,
- compare numerical accuracy against benchmark prices,
- test how grid resolution, damping, and time discretization affect the results.

Implemented methods:

- `price_black_scholes_1d(...)`: 1D BSDE-CFFT validation problem with analytic Black-Scholes comparison,
- `price_heston_2d(...)`: 2D BSDE-CFFT solver for the Heston stochastic-volatility model,
- `price_garch_2d(...)`: 2D BSDE-CFFT solver for a GARCH-diffusion stochastic-volatility model,
- `grid_damping_sensitivity(...)`: helper for sensitivity studies on grid and damping parameters.

Main parameters:

- `S0`: initial asset price,
- `K`: strike price,
- `r`: risk-free interest rate,
- `T`: maturity,
- `sigma`: volatility in the Black-Scholes model,
- `kappa`, `theta`, `xi`, `rho`, `V0`: Heston model parameters,
- `mu`, `a`, `b`, `c`, `rho`, `V0`: GARCH-diffusion model parameters,
- `N`: 1D spatial grid size,
- `Nx`, `Nv`: 2D grid sizes in log-price and variance,
- `L`, `Lx`, `Lv`: truncation widths of the computational domain,
- `n_steps`: number of backward time steps, with `dt = T / n_steps`,
- `alpha`, `alpha_x`, `alpha_v`: damping parameters used in the FFT-based scheme,
- `v_boundary`: boundary handling mode in the variance direction.

In practice:

- larger grids such as bigger `N`, `Nx`, `Nv` usually improve spatial accuracy but cost more,
- larger `n_steps` refines the time discretization but increases runtime,
- damping parameters such as `alpha_x` can materially affect stability and accuracy.

## Installation

Install from PyPI:

```bash
pip install bsde-cfft-sv
```

Install locally from the repository:

```bash
git clone https://github.com/ranshaodedatuimaomao/MATH5030.git
cd MATH5030
python3 -m pip install -e .
```

Requirements:

- Python `>=3.10`
- `numpy`
- `scipy`
- `pandas`
- `pyfeng>=0.3`

## Quick Start

Minimal Black-Scholes example:

```python
from bsde_cfft_sv import price_black_scholes_1d

result = price_black_scholes_1d(
    S0=100.0,
    K=100.0,
    sigma=0.2,
    T=1.0,
    N=4096,
    n_steps=1000,
)

print(result.price)
print(result.delta)
print(result.analytic_price)
print(result.abs_price_error)
```

Minimal Heston example:

```python
from bsde_cfft_sv import price_heston_2d

result = price_heston_2d(
    S0=100.0,
    K=100.0,
    V0=0.04,
    Nx=96,
    Nv=48,
    n_steps=700,
)

print(result.price)
print(result.delta)
print(result.z_x, result.z_v)
print(result.reference_price)
```

## API Reference

### Return Types

`OneDResult`

| Field | Type | Description |
| --- | --- | --- |
| `price` | `float` | Numerical option price |
| `delta` | `float` | Numerical delta |
| `analytic_price` | `float` | Closed-form Black-Scholes price |
| `analytic_delta` | `float` | Closed-form Black-Scholes delta |
| `abs_price_error` | `float` | Absolute error versus analytic price |
| `abs_delta_error` | `float` | Absolute error versus analytic delta |

`TwoDResult`

| Field | Type | Description |
| --- | --- | --- |
| `price` | `float` | Numerical option price |
| `delta` | `float` | Numerical delta |
| `z_x` | `float` | BSDE sensitivity in the log-price direction |
| `z_v` | `float` | BSDE sensitivity in the variance direction |
| `reference_price` | `float \| None` | Benchmark price when available |
| `abs_price_error` | `float \| None` | Absolute error versus the benchmark |

### `price_black_scholes_1d(...) -> OneDResult`

Prices a European call in the Black-Scholes model and compares the numerical result with the analytic formula.

Parameters:

| Parameter | Type | Description |
| --- | --- | --- |
| `S0` | `float` | Initial spot |
| `K` | `float` | Strike |
| `r` | `float` | Risk-free rate |
| `sigma` | `float` | Constant volatility |
| `T` | `float` | Maturity |
| `L` | `float` | Log-price truncation width |
| `N` | `int` | Number of spatial grid points |
| `n_steps` | `int` | Number of backward time steps |
| `alpha` | `float` | Damping parameter |

### `price_heston_2d(...) -> TwoDResult`

Prices a European call in the Heston model using the 2D BSDE-CFFT solver.

Parameters:

| Parameter | Type | Description |
| --- | --- | --- |
| `S0` | `float` | Initial spot |
| `K` | `float` | Strike |
| `r` | `float` | Risk-free rate |
| `kappa` | `float` | Mean-reversion speed of variance |
| `theta` | `float` | Long-run variance |
| `xi` | `float` | Vol-of-vol |
| `rho` | `float` | Correlation |
| `V0` | `float` | Initial variance |
| `T` | `float` | Maturity |
| `Nx`, `Nv` | `int` | Grid sizes in log-price and variance |
| `Lx`, `Lv` | `float` | Truncation widths |
| `n_steps` | `int` | Number of backward time steps |
| `alpha_x`, `alpha_v` | `float` | Damping parameters |
| `v_boundary` | `str` | Variance-boundary treatment |
| `fft_workers` | `int \| None` | Number of FFT workers |

### `price_garch_2d(...) -> TwoDResult`

Prices a European call in the GARCH-diffusion model using the same numerical interface.

Parameters:

| Parameter | Type | Description |
| --- | --- | --- |
| `S0` | `float` | Initial spot |
| `K` | `float` | Strike |
| `r` | `float` | Risk-free rate |
| `mu` | `float` | Drift parameter kept for model compatibility |
| `a` | `float` | Mean-reversion speed |
| `b` | `float` | Long-run variance level |
| `c` | `float` | Vol-of-vol coefficient |
| `rho` | `float` | Correlation |
| `V0` | `float` | Initial variance |
| `T` | `float` | Maturity |
| `Nx`, `Nv` | `int` | Grid sizes in log-price and variance |
| `Lx`, `Lv` | `float` | Truncation widths |
| `n_steps` | `int` | Number of backward time steps |
| `alpha_x`, `alpha_v` | `float` | Damping parameters |
| `v_boundary` | `str` | Variance-boundary treatment |
| `fft_workers` | `int \| None` | Number of FFT workers |

### `grid_damping_sensitivity(...) -> dict[str, list[dict[str, float | str | None]]]`

Runs lightweight sensitivity studies:

- Heston sensitivity with respect to `Lv`,
- GARCH sensitivity with respect to `alpha_x`.

Key parameters:

| Parameter | Type | Description |
| --- | --- | --- |
| `Nx`, `Nv` | `int` | 2D grid sizes |
| `heston_n_steps` | `int` | Time steps for Heston runs |
| `garch_n_steps` | `int` | Time steps for GARCH runs |
| `Lx` | `float` | Log-price domain width |
| `V0` | `float` | Initial variance |
| `heston_Lv_values` | `tuple[float, ...]` | Tested `Lv` values |
| `garch_alpha_values` | `tuple[float, ...]` | Tested damping values |

## License

MIT

## Demo Notebook

Notebook:

- [notebooks/Demo.ipynb](notebooks/Demo.ipynb)

Open in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ranshaodedatuimaomao/MATH5030/blob/main/notebooks/Demo.ipynb)
