# MATH5030 — `bsde-cfft-sv` (API skeleton)

**Source of truth for structure and behavior:** the reference tree **BSDE-CFFT-Method-For-Stochastic-Volatility-Models-main** (same import **`bsde_cfft_sv`**, distribution **`bsde-cfft-sv`** on PyPI in that project).

This **MATH5030** repo copies the **module and class layout** (under `src/bsde_cfft_sv/`) but **does not** ship:

- Numerical implementations (public callables raise `NotImplementedError` or are empty shells you can fill),
- **Tests** (`tests/` removed),
- **`notebooks/demo.ipynb`** (removed),
- A **console app** (no `app.py`, no `run_standalone.py`, no `cfft_bsde` package, no `[project.scripts]` — the upstream repo installs `bsde-cfft-sv` → `bsde_cfft_sv.cli:main`).

## Layout (mirrors upstream)

| Path | Role |
| --- | --- |
| `src/bsde_cfft_sv/__init__.py` | Public exports |
| `src/bsde_cfft_sv/_core.py` | `OneDResult`, `TwoDResult`, `price_*`, `grid_damping_sensitivity` |
| `src/bsde_cfft_sv/_bsde_cfft_1d.py` | `BSDECFFT1D`, `bs_call_*`, `run_bs_convergence_test` |
| `src/bsde_cfft_sv/_bsde_cfft_2d.py` | `TwoDimDensity`, `BSDECFFT2D`, `HestonBSDECFFT`, `GARCHDiffusionBSDECFFT` |
| `src/bsde_cfft_sv/_benchmarks.py` | Reference/benchmark symbol names |
| `src/bsde_cfft_sv/_experiments.py` | `run_experiments`, `PART_CHOICES` |
| `src/bsde_cfft_sv/__main__.py` | Stub message only (not the upstream CLI) |

## Install & import

```bash
python -m pip install -e .
python -c "import bsde_cfft_sv; print(bsde_cfft_sv.__version__)"
```

## PyPI

```bash
python -m pip install build
python -m build
```

Runtime dependencies are empty until you port code from the reference repo (upstream uses `numpy`, `scipy`, `pyfeng`).
