# MATH5030
MATH5030 @ columbia university MAFN 2026 spring

## Core Algorithm (CFFT-BSDE)

This project currently focuses on a core convolution-FFT backward solver for BSDEs with boundary control.

Brief implementation flow:

1. Define solver inputs: model drift/diffusion, BSDE driver, terminal condition, and numerical config.
2. Build uniform time, spatial, and frequency grids on a truncated log-price domain.
3. Apply centered discrete Fourier transforms using phase shifts for FFT-ready convolution.
4. Construct short-time characteristic-function multipliers for Y and Z updates.
5. Compute damping-shift recovery terms and solve per-step exponential shift parameters `(A, B)`.
6. Run backward time recursion: damp/shift -> Fourier-domain convolution -> inverse transform -> undamp/unshift -> driver update.
7. Return full `Y(t, x)` and `Z(t, x)` surfaces through `solve_core(...)`.

Main implementation: `src/cfft_bsde/cfft/core_algorithm.py`  
Console entrypoint: `src/cfft_bsde/cli.py`

Method variants exposed by CLI (CSV `method` column uses the same strings):

- **`new_boundary_control`** (default): boundary-error control scheme from the course paper—fixed damping plus a time-varying exponential shift `h(x)=A*exp(x)+B` with recovery terms.
- **`legacy_hyndman_2017`**: legacy convolution–FFT path after Hyndman and Oyono Ngou (2017)—adaptive damping heuristic and a **linear** shift `h(x)=a*x+b`.

(Older identifiers were `boundary_control` and `old_2017`; they are removed in favor of the names above.)

Benchmark comparison mode:

- Runs selected methods on the same parameter grid and writes a CSV with
  benchmark price/delta error metrics.
- Benchmark model is switchable via `--benchmark-model`:
  - `black_scholes_call` (default)
  - `intrinsic_call`
- Example:
  - `python -m cfft_bsde.cli --benchmark-compare --benchmark-methods "new_boundary_control,legacy_hyndman_2017" --benchmark-model "black_scholes_call" --benchmark-n-values "1000,2000" --benchmark-l-values "10,12,14" --benchmark-grid-values "1024,2048" --benchmark-output "results/benchmark.csv"`
- Optional full-surface mode (keeps default one-point mode unless enabled):
  - `python -m cfft_bsde.cli --benchmark-compare --benchmark-full-surface --surface-spot-min 60 --surface-spot-max 140 --surface-spot-points 81 --benchmark-output "results/benchmark_surface.csv"`

## Numerical benchmark outputs (CSV)

When you run `--benchmark-compare`, results are written to the path you pass via `--benchmark-output` (relative paths land under the repo root).

This repo currently includes small benchmark exports committed under `results/`:

- `results/numerical_results_quick.csv`: one-point (`spot_eval = 100`) sweep over a coarse grid intended for quick smoke testing.
- `results/numerical_results_surface_quick.csv`: the same benchmark metrics evaluated on a spot grid from 60 to 140 (41 points), intended to sanity-check boundary behavior quickly.
- `results/benchmark_smoke.csv`: minimal two-row smoke file (one solve per method) using the same `method` labels as the CLI.

PNG plots derived from these runs (paper-style layout) live in the same folder; see **Paper-style replication figures (PNG)** below.

Each row includes (among others) Black–Scholes reference values and errors:

- `price_benchmark`, `price_cfft`, `abs_price_err`
- `delta_benchmark`, `delta_z`, `abs_delta_z_err`, `rel_delta_z_err`
- `delta_fd`, `abs_delta_fd_err`, `rel_delta_fd_err`

### Snapshot summary (the two quick CSVs above)

These settings are **not** the fine grids used in `paper.pdf`; treat the numbers below as **smoke-test scale**, useful mainly to confirm the pipeline runs and to compare methods on identical inputs.

| CSV | `new_boundary_control` | `legacy_hyndman_2017` |
| --- | --- | --- |
| `results/numerical_results_quick.csv` (8 rows each) | mean `abs_price_err` ≈ 9.13 (max ≈ 9.41); mean `abs_delta_z_err` ≈ 0.557 (max ≈ 0.559) | mean `abs_price_err` ≈ 2475.88 (max ≈ 5824.38); mean `abs_delta_z_err` ≈ 7.28 (max ≈ 25.60) |
| `results/numerical_results_surface_quick.csv` (41 rows each) | mean `abs_price_err` ≈ 13.79 (max ≈ 42.39); mean `abs_delta_z_err` ≈ 0.511 (max ≈ 0.957) | mean `abs_price_err` ≈ 11.56 (max ≈ 38.43); mean `abs_delta_z_err` ≈ 2.55 (max ≈ 2.74) |

To move toward `paper.pdf`-scale replication, increase `--benchmark-n-values`, `--benchmark-l-values`, and `--benchmark-grid-values` (expect substantially longer runtimes).

## Paper-style replication figures (PNG)

Section 4 of `paper.pdf` shows call **price** and **delta** errors as a function of spot (Figures 1–2) and a **delta surface** over time and spot (Figure 3). This repo includes a small plotting helper that mirrors that layout using:

- **Figures 1–2**: curves read from a full-spot benchmark CSV (by default `results/numerical_results_surface_quick.csv`).
- **Figure 3**: a fresh `solve_core` run for `new_boundary_control` (same spirit as the paper’s surface plot; grid is controlled by CLI flags below).

Install Matplotlib (not required for the core solver), then run:

```powershell
python -m pip install matplotlib
python -m cfft_bsde.plot_paper_figures --surface-csv results/numerical_results_surface_quick.csv --output-dir results
```

Optional editable install with plotting extras:

```powershell
python -m pip install -e ".[plot]"
python -m cfft_bsde.plot_paper_figures
```

Generated files (also committed under `results/` when present):

- `results/figure_01_legacy_price_delta_errors.png`
- `results/figure_02_new_boundary_price_delta_errors.png`
- `results/figure_03_new_boundary_delta_surface.png`

## Simple Python Console App

### Windows

1) Install Python 3:
- Download from [python.org for Windows](https://www.python.org/downloads/windows/)
- During installation, check **Add Python to PATH**

2) Open PowerShell in the project folder and run:

```powershell
python --version
python -m pip install -r requirements.txt
python app.py
```

### macOS

1) Install Python 3:
- Option A (recommended): install Homebrew, then `brew install python`
- Option B: download from [python.org for macOS](https://www.python.org/downloads/macos/)

2) Open Terminal in the project folder and run:

```bash
python3 --version
python3 -m pip install -r requirements.txt
python3 app.py
```

## Install as a package (pip)

This repo is now a Python package (`cfft-bsde`) with an import name `cfft_bsde` and a console command `cfft-bsde`.

### Windows

```powershell
python -m pip install -U pip
python -m pip install -e .
cfft-bsde
```

### macOS

```bash
python3 -m pip install -U pip
python3 -m pip install -e .
cfft-bsde
```

## Standalone launcher script

Use `run_standalone.py` if you want one command that installs the local package and then starts the console app.

### Windows

```powershell
python run_standalone.py
# benchmark one-point comparison
python run_standalone.py --mode benchmark-point
# benchmark full-surface export
python run_standalone.py --mode benchmark-surface
# full replication bundle (CSVs + paper-style PNGs; needs matplotlib for figures)
python run_standalone.py --mode replication
# same CSV steps only (skip matplotlib)
python run_standalone.py --mode replication --skip-figures
```

### macOS

```bash
python3 run_standalone.py
# benchmark one-point comparison
python3 run_standalone.py --mode benchmark-point
# benchmark full-surface export
python3 run_standalone.py --mode benchmark-surface
# full replication bundle (CSVs + paper-style PNGs; needs matplotlib for figures)
python3 run_standalone.py --mode replication
python3 run_standalone.py --mode replication --skip-figures
```

The script runs `pip install -e .`, then imports and executes `cfft_bsde.cli.main()` (or, for `--mode replication`, runs the point benchmark, surface benchmark, then `cfft_bsde.plot_paper_figures` when Matplotlib is available).

`benchmark-point` / `benchmark-surface` write to `results/numerical_results_quick.csv` and `results/numerical_results_surface_quick.csv` respectively (same grids as the committed replication smoke run).

Use `--extra-cli-args` to append any additional CLI flags (each benchmark step in `replication` receives the same extras).

### Notes

- `requirements.txt` currently has no external packages, so installation is quick.
- If `pip` is outdated, you can update it with:
  - Windows: `python -m pip install --upgrade pip`
  - macOS: `python3 -m pip install --upgrade pip`
