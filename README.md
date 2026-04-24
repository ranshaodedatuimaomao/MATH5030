# MATH5030
MATH5030 @ columbia university MAFN 2026 spring

## Install with pip (library `bsde_cfft_sv`)

This repository is a normal Python project: **install it from the repo root** so you get the importable packages under `src/`.

| What you type | What you get |
| --- | --- |
| `python -m pip install .` | Non-editable install: builds a wheel and installs the **`bsde-cfft-sv`** distribution (PyPI-style name) into the active environment. |
| `python -m pip install -e .` | Editable install: same import paths while you edit the source. |
| `python -m pip install -e ".[plot,test]"` | Adds optional **matplotlib** (figures) and **pytest** (tests). |

After installation, you can use:

- `import bsde_cfft_sv` — upstream-style API (`BSDECFFT1D`, `run_experiments` via `python -m bsde_cfft_sv`, etc.)
- `import implementation_version_0` — v0 CFFT–BSDE `solve_core` core
- `import bsde_cfft_app` — MATH5030 course console modules
- The **`bsde-cfft-sv`** command — course menu / replication CLI (points at `bsde_cfft_app`, not the upstream `bsde_cfft_sv.cli` experiment driver)

**Names:** the **pip / PyPI name** is `bsde-cfft-sv` (hyphen). The **Python import name** for the main library is `bsde_cfft_sv` (underscore). `pip show bsde-cfft-sv` lists the distribution after a successful install.

**From a git URL (replace with your remote):**  
`python -m pip install "git+https://github.com/USER/REPO.git#egg=bsde-cfft-sv"`

**Smoke check:** `python -c "import bsde_cfft_sv; print(bsde_cfft_sv.__version__)"`

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

Main CFFT–BSDE core (v0): `src/implementation_version_0/cfft/core_algorithm.py`  
MATH5030 course console (menu, replication, benchmarks): `src/bsde_cfft_app/cli.py` (script: `bsde-cfft-sv`)  
Upstream-style library package: `src/bsde_cfft_sv/` (flat module layout; `python -m bsde_cfft_sv` runs the reference experiment CLI)

### Interactive menu (terminal)

From an **interactive terminal**, run **`bsde-cfft-sv`**, **`python -m bsde_cfft_app`**, or **`python app.py`** with **no extra arguments** to get: **(1)** open the v0 “first implementation” HTML report, **(2)** run the bundled CFFT–BSDE core in `implementation_version_0` (the `bsde_cfft_sv` package follows the upstream layout; the v0 solver is in the sibling `implementation_version_0` package), or **(3)** launch **`notebooks/demo.ipynb`**. BSM context lines print when a bundled quick CSV is present. The same menu appears with **`--menu`**. Full CSV replication is still available via **`--run-replication`**, not from this menu.

**`run_standalone.py`** uses the **same menu** by default: **`python run_standalone.py`** (or explicitly **`--mode auto`**) runs `pip install -e .` first, then shows the menu in a terminal. Use **`--mode menu`** to force the menu even when stdin is not a TTY. With **`--mode auto`** and a **non-interactive** stdin (CI, pipes), it falls back to a **core** preset solve instead of blocking on `input()`.

If **stdin is not a terminal** (pipelines, CI), an empty **`bsde-cfft-sv`** / **`python -m bsde_cfft_app`** command still runs a **single core solve** so automation keeps working.

Method variants exposed by CLI (CSV `method` column uses the same strings):

- **`new_boundary_control`** (default): boundary-error control scheme from the course paper—fixed damping plus a time-varying exponential shift `h(x)=A*exp(x)+B` with recovery terms.
- **`legacy_hyndman_2017`**: legacy convolution–FFT path after Hyndman and Oyono Ngou (2017)—adaptive damping heuristic and a **linear** shift `h(x)=a*x+b`.

(Older identifiers were `boundary_control` and `old_2017`; they are removed in favor of the names above.)

### Bundled replication (two CLI options)

After `pip install -e .`, use **exactly one** of the following primary actions (they are mutually exclusive with each other and with `--benchmark-compare`):

| Goal | Command |
| --- | --- |
| **1. Rerun** the full bundled replication (quick CSV + surface CSV + paper-style PNGs) | `python -m bsde_cfft_app --run-replication` |
| Same, but **skip figure generation** (CSVs only) | `python -m bsde_cfft_app --run-replication --skip-figures` |
| **2. View** the existing HTML report in your browser (no computation) | `python -m bsde_cfft_app --open-replication-report` |
| Optional explicit HTML path | `python -m bsde_cfft_app --open-replication-report --replication-report path/to/replication_report.html` |

Equivalent standalone launcher (also runs `pip install -e .` first): default **`python run_standalone.py`** opens the same interactive menu in a terminal; **`--mode replication`** / **`--mode open-report`** match **`--run-replication`** / **`--open-replication-report`**. See **Standalone launcher script** below.

### Custom benchmark comparison (`--benchmark-compare`)

- Runs selected methods on the same parameter grid and writes a CSV with
  benchmark price/delta error metrics.
- Benchmark model is switchable via `--benchmark-model`:
  - `black_scholes_call` (default)
  - `intrinsic_call`
- Example:
  - `python -m bsde_cfft_app --benchmark-compare --benchmark-methods "new_boundary_control,legacy_hyndman_2017" --benchmark-model "black_scholes_call" --benchmark-n-values "1000,2000" --benchmark-l-values "10,12,14" --benchmark-grid-values "1024,2048" --benchmark-output "src/implementation_version_0/results/benchmark.csv"`
- Optional full-surface mode (keeps default one-point mode unless enabled):
  - `python -m bsde_cfft_app --benchmark-compare --benchmark-full-surface --surface-spot-min 60 --surface-spot-max 140 --surface-spot-points 81 --benchmark-output "src/implementation_version_0/results/benchmark_surface.csv"`

## Numerical benchmark outputs (CSV)

When you run `--benchmark-compare`, results go to the path you pass via `--benchmark-output` (default: `implementation_version_0/results/benchmark_comparison.csv` under `src/`, independent of the current working directory).

This repo includes small benchmark exports committed under `src/implementation_version_0/results/`:

- `numerical_results_quick.csv`: one-point (`spot_eval = 100`) sweep over a coarse grid intended for quick smoke testing.
- `numerical_results_surface_quick.csv`: the same benchmark metrics evaluated on a spot grid from 60 to 140 (41 points), intended to sanity-check boundary behavior quickly.
- `benchmark_smoke.csv`: minimal two-row smoke file (one solve per method) using the same `method` labels as the CLI.
- `replication_report.html`: short HTML summary of the replication bundle (CSVs + figures). Regenerate inputs with `bsde-cfft-sv --run-replication` (or `run_standalone.py --mode replication`); open from the interactive menu (`bsde-cfft-sv` / `run_standalone.py` with no args in a terminal) or with `bsde-cfft-sv --open-replication-report` / `run_standalone.py --mode open-report`.

PNG plots derived from these runs (paper-style layout) live in the same folder; see **Paper-style replication figures (PNG)** below.

Each row includes (among others) Black–Scholes reference values and errors:

- `price_benchmark`, `price_cfft`, `abs_price_err`
- `delta_benchmark`, `delta_z`, `abs_delta_z_err`, `rel_delta_z_err`
- `delta_fd`, `abs_delta_fd_err`, `rel_delta_fd_err`

### Snapshot summary (the two quick CSVs above)

These settings are **not** the fine grids used in `paper.pdf`; treat the numbers below as **smoke-test scale**, useful mainly to confirm the pipeline runs and to compare methods on identical inputs.

| CSV | `new_boundary_control` | `legacy_hyndman_2017` |
| --- | --- | --- |
| `numerical_results_quick.csv` (8 rows each) | mean `abs_price_err` ≈ 9.13 (max ≈ 9.41); mean `abs_delta_z_err` ≈ 0.557 (max ≈ 0.559) | mean `abs_price_err` ≈ 2475.88 (max ≈ 5824.38); mean `abs_delta_z_err` ≈ 7.28 (max ≈ 25.60) |
| `numerical_results_surface_quick.csv` (41 rows each) | mean `abs_price_err` ≈ 13.79 (max ≈ 42.39); mean `abs_delta_z_err` ≈ 0.511 (max ≈ 0.957) | mean `abs_price_err` ≈ 11.56 (max ≈ 38.43); mean `abs_delta_z_err` ≈ 2.55 (max ≈ 2.74) |

To move toward `paper.pdf`-scale replication, increase `--benchmark-n-values`, `--benchmark-l-values`, and `--benchmark-grid-values` (expect substantially longer runtimes).

## Paper-style replication figures (PNG)

Section 4 of `paper.pdf` shows call **price** and **delta** errors as a function of spot (Figures 1–2) and a **delta surface** over time and spot (Figure 3). This repo includes a small plotting helper that mirrors that layout using:

- **Figures 1–2**: curves read from a full-spot benchmark CSV (by default `src/implementation_version_0/results/numerical_results_surface_quick.csv` when you run the module with no args after replication).
- **Figure 3**: a fresh `solve_core` run for `new_boundary_control` (same spirit as the paper’s surface plot; grid is controlled by CLI flags below).

The fastest way to regenerate figures together with the CSVs is `bsde-cfft-sv --run-replication` (or `run_standalone.py --mode replication`). Alternatively, after surface CSV exists, install Matplotlib and run:

```powershell
python -m pip install matplotlib
python -m bsde_cfft_app.plot_paper_figures
```

The figure helper defaults to reading and writing under `implementation_version_0/results/`. To override, pass e.g. `--surface-csv PATH --output-dir DIR`.

Optional editable install with plotting extras:

```powershell
python -m pip install -e ".[plot]"
python -m bsde_cfft_app.plot_paper_figures
```

Generated files (also committed under `src/implementation_version_0/results/` when present):

- `figure_01_legacy_price_delta_errors.png`
- `figure_02_new_boundary_price_delta_errors.png`
- `figure_03_new_boundary_delta_surface.png`

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

In a normal terminal window, `python app.py` with no arguments opens the **interactive menu** (see **Interactive menu (terminal)** above).

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

In a normal terminal, `python3 app.py` with no arguments opens the **interactive menu**.

## Install as a package (quick start)

`pip` installs the **`bsde-cfft-sv`** package (name on PyPI) which provides the **`bsde_cfft_sv`**, **`implementation_version_0`**, and **`bsde_cfft_app`** import packages. The console script **`bsde-cfft-sv`** is the MATH5030 course CLI (see the table at **Install with pip** above for `python -m bsde_cfft_sv` = upstream experiment runner).

### Windows

```powershell
python -m pip install -U pip
python -m pip install -e .
bsde-cfft-sv
# bundled replication: (1) rerun CSVs+figures  —or—  (2) open existing HTML report
bsde-cfft-sv --run-replication
bsde-cfft-sv --open-replication-report
```

### macOS

```bash
python3 -m pip install -U pip
python3 -m pip install -e .
bsde-cfft-sv
bsde-cfft-sv --run-replication
bsde-cfft-sv --open-replication-report
```

## Standalone launcher script

Use `run_standalone.py` for **one command** that runs `pip install -e .` and then either the **same interactive menu** as `bsde-cfft-sv` (default) or an explicit preset.

| Goal | Command |
| --- | --- |
| **Interactive menu** (open HTML / run replication / core demo) — default | `python run_standalone.py` or `python run_standalone.py --mode auto` |
| **Force** that menu (even if stdin might not look like a TTY) | `python run_standalone.py --mode menu` |
| **1. Rerun** full replication (same as `bsde-cfft-sv --run-replication`) | `python run_standalone.py --mode replication` |
| CSVs only (skip matplotlib figures) | `python run_standalone.py --mode replication --skip-figures` |
| **2. Open** existing `implementation_version_0/results/replication_report.html` (in the package tree) | `python run_standalone.py --mode open-report` |
| Core or benchmark **presets** only (no menu) | `python run_standalone.py --mode core` (or `benchmark-point` / `benchmark-surface`) |

With **`--mode auto`** or **`menu`**, do not pass **`--extra-cli-args`** (use an explicit `--mode` if you need extra CLI flags).

### Windows

```powershell
python run_standalone.py
python run_standalone.py --mode menu
python run_standalone.py --mode core
python run_standalone.py --mode benchmark-point
python run_standalone.py --mode benchmark-surface
python run_standalone.py --mode replication
python run_standalone.py --mode replication --skip-figures
python run_standalone.py --mode open-report
python run_standalone.py --mode open-report --replication-report path/to/replication_report.html
```

### macOS

```bash
python3 run_standalone.py
python3 run_standalone.py --mode menu
python3 run_standalone.py --mode core
python3 run_standalone.py --mode benchmark-point
python3 run_standalone.py --mode benchmark-surface
python3 run_standalone.py --mode replication
python3 run_standalone.py --mode replication --skip-figures
python3 run_standalone.py --mode open-report
```

The script runs `pip install -e .`, then uses `bsde_cfft_app.interactive_menu` for **`auto`**/**`menu`**, `bsde_cfft_app.replication_pipeline` / `bsde_cfft_app.replication_report` for **`replication`** / **`open-report`**, or **`bsde_cfft_app.cli.main()`** with preset argv for **`core`** and benchmark modes. (The upstream `python -m bsde_cfft_sv` runs the reference experiment CLI inside the `bsde_cfft_sv` package.)

`benchmark-point` / `benchmark-surface` write to `src/implementation_version_0/results/numerical_results_quick.csv` and `.../numerical_results_surface_quick.csv` respectively (same grids as the committed replication smoke run).

Use `--extra-cli-args` to append any additional CLI flags (each benchmark step in `replication` receives the same extras).

### Notes

- `requirements.txt` currently has no external packages, so installation is quick.
- If `pip` is outdated, you can update it with:
  - Windows: `python -m pip install --upgrade pip`
  - macOS: `python3 -m pip install --upgrade pip`
