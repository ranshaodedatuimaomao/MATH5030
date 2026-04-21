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

Method variants exposed by CLI:

- `boundary_control` (default): fixed damping + time-dependent exponential shift.
- `old_2017`: legacy-style path with adaptive damping heuristic + linear shift.

Benchmark comparison mode:

- Runs both methods on the same parameter grid and writes a CSV with
  Black-Scholes price/delta error metrics.
- Example:
  - `python -m cfft_bsde.cli --benchmark-compare --benchmark-n-values "1000,2000" --benchmark-l-values "10,12,14" --benchmark-grid-values "1024,2048" --benchmark-output "results/benchmark.csv"`

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
```

### macOS

```bash
python3 run_standalone.py
```

The script runs `pip install -e .`, then imports and executes `cfft_bsde.cli.main()`.

### Notes

- `requirements.txt` currently has no external packages, so installation is quick.
- If `pip` is outdated, you can update it with:
  - Windows: `python -m pip install --upgrade pip`
  - macOS: `python3 -m pip install --upgrade pip`
