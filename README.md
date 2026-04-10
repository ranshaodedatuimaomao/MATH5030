# MATH5030
MATH5030 @ columbia university MAFN 2026 spring

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
