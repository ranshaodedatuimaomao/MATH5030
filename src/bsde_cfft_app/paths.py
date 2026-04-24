"""Locations for v0 CFFT results and repo assets (console layer)."""

from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = _SRC / "implementation_version_0" / "results"
REPO_ROOT: Path = _SRC.parent
DEMO_NOTEBOOK: Path = REPO_ROOT / "notebooks" / "demo.ipynb"
