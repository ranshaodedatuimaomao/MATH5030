"""Shared terminal menu: v0 report, CFFT core, or notebook demo."""

from __future__ import annotations

import csv
import importlib.util
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from bsde_cfft_app.paths import DEMO_NOTEBOOK, REPO_ROOT, RESULTS_DIR

_RESULTS_LABEL = "implementation_version_0/results"


def _print_bsm_benchmark_line() -> None:
    """Show closed-form BSM reference from the bundled quick CSV when available."""

    quick = RESULTS_DIR / "numerical_results_quick.csv"
    if not quick.is_file():
        print(
            f"  (No bundled {quick.name} yet — run replication from the CLI if you need CSVs; "
            "replication still uses black_scholes_call as the benchmark model.)\n"
        )
        return
    try:
        with quick.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("benchmark_model") == "black_scholes_call":
                    pb = float(row["price_benchmark"])
                    db = float(row["delta_benchmark"])
                    se = row.get("spot_eval", "100")
                    print(
                        "  Black–Scholes (BSM) benchmark at spot_eval = "
                        f"{se} — price_benchmark ≈ {pb:.6f}, delta_benchmark ≈ {db:.6f} "
                        f"(from bundled `{_RESULTS_LABEL}/numerical_results_quick.csv`).\n"
                    )
                    return
    except (OSError, KeyError, ValueError):
        print("  (Could not read BSM reference line from numerical_results_quick.csv.)\n")


def _run_v0_cfft_core() -> None:
    """Run the working solver under ``implementation_version_0`` (course bundle)."""

    print(
        "Running the bundled CFFT–BSDE core in ``implementation_version_0.cfft``.\n"
        "Note: the `bsde_cfft_sv` public API (``BSDECFFT1D`` / ``price_black_scholes_1d``) is the "
        "upstream-style surface; the v0 `solve_core` engine lives in ``implementation_version_0``.\n"
    )
    from bsde_cfft_app import cli as cli_mod

    demo_args = cli_mod._build_parser().parse_args([])
    cli_mod._run_core(demo_args)


def _has_jupyter_notebook() -> bool:
    return importlib.util.find_spec("jupyter") is not None


def _launch_notebook_demo() -> int:
    """Start Jupyter for ``notebooks/demo.ipynb``, or open the file with the OS."""

    nb = DEMO_NOTEBOOK.resolve()
    if not nb.is_file():
        print(f"Demo notebook not found: {nb}", file=sys.stderr)
        return 1

    if _has_jupyter_notebook():
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "jupyter", "notebook", str(nb), "--no-browser"],
                cwd=str(REPO_ROOT),
            )
        except OSError as exc:  # pragma: no cover
            print(f"Could not start Jupyter: {exc}", file=sys.stderr)
        else:
            if proc.poll() is None:
                print(
                    f"Jupyter Notebook is starting for:\n  {nb}\n"
                    "Watch this console for a local URL and token, or run: python -m jupyter notebook list\n"
                )
                return 0

    try:
        if platform.system() == "Windows":
            os.startfile(str(nb))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(nb)], check=False)
        else:
            subprocess.run(["xdg-open", str(nb)], check=False)
    except OSError as exc:
        print(
            f"Install Jupyter to run the demo in-browser:\n  python -m pip install notebook\n"
            f"Or open the file yourself:\n  {nb}\n"
            f"({exc})",
            file=sys.stderr,
        )
        return 1

    print(f"Opened demo notebook (default app): {nb}")
    return 0


def run_replication_menu(
    *,
    title: str = "bsde-cfft-sv",
    extra_report_roots: Sequence[Path] | None = None,
    replication_extra_tail: Sequence[str] | None = None,
) -> int:
    """Prompt for v0 report, CFFT core, or notebook demo. Returns a process exit code."""

    _ = replication_extra_tail
    roots = tuple(extra_report_roots) if extra_report_roots else ()

    print(f"{title} - choose an action:\n")
    print(
        "  Benchmark context: replication CSVs use closed-form Black–Scholes (BSM) as the reference. "
        "The “first implementation” is v0 (``implementation_version_0``) tied to the course core.\n"
    )
    _print_bsm_benchmark_line()
    print("  1) First implementation (v0) — open the replication / summary HTML in your browser")
    print("  2) Run the bundled CFFT–BSDE core (``implementation_version_0.cfft``; `bsde_cfft_sv` is the public API package)")
    print("  3) Run the ``notebooks/demo.ipynb`` demo (Jupyter, or your default .ipynb app)")
    print("  q) Quit\n")

    while True:
        choice = input("Enter 1, 2, 3, or q: ").strip().lower()
        if choice in ("q", "quit", ""):
            print("Exiting.")
            return 0
        if choice == "1":
            from bsde_cfft_app.replication_report import open_replication_report_html

            return open_replication_report_html(None, extra_search_roots=roots)
        if choice == "2":
            _run_v0_cfft_core()
            return 0
        if choice == "3":
            return _launch_notebook_demo()
        print("Invalid choice; try again.\n")
