"""
Standalone launcher:
1) installs this package from the local repo
2) imports and runs the console app in selectable modes

Modes ``benchmark-point``, ``benchmark-surface``, and ``replication`` match the
committed replication CSVs under ``results/`` (see ``results/replication_report.html``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def install_local_package(repo_root: Path) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", str(repo_root)]
    )


def _shared_black_scholes_call_args() -> list[str]:
    """Market + benchmark model flags used across replication presets."""

    return [
        "--benchmark-model",
        "black_scholes_call",
        "--spot",
        "100",
        "--strike",
        "100",
        "--rate",
        "0.01",
        "--sigma",
        "0.2",
        "--maturity",
        "1",
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_standalone.py",
        description="Install local package and run cfft-bsde presets.",
    )
    parser.add_argument(
        "--mode",
        choices=["core", "benchmark-point", "benchmark-surface", "replication"],
        default="core",
        help=(
            "Preset: core (single solve); benchmark-point / benchmark-surface (CSV exports); "
            "replication (full pipeline: both CSVs + paper-style PNG figures)."
        ),
    )
    parser.add_argument(
        "--extra-cli-args",
        type=str,
        nargs="*",
        default=[],
        help="Additional raw CLI args appended to each benchmark/core invocation (optional)",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="With --mode replication: only regenerate CSV benchmarks, skip matplotlib figures",
    )
    return parser


def _preset_args(mode: str) -> list[str]:
    if mode == "core":
        return [
            "--n-time-steps",
            "32",
            "--n-space-points",
            "128",
            "--truncation-length",
            "12.0",
        ]
    if mode == "benchmark-point":
        return [
            "--benchmark-compare",
            "--benchmark-methods",
            "new_boundary_control,legacy_hyndman_2017",
            *_shared_black_scholes_call_args(),
            "--benchmark-n-values",
            "64,128",
            "--benchmark-l-values",
            "10,12",
            "--benchmark-grid-values",
            "128,256",
            "--benchmark-output",
            "results/numerical_results_quick.csv",
        ]
    return [
        "--benchmark-compare",
        "--benchmark-methods",
        "new_boundary_control,legacy_hyndman_2017",
        *_shared_black_scholes_call_args(),
        "--benchmark-full-surface",
        "--surface-spot-min",
        "60",
        "--surface-spot-max",
        "140",
        "--surface-spot-points",
        "41",
        "--benchmark-n-values",
        "64",
        "--benchmark-l-values",
        "10",
        "--benchmark-grid-values",
        "128",
        "--benchmark-output",
        "results/numerical_results_surface_quick.csv",
    ]


def _run_replication(
    *,
    extra_cli_args: list[str],
    passthrough: list[str],
    skip_figures: bool,
) -> None:
    """Regenerate committed-style CSVs and paper-style PNGs (``plot_paper_figures``)."""

    from cfft_bsde.cli import main as cli_main

    tail = list(extra_cli_args) + list(passthrough)

    print("[replication] Step 1/3: point benchmark -> results/numerical_results_quick.csv")
    cli_main(_preset_args("benchmark-point") + tail)

    print("[replication] Step 2/3: surface benchmark -> results/numerical_results_surface_quick.csv")
    cli_main(_preset_args("benchmark-surface") + tail)

    if skip_figures:
        print("[replication] Step 3/3: skipped (--skip-figures).")
        print("  Install matplotlib and run: python -m cfft_bsde.plot_paper_figures")
        return

    print("[replication] Step 3/3: paper-style figures -> results/figure_0*.png")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "[replication] matplotlib not installed; skipping figures.\n"
            "  Run: python -m pip install matplotlib\n"
            "  Then: python -m cfft_bsde.plot_paper_figures "
            "--surface-csv results/numerical_results_surface_quick.csv --output-dir results",
            file=sys.stderr,
        )
        return

    from cfft_bsde.plot_paper_figures import main as plot_main

    plot_main(
        [
            "--surface-csv",
            "results/numerical_results_surface_quick.csv",
            "--output-dir",
            "results",
        ]
    )

    print(
        "\n[replication] Done.\n"
        "  CSV:  results/numerical_results_quick.csv\n"
        "        results/numerical_results_surface_quick.csv\n"
        "  PNG:  results/figure_01_legacy_price_delta_errors.png\n"
        "        results/figure_02_new_boundary_price_delta_errors.png\n"
        "        results/figure_03_new_boundary_delta_surface.png\n"
        "  HTML: results/replication_report.html (static; open in a browser)\n"
    )


def main(argv: list[str] | None = None) -> None:
    args, passthrough = _build_parser().parse_known_args(argv)
    repo_root = Path(__file__).resolve().parent
    install_local_package(repo_root)

    if args.mode == "replication":
        _run_replication(
            extra_cli_args=args.extra_cli_args,
            passthrough=passthrough,
            skip_figures=args.skip_figures,
        )
        return

    from cfft_bsde.cli import main as cli_main

    cli_main(_preset_args(args.mode) + args.extra_cli_args + passthrough)


if __name__ == "__main__":
    main()
