"""
Standalone launcher:
1) installs this package from the local repo
2) imports and runs the console app in selectable modes

Two replication-focused modes:

- ``--mode replication`` — rerun the full replication (CSVs + figures), same as ``cfft-bsde --run-replication``.
- ``--mode open-report`` — open the existing ``results/replication_report.html`` in the default browser.

``benchmark-point`` / ``benchmark-surface`` use the same argv presets as the replication pipeline.
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_standalone.py",
        description="Install local package and run cfft-bsde presets.",
    )
    parser.add_argument(
        "--mode",
        choices=["core", "benchmark-point", "benchmark-surface", "replication", "open-report"],
        default="core",
        help=(
            "Preset: core (single solve); benchmark-point / benchmark-surface (CSV exports only); "
            "replication (rerun full pipeline: both CSVs + paper-style PNG figures); "
            "open-report (open existing results/replication_report.html in a browser)."
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
        help="Only with --mode replication: skip matplotlib figures (CSVs only)",
    )
    parser.add_argument(
        "--replication-report",
        type=Path,
        default=None,
        metavar="PATH",
        help="Only with --mode open-report: explicit path to replication_report.html (optional)",
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
    from cfft_bsde.replication_pipeline import replication_point_argv
    from cfft_bsde.replication_pipeline import replication_surface_argv

    if mode == "benchmark-point":
        return replication_point_argv()
    if mode == "benchmark-surface":
        return replication_surface_argv()
    raise ValueError(f"Unknown preset mode: {mode!r}")


def main(argv: list[str] | None = None) -> None:
    args, passthrough = _build_parser().parse_known_args(argv)
    repo_root = Path(__file__).resolve().parent
    install_local_package(repo_root)

    if args.skip_figures and args.mode != "replication":
        print("--skip-figures is only valid with --mode replication", file=sys.stderr)
        raise SystemExit(2)

    if args.mode == "replication":
        from cfft_bsde.replication_pipeline import run_full_replication

        run_full_replication(
            extra_tail=list(args.extra_cli_args) + list(passthrough),
            skip_figures=args.skip_figures,
        )
        return

    if args.mode == "open-report":
        from cfft_bsde.replication_report import open_replication_report_html

        code = open_replication_report_html(
            args.replication_report,
            extra_search_roots=[repo_root],
        )
        raise SystemExit(code)

    from cfft_bsde.cli import main as cli_main

    cli_main(_preset_args(args.mode) + args.extra_cli_args + passthrough)


if __name__ == "__main__":
    main()
