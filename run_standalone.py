"""
Standalone launcher:
1) installs this package from the local repo
2) imports and runs the console app in selectable modes
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
        choices=["core", "benchmark-point", "benchmark-surface"],
        default="core",
        help="Preset mode to pass to cfft_bsde.cli",
    )
    parser.add_argument(
        "--extra-cli-args",
        type=str,
        nargs="*",
        default=[],
        help="Additional raw CLI args appended to preset args (optional)",
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
            "boundary_control,old_2017",
            "--benchmark-model",
            "black_scholes_call",
            "--benchmark-n-values",
            "64",
            "--benchmark-l-values",
            "10,12",
            "--benchmark-grid-values",
            "128,256",
            "--benchmark-output",
            "results/standalone_benchmark_point.csv",
        ]
    return [
        "--benchmark-compare",
        "--benchmark-methods",
        "boundary_control,old_2017",
        "--benchmark-model",
        "black_scholes_call",
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
        "results/standalone_benchmark_surface.csv",
    ]


def main(argv: list[str] | None = None) -> None:
    args, passthrough = _build_parser().parse_known_args(argv)
    repo_root = Path(__file__).resolve().parent
    install_local_package(repo_root)

    from cfft_bsde.cli import main as cli_main

    cli_main(_preset_args(args.mode) + args.extra_cli_args + passthrough)


if __name__ == "__main__":
    main()
