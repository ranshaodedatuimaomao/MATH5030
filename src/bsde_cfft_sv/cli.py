"""Command-line interface for the BSDE-CFFT stochastic-volatility package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ._experiments import run_experiments


PART_CHOICES = ("all", "black-scholes", "heston", "garch", "sensitivity")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsde-cfft-sv",
        description="Run the BSDE-CFFT stochastic-volatility experiments.",
    )
    parser.add_argument("--part", choices=PART_CHOICES, default="all", help="Experiment section to run.")
    parser.add_argument("--run-1d", action="store_true", help="Run 1D Black-Scholes validation.")
    parser.add_argument("--run-heston", action="store_true", help="Run 2D Heston benchmark.")
    parser.add_argument("--run-garch", action="store_true", help="Run 2D GARCH diffusion benchmark.")
    parser.add_argument("--run-sensitivity", action="store_true", help="Run grid/damping sensitivity diagnostics.")
    return parser


def _selected_part(args: argparse.Namespace) -> str:
    flags = [
        ("black-scholes", args.run_1d),
        ("heston", args.run_heston),
        ("garch", args.run_garch),
        ("sensitivity", args.run_sensitivity),
    ]
    selected = [part for part, enabled in flags if enabled]
    if len(selected) > 1:
        raise SystemExit("Choose only one run flag, or use --part all.")
    if selected:
        return selected[0]
    return args.part


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_experiments(_selected_part(args))


if __name__ == "__main__":
    main()
