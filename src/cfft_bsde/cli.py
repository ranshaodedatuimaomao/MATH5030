"""CLI entry points mapping to each project section (placeholders)."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence


def _cmd_validate(_args: argparse.Namespace) -> None:
    from cfft_bsde.validation.reproduce import run_black_scholes_benchmarks

    run_black_scholes_benchmarks()


def _cmd_benchmark(_args: argparse.Namespace) -> None:
    from cfft_bsde.benchmarks.accuracy_runtime import run_study as run_accuracy_runtime

    run_accuracy_runtime()


def _cmd_scaling(_args: argparse.Namespace) -> None:
    from cfft_bsde.benchmarks.scaling import run_study as run_scaling

    run_scaling()


def _cmd_robustness(_args: argparse.Namespace) -> None:
    from cfft_bsde.robustness.sweep import run_sweep

    run_sweep()


def _cmd_adaptive(_args: argparse.Namespace) -> None:
    from cfft_bsde.adaptive.tuning import suggest_config

    suggest_config()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfft-bsde",
        description="Convolution-FFT BSDE project (scaffold).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="Literature reproduction / BS validation").set_defaults(
        _run=_cmd_validate
    )
    sub.add_parser("benchmark", help="Accuracy vs runtime study").set_defaults(_run=_cmd_benchmark)
    sub.add_parser("scaling", help="Grid / time-step scaling study").set_defaults(_run=_cmd_scaling)
    sub.add_parser("robustness", help="Parameter robustness sweep").set_defaults(_run=_cmd_robustness)
    sub.add_parser("adaptive", help="Adaptive parameter tuning (extension)").set_defaults(
        _run=_cmd_adaptive
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run: Callable[[argparse.Namespace], None] = args._run
    run(args)


if __name__ == "__main__":
    main()
