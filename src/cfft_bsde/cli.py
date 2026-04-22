"""Console entry point for running the core CFFT-BSDE algorithm."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from cfft_bsde.benchmark_driver import BenchmarkConfig
from cfft_bsde.benchmark_driver import MarketParams
from cfft_bsde.benchmark_driver import parse_csv_floats
from cfft_bsde.benchmark_driver import parse_csv_ints
from cfft_bsde.benchmark_driver import parse_methods
from cfft_bsde.benchmark_driver import run_benchmark_comparison
from cfft_bsde.cfft.core_algorithm import CoreConfig, CoreInputs, solve_core


def _run_benchmark_comparison(args: argparse.Namespace) -> None:
    """Run switchable benchmark comparison and write CSV."""

    benchmark_config = BenchmarkConfig(
        methods=parse_methods(args.benchmark_methods),
        n_values=parse_csv_ints(args.benchmark_n_values),
        l_values=parse_csv_floats(args.benchmark_l_values),
        grid_values=parse_csv_ints(args.benchmark_grid_values),
        damping_alpha=args.alpha,
        benchmark_model=args.benchmark_model,
        enforce_positivity=args.enforce_positivity,
        output_path=args.benchmark_output,
        full_surface=args.benchmark_full_surface,
        surface_spot_min=args.surface_spot_min,
        surface_spot_max=args.surface_spot_max,
        surface_spot_points=args.surface_spot_points,
    )
    market = MarketParams(
        spot=args.spot,
        strike=args.strike,
        rate=args.rate,
        sigma=args.sigma,
        maturity=args.maturity,
    )
    out_path = run_benchmark_comparison(benchmark_config, market)
    print(f"Benchmark comparison complete: {out_path}")


def _run_core(args: argparse.Namespace) -> None:
    """Run one core-algorithm solve with simple Black-Scholes-style inputs."""

    sigma_val = args.sigma
    rate = args.rate
    strike = args.strike
    spot = args.spot

    config = CoreConfig(
        maturity=args.maturity,
        n_time_steps=args.n_time_steps,
        truncation_length=args.truncation_length,
        n_space_points=args.n_space_points,
        damping_alpha=args.alpha,
    )
    inputs = CoreInputs(
        eta=lambda _t, _x: rate - 0.5 * sigma_val * sigma_val,
        sigma=lambda _t, _x: sigma_val,
        driver=lambda _t, _x, y, _z: -rate * y,
        terminal_condition=lambda x: max(math.exp(x) - strike, 0.0),
    )
    solution = solve_core(
        config=config,
        inputs=inputs,
        x_center=math.log(spot),
        enforce_positivity=args.enforce_positivity,
        method=args.method,
    )
    x0 = math.log(spot)
    idx = min(range(len(solution.x)), key=lambda i: abs(solution.x[i] - x0))
    y0 = solution.y[0][idx]
    z0 = solution.z[0][idx]
    print("Core solve complete")
    print(f"grid point x[{idx}]={solution.x[idx]:.6f}")
    print(f"Y(0, x0) ~= {y0:.6f}")
    print(f"Z(0, x0) ~= {z0:.6f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfft-bsde",
        description="Run the core CFFT-BSDE algorithm.",
    )

    parser.add_argument("--spot", type=float, default=100.0, help="Initial spot level")
    parser.add_argument("--strike", type=float, default=100.0, help="Option strike")
    parser.add_argument("--rate", type=float, default=0.01, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--maturity", type=float, default=1.0, help="Maturity T")
    parser.add_argument("--n-time-steps", type=int, default=64, help="Number of time steps")
    parser.add_argument("--n-space-points", type=int, default=128, help="Number of spatial points")
    parser.add_argument("--truncation-length", type=float, default=12.0, help="Spatial truncation length")
    parser.add_argument("--alpha", type=float, default=-1.5, help="Damping alpha (<0)")
    parser.add_argument(
        "--method",
        choices=["new_boundary_control", "legacy_hyndman_2017"],
        default="new_boundary_control",
        help="Numerical method variant: new_boundary_control (boundary error control) or legacy_hyndman_2017 (2017 convolution scheme)",
    )
    parser.add_argument(
        "--enforce-positivity",
        action="store_true",
        help="Apply max(Y,0) clamp in each backward step",
    )
    parser.add_argument(
        "--open-replication-report",
        action="store_true",
        help="Open results/replication_report.html in the default browser and exit (no solve)",
    )
    parser.add_argument(
        "--replication-report",
        type=Path,
        default=None,
        metavar="PATH",
        help="Explicit path to replication_report.html (optional; used with --open-replication-report)",
    )
    parser.add_argument(
        "--benchmark-compare",
        action="store_true",
        help="Run benchmark comparison for selected methods and write CSV",
    )
    parser.add_argument(
        "--benchmark-methods",
        type=str,
        default="new_boundary_control,legacy_hyndman_2017",
        help="Comma-separated methods to compare",
    )
    parser.add_argument(
        "--benchmark-model",
        choices=["black_scholes_call", "intrinsic_call"],
        default="black_scholes_call",
        help="Reference benchmark model used for error metrics",
    )
    parser.add_argument(
        "--benchmark-n-values",
        type=str,
        default="256,512",
        help="Comma-separated n_time_steps values for benchmark mode",
    )
    parser.add_argument(
        "--benchmark-l-values",
        type=str,
        default="10,12,14",
        help="Comma-separated truncation lengths for benchmark mode",
    )
    parser.add_argument(
        "--benchmark-grid-values",
        type=str,
        default="256,512",
        help="Comma-separated n_space_points values for benchmark mode",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default="results/benchmark_comparison.csv",
        help="Output CSV path for benchmark mode",
    )
    parser.add_argument(
        "--benchmark-full-surface",
        action="store_true",
        help="Evaluate benchmark metrics on a full spot surface instead of only at spot",
    )
    parser.add_argument("--surface-spot-min", type=float, default=60.0, help="Minimum spot for full-surface benchmark")
    parser.add_argument("--surface-spot-max", type=float, default=140.0, help="Maximum spot for full-surface benchmark")
    parser.add_argument("--surface-spot-points", type=int, default=81, help="Number of spot points for full surface")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.open_replication_report:
        from cfft_bsde.replication_report import open_replication_report_html

        raise SystemExit(open_replication_report_html(args.replication_report))

    args.benchmark_output = Path(args.benchmark_output)
    if args.benchmark_compare:
        _run_benchmark_comparison(args)
    else:
        _run_core(args)


if __name__ == "__main__":
    main()
