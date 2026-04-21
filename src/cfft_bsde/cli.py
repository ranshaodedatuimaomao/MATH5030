"""Console entry point for running the core CFFT-BSDE algorithm."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

from cfft_bsde.cfft.core_algorithm import CoreConfig, CoreInputs, solve_core


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
        "--enforce-positivity",
        action="store_true",
        help="Apply max(Y,0) clamp in each backward step",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _run_core(args)


if __name__ == "__main__":
    main()
