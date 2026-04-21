"""Console entry point for running the core CFFT-BSDE algorithm."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path

from cfft_bsde.cfft.core_algorithm import CoreConfig, CoreInputs, solve_core


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call_price_delta(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> tuple[float, float]:
    if maturity <= 0.0 or sigma <= 0.0 or spot <= 0.0 or strike <= 0.0:
        intrinsic = max(spot - strike, 0.0)
        delta = 1.0 if spot > strike else 0.0
        return intrinsic, delta
    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * maturity) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    price = spot * _norm_cdf(d1) - strike * math.exp(-rate * maturity) * _norm_cdf(d2)
    delta = _norm_cdf(d1)
    return price, delta


def _nearest_index(values: list[float], target: float) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _run_benchmark_comparison(args: argparse.Namespace) -> None:
    """Run old/new method benchmark comparison and write CSV."""

    n_values = _parse_csv_ints(args.benchmark_n_values)
    l_values = _parse_csv_floats(args.benchmark_l_values)
    grid_values = _parse_csv_ints(args.benchmark_grid_values)
    methods = ["boundary_control", "old_2017"]
    spot = args.spot
    strike = args.strike
    rate = args.rate
    sigma_val = args.sigma
    maturity = args.maturity
    x0 = math.log(spot)
    price_bs, delta_bs = _bs_call_price_delta(spot, strike, rate, sigma_val, maturity)

    rows: list[dict[str, float | int | str]] = []
    for n_time_steps in n_values:
        for truncation_length in l_values:
            for n_space_points in grid_values:
                config = CoreConfig(
                    maturity=maturity,
                    n_time_steps=n_time_steps,
                    truncation_length=truncation_length,
                    n_space_points=n_space_points,
                    damping_alpha=args.alpha,
                )
                inputs = CoreInputs(
                    eta=lambda _t, _x: rate - 0.5 * sigma_val * sigma_val,
                    sigma=lambda _t, _x: sigma_val,
                    driver=lambda _t, _x, y, _z: -rate * y,
                    terminal_condition=lambda x: max(math.exp(x) - strike, 0.0),
                )
                for method in methods:
                    solution = solve_core(
                        config=config,
                        inputs=inputs,
                        x_center=x0,
                        enforce_positivity=args.enforce_positivity,
                        method=method,
                    )
                    idx = _nearest_index(solution.x, x0)
                    y0 = solution.y[0][idx]
                    z0 = solution.z[0][idx]
                    delta_z = z0 / (sigma_val * spot)

                    if 0 < idx < len(solution.x) - 1:
                        dy_dx = (solution.y[0][idx + 1] - solution.y[0][idx - 1]) / (solution.x[idx + 1] - solution.x[idx - 1])
                    elif idx == 0:
                        dy_dx = (solution.y[0][1] - solution.y[0][0]) / (solution.x[1] - solution.x[0])
                    else:
                        dy_dx = (solution.y[0][-1] - solution.y[0][-2]) / (solution.x[-1] - solution.x[-2])
                    delta_fd = dy_dx / spot

                    abs_price_err = abs(y0 - price_bs)
                    abs_delta_z_err = abs(delta_z - delta_bs)
                    abs_delta_fd_err = abs(delta_fd - delta_bs)
                    rel_delta_z_err = abs_delta_z_err / abs(delta_bs) if delta_bs != 0.0 else float("nan")
                    rel_delta_fd_err = abs_delta_fd_err / abs(delta_bs) if delta_bs != 0.0 else float("nan")

                    rows.append(
                        {
                            "method": method,
                            "n": n_time_steps,
                            "L": truncation_length,
                            "N": n_space_points,
                            "price_bs": price_bs,
                            "price_cfft": y0,
                            "abs_price_err": abs_price_err,
                            "delta_bs": delta_bs,
                            "delta_z": delta_z,
                            "abs_delta_z_err": abs_delta_z_err,
                            "rel_delta_z_err": rel_delta_z_err,
                            "delta_fd": delta_fd,
                            "abs_delta_fd_err": abs_delta_fd_err,
                            "rel_delta_fd_err": rel_delta_fd_err,
                        }
                    )
                    print(
                        f"method={method} n={n_time_steps} L={truncation_length} N={n_space_points} "
                        f"price={y0:.6f} abs_price_err={abs_price_err:.3e} abs_delta_z_err={abs_delta_z_err:.3e}"
                    )

    out_path = Path(args.benchmark_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "n",
        "L",
        "N",
        "price_bs",
        "price_cfft",
        "abs_price_err",
        "delta_bs",
        "delta_z",
        "abs_delta_z_err",
        "rel_delta_z_err",
        "delta_fd",
        "abs_delta_fd_err",
        "rel_delta_fd_err",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
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
        choices=["boundary_control", "old_2017"],
        default="boundary_control",
        help="Numerical method variant to run",
    )
    parser.add_argument(
        "--enforce-positivity",
        action="store_true",
        help="Apply max(Y,0) clamp in each backward step",
    )
    parser.add_argument(
        "--benchmark-compare",
        action="store_true",
        help="Run benchmark comparison for boundary_control vs old_2017 and write CSV",
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.benchmark_compare:
        _run_benchmark_comparison(args)
    else:
        _run_core(args)


if __name__ == "__main__":
    main()
