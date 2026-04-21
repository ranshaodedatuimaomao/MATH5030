"""Benchmark driver for comparing CFFT-BSDE methods against reference models."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Literal

from cfft_bsde.cfft.core_algorithm import CoreConfig
from cfft_bsde.cfft.core_algorithm import CoreInputs
from cfft_bsde.cfft.core_algorithm import MethodName
from cfft_bsde.cfft.core_algorithm import solve_core


BenchmarkModelName = Literal["black_scholes_call", "intrinsic_call"]


@dataclass(frozen=True)
class MarketParams:
    """Inputs shared by numerical and benchmark models."""

    spot: float
    strike: float
    rate: float
    sigma: float
    maturity: float


@dataclass(frozen=True)
class BenchmarkConfig:
    """Driver configuration for grid sweeps and output."""

    methods: tuple[MethodName, ...]
    n_values: tuple[int, ...]
    l_values: tuple[float, ...]
    grid_values: tuple[int, ...]
    damping_alpha: float
    benchmark_model: BenchmarkModelName
    enforce_positivity: bool
    output_path: Path
    full_surface: bool = False
    surface_spot_min: float = 60.0
    surface_spot_max: float = 140.0
    surface_spot_points: int = 81


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(token.strip()) for token in raw.split(",") if token.strip())


def parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(token.strip()) for token in raw.split(",") if token.strip())


def parse_methods(raw: str) -> tuple[MethodName, ...]:
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    allowed = {"boundary_control", "old_2017"}
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"Unknown methods: {', '.join(invalid)}")
    if not values:
        raise ValueError("At least one method must be provided")
    return values  # type: ignore[return-value]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black_scholes_call(market: MarketParams) -> tuple[float, float]:
    if market.maturity <= 0.0 or market.sigma <= 0.0 or market.spot <= 0.0 or market.strike <= 0.0:
        intrinsic = max(market.spot - market.strike, 0.0)
        delta = 1.0 if market.spot > market.strike else 0.0
        return intrinsic, delta
    sqrt_t = math.sqrt(market.maturity)
    d1 = (math.log(market.spot / market.strike) + (market.rate + 0.5 * market.sigma * market.sigma) * market.maturity) / (
        market.sigma * sqrt_t
    )
    d2 = d1 - market.sigma * sqrt_t
    price = market.spot * _norm_cdf(d1) - market.strike * math.exp(-market.rate * market.maturity) * _norm_cdf(d2)
    delta = _norm_cdf(d1)
    return price, delta


def _intrinsic_call(market: MarketParams) -> tuple[float, float]:
    intrinsic = max(market.spot - market.strike, 0.0)
    delta = 1.0 if market.spot > market.strike else 0.0
    return intrinsic, delta


def _benchmark_pricer(name: BenchmarkModelName) -> Callable[[MarketParams], tuple[float, float]]:
    if name == "black_scholes_call":
        return _black_scholes_call
    if name == "intrinsic_call":
        return _intrinsic_call
    raise ValueError(f"Unsupported benchmark model: {name}")


def _nearest_index(values: list[float], target: float) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def _spot_grid(spot_min: float, spot_max: float, n_points: int) -> list[float]:
    if n_points < 2:
        raise ValueError("surface_spot_points must be >= 2")
    if spot_min <= 0.0 or spot_max <= 0.0:
        raise ValueError("surface_spot_min and surface_spot_max must be positive")
    if spot_max <= spot_min:
        raise ValueError("surface_spot_max must be greater than surface_spot_min")
    step = (spot_max - spot_min) / float(n_points - 1)
    return [spot_min + i * step for i in range(n_points)]


def _delta_fd_at(solution_y0: list[float], x_grid: list[float], idx: int, spot: float) -> float:
    if 0 < idx < len(x_grid) - 1:
        dy_dx = (solution_y0[idx + 1] - solution_y0[idx - 1]) / (x_grid[idx + 1] - x_grid[idx - 1])
    elif idx == 0:
        dy_dx = (solution_y0[1] - solution_y0[0]) / (x_grid[1] - x_grid[0])
    else:
        dy_dx = (solution_y0[-1] - solution_y0[-2]) / (x_grid[-1] - x_grid[-2])
    return dy_dx / spot


def run_benchmark_comparison(config: BenchmarkConfig, market: MarketParams) -> Path:
    """Run benchmark comparison and persist a CSV result file."""

    pricer = _benchmark_pricer(config.benchmark_model)
    if config.full_surface:
        spots = _spot_grid(config.surface_spot_min, config.surface_spot_max, config.surface_spot_points)
    else:
        spots = [market.spot]
    rows: list[dict[str, float | int | str]] = []

    for n_time_steps in config.n_values:
        for truncation_length in config.l_values:
            for n_space_points in config.grid_values:
                core_config = CoreConfig(
                    maturity=market.maturity,
                    n_time_steps=n_time_steps,
                    truncation_length=truncation_length,
                    n_space_points=n_space_points,
                    damping_alpha=config.damping_alpha,
                )
                core_inputs = CoreInputs(
                    eta=lambda _t, _x: market.rate - 0.5 * market.sigma * market.sigma,
                    sigma=lambda _t, _x: market.sigma,
                    driver=lambda _t, _x, y, _z: -market.rate * y,
                    terminal_condition=lambda x: max(math.exp(x) - market.strike, 0.0),
                )
                for method in config.methods:
                    solution = solve_core(
                        config=core_config,
                        inputs=core_inputs,
                        x_center=math.log(market.spot),
                        enforce_positivity=config.enforce_positivity,
                        method=method,
                    )
                    for eval_spot in spots:
                        eval_market = MarketParams(
                            spot=eval_spot,
                            strike=market.strike,
                            rate=market.rate,
                            sigma=market.sigma,
                            maturity=market.maturity,
                        )
                        price_bench, delta_bench = pricer(eval_market)
                        x_eval = math.log(eval_spot)
                        idx = _nearest_index(solution.x, x_eval)
                        price_cfft = solution.y[0][idx]
                        z0 = solution.z[0][idx]
                        delta_z = z0 / (market.sigma * eval_spot)
                        delta_fd = _delta_fd_at(solution.y[0], solution.x, idx, eval_spot)

                        abs_price_err = abs(price_cfft - price_bench)
                        abs_delta_z_err = abs(delta_z - delta_bench)
                        abs_delta_fd_err = abs(delta_fd - delta_bench)
                        rel_delta_z_err = abs_delta_z_err / abs(delta_bench) if delta_bench != 0.0 else float("nan")
                        rel_delta_fd_err = abs_delta_fd_err / abs(delta_bench) if delta_bench != 0.0 else float("nan")

                        rows.append(
                            {
                                "benchmark_model": config.benchmark_model,
                                "method": method,
                                "n": n_time_steps,
                                "L": truncation_length,
                                "N": n_space_points,
                                "spot_eval": eval_spot,
                                "x_eval": x_eval,
                                "price_benchmark": price_bench,
                                "price_cfft": price_cfft,
                                "abs_price_err": abs_price_err,
                                "delta_benchmark": delta_bench,
                                "delta_z": delta_z,
                                "abs_delta_z_err": abs_delta_z_err,
                                "rel_delta_z_err": rel_delta_z_err,
                                "delta_fd": delta_fd,
                                "abs_delta_fd_err": abs_delta_fd_err,
                                "rel_delta_fd_err": rel_delta_fd_err,
                            }
                        )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark_model",
        "method",
        "n",
        "L",
        "N",
        "spot_eval",
        "x_eval",
        "price_benchmark",
        "price_cfft",
        "abs_price_err",
        "delta_benchmark",
        "delta_z",
        "abs_delta_z_err",
        "rel_delta_z_err",
        "delta_fd",
        "abs_delta_fd_err",
        "rel_delta_fd_err",
    ]
    with config.output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return config.output_path
