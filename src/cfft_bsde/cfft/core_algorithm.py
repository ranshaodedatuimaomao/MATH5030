"""
Core implementation building blocks for the boundary-controlled CFFT-BSDE method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


RealFunc = Callable[[float], float]
DriverFunc = Callable[[float, float, float, float], float]


@dataclass(frozen=True)
class CoreConfig:
    """Minimal configuration for the core CFFT-BSDE backward solver."""

    maturity: float
    n_time_steps: int
    truncation_length: float
    n_space_points: int
    damping_alpha: float

    def validate(self) -> None:
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive")
        if self.n_time_steps < 1:
            raise ValueError("n_time_steps must be >= 1")
        if self.truncation_length <= 0.0:
            raise ValueError("truncation_length must be positive")
        if self.n_space_points < 4:
            raise ValueError("n_space_points must be >= 4 for boundary slopes")
        if self.damping_alpha >= 0.0:
            raise ValueError("damping_alpha must be negative")


@dataclass(frozen=True)
class CoreInputs:
    """Model and BSDE primitives needed by the core algorithm."""

    eta: Callable[[float, float], float]
    sigma: Callable[[float, float], float]
    driver: DriverFunc
    terminal_condition: RealFunc


@dataclass(frozen=True)
class CoreGrids:
    """Discretization grids and mesh sizes."""

    t: list[float]
    x: list[float]
    v: list[float]
    dt: float
    dx: float
    dv: float


def build_grids(config: CoreConfig, *, x_center: float) -> CoreGrids:
    """Build uniform time, space, and frequency grids used by the solver."""

    config.validate()
    dt = config.maturity / float(config.n_time_steps)
    dx = config.truncation_length / float(config.n_space_points)
    dv = 2.0 * 3.141592653589793 / config.truncation_length

    t = [k * dt for k in range(config.n_time_steps + 1)]
    x0 = x_center - 0.5 * config.truncation_length
    x = [x0 + i * dx for i in range(config.n_space_points)]
    n_half = config.n_space_points / 2.0
    v = [(j - n_half) * dv for j in range(config.n_space_points)]
    return CoreGrids(t=t, x=x, v=v, dt=dt, dx=dx, dv=dv)
