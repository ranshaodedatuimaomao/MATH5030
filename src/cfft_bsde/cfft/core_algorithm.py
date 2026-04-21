"""
Core implementation building blocks for the boundary-controlled CFFT-BSDE method.
"""

from __future__ import annotations

import cmath
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


@dataclass(frozen=True)
class FourierMultipliers:
    """Frequency-domain multipliers for Y and Z convolution updates."""

    psi_y: list[complex]
    psi_z: list[complex]


@dataclass(frozen=True)
class RecoveryTerms:
    """Recovery terms induced by exponential shifting h(x)=A exp(x)+B."""

    y_term: list[float]
    z_term: list[float]


@dataclass(frozen=True)
class ShiftParams:
    """Exponential shift parameters h(x)=A exp(x)+B."""

    a: float
    b: float


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


def _phase(i: int) -> float:
    return -1.0 if i % 2 else 1.0


def centered_dft(values: list[complex], x: list[float], v: list[float]) -> list[complex]:
    """Centered discrete Fourier transform with (-1)^n phase shift."""

    transformed: list[complex] = []
    for vk in v:
        acc = 0j
        for n, xn in enumerate(x):
            acc += (_phase(n) * values[n]) * cmath.exp(-1j * vk * xn)
        transformed.append(acc)
    return transformed


def centered_idft(values: list[complex], x: list[float], v: list[float]) -> list[complex]:
    """Centered inverse discrete Fourier transform with (-1)^n phase shift."""

    n_points = len(x)
    recovered: list[complex] = []
    for n, xn in enumerate(x):
        acc = 0j
        for k, vk in enumerate(v):
            acc += values[k] * cmath.exp(1j * vk * xn)
        recovered.append(_phase(n) * (acc / float(n_points)))
    return recovered


def _psi(v: complex, *, dt: float, eta: float, sigma: float) -> complex:
    return cmath.exp(dt * (eta * 1j * v - 0.5 * (sigma**2) * (v**2)))


def _psi_prime(v: complex, *, dt: float, eta: float, sigma: float) -> complex:
    psi_v = _psi(v, dt=dt, eta=eta, sigma=sigma)
    return dt * ((eta * 1j) - (sigma**2) * v) * psi_v


def build_multipliers(
    grids: CoreGrids,
    *,
    alpha: float,
    eta: float,
    sigma: float,
) -> FourierMultipliers:
    """Build Fourier multipliers from the short-time Gaussian characteristic function."""

    psi_y: list[complex] = []
    psi_z: list[complex] = []
    for vj in grids.v:
        shifted = complex(vj, alpha)
        psi_shift = _psi(shifted, dt=grids.dt, eta=eta, sigma=sigma)
        psi_y.append(psi_shift)
        psi_z.append(sigma * (1j * vj - alpha) * psi_shift)
    return FourierMultipliers(psi_y=psi_y, psi_z=psi_z)


def build_recovery_terms(
    grids: CoreGrids,
    *,
    eta: float,
    sigma: float,
) -> RecoveryTerms:
    """Precompute Y/Z recovery vectors from the characteristic function."""

    psi_neg_i = _psi(-1j, dt=grids.dt, eta=eta, sigma=sigma)
    psi_prime_neg_i = _psi_prime(-1j, dt=grids.dt, eta=eta, sigma=sigma)
    y_term: list[float] = []
    z_term: list[float] = []
    for xi in grids.x:
        exp_x = cmath.exp(xi).real
        y_term.append((exp_x * psi_neg_i).real)
        z_term.append(
            (
                (-eta * grids.dt * exp_x * psi_neg_i)
                + ((1j * exp_x * psi_prime_neg_i) / (sigma * grids.dt))
            ).real
        )
    return RecoveryTerms(y_term=y_term, z_term=z_term)


def _one_sided_boundary_slopes(y: list[float], dx: float) -> tuple[float, float]:
    left = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * dx)
    right = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / (2.0 * dx)
    return left, right


def solve_shift_params(x: list[float], y: list[float], *, alpha: float, dx: float) -> ShiftParams:
    """Solve A,B from periodicity constraints on damped-shifted value and slope."""

    x_left = x[0]
    x_right = x[-1]
    y_left = y[0]
    y_right = y[-1]
    yprime_left, yprime_right = _one_sided_boundary_slopes(y, dx)

    exp_left = cmath.exp(x_left).real
    exp_right = cmath.exp(x_right).real
    damp_left = cmath.exp(alpha * x_left).real
    damp_right = cmath.exp(alpha * x_right).real

    # Equation (value): e^{ax0}(Y0 - A e^{x0} - B) = e^{axN}(YN - A e^{xN} - B)
    c1 = damp_left * exp_left - damp_right * exp_right
    d1 = damp_left - damp_right
    r1 = damp_left * y_left - damp_right * y_right

    # Equation (slope): match first derivative of transformed function at boundaries.
    c2 = damp_left * (alpha * exp_left + exp_left) - damp_right * (alpha * exp_right + exp_right)
    d2 = alpha * (damp_left - damp_right)
    r2 = damp_left * (alpha * y_left + yprime_left) - damp_right * (alpha * y_right + yprime_right)

    det = c1 * d2 - c2 * d1
    if abs(det) < 1e-14:
        return ShiftParams(a=0.0, b=0.0)
    a_val = (r1 * d2 - r2 * d1) / det
    b_val = (c1 * r2 - c2 * r1) / det
    return ShiftParams(a=a_val, b=b_val)
