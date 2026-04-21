"""
Core implementation building blocks for the boundary-controlled CFFT-BSDE method.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import Literal
from typing import Callable


RealFunc = Callable[[float], float]
DriverFunc = Callable[[float, float, float, float], float]
MethodName = Literal["boundary_control", "old_2017"]


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


@dataclass
class SolverState:
    """Time-space storage for backward recursion arrays."""

    y: list[list[float]]
    z: list[list[float]]


@dataclass(frozen=True)
class CoreSolution:
    """Core algorithm output container."""

    t: list[float]
    x: list[float]
    y: list[list[float]]
    z: list[list[float]]


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

    c1 = damp_left * exp_left - damp_right * exp_right
    d1 = damp_left - damp_right
    r1 = damp_left * y_left - damp_right * y_right

    c2 = damp_left * (alpha * exp_left + exp_left) - damp_right * (alpha * exp_right + exp_right)
    d2 = alpha * (damp_left - damp_right)
    r2 = damp_left * (alpha * y_left + yprime_left) - damp_right * (alpha * y_right + yprime_right)

    det = c1 * d2 - c2 * d1
    if abs(det) < 1e-14:
        return ShiftParams(a=0.0, b=0.0)
    a_val = (r1 * d2 - r2 * d1) / det
    b_val = (c1 * r2 - c2 * r1) / det
    return ShiftParams(a=a_val, b=b_val)


def solve_linear_shift_params(
    x: list[float],
    y: list[float],
    *,
    alpha: float,
    dx: float,
) -> ShiftParams:
    """Solve A,B for h(x)=A*x+B using periodicity of damped value/slope."""

    x_left = x[0]
    x_right = x[-1]
    y_left = y[0]
    y_right = y[-1]
    yprime_left, yprime_right = _one_sided_boundary_slopes(y, dx)

    damp_left = cmath.exp(alpha * x_left).real
    damp_right = cmath.exp(alpha * x_right).real

    c1 = damp_left * x_left - damp_right * x_right
    d1 = damp_left - damp_right
    r1 = damp_left * y_left - damp_right * y_right

    c2 = damp_left * (alpha * x_left + 1.0) - damp_right * (alpha * x_right + 1.0)
    d2 = alpha * (damp_left - damp_right)
    r2 = damp_left * (alpha * y_left + yprime_left) - damp_right * (alpha * y_right + yprime_right)

    det = c1 * d2 - c2 * d1
    if abs(det) < 1e-14:
        return ShiftParams(a=0.0, b=0.0)
    a_val = (r1 * d2 - r2 * d1) / det
    b_val = (c1 * r2 - c2 * r1) / det
    return ShiftParams(a=a_val, b=b_val)


def _suggest_adaptive_alpha(
    x: list[float],
    y: list[float],
    *,
    default_alpha: float,
) -> float:
    """Heuristic per-step alpha update, clipped to stable negative range."""

    x_left = x[0]
    x_right = x[-1]
    y_left = max(abs(y[0]), 1e-12)
    y_right = max(abs(y[-1]), 1e-12)
    span = x_right - x_left
    if span <= 0.0:
        return default_alpha
    # Enforce alpha < 0 for damping while adapting to boundary growth.
    raw = -abs(cmath.log(y_right / y_left).real / span)
    if not (raw < 0.0):
        raw = default_alpha
    return min(-0.1, max(-8.0, raw))


def initialize_state(grids: CoreGrids, terminal_condition: RealFunc) -> SolverState:
    """Initialize Y at terminal time and Z with zeros."""

    n_time = len(grids.t)
    n_space = len(grids.x)
    y = [[0.0 for _ in range(n_space)] for _ in range(n_time)]
    z = [[0.0 for _ in range(n_space)] for _ in range(n_time)]
    y[-1] = [terminal_condition(xi) for xi in grids.x]
    return SolverState(y=y, z=z)


def run_backward_core(
    *,
    config: CoreConfig,
    inputs: CoreInputs,
    x_center: float,
    enforce_positivity: bool = False,
    method: MethodName = "boundary_control",
) -> tuple[CoreGrids, SolverState]:
    """Run the core backward CFFT recursion."""

    grids = build_grids(config, x_center=x_center)
    state = initialize_state(grids, inputs.terminal_condition)
    center_idx = len(grids.x) // 2
    x_ref = grids.x[center_idx]

    for k in range(len(grids.t) - 2, -1, -1):
        tk = grids.t[k]
        if method == "old_2017":
            alpha_k = _suggest_adaptive_alpha(grids.x, state.y[k + 1], default_alpha=config.damping_alpha)
        else:
            alpha_k = config.damping_alpha
        eta_k = inputs.eta(tk, x_ref)
        sigma_k = inputs.sigma(tk, x_ref)
        multipliers = build_multipliers(
            grids,
            alpha=alpha_k,
            eta=eta_k,
            sigma=sigma_k,
        )
        y_next = state.y[k + 1]
        if method == "old_2017":
            shift = solve_linear_shift_params(grids.x, y_next, alpha=alpha_k, dx=grids.dx)
        else:
            shift = solve_shift_params(grids.x, y_next, alpha=alpha_k, dx=grids.dx)
        recovery = build_recovery_terms(grids, eta=eta_k, sigma=sigma_k)

        y_tilde: list[complex] = []
        for i, xi in enumerate(grids.x):
            if method == "old_2017":
                shifted = shift.a * xi + shift.b
            else:
                shifted = shift.a * cmath.exp(xi).real + shift.b
            y_tilde.append(cmath.exp(alpha_k * xi) * (y_next[i] - shifted))

        y_hat = centered_dft(y_tilde, grids.x, grids.v)
        y_conv_hat = [y_hat[j] * multipliers.psi_y[j] for j in range(len(y_hat))]
        z_conv_hat = [y_hat[j] * multipliers.psi_z[j] for j in range(len(y_hat))]
        y_conv = centered_idft(y_conv_hat, grids.x, grids.v)
        z_conv = centered_idft(z_conv_hat, grids.x, grids.v)

        y_k: list[float] = []
        z_k: list[float] = []
        for i, xi in enumerate(grids.x):
            undamp = cmath.exp(-alpha_k * xi).real
            if method == "old_2017":
                y_pre = undamp * y_conv[i].real + shift.a * xi + shift.b
                z_pre = undamp * z_conv[i].real
            else:
                y_pre = undamp * y_conv[i].real + shift.a * recovery.y_term[i] + shift.b
                z_pre = undamp * z_conv[i].real + shift.a * recovery.z_term[i]
            y_new = y_pre + grids.dt * inputs.driver(tk, xi, y_pre, z_pre)
            if enforce_positivity:
                y_new = max(y_new, 0.0)
            y_k.append(y_new)
            z_k.append(z_pre)

        state.y[k] = y_k
        state.z[k] = z_k

    return grids, state


def solve_core(
    *,
    config: CoreConfig,
    inputs: CoreInputs,
    x_center: float,
    enforce_positivity: bool = False,
    method: MethodName = "boundary_control",
) -> CoreSolution:
    """Public core-only solve API."""

    grids, state = run_backward_core(
        config=config,
        inputs=inputs,
        x_center=x_center,
        enforce_positivity=enforce_positivity,
        method=method,
    )
    return CoreSolution(t=grids.t, x=grids.x, y=state.y, z=state.z)
