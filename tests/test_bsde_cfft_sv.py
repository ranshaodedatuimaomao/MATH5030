"""Smoke tests for the course layout: imports, stubs, and implementation_version_0 core."""

from __future__ import annotations

import math

import pytest

import bsde_cfft_sv
from bsde_cfft_sv.implementation_version_0.cfft.core_algorithm import CoreConfig, CoreInputs, solve_core


def test_package_import_and_version():
    assert bsde_cfft_sv.__version__
    assert hasattr(bsde_cfft_sv, "BSDECFFT1D")


def test_stub_api_raises():
    with pytest.raises(NotImplementedError):
        bsde_cfft_sv.price_black_scholes_1d()


def test_implementation_version_0_solve_core_smoke():
    config = CoreConfig(
        maturity=1.0,
        n_time_steps=8,
        truncation_length=10.0,
        n_space_points=32,
        damping_alpha=-1.5,
    )
    rate = 0.01
    sigma = 0.2
    strike = 100.0
    inputs = CoreInputs(
        eta=lambda _t, _x: rate - 0.5 * sigma * sigma,
        sigma=lambda _t, _x: sigma,
        driver=lambda _t, _x, y, _z: -rate * y,
        terminal_condition=lambda x: max(math.exp(x) - strike, 0.0),
    )
    sol = solve_core(config=config, inputs=inputs, x_center=math.log(100.0))
    assert len(sol.t) == config.n_time_steps + 1
    assert len(sol.x) == config.n_space_points
    assert math.isfinite(sol.y[0][len(sol.x) // 2])
