import math

from bsde_cfft_sv import price_black_scholes_1d, price_heston_2d, price_garch_2d, grid_damping_sensitivity

def test_bs_1d():
    result = price_black_scholes_1d()
    assert math.isfinite(result.price)
    assert math.isfinite(result.delta)
    assert result.price > 0
    assert result.abs_price_error < 0.1 
    assert 0 <= result.delta <= 1

def test_heston_2d():
    result = price_heston_2d()
    assert math.isfinite(result.price)
    assert math.isfinite(result.delta)
    assert math.isfinite(result.z_x)
    assert math.isfinite(result.z_v)
    assert result.price > 0.0
    assert 0.0 <= result.delta <= 1.0
    assert result.abs_price_error < 0.2

def test_garch_2d():
    result = price_garch_2d()
    assert math.isfinite(result.price)
    assert math.isfinite(result.delta)
    assert math.isfinite(result.z_x)
    assert math.isfinite(result.z_v)
    assert result.price > 0.0
    assert 0.0 <= result.delta <= 1.0
    assert result.abs_price_error < 0.2

def test_grid_damping_sensitivity_shape():
    sensitivity = grid_damping_sensitivity(Nx=24, Nv=12, heston_n_steps=60, garch_n_steps=60)
    assert "heston_lv_sensitivity" in sensitivity
    assert "garch_alpha_sensitivity" in sensitivity
    assert len(sensitivity["heston_lv_sensitivity"]) == 5
    assert len(sensitivity["garch_alpha_sensitivity"]) == 3




