"""Stub reference pricers (upstream ``_benchmarks`` names; full implementations not bundled here)."""

from __future__ import annotations


def _stub(name: str):
    def _f(*_args, **_kwargs):
        raise NotImplementedError(
            f"{name}: stub only — copy upstream bsde_cfft_sv._benchmarks or install the reference package."
        )

    _f.__name__ = name
    _f.__qualname__ = name
    return _f


heston_char_func = _stub("heston_char_func")
heston_call_price = _stub("heston_call_price")
heston_pyfeng_price = _stub("heston_pyfeng_price")
heston_delta_fd = _stub("heston_delta_fd")
heston_vderiv_fd = _stub("heston_vderiv_fd")
heston_z_fd = _stub("heston_z_fd")
garch_diffusion_approx_price = _stub("garch_diffusion_approx_price")
garch_diffusion_pyfeng_price = _stub("garch_diffusion_pyfeng_price")
garch_diffusion_pyfeng_delta_fd = _stub("garch_diffusion_pyfeng_delta_fd")
garch_diffusion_pyfeng_vderiv_fd = _stub("garch_diffusion_pyfeng_vderiv_fd")
garch_diffusion_pyfeng_z_fd = _stub("garch_diffusion_pyfeng_z_fd")
garch_diffusion_pyfeng_mc_price = _stub("garch_diffusion_pyfeng_mc_price")
heston_mc_milstein = _stub("heston_mc_milstein")
garch_diffusion_mc_milstein = _stub("garch_diffusion_mc_milstein")
