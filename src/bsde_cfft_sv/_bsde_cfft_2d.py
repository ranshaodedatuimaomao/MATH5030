"""Stub 2D BSDE-CFFT solvers (upstream API surface)."""

from __future__ import annotations


class HestonBSDECFFT:
    """Placeholder matching upstream ``HestonBSDECFFT``."""

    def __init__(self, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs

    def price_delta_z_at(self, *_args, **_kwargs):
        raise NotImplementedError("Stub: use upstream 2D Heston solver.")


class GARCHDiffusionBSDECFFT:
    """Placeholder matching upstream ``GARCHDiffusionBSDECFFT``."""

    def __init__(self, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs

    def price_delta_z_at(self, *_args, **_kwargs):
        raise NotImplementedError("Stub: use upstream 2D GARCH diffusion solver.")
