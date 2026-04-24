"""Stub 1D BSDE-CFFT solver (upstream API surface)."""

from __future__ import annotations


def bs_call_price(*args, **kwargs):
    raise NotImplementedError("Stub: use upstream bsde-cfft-sv or scipy-based reference.")


def bs_call_delta(*args, **kwargs):
    raise NotImplementedError("Stub: use upstream bsde-cfft-sv or scipy-based reference.")


class BSDECFFT1D:
    """Placeholder matching upstream ``BSDECFFT1D`` name; no numerical implementation here."""

    def __init__(self, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs

    def price_at(self, *_args, **_kwargs):
        raise NotImplementedError("Stub: use implementation_version_0 core or upstream 1D solver.")
