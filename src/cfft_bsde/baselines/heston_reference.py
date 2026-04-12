"""
Heston European prices via established routes (e.g. PyFENG / characteristic function FFT).

Placeholder for comparing CFFT-BSDE against fast SV pricers.
"""

from __future__ import annotations

from typing import Any, Mapping


def price_european(*, config: Mapping[str, Any] | None = None) -> float:
    """Placeholder: reference Heston call/put price."""
    raise NotImplementedError("Heston reference pricer")
