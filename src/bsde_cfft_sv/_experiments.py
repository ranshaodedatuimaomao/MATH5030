"""Stub experiment runner (upstream ``run_experiments`` entry point)."""

from __future__ import annotations


def run_experiments(*_args, **_kwargs) -> None:
    raise NotImplementedError(
        "Stub: full experiment suite is not bundled. Use the console app in "
        "``bsde_cfft_sv.cli`` (benchmark / replication) or the reference upstream repo."
    )
