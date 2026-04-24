"""
No CLI in this course repo — the upstream project exposes ``bsde_cfft_sv.cli:main`` via setuptools.

``python -m bsde_cfft_sv`` only prints this notice.
"""

from __future__ import annotations


def _main() -> None:
    print(
        "bsde_cfft_sv: stub package. Implement solvers in src/ or vendor code from the "
        "reference repo BSDE-CFFT-Method-For-Stochastic-Volatility-Models-main."
    )


if __name__ == "__main__":
    _main()
