"""Full replication workflow: quick + surface benchmark CSVs, then paper-style figures."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def _shared_black_scholes_call_argv() -> list[str]:
    return [
        "--benchmark-model",
        "black_scholes_call",
        "--spot",
        "100",
        "--strike",
        "100",
        "--rate",
        "0.01",
        "--sigma",
        "0.2",
        "--maturity",
        "1",
    ]


def replication_point_argv() -> list[str]:
    """Argv list for ``cfft_bsde.cli.main`` — point benchmark matching committed ``numerical_results_quick.csv``."""

    return [
        "--benchmark-compare",
        "--benchmark-methods",
        "new_boundary_control,legacy_hyndman_2017",
        *_shared_black_scholes_call_argv(),
        "--benchmark-n-values",
        "64,128",
        "--benchmark-l-values",
        "10,12",
        "--benchmark-grid-values",
        "128,256",
        "--benchmark-output",
        "results/numerical_results_quick.csv",
    ]


def replication_surface_argv() -> list[str]:
    """Argv list for ``cfft_bsde.cli.main`` — surface benchmark matching ``numerical_results_surface_quick.csv``."""

    return [
        "--benchmark-compare",
        "--benchmark-methods",
        "new_boundary_control,legacy_hyndman_2017",
        *_shared_black_scholes_call_argv(),
        "--benchmark-full-surface",
        "--surface-spot-min",
        "60",
        "--surface-spot-max",
        "140",
        "--surface-spot-points",
        "41",
        "--benchmark-n-values",
        "64",
        "--benchmark-l-values",
        "10",
        "--benchmark-grid-values",
        "128",
        "--benchmark-output",
        "results/numerical_results_surface_quick.csv",
    ]


def run_full_replication(
    *,
    extra_tail: Sequence[str],
    skip_figures: bool,
) -> None:
    """Run point CSV, surface CSV, then ``plot_paper_figures`` (unless ``skip_figures``)."""

    from cfft_bsde.cli import main as cli_main

    tail = list(extra_tail)

    print("[replication] Step 1/3: point benchmark -> results/numerical_results_quick.csv")
    cli_main(replication_point_argv() + tail)

    print("[replication] Step 2/3: surface benchmark -> results/numerical_results_surface_quick.csv")
    cli_main(replication_surface_argv() + tail)

    if skip_figures:
        print("[replication] Step 3/3: skipped (--skip-figures).")
        print("  Install matplotlib and run: python -m cfft_bsde.plot_paper_figures")
        _print_replication_done_footer()
        return

    print("[replication] Step 3/3: paper-style figures -> results/figure_0*.png")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "[replication] matplotlib not installed; skipping figures.\n"
            "  Run: python -m pip install matplotlib\n"
            "  Then: python -m cfft_bsde.plot_paper_figures "
            "--surface-csv results/numerical_results_surface_quick.csv --output-dir results",
            file=sys.stderr,
        )
        _print_replication_done_footer()
        return

    from cfft_bsde.plot_paper_figures import main as plot_main

    plot_main(
        [
            "--surface-csv",
            "results/numerical_results_surface_quick.csv",
            "--output-dir",
            "results",
        ]
    )

    _print_replication_done_footer()


def _print_replication_done_footer() -> None:
    print(
        "\n[replication] Done.\n"
        "  CSV:  results/numerical_results_quick.csv\n"
        "        results/numerical_results_surface_quick.csv\n"
        "  PNG:  results/figure_01_legacy_price_delta_errors.png\n"
        "        results/figure_02_new_boundary_price_delta_errors.png\n"
        "        results/figure_03_new_boundary_delta_surface.png\n"
        "  HTML: results/replication_report.html (open with: cfft-bsde --open-replication-report\n"
        "        or: python run_standalone.py --mode open-report)\n"
    )
