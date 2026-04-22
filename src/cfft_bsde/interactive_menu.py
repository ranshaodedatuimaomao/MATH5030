"""Shared terminal menu: open replication HTML, run full replication, or core demo."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def run_replication_menu(
    *,
    title: str = "cfft-bsde",
    extra_report_roots: Sequence[Path] | None = None,
    replication_extra_tail: Sequence[str] | None = None,
) -> int:
    """Prompt for replication actions or a core demo. Returns a process exit code."""

    roots = tuple(extra_report_roots) if extra_report_roots else ()
    rep_tail = list(replication_extra_tail) if replication_extra_tail else []

    print(f"{title} - choose an action:\n")
    print("  1) Open existing replication report (results/replication_report.html in browser)")
    print("  2) Run full replication (regenerate quick + surface CSVs, then paper-style PNGs)")
    print("  3) Run one core solve demo (default non-interactive behavior)")
    print("  q) Quit\n")

    while True:
        choice = input("Enter 1, 2, 3, or q: ").strip().lower()
        if choice in ("q", "quit", ""):
            print("Exiting.")
            return 0
        if choice == "1":
            from cfft_bsde.replication_report import open_replication_report_html

            return open_replication_report_html(None, extra_search_roots=roots)
        if choice == "2":
            skip = input("Skip matplotlib figures (CSVs only)? [y/N]: ").strip().lower()
            from cfft_bsde.replication_pipeline import run_full_replication

            run_full_replication(extra_tail=rep_tail, skip_figures=skip in ("y", "yes"))
            return 0
        if choice == "3":
            from cfft_bsde import cli as cli_mod

            demo_args = cli_mod._build_parser().parse_args([])
            cli_mod._run_core(demo_args)
            return 0
        print("Invalid choice; try again.\n")
