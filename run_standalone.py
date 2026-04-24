"""
Standalone launcher:
1) installs this package from the local repo
2) runs a preset or the same interactive menu as ``bsde-cfft-sv`` (when appropriate)

Default ``--mode auto`` (interactive terminal): same menu as the console app
(open HTML report / run replication / core demo). Use ``--mode menu`` to force
that menu even when stdin is not a TTY. Non-interactive ``auto`` falls back to
``--mode core``. Explicit modes match the CLI flags (replication, open-report, etc.).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def install_local_package(repo_root: Path) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", str(repo_root)]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_standalone.py",
        description="Install local package and run bsde-cfft-sv presets.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "auto",
            "menu",
            "core",
            "benchmark-point",
            "benchmark-surface",
            "replication",
            "open-report",
        ],
        default="auto",
        help=(
            "auto: in a terminal, same interactive menu as bsde-cfft-sv with no args; "
            "else preset core. "
            "menu: always show that menu. "
            "Other modes: core solve, benchmark CSV presets, full replication, or open HTML report."
        ),
    )
    parser.add_argument(
        "--extra-cli-args",
        type=str,
        nargs="*",
        default=[],
        help="Additional raw CLI args appended to each benchmark/core invocation (optional; not used with auto/menu)",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Only with --mode replication: skip matplotlib figures (CSVs only)",
    )
    parser.add_argument(
        "--replication-report",
        type=Path,
        default=None,
        metavar="PATH",
        help="Only with --mode open-report: explicit path to replication_report.html (optional)",
    )
    return parser


def _preset_args(mode: str) -> list[str]:
    if mode == "core":
        return [
            "--n-time-steps",
            "32",
            "--n-space-points",
            "128",
            "--truncation-length",
            "12.0",
        ]
    from bsde_cfft_sv.replication_pipeline import replication_point_argv
    from bsde_cfft_sv.replication_pipeline import replication_surface_argv

    if mode == "benchmark-point":
        return replication_point_argv()
    if mode == "benchmark-surface":
        return replication_surface_argv()
    raise ValueError(f"Unknown preset mode: {mode!r}")


def main(argv: list[str] | None = None) -> None:
    args, passthrough = _build_parser().parse_known_args(argv)
    repo_root = Path(__file__).resolve().parent
    install_local_package(repo_root)

    if args.skip_figures and args.mode != "replication":
        print("--skip-figures is only valid with --mode replication", file=sys.stderr)
        raise SystemExit(2)

    if args.mode in ("auto", "menu"):
        if args.extra_cli_args or passthrough:
            print(
                "Do not use --extra-cli-args with --mode auto or menu; "
                "pick an explicit --mode (e.g. core, replication) instead.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if args.mode == "menu" or (args.mode == "auto" and sys.stdin.isatty()):
            from bsde_cfft_sv.interactive_menu import run_replication_menu

            raise SystemExit(
                run_replication_menu(
                    title="run_standalone (after pip install -e .)",
                    extra_report_roots=(repo_root,),
                )
            )
        from bsde_cfft_sv.cli import main as cli_main

        cli_main(_preset_args("core"))
        return

    if args.mode == "replication":
        from bsde_cfft_sv.replication_pipeline import run_full_replication

        run_full_replication(
            extra_tail=list(args.extra_cli_args) + list(passthrough),
            skip_figures=args.skip_figures,
        )
        return

    if args.mode == "open-report":
        from bsde_cfft_sv.replication_report import open_replication_report_html

        code = open_replication_report_html(
            args.replication_report,
            extra_search_roots=[repo_root],
        )
        raise SystemExit(code)

    from bsde_cfft_sv.cli import main as cli_main

    cli_main(_preset_args(args.mode) + args.extra_cli_args + passthrough)


if __name__ == "__main__":
    main()
