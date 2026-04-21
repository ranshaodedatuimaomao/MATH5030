"""
Standalone launcher:
1) installs this package from the local repo
2) imports and runs the core algorithm console app
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def install_local_package(repo_root: Path) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", str(repo_root)]
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    install_local_package(repo_root)

    from cfft_bsde.cli import main as cli_main

    cli_main(
        [
            "--n-time-steps",
            "32",
            "--n-space-points",
            "128",
            "--truncation-length",
            "12.0",
        ]
    )


if __name__ == "__main__":
    main()
