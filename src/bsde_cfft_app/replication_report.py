"""Locate and open the bundled ``replication_report.html`` in the default browser."""

from __future__ import annotations

import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from bsde_cfft_app.paths import RESULTS_DIR


def _candidate_paths() -> list[Path]:
    """Ordered search locations for the static replication summary HTML."""

    out: list[Path] = [
        RESULTS_DIR / "replication_report.html",
        Path.cwd() / "results" / "replication_report.html",
    ]

    here = Path(__file__).resolve()
    p = here.parent
    for _ in range(10):
        out.append(p / "results" / "replication_report.html")
        if p.parent == p:
            break
        p = p.parent

    seen: set[str] = set()
    unique: list[Path] = []
    for cand in out:
        try:
            key = str(cand.resolve())
        except OSError:
            key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def resolve_replication_report_path(explicit: Path | None) -> Path | None:
    """Return an existing ``replication_report.html`` path, or ``None``."""

    if explicit is not None:
        p = explicit.expanduser().resolve()
        return p if p.is_file() else None
    for cand in _candidate_paths():
        if cand.is_file():
            return cand.resolve()
    return None


def open_replication_report_html(
    explicit: Path | None = None,
    *,
    extra_search_roots: Sequence[Path] | None = None,
) -> int:
    """Open the replication HTML in the system browser. Returns 0 on success, 1 on failure."""

    path = resolve_replication_report_path(explicit)
    if path is None and extra_search_roots:
        for root in extra_search_roots:
            for sub in (
                root / "src" / "implementation_version_0" / "results",
                root / "src" / "bsde_cfft_sv" / "implementation_version_0" / "results",
                root / "results",
            ):
                path = resolve_replication_report_path(sub / "replication_report.html")
                if path is not None:
                    break
            if path is not None:
                break
    if path is None:
        path = resolve_replication_report_path(None)

    if path is None:
        print(
            "Could not find replication_report.html. "
            "Expected src/implementation_version_0/results/replication_report.html, "
            "or legacy results/replication_report.html "
            "(or pass --replication-report PATH).",
            file=sys.stderr,
        )
        return 1

    url = path.as_uri()
    opened = webbrowser.open(url)
    if not opened:
        print(f"Browser could not be opened automatically. Open this file manually:\n  {path}", file=sys.stderr)
        return 1

    print(f"Opened replication report: {path}")
    return 0
