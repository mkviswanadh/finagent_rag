"""Timestamped archival of result/report files before a run overwrites them.

`run_pilot.py` writes `pilot_run_report.json` / `pilot_run.log` fresh on every invocation, and a
future full-run script will do the same to `Coding_Sheet_RESULTS.xlsx` — without this, a run that
crashes partway through, or simply performs worse than the last one, silently destroys the only
record of the previous run. `archive_file` copies whatever is about to be overwritten into
`archive/<category>/` with a timestamp in the filename first, so every run's output is recoverable,
not just the most recent one.

This is a copy, not a move — the caller's own write path is unaffected; call `archive_file` right
before opening the destination for writing.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from finagent.config import ARCHIVE_DIR

_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def archive_file(path: str | Path, *, category: str, archive_dir: Path = ARCHIVE_DIR) -> Path | None:
    """Copy `path` into `archive_dir/category/<stem>_<timestamp><suffix>` if it exists.

    Args:
        path: The file about to be overwritten.
        category: Subdirectory to group related archives under (e.g. "pilot", "full_run").
        archive_dir: Root archive directory — overridable for tests.

    Returns:
        The archived copy's path, or `None` if `path` doesn't exist yet (nothing to archive on a
        first run — not an error).
    """
    source = Path(path)
    if not source.exists():
        return None

    destination_dir = archive_dir / category
    destination_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
    destination = destination_dir / f"{source.stem}_{timestamp}{source.suffix}"
    shutil.copy2(source, destination)
    return destination
