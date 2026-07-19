"""Centralized logging setup for FinAgent-RAG.

Every module that wants to log calls `logging.getLogger(__name__)` as usual — this module's only
job is `configure_logging()`, called once at the start of a script (a pilot run, a full experiment
run, a Jupyter session), to attach a console handler and, optionally, a file handler so a run's
full stage-by-stage history is preserved on disk, not just scrolled past in the terminal.

Log level convention used throughout `finagent`:
- DEBUG: full prompt/response text, per-stage token/latency detail — verbose, off by default.
- INFO: stage transitions ("Retrieval: got 5 chunks in 0.12s"), per-question and per-experiment
  summaries, key results (generated answer, complexity route, headline metrics).
- WARNING: recoverable problems (a key rotation, a JSON repair retry, a metric that couldn't be
  computed for a question).
- ERROR: a question or experiment failed and had to be skipped.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_CONSOLE_DATE_FORMAT = "%H:%M:%S"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(filename)s:%(lineno)d]: %(message)s"


def configure_logging(
    *,
    log_file: str | Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """Attach a console handler (and, if requested, a file handler) to the root `finagent` logger.

    Safe to call more than once (e.g. at the top of every script) — existing handlers on the
    `finagent` logger are removed first, so repeated calls don't duplicate log lines.

    Args:
        log_file: If given, every log record at `file_level` or above is also written here, with
            full detail (filename/line number included) — the console output stays terse by
            comparison, since a long multi-experiment run is much easier to scan when the terminal
            isn't showing the same detail twice.
        console_level: Minimum level printed to the console. Defaults to INFO — stage transitions
            and summaries, not full prompt/response dumps.
        file_level: Minimum level written to `log_file`, if given. Defaults to DEBUG.
    """
    root = logging.getLogger("finagent")
    root.setLevel(min(console_level, file_level))
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_CONSOLE_DATE_FORMAT))
    root.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)

    root.propagate = False
