"""CLI logging and terminal output coordination.

Importing this module has no logging side effects.  The CLI is responsible for
calling :func:`configure_cli_logging`; library users can configure logging with
the standard :mod:`logging` APIs.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import Protocol, TextIO


class _ProgressRenderer(Protocol):
    """The small renderer surface needed by the terminal logging handler."""

    def suspend_for_log(self) -> None: ...

    def resume_after_log(self) -> None: ...

    def interrupt(self) -> None: ...


_TERMINAL_LOCK = RLock()
_ACTIVE_PROGRESS: _ProgressRenderer | None = None
_CLI_HANDLER_MARKER = "pystormtracker-cli-handler"
_QUIET_THIRD_PARTY_LOGGERS = ("dask", "distributed", "h5py", "numba", "xarray")


def terminal_lock() -> RLock:
    """Return the process-local lock shared by progress and CLI logging."""
    return _TERMINAL_LOCK


def register_progress(renderer: _ProgressRenderer) -> None:
    """Register the current live progress renderer."""
    global _ACTIVE_PROGRESS
    with _TERMINAL_LOCK:
        _ACTIVE_PROGRESS = renderer


def unregister_progress(renderer: _ProgressRenderer) -> None:
    """Remove a live progress renderer if it is still the active one."""
    global _ACTIVE_PROGRESS
    with _TERMINAL_LOCK:
        if _ACTIVE_PROGRESS is renderer:
            _ACTIVE_PROGRESS = None


def interrupt_active_progress() -> None:
    """Terminate the current progress display before CLI interruption output."""
    with _TERMINAL_LOCK:
        if _ACTIVE_PROGRESS is not None:
            _ACTIVE_PROGRESS.interrupt()


def write_terminal(text: str, *, stream: TextIO | None = None) -> None:
    """Write one operational message without splitting a progress line."""
    output = stream if stream is not None else sys.stderr
    with _TERMINAL_LOCK:
        progress = _ACTIVE_PROGRESS
        if progress is not None:
            progress.suspend_for_log()
        try:
            output.write(text)
            output.flush()
        finally:
            if progress is not None:
                progress.resume_after_log()


class _TerminalLoggingHandler(logging.StreamHandler[TextIO]):
    """Logging handler that suspends and redraws a live progress line."""

    marker = _CLI_HANDLER_MARKER

    def emit(self, record: logging.LogRecord) -> None:
        with _TERMINAL_LOCK:
            progress = _ACTIVE_PROGRESS
            if progress is not None:
                progress.suspend_for_log()
            try:
                super().emit(record)
            finally:
                if progress is not None:
                    progress.resume_after_log()


def configure_cli_logging(verbosity: int) -> None:
    """Configure the single operational logging stream used by the CLI.

    ``WARNING`` is the quiet default, ``INFO`` is selected by ``-v`` and
    ``DEBUG`` by ``-vv`` or any greater count.  The handler is always stderr so
    stdout remains available for command results and JSON output.
    """
    level = logging.WARNING if verbosity <= 0 else logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG

    root = logging.getLogger()
    root.setLevel(level)
    for name in _QUIET_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for handler in tuple(root.handlers):
        if getattr(handler, "marker", None) == _CLI_HANDLER_MARKER:
            root.removeHandler(handler)
            handler.close()

    handler = _TerminalLoggingHandler(sys.stderr)
    handler.marker = _CLI_HANDLER_MARKER
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)


@contextmanager
def terminal_output() -> Iterator[None]:
    """Hold the shared terminal lock for a coordinated output operation."""
    with _TERMINAL_LOCK:
        yield
