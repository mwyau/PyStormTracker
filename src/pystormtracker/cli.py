from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from . import __version__, compare, convert, sample, track
from .backends import defer_dask_interrupt_cleanup, drain_pending_dask_executors
from .utils.cli import add_cli_observability_options
from .utils.logging import (
    configure_cli_logging,
    interrupt_active_progress,
    write_terminal,
)

LOGGER = logging.getLogger(__name__)


def _first_interrupt(signum: int, frame: object) -> None:
    """Raise the normal interrupt in the CLI's main thread."""
    del signum, frame
    raise KeyboardInterrupt


def _emergency_interrupt(signum: int, frame: object) -> None:
    """Terminate immediately on the second interrupt during executor drain."""
    del signum, frame
    os._exit(130)


@contextmanager
def _cli_interrupt_handlers() -> Iterator[None]:
    """Install first-interrupt cancellation and second-interrupt emergency exit."""
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _first_interrupt)
    try:
        yield
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, _emergency_interrupt)
        interrupt_active_progress()
        write_terminal(
            "Interrupted; cancelling pending work.\n"
            "Waiting for active worker tasks to finish.\n"
            "Press Ctrl-C again to terminate immediately.\n"
        )
        drain_pending_dask_executors()
        raise
    finally:
        signal.signal(signal.SIGINT, previous)


def main() -> None:
    """
    Main entry point for the PyStormTracker CLI.

    Routes execution to the appropriate subcommand (track, sample, convert, compare).
    """
    parser = argparse.ArgumentParser(
        prog="stormtracker",
        description=(
            "PyStormTracker: A High-Performance Cyclone Tracker in Python "
            f"(v{__version__})"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_cli_observability_options(parser, version_string=__version__)
    parser.set_defaults(verbose=0)

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
        help="The command to run.",
    )

    # Register subcommands
    track.setup_parser(subparsers)
    sample.setup_parser(subparsers)
    convert.setup_parser(subparsers)
    compare.setup_parser(subparsers)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    configure_cli_logging(args.verbose)

    # Dispatch to the specific command's main function.
    try:
        with _cli_interrupt_handlers(), defer_dask_interrupt_cleanup():
            args.func(args)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        drain_pending_dask_executors()
        parser.exit(2, f"stormtracker: error: {exc}\n")
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        drain_pending_dask_executors()
        if args.verbose >= 2:
            LOGGER.exception("stormtracker: unexpected error")
            parser.exit(1)
        parser.exit(1, f"stormtracker: error: {exc}\n")


if __name__ == "__main__":
    main()
