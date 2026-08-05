from __future__ import annotations

import argparse
import sys

from . import __version__, compare, convert, sample, track


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

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

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

    # Dispatch to the specific command's main function
    try:
        args.func(args)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"stormtracker: error: {exc}\n")


if __name__ == "__main__":
    main()
