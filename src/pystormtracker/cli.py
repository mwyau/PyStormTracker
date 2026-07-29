from __future__ import annotations

import argparse
import sys

from . import compare, convert, sample, track


def main() -> None:
    """
    Main entry point for the PyStormTracker CLI.

    Routes execution to the appropriate subcommand (track, sample, convert, compare).
    """
    parser = argparse.ArgumentParser(
        prog="stormtracker",
        description="PyStormTracker: A High-Performance Cyclone Tracker in Python",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
