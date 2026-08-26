from __future__ import annotations

import argparse
import math
from importlib.metadata import PackageNotFoundError, version


def package_version() -> str:
    """Return the installed package version without importing the package root."""
    try:
        return version("pystormtracker")
    except PackageNotFoundError:
        return "unknown"


def add_cli_observability_options(
    parser: argparse.ArgumentParser,
    *,
    version_string: str | None = None,
) -> None:
    """Add the shared ``-v``/``-vv`` and ``-V`` CLI options to one parser."""
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {version_string or package_version()}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="Increase operational logging verbosity (-v: INFO, -vv: DEBUG).",
    )


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer for argparse."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    """Parse a strictly positive float for argparse."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    """Parse a nonnegative float for argparse."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def fraction(value: str) -> float:
    """Parse a float in the closed interval from zero to one."""
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


def finite_float(value: str) -> float:
    """Parse a finite float for argparse."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed
