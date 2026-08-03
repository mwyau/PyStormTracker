from __future__ import annotations

import argparse
from pathlib import Path

from .io.hodges import read_hodges
from .io.imilast import read_imilast
from .models.tracks import Tracks, TracksMetadata
from .models.units import canonical_unit_for


def generate_html(tracks: Tracks, outfile: str | Path, split: bool = False) -> None:
    raise ValueError("interactive JSON conversion is unavailable on this branch")


def setup_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Sets up the argument parser for the convert command."""
    parser = subparsers.add_parser(
        "convert",
        description=(
            "Convert PyStormTracker data between formats and generate "
            "text trajectory formats."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument(
        "-o", "--out", "--output", dest="output", required=True, help="Output file path"
    )
    parser.add_argument(
        "-f",
        "--in-format",
        choices=["imilast", "hodges"],
        required=True,
        help="Input file format",
    )
    parser.add_argument(
        "-F",
        "--out-format",
        choices=["imilast", "hodges"],
        required=True,
        help="Output file format",
    )
    parser.add_argument(
        "-v", "--var", help="Override track variable / type (e.g., msl or vo)"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Retained for compatibility; HTML output is unavailable on this branch.",
    )
    parser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the convert command.

    Supports conversion between the IMILAST and Hodges text formats.
    """
    if args.split:
        raise ValueError("--split requires the JSON format branch")

    print(f"Reading {args.input} (format: {args.in_format})...")

    if args.in_format == "imilast":
        tracks = read_imilast(args.input)
    else:
        tracks = read_hodges(args.input)

    if args.var:
        variable_name = args.var.lower()
        variables = dict(tracks.variables)
        if variable_name not in variables and "Intensity1" in variables:
            variables[variable_name] = variables.pop("Intensity1")
        units = dict(tracks.units)
        if variable_name in variables:
            units[variable_name] = canonical_unit_for(variable_name) or units.get(
                "Intensity1", "1"
            )
        tracks = Tracks(
            ids=tracks.ids,
            offsets=tracks.offsets,
            times=tracks.times,
            lats=tracks.lats,
            lons=tracks.lons,
            variables=variables,
            metadata=TracksMetadata(variable_name, tracks.mode, units),
        )

    print(f"Loaded {len(tracks)} tracks. Primary variable: {tracks.primary_var}")
    print(f"Writing to {args.output} (format: {args.out_format})...")

    tracks.write(args.output, format=args.out_format)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_parser(subparsers)
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
