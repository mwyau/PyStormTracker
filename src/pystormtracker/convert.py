from __future__ import annotations

import argparse

from .io.format import infer_format, load_tracks, save_tracks
from .io.json import infer_track_type


def setup_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Sets up the argument parser for the convert command."""
    parser = subparsers.add_parser(
        "convert",
        description=(
            "Convert PyStormTracker trajectory data between supported formats "
            "(TrackJSON, GeoJSON, IMILAST, Hodges). Formats are automatically inferred "
            "from file extensions if omitted."
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
        choices=["json", "geojson", "imilast"],
        default=None,
        help="Input file format (auto-inferred from extension if omitted)",
    )
    parser.add_argument(
        "-F",
        "--out-format",
        choices=["json", "geojson", "imilast", "hodges"],
        default=None,
        help="Output file format (auto-inferred from extension if omitted)",
    )
    parser.add_argument(
        "-v", "--var", help="Override track variable / type (e.g., msl or vo)"
    )
    parser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    """Main entry point for the convert command."""
    in_fmt = args.in_format or infer_format(args.input)
    out_fmt = args.out_format or infer_format(args.output)

    print(f"Reading {args.input} (format: {in_fmt})...")
    tracks = load_tracks(args.input, format=args.in_format)

    # Track Variable / Type Override
    if args.var:
        tracks.track_type = args.var.lower()
    else:
        tracks.track_type = infer_track_type(tracks)

    # Normalize a known primary variable name after importing text formats.
    if tracks.track_type != "unknown":
        matching_names = [
            name for name in tracks.vars if name.lower() == tracks.track_type
        ]
        if len(matching_names) == 1 and matching_names[0] != tracks.track_type:
            tracks.vars[tracks.track_type] = tracks.vars.pop(matching_names[0])
        elif "Intensity1" in tracks.vars:
            tracks.vars[tracks.track_type] = tracks.vars.pop("Intensity1")

    print(f"Loaded {len(tracks)} tracks. Detected type: {tracks.track_type}")
    print(f"Writing to {args.output} (format: {out_fmt})...")

    save_tracks(tracks, args.output, format=args.out_format)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_parser(subparsers)
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
