from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io.imilast import read_imilast
from .io.json import read_json, write_json
from .metrics.compare import match_tracks
from .models.tracks import Tracks
from .utils.cli import fraction, positive_float


def setup_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Sets up the argument parser for the compare command."""
    parser = subparsers.add_parser(
        "compare",
        description=(
            "Compare tracks from a comparison set to a reference set based on "
            "spatial proximity and temporal overlap."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ref", required=True, help="Reference track file (JSON or Imilast)."
    )
    parser.add_argument(
        "--comp", required=True, help="Comparison track file (JSON or Imilast)."
    )
    parser.add_argument(
        "-o", "--output", help="Output filtered comparison track file (JSON)."
    )
    parser.add_argument(
        "--max-dist",
        type=positive_float,
        default=440.0,
        help="Maximum mean geodetic distance (km) allowed for a match.",
    )
    parser.add_argument(
        "--min-overlap",
        type=fraction,
        default=0.1,
        help="Minimum overlap ratio required for a match.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output match mapping as JSON to stdout."
    )
    parser.set_defaults(func=main)


def _load_tracks(path: str) -> Tracks:
    """Helper to load tracks from either JSON or Imilast format."""
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return read_json(path)
    if suffix in (".txt", ".dat"):
        return read_imilast(path)
    raise ValueError(f"Unsupported track file extension: '{suffix or '<none>'}'")


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the compare command.

    Loads a reference set and a comparison set of tracks, performs spatial and
    temporal matching, and optionally outputs the matched tracks or a mapping.
    """

    def log(message: str) -> None:
        print(message, file=sys.stderr if args.json else sys.stdout)

    log(f"Loading reference tracks from {args.ref}...")
    tracks_ref = _load_tracks(args.ref)

    log(f"Loading comparison tracks from {args.comp}...")
    tracks_comp = _load_tracks(args.comp)

    log(
        f"Matching tracks (max_dist={args.max_dist} km, "
        f"min_overlap={args.min_overlap})..."
    )
    matches = match_tracks(
        tracks_ref,
        tracks_comp,
        max_dist_km=args.max_dist,
        min_overlap_fraction=args.min_overlap,
    )

    log(f"Matched {len(matches)} out of {len(tracks_comp)} comparison tracks.")

    if args.json:
        print(json.dumps(matches, indent=2))

    if args.output:
        log(f"Filtering comparison tracks and saving to {args.output}...")
        # Create a new Tracks object containing only matched tracks
        matched_tracks = Tracks(track_type=tracks_comp.track_type)
        matched_ids = set(matches.keys())

        for track in tracks_comp:
            if track.track_id in matched_ids:
                matched_tracks.append(track)

        write_json(matched_tracks, args.output)
        log("Done!")
