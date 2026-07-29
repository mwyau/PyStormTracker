from __future__ import annotations

import argparse
import json

from .io.imilast import read_imilast
from .io.json import read_json, write_json
from .metrics.compare import match_tracks
from .models.tracks import Tracks


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
        type=float,
        default=440.0,
        help="Maximum mean geodetic distance (km) allowed for a match.",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.1,
        help="Minimum overlap ratio required for a match.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output match mapping as JSON to stdout."
    )
    parser.set_defaults(func=main)


def _load_tracks(path: str) -> Tracks:
    """Helper to load tracks from either JSON or Imilast format."""
    if path.endswith(".json"):
        return read_json(path)
    # Default to imilast for .txt or other extensions
    return read_imilast(path)


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the compare command.

    Loads a reference set and a comparison set of tracks, performs spatial and
    temporal matching, and optionally outputs the matched tracks or a mapping.
    """
    print(f"Loading reference tracks from {args.ref}...")
    tracks_ref = _load_tracks(args.ref)

    print(f"Loading comparison tracks from {args.comp}...")
    tracks_comp = _load_tracks(args.comp)

    print(
        f"Matching tracks (max_dist={args.max_dist} km, "
        f"min_overlap={args.min_overlap})..."
    )
    matches = match_tracks(
        tracks_ref,
        tracks_comp,
        max_dist_km=args.max_dist,
        min_overlap_fraction=args.min_overlap,
    )

    print(f"Matched {len(matches)} out of {len(tracks_comp)} comparison tracks.")

    if args.json:
        print("\nMatch Mapping (comp_id -> ref_id):")
        print(json.dumps(matches, indent=2))

    if args.output:
        print(f"Filtering comparison tracks and saving to {args.output}...")
        # Create a new Tracks object containing only matched tracks
        matched_tracks = Tracks()
        matched_ids = set(matches.keys())

        for track in tracks_comp:
            if track.track_id in matched_ids:
                matched_tracks.append(track)

        write_json(matched_tracks, args.output)
        print("Done!")
