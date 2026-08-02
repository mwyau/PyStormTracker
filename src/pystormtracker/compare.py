"""Command-line interface for trajectory intercomparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io.imilast import read_imilast
from .io.json import read_json, write_json
from .metrics.compare import TrackComparison, TrackComparisonConfig, compare_tracks
from .models.tracks import Tracks
from .utils.cli import fraction, positive_float


def setup_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Set up the comparison command parser."""
    parser = subparsers.add_parser(
        "compare",
        description=(
            "Compare reference tracks with their closest eligible candidates "
            "using temporal overlap and mean geodesic separation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-r",
        "--ref",
        dest="reference",
        required=True,
        help="Reference track file (JSON or IMILAST).",
    )
    parser.add_argument(
        "-c",
        "--cand",
        dest="candidate",
        required=True,
        help="Candidate track file (JSON or IMILAST).",
    )
    parser.add_argument(
        "-s",
        "--max-sep",
        dest="max_mean_separation",
        type=positive_float,
        default=2.0,
        help="Maximum mean great-circle separation in degrees.",
    )
    parser.add_argument(
        "-l",
        "--min-overlap",
        type=fraction,
        default=0.6,
        help="Minimum overlap fraction: 2 * overlap / (n_ref + n_candidate).",
    )
    parser.add_argument(
        "-v",
        "--var",
        help="Common trajectory variable used for vorticity/intensity statistics.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["auto", "min", "max"],
        default="auto",
        help="Extremum mode for peak intensity calculation ('auto', 'min', 'max').",
    )
    parser.add_argument(
        "-o", "--out", dest="report", help="Write the full comparison report as JSON."
    )
    parser.add_argument(
        "-M",
        "--matched-out",
        dest="matched_candidate_output",
        help="Write candidate tracks selected by at least one reference as JSON.",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Print the full comparison report as JSON.",
    )
    parser.set_defaults(func=main)


def _load_tracks(path: str) -> Tracks:
    """Load a JSON or IMILAST trajectory file."""
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return read_json(path)
    if suffix in (".txt", ".dat"):
        return read_imilast(path)
    raise ValueError(f"Unsupported track file extension: '{suffix or '<none>'}'")


def _matched_candidate_tracks(tracks: Tracks, candidate_ids: set[int]) -> Tracks:
    """Return candidate tracks selected by at least one reference track."""
    matched = Tracks(track_type=tracks.track_type)
    for track in tracks:
        if track.track_id in candidate_ids:
            matched.append(track)
    return matched


def _print_summary(result: TrackComparison) -> None:
    """Print the concise human-readable comparison summary."""
    print(
        "Matched "
        f"{result.match_count} of {result.reference_count} reference and "
        f"{result.candidate_count} candidate tracks."
    )
    print(
        f"Reference coverage: {result.reference_coverage:.1%}; candidate coverage: "
        f"{result.candidate_coverage:.1%}."
    )


def main(args: argparse.Namespace) -> None:
    """Run the trajectory comparison command."""

    def log(message: str) -> None:
        print(message, file=sys.stderr if args.json else sys.stdout)

    log(f"Loading reference tracks from {args.reference}...")
    reference = _load_tracks(args.reference)
    log(f"Loading candidate tracks from {args.candidate}...")
    candidate = _load_tracks(args.candidate)

    config = TrackComparisonConfig(
        max_mean_separation_deg=args.max_mean_separation,
        min_overlap_fraction=args.min_overlap,
        var=args.var,
        mode=args.mode,
    )
    result = compare_tracks(reference, candidate, config=config)
    result_json = result.to_dict()

    if args.json:
        print(json.dumps(result_json, indent=2, sort_keys=True))
    else:
        _print_summary(result)

    if args.report:
        log(f"Writing comparison report to {args.report}...")
        with open(args.report, "w", encoding="utf-8") as report_file:
            json.dump(result_json, report_file, indent=2, sort_keys=True)

    if args.matched_candidate_output:
        log(f"Writing matched candidate tracks to {args.matched_candidate_output}...")
        candidate_ids = {match.candidate_id for match in result.matches}
        write_json(
            _matched_candidate_tracks(candidate, candidate_ids),
            args.matched_candidate_output,
        )
