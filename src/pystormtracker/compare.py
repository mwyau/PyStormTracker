"""Command-line interface for trajectory intercomparison."""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from .io.format import load_tracks
from .io.trackjson import write_trackjson
from .metrics.compare import (
    MatchingMethod,
    TrackComparison,
    TrackComparisonConfig,
    compare_tracks,
)
from .models.tracks import Tracks
from .utils.cli import add_cli_observability_options, fraction, positive_float

LOGGER = logging.getLogger(__name__)


def setup_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Set up the comparison command parser."""
    parser = subparsers.add_parser(
        "compare",
        description=(
            "Compare reference and candidate tracks using nearest, "
            "mutual-nearest, or global-assignment matching."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_cli_observability_options(parser)
    parser.add_argument(
        "-r",
        "--reference",
        dest="reference",
        required=True,
        help="Reference track file (TrackJSON, IMILAST, or Hodges/tdump).",
    )
    parser.add_argument(
        "-c",
        "--candidate",
        dest="candidate",
        required=True,
        help="Candidate track file (TrackJSON, IMILAST, or Hodges/tdump).",
    )
    parser.add_argument(
        "--matching",
        dest="matching",
        choices=["nearest", "mutual_nearest", "global_assignment"],
        default="nearest",
        help=(
            "Trajectory matching method ('nearest', 'mutual_nearest', "
            "'global_assignment')."
        ),
    )
    parser.add_argument(
        "-s",
        "--max-mean-separation",
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
        "--variable",
        dest="variable",
        help="Common trajectory variable used for vorticity/intensity statistics.",
    )
    parser.add_argument(
        "-m",
        "--detection-mode",
        dest="detection_mode",
        choices=["auto", "min", "max"],
        default="auto",
        help="Extremum mode for peak intensity calculation ('auto', 'min', 'max').",
    )
    parser.add_argument(
        "-o",
        "--output-report",
        dest="report",
        help="Write the full comparison report as JSON.",
    )
    parser.add_argument(
        "-M",
        "--matched-output",
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
    """Load a supported trajectory file."""
    return load_tracks(path)


def _matched_candidate_tracks(tracks: Tracks, candidate_ids: set[int]) -> Tracks:
    """Return candidate tracks selected by at least one reference track."""
    indices = np.asarray(
        [
            index
            for index, track in enumerate(tracks)
            if track.track_id in candidate_ids
        ],
        dtype=np.int64,
    )
    return tracks.subset(indices)


def _print_summary(result: TrackComparison) -> None:
    """Print the concise human-readable comparison summary."""
    if result.matching == "nearest":
        print(
            f"Matched {result.match_count} of {result.reference_count} "
            f"reference tracks against {result.candidate_count} candidate tracks."
        )
        print(f"Reference coverage: {result.reference_coverage:.1%}")
        print(f"Candidate coverage: {result.candidate_coverage:.1%}")
        print(
            f"Unique candidates: {result.unique_candidate_count}; "
            f"Reused candidates: {result.reused_candidate_count} "
            f"({result.reused_candidate_assignments} duplicate assignments)"
        )
    elif result.matching == "mutual_nearest":
        print(
            f"Mutual matches: {result.match_count} "
            f"(Ref: {result.reference_count}, Cand: {result.candidate_count})"
        )
        print(f"Agreement: {result.agreement:.1%}")
        print(
            f"Unmatched reference: {result.unmatched_reference_count}; "
            f"Unmatched candidate: {result.unmatched_candidate_count}"
        )
    elif result.matching == "global_assignment":
        print(
            f"Assigned pairs: {result.match_count} "
            f"(Ref: {result.reference_count}, Cand: {result.candidate_count})"
        )
        print(f"Precision: {result.precision:.1%}")
        print(f"Recall: {result.recall:.1%}")
        print(f"F1: {result.f1:.1%}")
        print(
            f"TP: {result.tp}, FP: {result.fp}, FN: {result.fn}; "
            f"Topology-identical pairs: {result.topology_identical_count}"
        )


def main(args: argparse.Namespace) -> None:
    """Run the trajectory comparison command."""

    LOGGER.info("Loading reference tracks from %s", args.reference)
    reference = _load_tracks(args.reference)
    LOGGER.info("Loading candidate tracks from %s", args.candidate)
    candidate = _load_tracks(args.candidate)

    matching_method: MatchingMethod = getattr(args, "matching", "nearest")
    config = TrackComparisonConfig(
        matching=matching_method,
        max_mean_separation_deg=args.max_mean_separation,
        min_overlap_fraction=args.min_overlap,
        variable=args.variable,
        mode=args.detection_mode,
    )
    result = compare_tracks(reference, candidate, config=config)
    result_json = result.to_dict()

    if args.json:
        print(json.dumps(result_json, indent=2, sort_keys=True))
    else:
        _print_summary(result)

    if args.report:
        LOGGER.info("Writing comparison report to %s", args.report)
        with open(args.report, "w", encoding="utf-8") as report_file:
            json.dump(result_json, report_file, indent=2, sort_keys=True)

    if args.matched_candidate_output:
        LOGGER.info(
            "Writing matched candidate tracks to %s", args.matched_candidate_output
        )
        candidate_ids = {match.candidate_id for match in result.matches}
        write_trackjson(
            _matched_candidate_tracks(candidate, candidate_ids),
            args.matched_candidate_output,
        )
