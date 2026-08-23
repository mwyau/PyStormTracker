#!/usr/bin/env python3
"""Summarize measured PST runs and compactly compare them with TRACK."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr
from run_pst_repeat import CASES

from pystormtracker.io import load_tracks
from pystormtracker.metrics.compare import (
    TrackComparison,
    TrackComparisonConfig,
    compare_tracks,
)
from pystormtracker.models.tracks import Tracks

TRACK_RESULT_BASE: Final[Path] = Path(
    "/home/albert/PyStormTracker-Validation/results/track_comparison-20260818/repeats"
)
PST_RESULT_BASE: Final[Path] = Path(
    "/home/albert/PyStormTracker-Validation/results/pst_track_comparison-20260819-corrected"
)
OUTPUT_BASE: Final[Path] = Path("benchmarks/track_comparison/results")
TRACK_WORKFLOW_SECONDS: Final[dict[str, float]] = {
    "f320_to_t42_january": 4.949,
    "f320_to_t42_full_year": 59.722,
    "f320_to_f320_january": 165.647,
    "f320_to_f320_full_year": 1973.061,
    "regular-2p5-dec": 12.697,
    "regular-2p5-season": 29.288,
    "regular-0p25-dec": 555.266,
    "regular-0p25-season": 1205.891,
}
METHODS: Final[tuple[str, ...]] = (
    "nearest",
    "mutual_nearest",
    "global_assignment",
)
PERFORMANCE_FIELDS: Final[tuple[str, ...]] = (
    "backend",
    "case",
    "frames",
    "target_grid",
    "frame_workers",
    "sht_threads",
    "mge_workers",
    "tracks",
    "points",
    "source_graph_prep_median_s",
    "frame_stage_run1_s",
    "frame_stage_run2_s",
    "frame_stage_run3_s",
    "frame_stage_median_s",
    "mge_stage_run1_s",
    "mge_stage_run2_s",
    "mge_stage_run3_s",
    "mge_stage_median_s",
    "merge_splice_median_s",
    "trackjson_write_median_s",
    "total_run1_s",
    "total_run2_s",
    "total_run3_s",
    "total_median_s",
    "gnu_wall_median_s",
    "gnu_user_median_s",
    "gnu_system_median_s",
    "gnu_max_rss_median_kb",
    "track_workflow_median_s",
    "pst_to_track_workflow_ratio",
)
PARITY_FIELDS: Final[tuple[str, ...]] = (
    "backend",
    "case",
    "method",
    "reference_tracks",
    "reference_points",
    "candidate_tracks",
    "candidate_points",
    "match_count",
    "unmatched_reference_count",
    "unmatched_candidate_count",
    "topology_identical_count",
    "topology_identical_fraction",
    "reference_coverage",
    "candidate_coverage",
    "unique_candidate_count",
    "reused_candidate_count",
    "reused_candidate_assignments",
    "mutual_agreement",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "median_matched_mean_separation_km",
    "p95_matched_mean_separation_km",
    "maximum_matched_point_separation_km",
    "comparison_source_backend",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pst-base", type=Path, default=PST_RESULT_BASE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_BASE,
        help="Checked-in CSV/JSON directory.",
    )
    return parser.parse_args()


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _read_time_file(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, value = line.split("=", 1)
        values[name] = float(value)
    if values.get("exit_status") != 0.0:
        raise RuntimeError(f"nonzero benchmark exit in {path}: {values}")
    return values


def _load_reference(case_name: str) -> Tracks:
    product_file = TRACK_RESULT_BASE / case_name / "run1" / "products.tsv"
    track_path: Path | None = None
    for line in product_file.read_text(encoding="utf-8").splitlines():
        product, sign, path, _size, _counts = line.split("\t")
        if product == "tr_trs" and sign == "neg":
            track_path = Path(path)
            break
    if track_path is None:
        raise RuntimeError(f"no TRACK tr_trs_neg product in {product_file}")

    case = CASES[case_name]
    with xr.open_dataset(case.source) as dataset:
        data = dataset["msl"]
        time_dimension = next(
            name for name in ("time", "valid_time") if name in data.dims
        )
        times = np.asarray(
            data.sel({time_dimension: slice(case.start, case.end)})[
                time_dimension
            ].values,
            dtype="datetime64[ms]",
        )
    return load_tracks(
        track_path,
        format="track",
        primary_variable="Intensity1",
        mode="min",
        track_numeric_time="frame_index",
        track_frame_times=times,
    )


def _candidate_paths(base: Path, case_name: str) -> list[Path]:
    return [
        base / case_name / run / "output.trackjson" for run in ("run1", "run2", "run3")
    ]


def _load_candidates(base: Path, case_name: str) -> Tracks:
    paths = _candidate_paths(base, case_name)
    candidates = [load_tracks(path, format="json") for path in paths]
    if not (candidates[0] == candidates[1] == candidates[2]):
        raise RuntimeError(
            f"repeated PST outputs are not identical for {case_name} in {base}"
        )
    return candidates[0]


def _performance_row(base: Path, case_name: str, backend: str) -> dict[str, object]:
    metadata = [
        json.loads(
            (base / case_name / run / "metadata.json").read_text(encoding="utf-8")
        )
        for run in ("run1", "run2", "run3")
    ]
    gnu = [
        _read_time_file(base / case_name / run / "workflow.time")
        for run in ("run1", "run2", "run3")
    ]
    timing = [item["timing"] for item in metadata]
    frame_stage = [float(item["frame_stage_wall_seconds"]) for item in timing]
    mge_stage = [float(item["mge_segment_stage_wall_seconds"]) for item in timing]
    workflow = [float(item["total_end_to_end_wall_seconds"]) for item in timing]
    track_workflow = TRACK_WORKFLOW_SECONDS[case_name]
    hashes = [str(item["output"]["trackjson_sha256"]) for item in metadata]
    if len(set(hashes)) != 1:
        raise RuntimeError(
            f"TrackJSON hashes differ between repetitions for {case_name}"
        )
    row: dict[str, object] = {
        "backend": backend,
        "case": case_name,
        "frames": metadata[0]["input"]["frame_count"],
        "target_grid": metadata[0]["target_grid"]["name"],
        "frame_workers": metadata[0]["execution"]["resolved_frame_workers"],
        "sht_threads": metadata[0]["execution"]["resolved_sht_threads"],
        "mge_workers": metadata[0]["execution"]["resolved_mge_workers"],
        "tracks": metadata[0]["output"]["tracks"],
        "points": metadata[0]["output"]["points"],
        "source_graph_prep_median_s": _median(
            [
                float(item["source_open_graph_preparation_wall_seconds"])
                for item in timing
            ]
        ),
        "frame_stage_run1_s": frame_stage[0],
        "frame_stage_run2_s": frame_stage[1],
        "frame_stage_run3_s": frame_stage[2],
        "frame_stage_median_s": _median(frame_stage),
        "mge_stage_run1_s": mge_stage[0],
        "mge_stage_run2_s": mge_stage[1],
        "mge_stage_run3_s": mge_stage[2],
        "mge_stage_median_s": _median(mge_stage),
        "merge_splice_median_s": _median(
            [float(item["merge_splice_wall_seconds"]) for item in timing]
        ),
        "trackjson_write_median_s": _median(
            [float(item["trackjson_write_wall_seconds"]) for item in timing]
        ),
        "total_run1_s": workflow[0],
        "total_run2_s": workflow[1],
        "total_run3_s": workflow[2],
        "total_median_s": _median(workflow),
        "gnu_wall_median_s": _median([item["wall_seconds"] for item in gnu]),
        "gnu_user_median_s": _median([item["user_seconds"] for item in gnu]),
        "gnu_system_median_s": _median([item["system_seconds"] for item in gnu]),
        "gnu_max_rss_median_kb": _median([item["max_rss_kb"] for item in gnu]),
        "track_workflow_median_s": track_workflow,
        "pst_to_track_workflow_ratio": _median(workflow) / track_workflow,
    }
    return row


def _comparison_row(
    case_name: str, backend: str, comparison: TrackComparison
) -> dict[str, object]:
    matches = comparison.matches
    mean_separations = [match.mean_separation_km for match in matches]
    p95_separations = [match.p95_separation_km for match in matches]
    maximum_separations = [match.maximum_separation_km for match in matches]
    topology_fraction = (
        comparison.topology_identical_count / comparison.match_count
        if comparison.match_count
        else 0.0
    )
    values: dict[str, object] = {
        "backend": backend,
        "case": case_name,
        "method": comparison.matching,
        "reference_tracks": comparison.reference_count,
        "reference_points": "",
        "candidate_tracks": comparison.candidate_count,
        "candidate_points": "",
        "match_count": comparison.match_count,
        "unmatched_reference_count": comparison.unmatched_reference_count,
        "unmatched_candidate_count": comparison.unmatched_candidate_count,
        "topology_identical_count": comparison.topology_identical_count,
        "topology_identical_fraction": topology_fraction,
        "reference_coverage": comparison.reference_coverage
        if comparison.matching == "nearest"
        else "",
        "candidate_coverage": comparison.candidate_coverage
        if comparison.matching == "nearest"
        else "",
        "unique_candidate_count": comparison.unique_candidate_count
        if comparison.matching == "nearest"
        else "",
        "reused_candidate_count": comparison.reused_candidate_count
        if comparison.matching == "nearest"
        else "",
        "reused_candidate_assignments": comparison.reused_candidate_assignments
        if comparison.matching == "nearest"
        else "",
        "mutual_agreement": comparison.agreement
        if comparison.matching == "mutual_nearest"
        else "",
        "tp": comparison.tp if comparison.matching == "global_assignment" else "",
        "fp": comparison.fp if comparison.matching == "global_assignment" else "",
        "fn": comparison.fn if comparison.matching == "global_assignment" else "",
        "precision": comparison.precision
        if comparison.matching == "global_assignment"
        else "",
        "recall": comparison.recall
        if comparison.matching == "global_assignment"
        else "",
        "f1": comparison.f1 if comparison.matching == "global_assignment" else "",
        "median_matched_mean_separation_km": _median(mean_separations)
        if mean_separations
        else "",
        "p95_matched_mean_separation_km": _median(p95_separations)
        if p95_separations
        else "",
        "maximum_matched_point_separation_km": max(maximum_separations)
        if maximum_separations
        else "",
        "comparison_source_backend": backend,
    }
    return values


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    dask_base = args.pst_base
    serial_base = args.pst_base / "serial"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    performance_rows: list[dict[str, object]] = []
    parity_rows: list[dict[str, object]] = []
    comparisons: dict[str, dict[str, object]] = {}
    for case_name in CASES:
        dask_candidate = _load_candidates(dask_base, case_name)
        serial_candidate = _load_candidates(serial_base, case_name)
        if dask_candidate != serial_candidate:
            raise RuntimeError(
                f"Dask and serial canonical outputs differ for {case_name}"
            )
        reference = _load_reference(case_name)
        candidate_points = int(dask_candidate.times.size)
        reference_points = int(reference.times.size)
        for backend, base in (("dask", dask_base), ("serial", serial_base)):
            performance_rows.append(_performance_row(base, case_name, backend))
        for method in METHODS:
            comparison = compare_tracks(
                reference,
                dask_candidate,
                config=TrackComparisonConfig(
                    matching=method,
                    max_mean_separation_deg=2.0,
                    min_overlap_fraction=0.6,
                    variable=None,
                    mode="min",
                ),
            )
            row = _comparison_row(case_name, "dask", comparison)
            row["reference_points"] = reference_points
            row["candidate_points"] = candidate_points
            parity_rows.append(row)
            serial_row = dict(row)
            serial_row["backend"] = "serial"
            serial_row["comparison_source_backend"] = "dask"
            parity_rows.append(serial_row)
            comparisons[f"{case_name}:{method}"] = {
                key: value for key, value in row.items() if key not in ("backend",)
            }

    _write_csv(
        args.output_dir / "pst_performance.csv", PERFORMANCE_FIELDS, performance_rows
    )
    _write_csv(args.output_dir / "pst_vs_track.csv", PARITY_FIELDS, parity_rows)
    (args.output_dir / "pst_vs_track.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
