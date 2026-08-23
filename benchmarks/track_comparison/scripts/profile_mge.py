"""Profile one serial Hodges MGE link with optional native counters.

The profiler deliberately runs after detection and preprocessing.  It reports
the hot MGE stages without adding logging or counters to ordinary tracking.
The first link is an untimed Numba warmup; the reported link uses the same
detected frames and warmed native kernels.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from run_pst_repeat import CASES, _filter_data, _open_selected_data, _tracker

from pystormtracker.hodges.detections import HodgesCenterFrame
from pystormtracker.hodges.linker import (
    HodgesLinker,
    HodgesLinkInput,
    _MGEInitializationWorkspace,
)
from pystormtracker.hodges.tracker import HodgesTracker
from pystormtracker.models.geo import SpatialBounds, spatial_bounds_from_xarray
from pystormtracker.models.tracks import ProcessingStep, ResolvedDetectionMode, Tracks

STAT_PAIR_EVALUATIONS: Final[int] = 0
STAT_ACCEPTED_EXCHANGES: Final[int] = 1
STAT_TRACK_FAIL_CALLS: Final[int] = 2
STAT_FORWARD_SWEEPS: Final[int] = 3
STAT_BACKWARD_SWEEPS: Final[int] = 4
STAT_OUTER_ITERATIONS: Final[int] = 5
STAT_COUNT: Final[int] = 6


class _ProfiledHodgesLinker(HodgesLinker):
    """Hodges linker with benchmark-only stage timing and counters."""

    def __init__(
        self,
        *,
        w1: float,
        w2: float,
        dmax: float,
        phimax: float,
        mge_max_iterations: int,
        dmax_zones: NDArray[np.float64],
        adaptive_smoothness: NDArray[np.float64],
        missing_frame_parameters: NDArray[np.float64],
    ) -> None:
        super().__init__(
            w1=w1,
            w2=w2,
            dmax=dmax,
            phimax=phimax,
            mge_max_iterations=mge_max_iterations,
            dmax_zones=dmax_zones,
            adaptive_smoothness=adaptive_smoothness,
            missing_frame_parameters=missing_frame_parameters,
        )
        self.stats = np.zeros(STAT_COUNT, dtype=np.int64)
        self.timings: dict[str, float] = {}
        self.raw_feature_count = 0
        self.retained_feature_count = 0
        self.workspace_rows = 0
        self.workspace_real_rows = 0
        self.workspace_phantom_rows = 0
        self.workspace_real_cells = 0
        self.workspace_phantom_cells = 0
        self.split_calls = 0
        self.split_row_operations = 0
        self.constraint_filter_calls = 0
        self.directional_stage_count = 0
        self._in_directional = False

    def _time(self, name: str, started: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + (
            time.perf_counter() - started
        )

    def _filter_feature_points_numba(
        self,
        detections: Sequence[HodgesLinkInput],
        missing_input_counts: NDArray[np.int64] | None = None,
    ) -> list[HodgesCenterFrame]:
        self.raw_feature_count = sum(step.values.size for step in detections)
        started = time.perf_counter()
        filtered = super()._filter_feature_points_numba(
            detections,
            missing_input_counts,
        )
        self._time("prefilter_wall_seconds", started)
        self.retained_feature_count = sum(step.values.size for step in filtered)
        return filtered

    def _initialize_mge_workspace_numba(
        self,
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        step_offsets: NDArray[np.int64],
        missing_input_counts: NDArray[np.int64] | None = None,
    ) -> _MGEInitializationWorkspace:
        started = time.perf_counter()
        workspace = super()._initialize_mge_workspace_numba(
            features_lat,
            features_lon,
            step_offsets,
            missing_input_counts,
        )
        self._time("workspace_wall_seconds", started)
        assignments = workspace.assignments
        self.workspace_rows = assignments.shape[0]
        self.workspace_real_rows = int(
            np.count_nonzero(np.any(assignments != -1, axis=1))
        )
        self.workspace_phantom_rows = self.workspace_rows - self.workspace_real_rows
        self.workspace_real_cells = int(np.count_nonzero(assignments != -1))
        self.workspace_phantom_cells = int(assignments.size - self.workspace_real_cells)
        return workspace

    def _split_track_sections(  # type: ignore[override]
        self,
        tracks: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        started = time.perf_counter()
        split = HodgesLinker._split_track_sections(tracks)
        elapsed = time.perf_counter() - started
        if self._in_directional:
            self.timings["adaptive_split_wall_seconds"] = (
                self.timings.get("adaptive_split_wall_seconds", 0.0) + elapsed
            )
        else:
            self.timings["final_split_wall_seconds"] = (
                self.timings.get("final_split_wall_seconds", 0.0) + elapsed
            )
        self.split_calls += 1
        self.split_row_operations += max(0, split.shape[0] - tracks.shape[0])
        return split

    def _apply_track_constraint_filter(
        self,
        tracks: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        *,
        direction: Literal["forward", "backward"],
    ) -> NDArray[np.int64]:
        started = time.perf_counter()
        filtered = super()._apply_track_constraint_filter(
            tracks,
            features_lat,
            features_lon,
            direction=direction,
        )
        self._time("adaptive_filter_wall_seconds", started)
        self.constraint_filter_calls += 1
        return filtered

    def _run_mge_direction_until_stable(
        self,
        tracks: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        n_frames: int,
        missing_input_counts: NDArray[np.int64] | None = None,
        *,
        direction: Literal["forward", "backward"],
        profile_stats: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.int64], bool]:
        started = time.perf_counter()
        result = super()._run_mge_direction_until_stable(
            tracks,
            features_lat,
            features_lon,
            n_frames,
            missing_input_counts,
            direction=direction,
            profile_stats=self.stats,
        )
        self._time(f"{direction}_stage_wall_seconds", started)
        self.directional_stage_count += 1
        return result

    def _run_directional_mge(
        self,
        track_matrix: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        n_frames: int,
        missing_input_counts: NDArray[np.int64] | None = None,
        profile_stats: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        self._in_directional = True
        try:
            return super()._run_directional_mge(
                track_matrix,
                features_lat,
                features_lon,
                n_frames,
                missing_input_counts,
                profile_stats=self.stats,
            )
        finally:
            self._in_directional = False

    def _build_tracks(  # type: ignore[override]
        self,
        track_matrix: NDArray[np.int64],
        times_packed: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        features_val: NDArray[np.float64],
        feature_diagnostics: dict[str, NDArray[np.float64]],
        *,
        primary_variable: str,
        mode: ResolvedDetectionMode,
        bounds: SpatialBounds | None,
        units: dict[str, str],
        processing: tuple[ProcessingStep, ...],
    ) -> Tracks:
        started = time.perf_counter()
        tracks = HodgesLinker._build_tracks(
            track_matrix,
            times_packed,
            features_lat,
            features_lon,
            features_val,
            feature_diagnostics,
            primary_variable=primary_variable,
            mode=mode,
            bounds=bounds,
            units=units,
            processing=processing,
        )
        self._time("materialization_wall_seconds", started)
        return tracks


def _profile_link(
    detections: list[HodgesCenterFrame],
    tracker: HodgesTracker,
    bounds: SpatialBounds | None,
) -> tuple[_ProfiledHodgesLinker, Tracks]:
    linker = _ProfiledHodgesLinker(
        w1=tracker.w1,
        w2=tracker.w2,
        dmax=tracker.dmax,
        phimax=tracker.phimax,
        mge_max_iterations=tracker.mge_max_iterations,
        dmax_zones=tracker.dmax_zones,
        adaptive_smoothness=tracker.adaptive_smoothness,
        missing_frame_parameters=tracker.missing_frame_parameters,
    )
    started = time.perf_counter()
    tracks = linker.link(
        detections,
        primary_variable="msl",
        mode="min",
        bounds=bounds,
        unit="Pa",
    )
    linker.timings["link_wall_seconds"] = time.perf_counter() - started
    return linker, tracks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), default="f320_to_t42_january")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    case = CASES[args.case]
    dataset, selected, _metadata = _open_selected_data(case, "serial")
    try:
        if args.frames is not None:
            if args.frames < 4 or args.frames > case.frames:
                raise ValueError(f"--frames must be in [4, {case.frames}]")
            time_dim = "time" if "time" in selected.dims else "valid_time"
            selected = selected.isel({time_dim: slice(0, args.frames)})
        tracker = _tracker(
            "serial",
            frame_workers=None,
            sht_threads=1,
            mge_workers=None,
        )
        filtered = _filter_data(selected, case, "serial", 1)
        detections = tracker._detect_frames(
            filtered,
            primary_variable="msl",
            mode="min",
            threshold=-100.0,
        )
        bounds = spatial_bounds_from_xarray(filtered)
        _profile_link(detections, tracker, bounds)
        linker, tracks = _profile_link(detections, tracker, bounds)
    finally:
        dataset.close()

    report: dict[str, object] = {
        "case": args.case,
        "frames": len(detections),
        "mge_max_iterations": tracker.mge_max_iterations,
        "outer_iterations": int(linker.stats[STAT_OUTER_ITERATIONS]),
        "raw_feature_count": linker.raw_feature_count,
        "retained_feature_count": linker.retained_feature_count,
        "workspace_rows": linker.workspace_rows,
        "workspace_real_rows": linker.workspace_real_rows,
        "workspace_phantom_rows": linker.workspace_phantom_rows,
        "workspace_real_cells": linker.workspace_real_cells,
        "workspace_phantom_cells": linker.workspace_phantom_cells,
        "directional_stage_count": linker.directional_stage_count,
        "forward_sweeps": int(linker.stats[STAT_FORWARD_SWEEPS]),
        "backward_sweeps": int(linker.stats[STAT_BACKWARD_SWEEPS]),
        "pair_evaluations": int(linker.stats[STAT_PAIR_EVALUATIONS]),
        "accepted_exchanges": int(linker.stats[STAT_ACCEPTED_EXCHANGES]),
        "track_fail_calls": int(linker.stats[STAT_TRACK_FAIL_CALLS]),
        "split_calls": linker.split_calls,
        "split_row_operations": linker.split_row_operations,
        "constraint_filter_calls": linker.constraint_filter_calls,
        "track_count": len(tracks),
        "point_count": int(tracks.times.size),
        "timing": linker.timings,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
