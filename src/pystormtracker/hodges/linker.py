from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..models.geo import DEG_TO_RAD, SpatialBounds, geod_dist
from ..models.time import encode_time_values
from ..models.tracker import CenterFrame
from ..models.tracks import (
    ProcessingStep,
    ResolvedDetectionMode,
    Tracks,
    TracksMetadata,
    _TracksBuilder,
)
from ..models.units import canonical_unit_for
from . import constants
from .detections import HodgesCenterFrame
from .detector import DUFF_FEATURE_CUTOFF
from .mge import (
    _NO_PROFILE_STATS,
    _backward_mge_sweep,
    _compute_adaptive_phimax,
    _filter_feature_points_native,
    _forward_mge_sweep,
    _initialize_mge_workspace_native,
    _select_regional_dmax,
    geod_dev,
)

type MGEDirection = Literal["forward", "backward"]
type HodgesLinkInput = CenterFrame | HodgesCenterFrame


class _MGEInitializationWorkspace(NamedTuple):
    """TRACK-shaped real/phantom workspace before MGE exchange sweeps."""

    assignments: NDArray[np.int64]


class HodgesLinker:
    """Implement the Modified Greedy Exchange (MGE) tracking algorithm.

    Hodges (1999) establishes the regional upper-bound displacement and
    adaptive track-smoothness lineage.  TRACK 1.5.4 establishes exact
    initialization, restart, missing-frame, failure, splitting, and bounded
    iteration semantics; the source map points to ``src/mge_tracks.c``,
    ``src/initialize_mge.c``, ``src/geod_dev.c``, ``src/track_fail.c``, and
    ``src/track_split.c``:
    https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/src/mge_tracks.c
    https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/src/initialize_mge.c
    https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/src/geod_dev.c

    PST supplies the packed-data integration and execution orchestration;
    those engineering layers are not scientific provenance.
    """

    def __init__(
        self,
        w1: float = constants.W1_DEFAULT,
        w2: float = constants.W2_DEFAULT,
        dmax: float = constants.DMAX_DEFAULT,
        phimax: float = constants.PHIMAX_DEFAULT,
        mge_max_iterations: int = constants.MGE_MAX_ITERATIONS_DEFAULT,
        dmax_zones: NDArray[np.float64] | None = None,
        adaptive_smoothness: NDArray[np.float64] | None = None,
        missing_frame_parameters: NDArray[np.float64] | None = None,
        time_step_ms: int | None = None,
    ) -> None:
        """
        Initialize the MGE linker.

        Args:
            w1, w2: Weights for the cost function.
            adaptive_smoothness: Piecewise linear adaptive smoothness parameters (2xN).
            missing_frame_parameters: TRACK-style ``(dmax, phimax)`` rows
                selected by known missing-input-frame count.
            time_step_ms: Expected source-input cadence in milliseconds.
                It is required with multiple missing-frame parameter rows;
                otherwise the smallest positive observed cadence is used.
        """
        if mge_max_iterations <= 0:
            raise ValueError("mge_max_iterations must be positive")

        dmax_zones_arr = (
            dmax_zones.copy()
            if dmax_zones is not None
            else constants.DEFAULT_DMAX_ZONES.copy()
        )
        adaptive_smoothness_arr = (
            adaptive_smoothness.copy()
            if adaptive_smoothness is not None
            else constants.DEFAULT_ADAPTIVE_SMOOTHNESS.copy()
        )
        self._validate_dmax_zones(dmax_zones_arr)
        self._validate_adaptive_smoothness(adaptive_smoothness_arr)
        parameters = self._validate_missing_frame_parameters(
            missing_frame_parameters,
            dmax=dmax,
            phimax=phimax,
        )
        if time_step_ms is not None and time_step_ms <= 0:
            raise ValueError("time_step_ms must be positive")
        if parameters.shape[0] > 1 and (
            dmax_zones_arr.shape[0] > 0 or adaptive_smoothness_arr.shape[1] > 0
        ):
            raise ValueError(
                "multiple missing-frame parameter sets require dmax_zones and "
                "adaptive_smoothness to be disabled; per-parameter zone and "
                "adaptive tables are not implemented"
            )
        if parameters.shape[0] > 1 and time_step_ms is None:
            raise ValueError(
                "time_step_ms is required with multiple missing-frame parameter sets"
            )
        if adaptive_smoothness_arr.shape[1] == 4:
            parameters = parameters.copy()
            parameters[:, 1] = np.maximum(
                parameters[:, 1],
                np.max(adaptive_smoothness_arr[1]),
            )

        self.w1 = w1
        self.w2 = w2
        self.dmax_parameters = parameters[:, 0]
        self.phimax_parameters = parameters[:, 1]
        self.dmax = (
            float(np.max(dmax_zones_arr[:, 4]))
            if dmax_zones_arr.shape[0] > 0
            else float(self.dmax_parameters[0])
        )
        self.phimax = max(
            float(self.phimax_parameters[0]),
            float(np.max(adaptive_smoothness_arr[1]))
            if adaptive_smoothness_arr.shape[1] == 4
            else float(self.phimax_parameters[0]),
        )
        self.mge_max_iterations = mge_max_iterations
        self.dmax_zones = dmax_zones_arr
        self.adaptive_smoothness = adaptive_smoothness_arr
        self.time_step_ms = time_step_ms

    def link(
        self,
        detections: Sequence[HodgesLinkInput],
        *,
        primary_variable: str,
        mode: ResolvedDetectionMode,
        bounds: SpatialBounds | None = None,
        unit: str | None = None,
        processing: tuple[ProcessingStep, ...] = (),
    ) -> Tracks:
        """
        Links raw detections into trajectories using MGE optimization.

        Args:
            detections: List of (time, lats, lons, values) for each frame.

        Returns:
            A Tracks object containing the optimized trajectories.
        """
        normalized_detections = self._normalize_detections(detections)
        diagnostic_units = self._diagnostic_units(
            normalized_detections,
            primary_variable,
            unit,
        )
        units = {primary_variable: unit or canonical_unit_for(primary_variable) or "1"}
        units.update(diagnostic_units)

        n_frames = len(normalized_detections)
        if n_frames == 0:
            return Tracks.empty(
                TracksMetadata(
                    primary_variable,
                    mode,
                    units,
                    bounds,
                    processing,
                )
            )

        times_packed = encode_time_values([step.time for step in normalized_detections])
        missing_input_counts = self._infer_missing_input_counts(times_packed)
        for step in normalized_detections:
            self._validate_zone_coverage(step.latitudes, step.longitudes)

        # 1. Apply TRACK's pre-initialization feature-population filter, then
        # flatten retained features for mapping to the MGE workspace.
        filtered_detections = self._filter_feature_points_numba(
            normalized_detections,
            missing_input_counts,
        )
        all_lats: list[float] = []
        all_lons: list[float] = []
        all_vals: list[float] = []
        all_diagnostic_values: dict[str, list[float]] = {
            name: [] for name in diagnostic_units
        }
        step_offsets = np.zeros(n_frames + 1, dtype=np.int64)
        for i, step in enumerate(filtered_detections):
            all_lats.extend(step.latitudes)
            all_lons.extend(step.longitudes)
            all_vals.extend(step.values)
            for name, diagnostic_values in all_diagnostic_values.items():
                diagnostic_values.extend(step.diagnostics[name])
            step_offsets[i + 1] = step_offsets[i] + step.values.size

        features_lat = np.array(all_lats, dtype=np.float64)
        features_lon = np.array(all_lons, dtype=np.float64)
        features_val = np.array(all_vals, dtype=np.float64)
        feature_diagnostics = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in all_diagnostic_values.items()
        }
        n_features = len(features_lat)

        if n_features == 0:
            return Tracks.empty(
                TracksMetadata(
                    primary_variable,
                    mode,
                    units,
                    bounds,
                    processing,
                )
            )

        # 2. Source-shaped nearest-neighbor initialization, including paired
        # all-phantom rows that MGE may exchange with real rows.
        workspace = self._initialize_mge_workspace_numba(
            features_lat,
            features_lon,
            step_offsets,
            missing_input_counts,
        )
        track_matrix = workspace.assignments

        # 3. Directional MGE Optimization
        # Forward and backward passes alternate based on whether exchanges
        # occurred. This is the Hodges (1999) adaptive-constraint lineage;
        # exact control flow is sourced to TRACK 1.5.4 MGE implementation.
        track_matrix = self._run_directional_mge(
            track_matrix,
            features_lat,
            features_lon,
            n_frames,
            missing_input_counts,
        )

        # TRACK's track_split() runs once after the final MGE outer round.
        # It splits real sections around phantom gaps while preserving the
        # paired all-phantom workspace rows used during MGE.
        if n_frames > 3:
            track_matrix = self._split_track_sections(track_matrix)

        # 4. Build final Tracks output
        return self._build_tracks(
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

    @staticmethod
    def _build_tracks(
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
        """Materialize packed tracks by shared feature index after MGE."""
        builder = _TracksBuilder(
            TracksMetadata(primary_variable, mode, units, bounds, processing)
        )

        for row in range(track_matrix.shape[0]):
            feature_indices = track_matrix[row, :]
            valid_mask = feature_indices != -1

            if not np.any(valid_mask):
                continue

            track_times = times_packed[valid_mask]
            valid_feats = feature_indices[valid_mask]

            track_lats = features_lat[valid_feats]
            track_lons = features_lon[valid_feats]
            track_vals = features_val[valid_feats]

            track_variables: dict[str, NDArray[np.float64]] = {
                primary_variable: track_vals,
            }
            for name, values in feature_diagnostics.items():
                track_variables[name] = values[valid_feats]
            builder.add_track(
                row + 1,
                track_times,
                track_lats,
                track_lons,
                track_variables,
            )

        return builder.finish()

    @staticmethod
    def _normalize_detections(
        detections: Sequence[HodgesLinkInput],
    ) -> list[HodgesCenterFrame]:
        """Lift generic raw detections into Hodges aligned feature tables."""
        normalized: list[HodgesCenterFrame] = []
        for step in detections:
            if isinstance(step, HodgesCenterFrame):
                normalized.append(step)
            else:
                normalized.append(
                    HodgesCenterFrame(
                        step.time,
                        step.latitudes,
                        step.longitudes,
                        step.values,
                    )
                )
        return normalized

    @staticmethod
    def _diagnostic_units(
        detections: Sequence[HodgesCenterFrame],
        primary_variable: str,
        unit: str | None,
    ) -> dict[str, str]:
        """Validate a stable diagnostic schema and resolve public units."""
        if not detections:
            return {}
        first = detections[0]
        names = tuple(first.diagnostics)
        if primary_variable in first.diagnostics:
            raise ValueError("Hodges diagnostic name conflicts with primary variable")
        primary_unit = unit or canonical_unit_for(primary_variable) or "1"
        resolved = {
            name: first.diagnostic_units[name] or primary_unit for name in names
        }
        for step in detections[1:]:
            if tuple(step.diagnostics) != names:
                raise ValueError(
                    "Hodges diagnostic columns must be identical per frame"
                )
            if primary_variable in step.diagnostics:
                raise ValueError(
                    "Hodges diagnostic name conflicts with primary variable"
                )
            for name in names:
                candidate_unit = step.diagnostic_units[name] or primary_unit
                if candidate_unit != resolved[name]:
                    raise ValueError(
                        "Hodges diagnostic units must be identical per frame"
                    )
        return resolved

    @staticmethod
    def _validate_dmax_zones(zones: NDArray[np.float64]) -> None:
        """Validate the compact in-memory representation of TRACK zones."""
        if zones.ndim != 2 or zones.shape[1] != 5:
            raise ValueError("dmax_zones must have shape (n, 5)")
        if not np.all(np.isfinite(zones)):
            raise ValueError("dmax_zones must contain only finite values")
        if np.any(zones[:, 4] <= 0.0):
            raise ValueError("dmax_zones displacement limits must be positive")

    @staticmethod
    def _validate_adaptive_smoothness(
        adaptive_smoothness: NDArray[np.float64],
    ) -> None:
        """Accept TRACK's disabled or four-knot adaptive configuration."""
        if adaptive_smoothness.ndim != 2 or adaptive_smoothness.shape[0] != 2:
            raise ValueError("adaptive_smoothness must have shape (2, 0) or (2, 4)")
        if adaptive_smoothness.shape[1] not in (0, 4):
            raise ValueError("adaptive_smoothness must have shape (2, 0) or (2, 4)")
        if not np.all(np.isfinite(adaptive_smoothness)):
            raise ValueError("adaptive_smoothness must contain only finite values")
        if adaptive_smoothness.shape[1] == 4 and np.any(
            np.diff(adaptive_smoothness[0]) <= 0.0
        ):
            raise ValueError(
                "adaptive_smoothness distance thresholds must be strictly increasing"
            )

    @staticmethod
    def _validate_missing_frame_parameters(
        parameters: NDArray[np.float64] | None,
        *,
        dmax: float,
        phimax: float,
    ) -> NDArray[np.float64]:
        """Return TRACK's one-or-more missing-frame parameter pairs."""
        if parameters is None:
            return np.array([[dmax, phimax]], dtype=np.float64)
        normalized = np.asarray(parameters, dtype=np.float64)
        if normalized.ndim != 2 or normalized.shape[1] != 2 or normalized.shape[0] == 0:
            raise ValueError("missing_frame_parameters must have shape (n, 2), n >= 1")
        if not np.all(np.isfinite(normalized)):
            raise ValueError("missing_frame_parameters must contain only finite values")
        if np.any(normalized[:, 0] <= 0.0):
            raise ValueError("missing-frame dmax values must be positive")
        if np.any(normalized[:, 1] < 0.0):
            raise ValueError("missing-frame phimax values must be nonnegative")
        return normalized

    def _parameter_index(self, missing_count: int) -> int:
        """Match TRACK's cap of ``nmiss`` at the final parameter-set index."""
        return min(max(missing_count, 0), self.dmax_parameters.size - 1)

    def _dmax_for_missing_count(self, missing_count: int) -> float:
        return float(self.dmax_parameters[self._parameter_index(missing_count)])

    def _infer_missing_input_counts(
        self,
        times_packed: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Map source-time gaps to TRACK's preceding-frame ``nmiss`` values."""
        missing_counts = np.zeros(times_packed.size, dtype=np.int64)
        if times_packed.size <= 1:
            return missing_counts

        intervals = np.diff(times_packed)
        if np.any(intervals <= 0):
            raise ValueError("detection times must be strictly increasing")
        cadence = (
            self.time_step_ms
            if self.time_step_ms is not None
            else int(np.min(intervals))
        )
        assert cadence is not None
        for frame_index, interval in enumerate(intervals):
            if self.time_step_ms is not None and int(interval) % cadence != 0:
                raise ValueError(
                    "detection interval is not an integral multiple of time_step_ms"
                )
            missing_counts[frame_index] = (int(interval) - 1) // cadence
        return missing_counts

    def _validate_zone_coverage(
        self,
        latitudes: NDArray[np.float64],
        longitudes: NDArray[np.float64],
    ) -> None:
        """Reject zone configurations that do not cover retained features.

        TRACK's ``dmaxx()`` terminates when either endpoint lies outside every
        configured zone.  Empty zones select the global
        ``dmax`` path instead.
        """
        if self.dmax_zones.shape[0] == 0:
            return

        use_360_longitudes = bool(np.all(self.dmax_zones[:, :2] >= 0.0))
        for latitude, longitude in zip(latitudes, longitudes, strict=True):
            zone_longitude = float(longitude)
            if use_360_longitudes and zone_longitude < 0.0:
                zone_longitude = float(np.mod(zone_longitude, 360.0))

            covered = False
            for lon_min, lon_max, lat_min, lat_max, _dmax in self.dmax_zones:
                if latitude < lat_min or latitude > lat_max:
                    continue
                if lon_min > lon_max:
                    in_longitude = (
                        zone_longitude >= lon_min or zone_longitude <= lon_max
                    )
                else:
                    in_longitude = lon_min <= zone_longitude <= lon_max
                if in_longitude:
                    covered = True
                    break

            if not covered:
                raise ValueError(
                    "configured dmax_zones do not cover feature at "
                    f"latitude {latitude:.8g}, longitude {longitude:.8g}"
                )

    def _filter_feature_points(
        self,
        detections: Sequence[HodgesLinkInput],
        missing_input_counts: NDArray[np.int64] | None = None,
    ) -> list[HodgesCenterFrame]:
        """Retain only features that TRACK can connect to an adjacent frame.

        ``feature_pt_filter.c`` checks the next frame first and checks the
        previous frame only when no next-frame connection exists.  It retains a
        candidate on the inclusive displacement boundary.  Raw detection steps
        currently represent present input frames only, so this Plan 04 path
        uses the ``nmiss``-selected displacement parameter for the preceding
        source frame, while retaining input frames rather than inserting empty
        frames for temporal jumps.
        """
        normalized_detections = self._normalize_detections(detections)
        if not normalized_detections:
            return []
        if missing_input_counts is None:
            missing_input_counts = np.zeros(
                len(normalized_detections),
                dtype=np.int64,
            )

        retained: list[HodgesCenterFrame] = []
        for frame_index, step in enumerate(normalized_detections):
            keep = np.zeros(step.values.size, dtype=bool)
            for feature_index in range(step.values.size):
                if step.values[feature_index] < DUFF_FEATURE_CUTOFF:
                    continue
                latitude = float(step.latitudes[feature_index])
                longitude = float(step.longitudes[feature_index])
                connected = False
                if frame_index + 1 < len(normalized_detections):
                    connected = self._has_adjacent_feature_within_dmax(
                        latitude,
                        longitude,
                        normalized_detections[frame_index + 1],
                        int(missing_input_counts[frame_index]),
                    )
                if not connected and frame_index > 0:
                    connected = self._has_adjacent_feature_within_dmax(
                        latitude,
                        longitude,
                        normalized_detections[frame_index - 1],
                        int(missing_input_counts[frame_index - 1]),
                    )
                keep[feature_index] = connected
            retained.append(step.with_feature_mask(keep))
        return retained

    def _filter_feature_points_numba(
        self,
        detections: Sequence[HodgesLinkInput],
        missing_input_counts: NDArray[np.int64] | None = None,
    ) -> list[HodgesCenterFrame]:
        """Apply the TRACK feature filter using the native flattened scan."""
        normalized_detections = self._normalize_detections(detections)
        if not normalized_detections:
            return []
        if missing_input_counts is None:
            missing_input_counts = np.zeros(
                len(normalized_detections),
                dtype=np.int64,
            )

        step_offsets = np.zeros(len(normalized_detections) + 1, dtype=np.int64)
        for frame_index, step in enumerate(normalized_detections):
            step_offsets[frame_index + 1] = step_offsets[frame_index] + step.values.size
        features_lat = np.concatenate(
            [step.latitudes for step in normalized_detections]
        )
        features_lon = np.concatenate(
            [step.longitudes for step in normalized_detections]
        )
        features_values = np.concatenate(
            [step.values for step in normalized_detections]
        )
        keep = _filter_feature_points_native(
            features_lat,
            features_lon,
            features_values,
            step_offsets,
            missing_input_counts,
            self.dmax_parameters,
            self.dmax_zones,
            DUFF_FEATURE_CUTOFF,
        )
        return [
            step.with_feature_mask(keep[step_offsets[index] : step_offsets[index + 1]])
            for index, step in enumerate(normalized_detections)
        ]

    def _has_adjacent_feature_within_dmax(
        self,
        latitude: float,
        longitude: float,
        adjacent: HodgesCenterFrame,
        missing_count: int,
    ) -> bool:
        """Return whether one adjacent feature is within TRACK's inclusive cap."""
        for adjacent_index in range(adjacent.values.size):
            adjacent_latitude = float(adjacent.latitudes[adjacent_index])
            adjacent_longitude = float(adjacent.longitudes[adjacent_index])
            dmax = self._endpoint_average_dmax(
                latitude,
                longitude,
                adjacent_latitude,
                adjacent_longitude,
                missing_count=missing_count,
            )
            if (
                geod_dist(latitude, longitude, adjacent_latitude, adjacent_longitude)
                <= dmax * DEG_TO_RAD
            ):
                return True
        return False

    def _endpoint_average_dmax(
        self,
        first_latitude: float,
        first_longitude: float,
        second_latitude: float,
        second_longitude: float,
        *,
        missing_count: int = 0,
    ) -> float:
        """Match TRACK ``dmaxx()`` by averaging endpoint-zone limits."""
        default_dmax = self._dmax_for_missing_count(missing_count)
        first_dmax = _select_regional_dmax(
            first_latitude,
            first_longitude,
            self.dmax_zones,
            default_dmax,
        )
        second_dmax = _select_regional_dmax(
            second_latitude,
            second_longitude,
            self.dmax_zones,
            default_dmax,
        )
        return 0.5 * (first_dmax + second_dmax)

    def _initialize_mge_workspace(
        self,
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        step_offsets: NDArray[np.int64],
        missing_input_counts: NDArray[np.int64] | None = None,
    ) -> _MGEInitializationWorkspace:
        """Create TRACK's ordered paired real/phantom initialization workspace."""
        n_frames = step_offsets.size - 1
        if missing_input_counts is None:
            missing_input_counts = np.zeros(n_frames, dtype=np.int64)
        rows: list[NDArray[np.int64]] = []
        feature_rows = np.full(features_lat.size, -1, dtype=np.int64)

        def new_real_row(feature_index: int, frame_index: int) -> int:
            row = np.full(n_frames, -1, dtype=np.int64)
            row[frame_index] = feature_index
            rows.append(row)
            rows.append(np.full(n_frames, -1, dtype=np.int64))
            return len(rows) - 2

        for feature_index in range(int(step_offsets[1])):
            feature_rows[feature_index] = new_real_row(feature_index, 0)

        for frame_index in range(n_frames - 1):
            current_start = int(step_offsets[frame_index])
            current_end = int(step_offsets[frame_index + 1])
            next_start = int(step_offsets[frame_index + 1])
            next_end = int(step_offsets[frame_index + 2])

            for feature_index in range(current_start, current_end):
                row_index = int(feature_rows[feature_index])
                if row_index < 0:
                    continue
                closest_distance = float("inf")
                closest_feature = -1
                closest_dmax = self._dmax_for_missing_count(
                    int(missing_input_counts[frame_index])
                )
                for candidate_index in range(next_start, next_end):
                    if feature_rows[candidate_index] >= 0:
                        continue
                    distance = geod_dist(
                        features_lat[feature_index],
                        features_lon[feature_index],
                        features_lat[candidate_index],
                        features_lon[candidate_index],
                    )
                    # TRACK uses <=, so later source-order candidates win an
                    # equal-distance tie before the displacement test.
                    if distance <= closest_distance:
                        closest_distance = distance
                        closest_feature = candidate_index
                        closest_dmax = self._endpoint_average_dmax(
                            features_lat[feature_index],
                            features_lon[feature_index],
                            features_lat[candidate_index],
                            features_lon[candidate_index],
                            missing_count=int(missing_input_counts[frame_index]),
                        )
                if (
                    closest_feature >= 0
                    and closest_distance <= closest_dmax * DEG_TO_RAD
                ):
                    rows[row_index][frame_index + 1] = closest_feature
                    feature_rows[closest_feature] = row_index

            for candidate_index in range(next_start, next_end):
                if feature_rows[candidate_index] < 0:
                    feature_rows[candidate_index] = new_real_row(
                        candidate_index,
                        frame_index + 1,
                    )

        if not rows:
            return _MGEInitializationWorkspace(np.empty((0, n_frames), dtype=np.int64))
        return _MGEInitializationWorkspace(np.stack(rows))

    def _initialize_mge_workspace_numba(
        self,
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        step_offsets: NDArray[np.int64],
        missing_input_counts: NDArray[np.int64] | None = None,
    ) -> _MGEInitializationWorkspace:
        """Build the TRACK workspace using the native ordered candidate scan."""
        n_frames = step_offsets.size - 1
        if missing_input_counts is None:
            missing_input_counts = np.zeros(n_frames, dtype=np.int64)
        rows, n_rows = _initialize_mge_workspace_native(
            features_lat,
            features_lon,
            step_offsets,
            missing_input_counts,
            self.dmax_parameters,
            self.dmax_zones,
        )
        return _MGEInitializationWorkspace(rows[:n_rows, :].copy())

    def _has_excess_displacement(
        self,
        first_feature: int,
        second_feature: int,
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        *,
        use_regional_dmax: bool = True,
        missing_count: int = 0,
    ) -> bool:
        """Apply TRACK's real-point displacement failure comparison.

        ``disp.c`` returns the supplied maximum displacement if either point is
        phantom, so such links never cause ``track_fail()``.
        """
        if first_feature == -1 or second_feature == -1:
            return False

        if use_regional_dmax:
            dmax = self._endpoint_average_dmax(
                float(features_lat[first_feature]),
                float(features_lon[first_feature]),
                float(features_lat[second_feature]),
                float(features_lon[second_feature]),
                missing_count=missing_count,
            )
        else:
            dmax = self._dmax_for_missing_count(missing_count)
        distance = geod_dist(
            float(features_lat[first_feature]),
            float(features_lon[first_feature]),
            float(features_lat[second_feature]),
            float(features_lon[second_feature]),
        )
        return distance > dmax * DEG_TO_RAD

    @staticmethod
    def _first_compatible_empty_row(
        tracks: NDArray[np.int64],
        first_frame: int,
        last_frame: int,
    ) -> int:
        """Return TRACK's first row empty across an inclusive section."""
        for row_index in range(tracks.shape[0]):
            if np.all(tracks[row_index, first_frame : last_frame + 1] == -1):
                return row_index
        return -1

    def _apply_track_fail(
        self,
        tracks: NDArray[np.int64],
        track_index: int,
        middle_frame: int,
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        *,
        direction: MGEDirection,
        missing_count: int = 0,
    ) -> NDArray[np.int64]:
        """Apply TRACK ``track_fail.c`` to an already exchanged row.

        TRACK moves only the contiguous real section beyond the failed link to
        the first existing row that is empty across the required adjacent
        section.  Its preallocated all-phantom rows provide the normal target;
        unlike the historical Python helper, this operation never allocates a
        standalone row.
        """
        n_frames = tracks.shape[1]
        if direction == "forward":
            first_frame = middle_frame + 2
            if first_frame >= n_frames:
                return tracks
            if not self._has_excess_displacement(
                int(tracks[track_index, middle_frame + 1]),
                int(tracks[track_index, first_frame]),
                features_lat,
                features_lon,
                missing_count=missing_count,
            ):
                return tracks

            last_frame = n_frames
            for frame_index in range(first_frame + 1, n_frames):
                if tracks[track_index, frame_index] == -1:
                    last_frame = frame_index
                    break
        else:
            last_frame = middle_frame - 1
            if last_frame <= 0:
                return tracks
            if not self._has_excess_displacement(
                int(tracks[track_index, last_frame]),
                int(tracks[track_index, middle_frame - 2]),
                features_lat,
                features_lon,
                missing_count=missing_count,
            ):
                return tracks

            first_frame = 0
            if last_frame - 1 != 0:
                for frame_index in range(last_frame - 1, -1, -1):
                    if tracks[track_index, frame_index] == -1:
                        first_frame = frame_index + 1
                        break

        section_length = last_frame - first_frame
        if section_length <= 0:
            return tracks

        compatible_first = max(first_frame - 1, 0)
        compatible_last = last_frame - 1 if last_frame == n_frames else last_frame
        destination = self._first_compatible_empty_row(
            tracks,
            compatible_first,
            compatible_last,
        )

        if destination == -1:
            # The source makes a second pass restricted to all-phantom rows.
            # It does not allocate if the MGE workspace has been corrupted.
            for row_index in range(tracks.shape[0]):
                if np.all(tracks[row_index] == -1):
                    destination = row_index
                    break
        if destination == -1:
            return tracks

        tracks[destination, first_frame:last_frame] = tracks[
            track_index, first_frame:last_frame
        ]
        tracks[track_index, first_frame:last_frame] = -1
        return tracks

    @staticmethod
    def _split_track_sections(tracks: NDArray[np.int64]) -> NDArray[np.int64]:
        """Reproduce final ``track_split.c`` real-section extraction.

        The source scans only the rows present on entry.  Each extracted
        earlier contiguous real section appends one real row followed by an
        all-phantom row; the original row retains its final section.
        """
        n_original_rows, n_frames = tracks.shape
        rows = [tracks[row_index].copy() for row_index in range(n_original_rows)]

        for row_index in range(n_original_rows):
            row = rows[row_index]
            while np.count_nonzero(row != -1) > 1:
                first_frame = -1
                for frame_index in range(n_frames):
                    if row[frame_index] != -1:
                        first_frame = frame_index
                        break
                if first_frame == -1:
                    break

                last_frame = n_frames
                for frame_index in range(first_frame + 1, n_frames):
                    if row[frame_index] == -1:
                        last_frame = frame_index
                        break

                section_length = last_frame - first_frame
                if section_length >= np.count_nonzero(row != -1):
                    break

                section = np.full(n_frames, -1, dtype=np.int64)
                section[first_frame:last_frame] = row[first_frame:last_frame]
                row[first_frame:last_frame] = -1
                rows.append(section)
                rows.append(np.full(n_frames, -1, dtype=np.int64))

        return np.stack(rows)

    def _apply_track_constraint_filter(
        self,
        tracks: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        *,
        direction: MGEDirection,
    ) -> NDArray[np.int64]:
        """Apply TRACK's directional adaptive preprocessing filter.

        When adaptive tracking is enabled, ``mge_tracks.c`` applies
        ``track_split()`` and ``tr_zonal_filter()`` immediately before each
        directional sweep.  This compact representation has already split
        phantom gaps, so a failed constraint moves the relevant contiguous
        prefix or suffix to a newly appended real/phantom row pair.
        """
        track_index = 0
        n_frames = tracks.shape[1]
        while track_index < tracks.shape[0]:
            row = tracks[track_index]
            real_frames = np.flatnonzero(row != -1)
            if real_frames.size <= 2:
                track_index += 1
                continue

            if direction == "forward":
                central_frames = real_frames[1:-1]
            else:
                central_frames = real_frames[-2:0:-1]

            for central_frame in central_frames:
                if direction == "forward":
                    adjacent_frame = central_frame + 1
                    first_frame = central_frame + 1
                    last_frame = n_frames
                else:
                    adjacent_frame = central_frame - 1
                    first_frame = 0
                    last_frame = central_frame

                central_feature = int(row[central_frame])
                adjacent_feature = int(row[adjacent_frame])
                if self._has_excess_displacement(
                    central_feature,
                    adjacent_feature,
                    features_lat,
                    features_lon,
                    use_regional_dmax=False,
                ):
                    violates_constraint = True
                else:
                    previous_frame = (
                        central_frame - 1
                        if direction == "forward"
                        else central_frame + 1
                    )
                    previous_feature = int(row[previous_frame])
                    cost = geod_dev(
                        float(features_lat[previous_feature]),
                        float(features_lon[previous_feature]),
                        float(features_lat[central_feature]),
                        float(features_lon[central_feature]),
                        float(features_lat[adjacent_feature]),
                        float(features_lon[adjacent_feature]),
                        self.w1,
                        self.w2,
                    )
                    first_distance = geod_dist(
                        float(features_lat[previous_feature]),
                        float(features_lon[previous_feature]),
                        float(features_lat[central_feature]),
                        float(features_lon[central_feature]),
                    )
                    second_distance = geod_dist(
                        float(features_lat[central_feature]),
                        float(features_lon[central_feature]),
                        float(features_lat[adjacent_feature]),
                        float(features_lon[adjacent_feature]),
                    )
                    phimax = _compute_adaptive_phimax(
                        0.5 * (first_distance + second_distance) / DEG_TO_RAD,
                        self.adaptive_smoothness,
                        self.phimax,
                    )
                    violates_constraint = cost > phimax

                if not violates_constraint:
                    continue

                moved = np.full(n_frames, -1, dtype=np.int64)
                moved[first_frame:last_frame] = row[first_frame:last_frame]
                row[first_frame:last_frame] = -1
                tracks = np.vstack(
                    (tracks, moved, np.full(n_frames, -1, dtype=np.int64))
                )
                break

            track_index += 1

        return tracks

    def _run_directional_mge(
        self,
        track_matrix: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        n_frames: int,
        missing_input_counts: NDArray[np.int64] | None = None,
        profile_stats: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Run TRACK-style bounded forward/backward MGE optimization.

        Each active direction runs repeatedly until a complete sweep performs no
        exchange. Successful exchanges reactivate the opposite direction.

        ``max_iterations`` limits the number of outer directional rounds. The
        final permitted round is forward-only, matching ``mge_tracks.c`` where
        backward MGE runs only while ``tot_count < tot_term``.

        Reaching the outer iteration bound is a normal termination condition.
        """
        if n_frames <= 3:
            return track_matrix
        if missing_input_counts is None:
            missing_input_counts = np.zeros(n_frames, dtype=np.int64)

        forward_active = True
        backward_active = True

        for outer_iteration in range(self.mge_max_iterations):
            if profile_stats is not None:
                profile_stats[5] += 1
            if not (forward_active or backward_active):
                break

            if forward_active:
                if self.adaptive_smoothness.shape[1] == 4:
                    track_matrix = self._split_track_sections(track_matrix)
                    track_matrix = self._apply_track_constraint_filter(
                        track_matrix,
                        features_lat,
                        features_lon,
                        direction="forward",
                    )
                track_matrix, forward_changed = self._run_mge_direction_until_stable(
                    track_matrix,
                    features_lat,
                    features_lon,
                    n_frames,
                    missing_input_counts,
                    direction="forward",
                    profile_stats=profile_stats,
                )

                # The forward direction has converged for the current state.
                forward_active = False

                # TRACK's fel_mge() reactivates backward processing when at least
                # one forward exchange occurred.
                if forward_changed:
                    backward_active = True

            # TRACK skips bel_mge() during the final permitted outer round.
            if outer_iteration == self.mge_max_iterations - 1:
                break

            if backward_active:
                if self.adaptive_smoothness.shape[1] == 4:
                    track_matrix = self._split_track_sections(track_matrix)
                    track_matrix = self._apply_track_constraint_filter(
                        track_matrix,
                        features_lat,
                        features_lon,
                        direction="backward",
                    )
                track_matrix, backward_changed = self._run_mge_direction_until_stable(
                    track_matrix,
                    features_lat,
                    features_lon,
                    n_frames,
                    missing_input_counts,
                    direction="backward",
                    profile_stats=profile_stats,
                )

                # The backward direction has converged for the current state.
                backward_active = False

                # TRACK's bel_mge() reactivates forward processing when at least
                # one backward exchange occurred.
                if backward_changed:
                    forward_active = True

        return track_matrix

    def _run_mge_direction_until_stable(
        self,
        tracks: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        n_frames: int,
        missing_input_counts: NDArray[np.int64] | None = None,
        *,
        direction: MGEDirection,
        profile_stats: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.int64], bool]:
        """Repeat one directional MGE sweep until it makes no exchange."""
        changed_any = False
        if missing_input_counts is None:
            missing_input_counts = np.zeros(n_frames, dtype=np.int64)

        while True:
            if direction == "forward":
                if profile_stats is None:
                    tracks, sweep_changed = self._run_forward_mge_iteration(
                        tracks,
                        features_lat,
                        features_lon,
                        n_frames,
                        missing_input_counts,
                    )
                else:
                    tracks, sweep_changed = self._run_forward_mge_iteration(
                        tracks,
                        features_lat,
                        features_lon,
                        n_frames,
                        missing_input_counts,
                        profile_stats=profile_stats,
                    )
            else:
                if profile_stats is None:
                    tracks, sweep_changed = self._run_backward_mge_iteration(
                        tracks,
                        features_lat,
                        features_lon,
                        n_frames,
                        missing_input_counts,
                    )
                else:
                    tracks, sweep_changed = self._run_backward_mge_iteration(
                        tracks,
                        features_lat,
                        features_lon,
                        n_frames,
                        missing_input_counts,
                        profile_stats=profile_stats,
                    )

            if not sweep_changed:
                return tracks, changed_any

            changed_any = True

    def _run_forward_mge_iteration(
        self,
        tracks: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        n_frames: int,
        missing_input_counts: NDArray[np.int64] | None = None,
        profile_stats: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.int64], bool]:
        """Run a single forward-direction MGE iteration in Numba."""
        if missing_input_counts is None:
            missing_input_counts = np.zeros(n_frames, dtype=np.int64)

        stats = _NO_PROFILE_STATS if profile_stats is None else profile_stats
        changed = _forward_mge_sweep(
            tracks,
            features_lat,
            features_lon,
            n_frames,
            missing_input_counts,
            self.w1,
            self.w2,
            self.dmax_parameters,
            self.phimax_parameters,
            self.dmax_zones,
            self.adaptive_smoothness,
            stats,
        )
        return tracks, changed

    def _run_backward_mge_iteration(
        self,
        tracks: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        n_frames: int,
        missing_input_counts: NDArray[np.int64] | None = None,
        profile_stats: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.int64], bool]:
        """Run a single backward-direction MGE iteration in Numba."""
        if missing_input_counts is None:
            missing_input_counts = np.zeros(n_frames, dtype=np.int64)

        stats = _NO_PROFILE_STATS if profile_stats is None else profile_stats
        changed = _backward_mge_sweep(
            tracks,
            features_lat,
            features_lon,
            n_frames,
            missing_input_counts,
            self.w1,
            self.w2,
            self.dmax_parameters,
            self.phimax_parameters,
            self.dmax_zones,
            self.adaptive_smoothness,
            stats,
        )
        return tracks, changed
