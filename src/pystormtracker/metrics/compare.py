"""Trajectory intercomparison using directed nearest-candidate selection.

Candidate eligibility follows the documented reference comparison utilities:
trajectories must overlap in time, satisfy the symmetric overlap fraction, and
remain within a mean geodesic-separation threshold. Each reference trajectory
selects its closest eligible candidate independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..models.constants import R_EARTH_KM
from ..models.tracks import Tracks
from ..models.units import Mode, ModeOption


@dataclass(frozen=True, slots=True)
class TrackComparisonConfig:
    """Configuration for track-pair eligibility."""

    max_mean_separation_deg: float = 2.0
    min_overlap_fraction: float = 0.6
    var: str | None = None
    mode: ModeOption | None = None

    def __post_init__(self) -> None:
        if self.max_mean_separation_deg <= 0.0:
            raise ValueError("max_mean_separation_deg must be greater than zero")
        if not 0.0 <= self.min_overlap_fraction <= 1.0:
            raise ValueError("min_overlap_fraction must be between zero and one")
        if self.var == "":
            raise ValueError("var must be a non-empty name or None")
        if self.mode is not None and self.mode not in ("auto", "min", "max"):
            raise ValueError("mode must be 'auto', 'min', 'max', or None")


@dataclass(frozen=True, slots=True)
class TrackProperties:
    """Lifecycle and intensity characteristics of one trajectory."""

    point_count: int
    duration_hours: float
    path_length_km: float
    mean_speed_kmh: float
    mean_intensity: float | None
    peak_intensity: float | None


@dataclass(frozen=True, slots=True)
class IntensityDifference:
    """Pointwise candidate-minus-reference intensity statistics."""

    bias: float
    mae: float
    rmse: float
    correlation: float | None


@dataclass(frozen=True, slots=True)
class TrackMatch:
    """The closest eligible candidate selected for one reference trajectory."""

    reference_id: int
    candidate_id: int
    eligible_candidate_count: int
    overlap_count: int
    overlap_fraction: float
    mean_separation_deg: float
    mean_separation_km: float
    minimum_separation_km: float
    median_separation_km: float
    p95_separation_km: float
    maximum_separation_km: float
    reference: TrackProperties
    candidate: TrackProperties
    intensity_difference: IntensityDifference | None


@dataclass(frozen=True, slots=True)
class TrackComparison:
    """Result of comparing one reference trajectory set with one candidate set."""

    config: TrackComparisonConfig
    reference_count: int
    candidate_count: int
    matches: tuple[TrackMatch, ...]
    unmatched_reference_ids: tuple[int, ...]
    unmatched_candidate_ids: tuple[int, ...]

    @property
    def match_count(self) -> int:
        """Return the number of reference tracks with a selected candidate."""
        return len(self.matches)

    @property
    def reference_coverage(self) -> float:
        """Return the fraction of reference tracks with an assigned candidate."""
        if self.reference_count == 0:
            return 0.0
        return self.match_count / self.reference_count

    @property
    def candidate_coverage(self) -> float:
        """Return the fraction of candidate tracks selected at least once."""
        if self.candidate_count == 0:
            return 0.0
        return 1.0 - len(self.unmatched_candidate_ids) / self.candidate_count

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the comparison."""

        def properties_to_dict(properties: TrackProperties) -> dict[str, object]:
            return {
                "point_count": properties.point_count,
                "duration_hours": properties.duration_hours,
                "path_length_km": properties.path_length_km,
                "mean_speed_kmh": properties.mean_speed_kmh,
                "mean_intensity": properties.mean_intensity,
                "peak_intensity": properties.peak_intensity,
            }

        def match_to_dict(match: TrackMatch) -> dict[str, object]:
            intensity_difference: dict[str, object] | None = None
            if match.intensity_difference is not None:
                intensity_difference = {
                    "bias": match.intensity_difference.bias,
                    "mae": match.intensity_difference.mae,
                    "rmse": match.intensity_difference.rmse,
                    "correlation": match.intensity_difference.correlation,
                }
            return {
                "reference_id": match.reference_id,
                "candidate_id": match.candidate_id,
                "eligible_candidate_count": match.eligible_candidate_count,
                "overlap_count": match.overlap_count,
                "overlap_fraction": match.overlap_fraction,
                "mean_separation_deg": match.mean_separation_deg,
                "mean_separation_km": match.mean_separation_km,
                "minimum_separation_km": match.minimum_separation_km,
                "median_separation_km": match.median_separation_km,
                "p95_separation_km": match.p95_separation_km,
                "maximum_separation_km": match.maximum_separation_km,
                "reference": properties_to_dict(match.reference),
                "candidate": properties_to_dict(match.candidate),
                "intensity_difference": intensity_difference,
            }

        return {
            "config": {
                "max_mean_separation_deg": self.config.max_mean_separation_deg,
                "min_overlap_fraction": self.config.min_overlap_fraction,
                "var": self.config.var,
                "mode": self.config.mode,
            },
            "reference_count": self.reference_count,
            "candidate_count": self.candidate_count,
            "match_count": self.match_count,
            "reference_coverage": self.reference_coverage,
            "candidate_coverage": self.candidate_coverage,
            "matches": [match_to_dict(match) for match in self.matches],
            "unmatched_reference_ids": list(self.unmatched_reference_ids),
            "unmatched_candidate_ids": list(self.unmatched_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class _TrackData:
    track_id: int
    times: NDArray[np.int64]
    lats: NDArray[np.float64]
    lons: NDArray[np.float64]
    intensity: NDArray[np.float64] | None
    mode: Mode


@dataclass(frozen=True, slots=True)
class _CandidatePair:
    reference_index: int
    candidate_index: int
    overlap_reference_indices: NDArray[np.int64]
    overlap_candidate_indices: NDArray[np.int64]
    overlap_fraction: float
    separation_km: NDArray[np.float64]


def _great_circle_distances_km(
    lat1: NDArray[np.float64],
    lon1: NDArray[np.float64],
    lat2: NDArray[np.float64],
    lon2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return clamped great-circle distances for aligned coordinate arrays."""
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    delta_lon = np.deg2rad(lon1 - lon2)
    dot = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lon)
    return np.asarray(
        np.arccos(np.clip(dot, -1.0, 1.0)) * R_EARTH_KM,
        dtype=np.float64,
    )


def _collect_tracks(tracks: Tracks, var: str | None) -> tuple[_TrackData, ...]:
    """Extract time-ordered track arrays in their source-file order."""
    if var is not None and var not in tracks.variables:
        raise ValueError(f"intensity variable '{var}' is not present")

    extracted: list[_TrackData] = []
    for track in tracks:
        times = track.times
        if np.unique(times).size != times.size:
            raise ValueError(f"track {track.track_id} contains duplicate timestamps")
        if np.any(times[1:] < times[:-1]):
            raise ValueError(f"track {track.track_id} timestamps must be ordered")
        intensity = (
            np.asarray(tracks.variables[var][track.point_slice], dtype=np.float64)
            if var is not None
            else None
        )
        extracted.append(
            _TrackData(
                track_id=track.track_id,
                times=times,
                lats=np.asarray(track.lats, dtype=np.float64),
                lons=np.asarray(track.lons, dtype=np.float64),
                intensity=intensity,
                mode=tracks.mode,
            )
        )
    return tuple(extracted)


def _candidate_pair(
    reference: _TrackData,
    candidate: _TrackData,
    reference_index: int,
    candidate_index: int,
    config: TrackComparisonConfig,
) -> _CandidatePair | None:
    """Return an eligible pair using the reference overlap section procedure."""
    overlap_start = max(reference.times[0], candidate.times[0])
    overlap_end = min(reference.times[-1], candidate.times[-1])
    if overlap_start > overlap_end:
        return None

    reference_indices = np.where(
        (reference.times >= overlap_start) & (reference.times <= overlap_end)
    )[0]
    candidate_indices = np.where(
        (candidate.times >= overlap_start) & (candidate.times <= overlap_end)
    )[0]
    if reference_indices.size != candidate_indices.size:
        raise ValueError(
            "overlapping track sections must have equal point counts; "
            "resample inputs to a common cadence before comparison"
        )

    overlap_fraction = (
        2.0 * reference_indices.size / (reference.times.size + candidate.times.size)
    )
    if overlap_fraction < config.min_overlap_fraction:
        return None

    separations_km = _great_circle_distances_km(
        reference.lats[reference_indices],
        reference.lons[reference_indices],
        candidate.lats[candidate_indices],
        candidate.lons[candidate_indices],
    )
    mean_separation_deg = float(np.mean(separations_km) / R_EARTH_KM * 180.0 / np.pi)
    if (
        not np.isfinite(mean_separation_deg)
        or mean_separation_deg > config.max_mean_separation_deg
    ):
        return None

    return _CandidatePair(
        reference_index=reference_index,
        candidate_index=candidate_index,
        overlap_reference_indices=np.asarray(reference_indices, dtype=np.int64),
        overlap_candidate_indices=np.asarray(candidate_indices, dtype=np.int64),
        overlap_fraction=float(overlap_fraction),
        separation_km=separations_km,
    )


def _track_properties(track: _TrackData, intensity_mode: Mode) -> TrackProperties:
    """Calculate lifecycle, path, and optional intensity properties."""
    duration_hours = float((int(track.times[-1]) - int(track.times[0])) / 3_600_000.0)
    if track.times.size > 1:
        path_length_km = float(
            np.sum(
                _great_circle_distances_km(
                    track.lats[:-1],
                    track.lons[:-1],
                    track.lats[1:],
                    track.lons[1:],
                )
            )
        )
    else:
        path_length_km = 0.0
    mean_speed_kmh = path_length_km / duration_hours if duration_hours > 0.0 else 0.0

    if track.intensity is None:
        mean_intensity = None
        peak_intensity = None
    else:
        finite_intensity = track.intensity[np.isfinite(track.intensity)]
        mean_intensity = (
            float(np.mean(finite_intensity)) if finite_intensity.size else None
        )
        if finite_intensity.size:
            peak_intensity = float(
                np.min(finite_intensity)
                if intensity_mode == "min"
                else np.max(finite_intensity)
            )
        else:
            peak_intensity = None

    return TrackProperties(
        point_count=int(track.times.size),
        duration_hours=duration_hours,
        path_length_km=path_length_km,
        mean_speed_kmh=mean_speed_kmh,
        mean_intensity=mean_intensity,
        peak_intensity=peak_intensity,
    )


def _intensity_difference(
    reference: _TrackData,
    candidate: _TrackData,
    pair: _CandidatePair,
) -> IntensityDifference | None:
    """Calculate aligned intensity statistics for an assigned pair."""
    if reference.intensity is None or candidate.intensity is None:
        return None
    reference_values = reference.intensity[pair.overlap_reference_indices]
    candidate_values = candidate.intensity[pair.overlap_candidate_indices]
    valid = np.isfinite(reference_values) & np.isfinite(candidate_values)
    if not np.any(valid):
        return None
    difference = candidate_values[valid] - reference_values[valid]
    if difference.size > 1:
        correlation_value = np.corrcoef(
            reference_values[valid], candidate_values[valid]
        )[0, 1]
        correlation = (
            float(correlation_value) if np.isfinite(correlation_value) else None
        )
    else:
        correlation = None
    return IntensityDifference(
        bias=float(np.mean(difference)),
        mae=float(np.mean(np.abs(difference))),
        rmse=float(np.sqrt(np.mean(np.square(difference)))),
        correlation=correlation,
    )


def compare_tracks(
    reference: Tracks,
    candidate: Tracks,
    *,
    config: TrackComparisonConfig | None = None,
) -> TrackComparison:
    """Compare track sets by selecting each reference's closest candidate.

    The overlap is the contiguous interval shared by the two track lifecycles.
    Its two sections must have equal point counts and are paired by position.
    This reproduces the assumptions of the reference comparison program.
    """
    effective_config = config if config is not None else TrackComparisonConfig()
    if effective_config.var is not None:
        variable = effective_config.var
        if variable not in reference.units or variable not in candidate.units:
            raise ValueError(
                f"comparison intensity variable {variable!r} must be present in "
                "both tracks"
            )
        if reference.units[variable] != candidate.units[variable]:
            raise ValueError(
                f"comparison intensity variable {variable!r} requires matching units; "
                f"got {reference.units[variable]!r} and {candidate.units[variable]!r}"
            )
    reference_tracks = _collect_tracks(reference, effective_config.var)
    candidate_tracks = _collect_tracks(candidate, effective_config.var)
    matches: list[TrackMatch] = []
    matched_candidate_indices: set[int] = set()
    matched_reference_indices: set[int] = set()
    for reference_index, reference_track in enumerate(reference_tracks):
        eligible_pairs = [
            pair
            for candidate_index, candidate_track in enumerate(candidate_tracks)
            if (
                pair := _candidate_pair(
                    reference_track,
                    candidate_track,
                    reference_index,
                    candidate_index,
                    effective_config,
                )
            )
            is not None
        ]
        if not eligible_pairs:
            continue
        pair = min(
            eligible_pairs,
            key=lambda candidate_pair: np.mean(candidate_pair.separation_km),
        )
        candidate_track = candidate_tracks[pair.candidate_index]
        separations = pair.separation_km
        ref_mode: Mode = (
            effective_config.mode
            if effective_config.mode in ("min", "max")
            else reference_track.mode
        )
        cand_mode: Mode = (
            effective_config.mode
            if effective_config.mode in ("min", "max")
            else candidate_track.mode
        )
        matches.append(
            TrackMatch(
                reference_id=reference_track.track_id,
                candidate_id=candidate_track.track_id,
                eligible_candidate_count=len(eligible_pairs),
                overlap_count=int(separations.size),
                overlap_fraction=pair.overlap_fraction,
                mean_separation_deg=float(
                    np.mean(separations) / R_EARTH_KM * 180.0 / np.pi
                ),
                mean_separation_km=float(np.mean(separations)),
                minimum_separation_km=float(np.min(separations)),
                median_separation_km=float(np.median(separations)),
                p95_separation_km=float(np.percentile(separations, 95)),
                maximum_separation_km=float(np.max(separations)),
                reference=_track_properties(reference_track, ref_mode),
                candidate=_track_properties(candidate_track, cand_mode),
                intensity_difference=_intensity_difference(
                    reference_track, candidate_track, pair
                ),
            )
        )
        matched_reference_indices.add(pair.reference_index)
        matched_candidate_indices.add(pair.candidate_index)

    return TrackComparison(
        config=effective_config,
        reference_count=len(reference_tracks),
        candidate_count=len(candidate_tracks),
        matches=tuple(matches),
        unmatched_reference_ids=tuple(
            track.track_id
            for index, track in enumerate(reference_tracks)
            if index not in matched_reference_indices
        ),
        unmatched_candidate_ids=tuple(
            track.track_id
            for index, track in enumerate(candidate_tracks)
            if index not in matched_candidate_indices
        ),
    )
