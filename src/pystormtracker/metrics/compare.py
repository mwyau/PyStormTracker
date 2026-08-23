"""Trajectory intercomparison using three explicit matching strategies.

Mutual-nearest comparison has lineage from Blender and Schubert (2000) and
the intercomparison definitions discussed by Neu et al. (2013).  The exact
distance and eligibility rules in this module are PyStormTracker definitions;
this module does not claim to reproduce either published matcher when those
definitions differ.  Global assignment uses standard optimization primitives
with a PST-specific lexicographic objective.

References:
    Blender, R., and M. Schubert (2000). Cyclone Tracking in Different
        Spatial and Temporal Resolutions. *Monthly Weather Review*, 128(2),
        377--384. https://doi.org/10.1175/1520-0493(2000)128<0377:CTIDSA>2.0.CO;2
    Neu, U., et al. (2013). IMILAST: A Community Effort to Intercompare
        Extratropical Cyclone Detection and Tracking Algorithms. *Bulletin
        of the American Meteorological Society*, 94(4), 529--547.
        https://doi.org/10.1175/BAMS-D-11-00154.1
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..models.geo import R_EARTH_KM
from ..models.tracks import DetectionMode, ResolvedDetectionMode, Tracks

type MatchingMethod = Literal["nearest", "mutual_nearest", "global_assignment"]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrackComparisonConfig:
    """Configuration for track comparison and pair eligibility."""

    matching: MatchingMethod = "nearest"
    max_mean_separation_deg: float = 2.0
    min_overlap_fraction: float = 0.6
    variable: str | None = None
    mode: DetectionMode | None = None

    def __post_init__(self) -> None:
        if self.matching not in (
            "nearest",
            "mutual_nearest",
            "global_assignment",
        ):
            raise ValueError(
                f"unsupported matching method {self.matching!r}; "
                "expected 'nearest', 'mutual_nearest', or 'global_assignment'"
            )
        if self.max_mean_separation_deg <= 0.0:
            raise ValueError("max_mean_separation_deg must be greater than zero")
        if not 0.0 <= self.min_overlap_fraction <= 1.0:
            raise ValueError("min_overlap_fraction must be between zero and one")
        if self.variable == "":
            raise ValueError("variable must be a non-empty name or None")
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
    """Matched candidate trajectory for one reference trajectory."""

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
    same_time_range: bool = False
    same_point_count: bool = False
    topology_identical: bool = False


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
    def matching(self) -> MatchingMethod:
        """Return the matching method configured for this comparison."""
        return self.config.matching

    @property
    def match_count(self) -> int:
        """Return the number of matched trajectory pairs."""
        return len(self.matches)

    @property
    def unique_candidate_count(self) -> int:
        """Return the count of unique candidate trajectories selected."""
        return len({match.candidate_id for match in self.matches})

    @property
    def reused_candidate_count(self) -> int:
        """Return the count of candidates selected by more than one reference."""
        counts = Counter(match.candidate_id for match in self.matches)
        return sum(1 for cnt in counts.values() if cnt > 1)

    @property
    def reused_candidate_assignments(self) -> int:
        """Return count of duplicate assignments to already-selected candidates."""
        counts = Counter(match.candidate_id for match in self.matches)
        return sum(cnt - 1 for cnt in counts.values() if cnt > 1)

    @property
    def unmatched_reference_count(self) -> int:
        """Return the count of unmatched reference trajectories."""
        return len(self.unmatched_reference_ids)

    @property
    def unmatched_candidate_count(self) -> int:
        """Return the count of unmatched candidate trajectories."""
        return len(self.unmatched_candidate_ids)

    @property
    def topology_identical_count(self) -> int:
        """Return the count of matched pairs with identical timestamp topology."""
        return sum(1 for match in self.matches if match.topology_identical)

    # Nearest-specific properties
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
        return self.unique_candidate_count / self.candidate_count

    # Mutual-specific properties
    @property
    def agreement(self) -> float:
        """Return symmetric mutual match agreement: match_count / min(N_ref, N_cand)."""
        denominator = min(self.reference_count, self.candidate_count)
        if denominator == 0:
            return 0.0
        return self.match_count / denominator

    # Assignment-specific properties
    @property
    def tp(self) -> int:
        """True positives (matched pairs) under assignment."""
        return self.match_count

    @property
    def fn(self) -> int:
        """False negatives (unmatched reference trajectories) under assignment."""
        return self.unmatched_reference_count

    @property
    def fp(self) -> int:
        """False positives (unmatched candidate trajectories) under assignment."""
        return self.unmatched_candidate_count

    @property
    def precision(self) -> float:
        """Candidate-side precision: matched / candidate_count."""
        if self.candidate_count == 0:
            return 0.0
        return self.match_count / self.candidate_count

    @property
    def recall(self) -> float:
        """Recall relative to reference: TP / (TP + FN) = matched / reference_count."""
        if self.reference_count == 0:
            return 0.0
        return self.match_count / self.reference_count

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p = self.precision
        r = self.recall
        if p + r == 0.0:
            return 0.0
        return 2.0 * p * r / (p + r)

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
                "same_time_range": match.same_time_range,
                "same_point_count": match.same_point_count,
                "topology_identical": match.topology_identical,
                "reference": properties_to_dict(match.reference),
                "candidate": properties_to_dict(match.candidate),
                "intensity_difference": intensity_difference,
            }

        output: dict[str, object] = {
            "config": {
                "matching": self.config.matching,
                "max_mean_separation_deg": self.config.max_mean_separation_deg,
                "min_overlap_fraction": self.config.min_overlap_fraction,
                "variable": self.config.variable,
                "mode": self.config.mode,
            },
            "matching": self.matching,
            "reference_count": self.reference_count,
            "candidate_count": self.candidate_count,
            "match_count": self.match_count,
            "unmatched_reference_count": self.unmatched_reference_count,
            "unmatched_candidate_count": self.unmatched_candidate_count,
            "topology_identical_count": self.topology_identical_count,
        }

        if self.matching == "nearest":
            output["reference_coverage"] = self.reference_coverage
            output["candidate_coverage"] = self.candidate_coverage
            output["unique_candidate_count"] = self.unique_candidate_count
            output["reused_candidate_count"] = self.reused_candidate_count
            output["reused_candidate_assignments"] = self.reused_candidate_assignments
        elif self.matching == "mutual_nearest":
            output["agreement"] = self.agreement
        elif self.matching == "global_assignment":
            output["tp"] = self.tp
            output["fn"] = self.fn
            output["fp"] = self.fp
            output["precision"] = self.precision
            output["recall"] = self.recall
            output["f1"] = self.f1

        output["matches"] = [match_to_dict(match) for match in self.matches]
        output["unmatched_reference_ids"] = list(self.unmatched_reference_ids)
        output["unmatched_candidate_ids"] = list(self.unmatched_candidate_ids)
        return output


@dataclass(frozen=True, slots=True)
class _TrackData:
    track_id: int
    times: NDArray[np.int64]
    lats: NDArray[np.float64]
    lons: NDArray[np.float64]
    intensity: NDArray[np.float64] | None
    mode: ResolvedDetectionMode


@dataclass(frozen=True, slots=True)
class _CandidatePair:
    reference_index: int
    candidate_index: int
    overlap_reference_indices: NDArray[np.int64]
    overlap_candidate_indices: NDArray[np.int64]
    overlap_fraction: float
    separation_km: NDArray[np.float64]
    overlap_score: Fraction | None = None


type _LexicographicCost = tuple[int, Fraction, Fraction, int]


@dataclass(slots=True)
class _FlowEdge:
    """One residual edge in the lexicographic matching network."""

    target: int
    reverse_index: int
    capacity: int
    cost: _LexicographicCost


def _cost_add(
    left: _LexicographicCost, right: _LexicographicCost
) -> _LexicographicCost:
    """Add two lexicographic edge costs componentwise."""
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
        left[3] + right[3],
    )


def _cost_negate(cost: _LexicographicCost) -> _LexicographicCost:
    """Return the residual reverse-edge cost."""
    return (-cost[0], -cost[1], -cost[2], -cost[3])


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
    cost: _LexicographicCost,
) -> _FlowEdge:
    """Add a unit-capacity edge and its residual reverse edge."""
    forward = _FlowEdge(target, len(graph[target]), 1, cost)
    reverse = _FlowEdge(source, len(graph[source]), 0, _cost_negate(cost))
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _shortest_residual_path(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
) -> tuple[list[int], list[int]] | None:
    """Find a minimum lexicographic-cost augmenting path.

    The network starts with no residual cycles of positive capacity. After each
    augmentation the selected flow is optimal for its cardinality, so the
    standard successive-shortest-augmenting-path method remains valid. The
    costs are tuples, not scalarized scientific weights; Python tuple ordering
    implements the required exact lexicographic priority.
    """
    distances: list[_LexicographicCost | None] = [None] * len(graph)
    previous_nodes = [-1] * len(graph)
    previous_edges = [-1] * len(graph)
    distances[source] = (0, Fraction(0), Fraction(0), 0)

    for _ in range(len(graph) - 1):
        changed = False
        for node, distance in enumerate(distances):
            if distance is None or node == target:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity == 0 or edge.target == source:
                    continue
                candidate = _cost_add(distance, edge.cost)
                current = distances[edge.target]
                if current is None or candidate < current:
                    distances[edge.target] = candidate
                    previous_nodes[edge.target] = node
                    previous_edges[edge.target] = edge_index
                    changed = True
        if not changed:
            break

    if distances[target] is None:
        return None

    path_nodes: list[int] = []
    path_edges: list[int] = []
    node = target
    while node != source:
        previous = previous_nodes[node]
        edge_index = previous_edges[node]
        if previous < 0 or edge_index < 0:
            return None
        path_nodes.append(previous)
        path_edges.append(edge_index)
        node = previous
    path_nodes.reverse()
    path_edges.reverse()
    return path_nodes, path_edges


def _lexicographic_component_assignment(
    comp_ref: list[int],
    comp_cand: list[int],
    pair_map: dict[tuple[int, int], _CandidatePair],
) -> list[tuple[int, int]]:
    """Return a maximum-cardinality lexicographic matching for one component.

    The flow objective is the additive tuple ``(-1, -overlap, separation,
    deterministic_tie)`` on each eligible reference/candidate edge. Sending
    flow until no augmenting path remains therefore first maximizes cardinality,
    then total temporal-overlap fraction, then minimizes total mean separation.
    No finite scientific scalar can override an earlier objective.
    """
    n_ref = len(comp_ref)
    n_cand = len(comp_cand)
    source = n_ref + n_cand
    target = source + 1
    graph: list[list[_FlowEdge]] = [[] for _ in range(target + 1)]

    for ref_local in range(n_ref):
        _add_flow_edge(graph, source, ref_local, (0, Fraction(0), Fraction(0), 0))
    for cand_local in range(n_cand):
        _add_flow_edge(
            graph, n_ref + cand_local, target, (0, Fraction(0), Fraction(0), 0)
        )

    assignment_edges: list[tuple[int, int, _FlowEdge]] = []
    tie_order = 0
    for ref_local, ref_idx in enumerate(comp_ref):
        for cand_local, cand_idx in enumerate(comp_cand):
            pair = pair_map.get((ref_idx, cand_idx))
            if pair is None:
                continue
            mean_separation_deg = float(
                np.mean(pair.separation_km) / R_EARTH_KM * 180.0 / np.pi
            )
            overlap_score = (
                pair.overlap_score
                if pair.overlap_score is not None
                else Fraction(str(pair.overlap_fraction))
            )
            edge = _add_flow_edge(
                graph,
                ref_local,
                n_ref + cand_local,
                (
                    -1,
                    -overlap_score,
                    Fraction.from_float(mean_separation_deg),
                    1 << tie_order,
                ),
            )
            assignment_edges.append((ref_idx, cand_idx, edge))
            tie_order += 1

    while True:
        path = _shortest_residual_path(graph, source, target)
        if path is None:
            break
        path_nodes, path_edges = path
        for node, edge_index in zip(path_nodes, path_edges, strict=True):
            edge = graph[node][edge_index]
            edge.capacity = 0
            graph[edge.target][edge.reverse_index].capacity = 1

    return [
        (ref_idx, cand_idx)
        for ref_idx, cand_idx, edge in assignment_edges
        if edge.capacity == 0
    ]


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


def _collect_tracks(tracks: Tracks, variable: str | None) -> tuple[_TrackData, ...]:
    """Extract time-ordered track arrays in their source-file order."""
    if variable is not None and variable not in tracks.variables:
        raise ValueError(f"intensity variable '{variable}' is not present")

    extracted: list[_TrackData] = []
    for track in tracks:
        times = track.times
        if np.unique(times).size != times.size:
            raise ValueError(f"track {track.track_id} contains duplicate timestamps")
        if np.any(times[1:] < times[:-1]):
            raise ValueError(f"track {track.track_id} timestamps must be ordered")
        intensity = (
            np.asarray(tracks.variables[variable][track.point_slice], dtype=np.float64)
            if variable is not None
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

    # Coordinates are aligned only where the two trajectories have the exact
    # same timestamp.  Equal counts between the first and last overlapping
    # time are not evidence that two different cadences describe concurrent
    # points.
    common_times, reference_indices, candidate_indices = np.intersect1d(
        reference.times,
        candidate.times,
        assume_unique=True,
        return_indices=True,
    )
    if common_times.size == 0:
        return None

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
        overlap_score=Fraction(
            2 * int(reference_indices.size),
            int(reference.times.size + candidate.times.size),
        ),
    )


def _track_properties(
    track: _TrackData, intensity_mode: ResolvedDetectionMode
) -> TrackProperties:
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


def _build_match(
    pair: _CandidatePair,
    reference_track: _TrackData,
    candidate_track: _TrackData,
    config: TrackComparisonConfig,
    eligible_candidate_count: int,
) -> TrackMatch:
    """Construct one matched pair with complete statistics and topology flags."""
    separations = pair.separation_km
    ref_mode: ResolvedDetectionMode = (
        config.mode if config.mode in ("min", "max") else reference_track.mode
    )
    cand_mode: ResolvedDetectionMode = (
        config.mode if config.mode in ("min", "max") else candidate_track.mode
    )
    same_time_range = bool(
        reference_track.times[0] == candidate_track.times[0]
        and reference_track.times[-1] == candidate_track.times[-1]
    )
    same_point_count = bool(reference_track.times.size == candidate_track.times.size)
    topology_identical = bool(
        np.array_equal(reference_track.times, candidate_track.times)
    )

    return TrackMatch(
        reference_id=reference_track.track_id,
        candidate_id=candidate_track.track_id,
        eligible_candidate_count=eligible_candidate_count,
        overlap_count=int(separations.size),
        overlap_fraction=pair.overlap_fraction,
        mean_separation_deg=float(np.mean(separations) / R_EARTH_KM * 180.0 / np.pi),
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
        same_time_range=same_time_range,
        same_point_count=same_point_count,
        topology_identical=topology_identical,
    )


def _match_nearest(
    reference_tracks: tuple[_TrackData, ...],
    candidate_tracks: tuple[_TrackData, ...],
    eligible_pairs_by_ref: dict[int, list[_CandidatePair]],
    config: TrackComparisonConfig,
) -> list[TrackMatch]:
    """Perform directed nearest-candidate matching for each reference track."""
    matches: list[TrackMatch] = []
    for ref_idx, ref_track in enumerate(reference_tracks):
        pairs = eligible_pairs_by_ref.get(ref_idx)
        if not pairs:
            continue
        best_pair = min(
            pairs,
            key=lambda candidate_pair: np.mean(candidate_pair.separation_km),
        )
        cand_track = candidate_tracks[best_pair.candidate_index]
        matches.append(
            _build_match(
                best_pair,
                ref_track,
                cand_track,
                config,
                len(pairs),
            )
        )
    return matches


def _match_mutual(
    reference_tracks: tuple[_TrackData, ...],
    candidate_tracks: tuple[_TrackData, ...],
    eligible_pairs_by_ref: dict[int, list[_CandidatePair]],
    eligible_pairs_by_cand: dict[int, list[_CandidatePair]],
    config: TrackComparisonConfig,
) -> list[TrackMatch]:
    """Perform mutual-nearest trajectory matching.

    A pair (R_i, C_j) is accepted if and only if C_j is R_i's nearest eligible candidate
    and R_i is C_j's nearest eligible reference track by mean geodesic separation.

    This comparison has lineage from Blender and Schubert (2000) and Neu et
    al. (2013).  The current PST distance, overlap eligibility, boundary
    behavior, and ordering are implementation details, not claims that the
    published matchers are reproduced exactly.
    """
    nearest_for_ref: dict[int, _CandidatePair] = {}
    for ref_idx, pairs in eligible_pairs_by_ref.items():
        if pairs:
            nearest_for_ref[ref_idx] = min(
                pairs,
                key=lambda p: float(np.mean(p.separation_km)),
            )

    nearest_for_cand: dict[int, _CandidatePair] = {}
    for cand_idx, pairs in eligible_pairs_by_cand.items():
        if pairs:
            nearest_for_cand[cand_idx] = min(
                pairs,
                key=lambda p: float(np.mean(p.separation_km)),
            )

    matches: list[TrackMatch] = []
    for ref_idx, pair_from_ref in sorted(nearest_for_ref.items()):
        cand_idx = pair_from_ref.candidate_index
        pair_from_cand = nearest_for_cand.get(cand_idx)
        if pair_from_cand is not None and pair_from_cand.reference_index == ref_idx:
            ref_track = reference_tracks[ref_idx]
            cand_track = candidate_tracks[cand_idx]
            matches.append(
                _build_match(
                    pair_from_ref,
                    ref_track,
                    cand_track,
                    config,
                    len(eligible_pairs_by_ref[ref_idx]),
                )
            )
    return matches


def _match_assignment(
    reference_tracks: tuple[_TrackData, ...],
    candidate_tracks: tuple[_TrackData, ...],
    eligible_pairs_by_ref: dict[int, list[_CandidatePair]],
    config: TrackComparisonConfig,
) -> list[TrackMatch]:
    """Perform the PST lexicographic one-to-one bipartite assignment.

    Uses a lexicographic maximum-cardinality flow objective:
      1. Maximize number of matched trajectories
      2. Maximize total temporal overlap fraction
      3. Minimize total mean geodesic separation

    Decomposes the bipartite graph into connected components. Each component is
    solved independently with successive shortest augmenting paths whose costs
    are scientific-priority tuples, followed by a deterministic tie component.

    The project-specific objective is to maximize matched tracks, then
    temporal overlap, then minimize spatial separation.  The flow/assignment
    primitives are standard numerical tools; the comparison objective and
    eligibility definitions belong to PyStormTracker.
    """
    if not reference_tracks or not candidate_tracks or not eligible_pairs_by_ref:
        return []

    pair_map: dict[tuple[int, int], _CandidatePair] = {}
    ref_adj: dict[int, list[int]] = {i: [] for i in range(len(reference_tracks))}
    cand_adj: dict[int, list[int]] = {j: [] for j in range(len(candidate_tracks))}

    for ref_idx, pairs in eligible_pairs_by_ref.items():
        for pair in pairs:
            cand_idx = pair.candidate_index
            pair_map[(ref_idx, cand_idx)] = pair
            ref_adj[ref_idx].append(cand_idx)
            cand_adj[cand_idx].append(ref_idx)

    visited_ref: set[int] = set()
    visited_cand: set[int] = set()
    matches: list[TrackMatch] = []

    for start_ref in range(len(reference_tracks)):
        if start_ref in visited_ref or not ref_adj[start_ref]:
            continue

        comp_ref: list[int] = []
        comp_cand: list[int] = []
        queue_ref = [start_ref]
        visited_ref.add(start_ref)

        while queue_ref:
            curr_ref = queue_ref.pop()
            comp_ref.append(curr_ref)
            for c in ref_adj[curr_ref]:
                if c not in visited_cand:
                    visited_cand.add(c)
                    comp_cand.append(c)
                    for r in cand_adj[c]:
                        if r not in visited_ref:
                            visited_ref.add(r)
                            queue_ref.append(r)

        comp_ref.sort()
        comp_cand.sort()

        if not comp_ref or not comp_cand:
            continue

        for r_idx, c_idx in _lexicographic_component_assignment(
            comp_ref, comp_cand, pair_map
        ):
            matched_pair = pair_map[(r_idx, c_idx)]
            matches.append(
                _build_match(
                    matched_pair,
                    reference_tracks[r_idx],
                    candidate_tracks[c_idx],
                    config,
                    len(eligible_pairs_by_ref[r_idx]),
                )
            )

    matches.sort(key=lambda match: match.reference_id)
    return matches


def compare_tracks(
    reference: Tracks,
    candidate: Tracks,
    *,
    config: TrackComparisonConfig | None = None,
) -> TrackComparison:
    """Compare track sets using the specified matching method.

    Supported matching methods:
      - ``"nearest"``: Directed correspondence where each reference selects its
        closest eligible candidate independently.
      - ``"mutual_nearest"``: Reciprocal nearest-neighbor matching where pairs are
        accepted only when both tracks are each other's closest candidate.
      - ``"global_assignment"``: Globally optimal 1-to-1 bipartite assignment using a
        lexicographic objective (maximize matched count, maximize overlap,
        minimize separation).

    The mutual-nearest method has published comparison lineage (Blender and
    Schubert, 2000; Neu et al., 2013), but the current PST eligibility and
    distance definitions are not asserted to be exact implementations of
    those matchers.  The global-assignment objective is PST-specific.
    """
    effective_config = config if config is not None else TrackComparisonConfig()
    if effective_config.variable is not None:
        variable = effective_config.variable
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
    reference_tracks = _collect_tracks(reference, effective_config.variable)
    candidate_tracks = _collect_tracks(candidate, effective_config.variable)

    eligible_pairs_by_ref: dict[int, list[_CandidatePair]] = {
        i: [] for i in range(len(reference_tracks))
    }
    eligible_pairs_by_cand: dict[int, list[_CandidatePair]] = {
        j: [] for j in range(len(candidate_tracks))
    }

    if candidate_tracks:
        cand_start_times = np.fromiter(
            (c.times[0] for c in candidate_tracks),
            dtype=np.int64,
            count=len(candidate_tracks),
        )
        cand_end_times = np.fromiter(
            (c.times[-1] for c in candidate_tracks),
            dtype=np.int64,
            count=len(candidate_tracks),
        )
    else:
        cand_start_times = np.empty(0, dtype=np.int64)
        cand_end_times = np.empty(0, dtype=np.int64)

    for ref_idx, ref_track in enumerate(reference_tracks):
        ref_start = ref_track.times[0]
        ref_end = ref_track.times[-1]
        overlapping_indices = np.where(
            (cand_start_times <= ref_end) & (cand_end_times >= ref_start)
        )[0]

        for cand_idx in overlapping_indices:
            cand_track = candidate_tracks[int(cand_idx)]
            pair = _candidate_pair(
                ref_track,
                cand_track,
                ref_idx,
                int(cand_idx),
                effective_config,
            )
            if pair is not None:
                eligible_pairs_by_ref[ref_idx].append(pair)
                eligible_pairs_by_cand[int(cand_idx)].append(pair)

    if effective_config.matching == "nearest":
        matches = _match_nearest(
            reference_tracks,
            candidate_tracks,
            eligible_pairs_by_ref,
            effective_config,
        )
    elif effective_config.matching == "mutual_nearest":
        matches = _match_mutual(
            reference_tracks,
            candidate_tracks,
            eligible_pairs_by_ref,
            eligible_pairs_by_cand,
            effective_config,
        )
    elif effective_config.matching == "global_assignment":
        matches = _match_assignment(
            reference_tracks,
            candidate_tracks,
            eligible_pairs_by_ref,
            effective_config,
        )
    else:
        raise ValueError(f"unsupported matching method: {effective_config.matching}")

    matched_ref_ids = {match.reference_id for match in matches}
    matched_cand_ids = {match.candidate_id for match in matches}

    LOGGER.info(
        "Comparison method=%s reference=%d candidate=%d matched=%d",
        effective_config.matching,
        len(reference_tracks),
        len(candidate_tracks),
        len(matches),
    )

    return TrackComparison(
        config=effective_config,
        reference_count=len(reference_tracks),
        candidate_count=len(candidate_tracks),
        matches=tuple(matches),
        unmatched_reference_ids=tuple(
            track.track_id
            for track in reference_tracks
            if track.track_id not in matched_ref_ids
        ),
        unmatched_candidate_ids=tuple(
            track.track_id
            for track in candidate_tracks
            if track.track_id not in matched_cand_ids
        ),
    )
