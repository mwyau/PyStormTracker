from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, NamedTuple

import numba as nb
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.data_loader import DataLoader
from ..models.geo import DEG_TO_RAD, KM_PER_DEG, R_EARTH_KM, geod_dist
from ..models.time import TimeInput, TimeRange, coerce_time_input, select_time_range
from ..models.tracks import ResolvedDetectionMode
from ..refinement.bspline import (
    BsplineRefinementStatus,
    RectangularGridPreparation,
    build_bspline_surface,
    build_spherical_bspline_surface,
    prepare_rectangular_grid,
    refine_bspline_feature_point,
    refine_spherical_bspline_feature_point,
)
from ..refinement.quadratic import (
    SphericalQuadraticRefinementStatus,
    refine_quadratic_feature_coordinates,
    refine_spherical_quadratic_feature_points,
    spherical_quadratic_status_name,
)
from .constants import (
    DEFAULT_MSL_OBJECT_THRESHOLD,
    DEFAULT_VO_OBJECT_THRESHOLD,
    MIN_OBJECT_GRID_POINTS_DEFAULT,
    TRACK_SMOOPY_OPTIMIZATION_SCALE_DEFAULT,
)
from .detections import HodgesCenterFrame

LOGGER = logging.getLogger(__name__)

DEFAULT_SEARCH_WINDOW_SIZE: Final[int] = 3

# TRACK's non_lin_opt.c marks a later feature in the same object as unusable
# when its optimized center is closer than TOLSEP. feature_pt_filter.c removes
# these DUFF_PT entries before MGE initialization.
FEATURE_DUPLICATE_TOLERANCE_RAD: Final[float] = 1.0e-4
DUFF_FEATURE_VALUE: Final[float] = -1.0e12
DUFF_FEATURE_CUTOFF: Final[float] = -1.0e10

type HodgesFeatureRefinement = Literal[
    "grid",
    "quadratic",
    "spherical_quadratic",
    "bspline",
    "spherical_bspline",
]
type RefinementDiagnosticStatus = (
    BsplineRefinementStatus | SphericalQuadraticRefinementStatus
)


class HodgesFeaturePointDiagnostic(NamedTuple):
    """Observable outcome for each refinement attempt."""

    grid_latitude: float
    grid_longitude: float
    status: RefinementDiagnosticStatus
    failure_status: RefinementDiagnosticStatus | None


# ---------------------------------------------------------------------------
# Compiled Hodges Detection Numerics
# ---------------------------------------------------------------------------


@nb.njit(cache=True, nogil=True)
def _label_connected_components(
    frame: NDArray[np.float64],
    threshold: float,
    is_min: bool,
    periodic_x: bool = True,
) -> tuple[NDArray[np.int32], int]:
    """Connected Component Labeling (CCL) fused with threshold evaluation.

    Evaluates TRACK-compatible inclusive threshold membership without allocating
    intermediate binary masks.

    Inclusive TRACK threshold semantics:
        minima: frame <= threshold
        maxima: frame >= threshold
    NaN values are rejected as background.

    Args:
        frame: 2D float64 data array.
        threshold: Intensity threshold.
        is_min: True for local minima (frame <= threshold),
            False for maxima (frame >= threshold).
        periodic_x: Whether the first and final columns are adjacent.

    Returns:
        (labeled_mask, num_objects)
    """
    ny, nx = frame.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    label_count = 0

    # Initial labeling pass evaluating threshold predicate directly
    for i in range(ny):
        for j in range(nx):
            val = frame[i, j]
            if np.isnan(val):
                continue
            if is_min:
                if val <= threshold:
                    label_count += 1
                    labels[i, j] = label_count
            else:
                if val >= threshold:
                    label_count += 1
                    labels[i, j] = label_count

    if label_count == 0:
        return labels, 0

    # Iterative propagation until convergence
    changed = True
    while changed:
        changed = False
        # Forward pass
        for i in range(ny):
            for j in range(nx):
                if labels[i, j] == 0:
                    continue
                cur = labels[i, j]
                for di in range(-1, 2):
                    ni = i + di
                    if ni < 0 or ni >= ny:
                        continue
                    for dj in range(-1, 2):
                        nj = j + dj
                        if periodic_x:
                            nj %= nx
                        elif nj < 0 or nj >= nx:
                            continue
                        if labels[ni, nj] > 0 and labels[ni, nj] < cur:
                            cur = labels[ni, nj]
                if cur != labels[i, j]:
                    labels[i, j] = cur
                    changed = True

        # Backward pass
        for i in range(ny - 1, -1, -1):
            for j in range(nx - 1, -1, -1):
                if labels[i, j] == 0:
                    continue
                cur = labels[i, j]
                for di in range(-1, 2):
                    ni = i + di
                    if ni < 0 or ni >= ny:
                        continue
                    for dj in range(-1, 2):
                        nj = j + dj
                        if periodic_x:
                            nj %= nx
                        elif nj < 0 or nj >= nx:
                            continue
                        if labels[ni, nj] > 0 and labels[ni, nj] < cur:
                            cur = labels[ni, nj]
                if cur != labels[i, j]:
                    labels[i, j] = cur
                    changed = True

    # Compact labels to sequential 1..N
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    num_objects = len(unique_labels)
    label_map = np.zeros(label_count + 1, dtype=np.int32)
    for i in range(num_objects):
        label_map[unique_labels[i]] = i + 1

    for i in range(ny):
        for j in range(nx):
            labels[i, j] = label_map[labels[i, j]]

    return labels, num_objects


@nb.njit(cache=True, nogil=True)
def _find_object_first_indices(
    labeled_mask: NDArray[np.int32],
    num_objects: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return each object's first column and row in TRACK scan ordering.

    ``hierarc_segment()`` emits objects from west to east for the regular
    global workflows supported by the Hodges tracker.  MGE initialization is
    sensitive to this feature ordering when costs tie, so the detector keeps
    the source object order instead of its incidental row-major label order.
    """
    ny, nx = labeled_mask.shape
    first_rows = np.full(num_objects + 1, ny, dtype=np.int64)
    first_columns = np.full(num_objects + 1, nx, dtype=np.int64)
    for row in range(ny):
        for column in range(nx):
            object_id = labeled_mask[row, column]
            if object_id == 0:
                continue
            if column < first_columns[object_id]:
                first_columns[object_id] = column
                first_rows[object_id] = row
            elif column == first_columns[object_id] and row < first_rows[object_id]:
                first_rows[object_id] = row
    return first_rows, first_columns


@nb.njit(cache=True, nogil=True)
def _find_object_extrema(
    frame: NDArray[np.float64],
    labeled_mask: NDArray[np.int32],
    num_objects: int,
    size: int,
    is_min: bool,
    min_points: int,
    periodic_x: bool = True,
    exclude_boundary_extrema: bool = False,
) -> NDArray[np.float64]:
    """Find local extrema within objects matching TRACK feature identification."""
    ny, nx = frame.shape
    extrema = np.zeros_like(frame)
    half = size // 2

    # Calculate object sizes
    object_sizes = np.zeros(num_objects + 1, dtype=np.int32)
    for i in range(ny):
        for j in range(nx):
            if labeled_mask[i, j] > 0:
                object_sizes[labeled_mask[i, j]] += 1

    for i in range(ny):
        for j in range(nx):
            obj_id = labeled_mask[i, j]
            if obj_id == 0 or object_sizes[obj_id] < min_points:
                continue

            val = frame[i, j]
            is_extrema = True
            for di in range(-half, half + 1):
                ni = i + di
                if ni < 0 or ni >= ny:
                    continue
                for dj in range(-half, half + 1):
                    if di == 0 and dj == 0:
                        continue
                    nj = j + dj
                    if periodic_x:
                        nj %= nx
                    elif nj < 0 or nj >= nx:
                        continue
                    if labeled_mask[ni, nj] != obj_id:
                        if exclude_boundary_extrema:
                            is_extrema = False
                            break
                        continue
                    nval = frame[ni, nj]

                    if is_min:
                        if nval < val:
                            is_extrema = False
                            break
                    else:
                        if nval > val:
                            is_extrema = False
                            break
                if not is_extrema:
                    break

            if is_extrema:
                extrema[i, j] = 1.0

    return extrema


@nb.njit(cache=True, nogil=True)
def _group_object_extrema(
    extrema: NDArray[np.float64],
    labeled_mask: NDArray[np.int32],
    frame: NDArray[np.float64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    periodic_x: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
]:
    """Group 8-connected extrema separately within each threshold object.

    This is the TRACK ``tf = 4`` local-maximum grouping mode. Group positions
    are means of the underlying coordinate values; the primary value is the
    first row-major member's value.
    """
    ny, nx = extrema.shape
    component_labels = np.zeros((ny, nx), dtype=np.int32)
    label_count = 0

    for i in range(ny):
        for j in range(nx):
            if extrema[i, j] > 0.0:
                label_count += 1
                component_labels[i, j] = label_count

    if label_count == 0:
        empty_float = np.empty(0, dtype=np.float64)
        return empty_float, empty_float, empty_float, np.empty(0, dtype=np.int32)

    changed = True
    while changed:
        changed = False
        for i in range(ny):
            for j in range(nx):
                current = component_labels[i, j]
                if current == 0:
                    continue
                object_id = labeled_mask[i, j]
                lowest = current
                for di in range(-1, 2):
                    ni = i + di
                    if ni < 0 or ni >= ny:
                        continue
                    for dj in range(-1, 2):
                        nj = j + dj
                        if periodic_x:
                            nj %= nx
                        elif nj < 0 or nj >= nx:
                            continue
                        neighbor = component_labels[ni, nj]
                        if (
                            neighbor > 0
                            and labeled_mask[ni, nj] == object_id
                            and neighbor < lowest
                        ):
                            lowest = neighbor
                if lowest != current:
                    component_labels[i, j] = lowest
                    changed = True

        for i in range(ny - 1, -1, -1):
            for j in range(nx - 1, -1, -1):
                current = component_labels[i, j]
                if current == 0:
                    continue
                object_id = labeled_mask[i, j]
                lowest = current
                for di in range(-1, 2):
                    ni = i + di
                    if ni < 0 or ni >= ny:
                        continue
                    for dj in range(-1, 2):
                        nj = j + dj
                        if periodic_x:
                            nj %= nx
                        elif nj < 0 or nj >= nx:
                            continue
                        neighbor = component_labels[ni, nj]
                        if (
                            neighbor > 0
                            and labeled_mask[ni, nj] == object_id
                            and neighbor < lowest
                        ):
                            lowest = neighbor
                if lowest != current:
                    component_labels[i, j] = lowest
                    changed = True

    unique_labels = np.unique(component_labels)
    unique_labels = unique_labels[unique_labels > 0]
    group_count = len(unique_labels)
    label_map = np.zeros(label_count + 1, dtype=np.int32)
    for group_index in range(group_count):
        label_map[unique_labels[group_index]] = group_index + 1

    group_lats = np.zeros(group_count, dtype=np.float64)
    group_lons = np.zeros(group_count, dtype=np.float64)
    group_values = np.zeros(group_count, dtype=np.float64)
    group_object_ids = np.zeros(group_count, dtype=np.int32)
    group_counts = np.zeros(group_count, dtype=np.int32)
    reference_lons = np.zeros(group_count, dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            component = component_labels[i, j]
            if component == 0:
                continue
            group_index = label_map[component] - 1
            if group_counts[group_index] == 0:
                reference_lons[group_index] = lon[j]
                group_values[group_index] = frame[i, j]
                group_object_ids[group_index] = labeled_mask[i, j]
            group_lats[group_index] += lat[i]
            group_lon = lon[j]
            if periodic_x:
                reference_lon = reference_lons[group_index]
                group_lon = (
                    reference_lon + (group_lon - reference_lon + 180.0) % 360.0 - 180.0
                )
            group_lons[group_index] += group_lon
            group_counts[group_index] += 1

    for group_index in range(group_count):
        group_lats[group_index] /= group_counts[group_index]
        group_lons[group_index] /= group_counts[group_index]

    return group_lats, group_lons, group_values, group_object_ids


@nb.njit(cache=True, nogil=True)
def _compute_cell_area(lat: float, dlat: float, dlon: float) -> float:
    """Calculate the area of a grid cell in km^2."""
    return float(
        R_EARTH_KM**2
        * np.cos(lat * DEG_TO_RAD)
        * (dlat * DEG_TO_RAD)
        * (dlon * DEG_TO_RAD)
    )


@nb.njit(cache=True, nogil=True)
def _compute_object_properties(
    frame: NDArray[np.float64],
    labeled_mask: NDArray[np.int32],
    num_objects: int,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    threshold: float,
    is_min: bool,
    spherical_coords: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Calculate object properties: raw_size, fitted_size, axes, orientation."""
    ny, nx = frame.shape
    raw_areas = np.zeros(num_objects + 1, dtype=np.float64)
    fitted_areas = np.zeros(num_objects + 1, dtype=np.float64)
    major_axes = np.zeros(num_objects + 1, dtype=np.float64)
    minor_axes = np.zeros(num_objects + 1, dtype=np.float64)
    orientations = np.zeros(num_objects + 1, dtype=np.float64)

    # Moments for each object
    m00 = np.zeros(num_objects + 1, dtype=np.float64)
    m10 = np.zeros(num_objects + 1, dtype=np.float64)
    m01 = np.zeros(num_objects + 1, dtype=np.float64)
    m20 = np.zeros(num_objects + 1, dtype=np.float64)
    m02 = np.zeros(num_objects + 1, dtype=np.float64)
    m11 = np.zeros(num_objects + 1, dtype=np.float64)
    reference_x = np.zeros(num_objects + 1, dtype=np.float64)
    has_reference_x = np.zeros(num_objects + 1, dtype=np.bool_)

    if spherical_coords:
        for i in range(ny):
            for j in range(nx):
                obj_id = labeled_mask[i, j]
                if obj_id > 0 and not has_reference_x[obj_id]:
                    reference_x[obj_id] = lon[j]
                    has_reference_x[obj_id] = True

    dy = abs(lat[1] - lat[0]) if ny > 1 else 1.0
    dx = abs(lon[1] - lon[0]) if nx > 1 else 1.0

    for i in range(ny):
        area_cell = _compute_cell_area(lat[i], dy, dx) if spherical_coords else dy * dx
        for j in range(nx):
            obj_id = labeled_mask[i, j]
            if obj_id == 0:
                continue

            raw_areas[obj_id] += area_cell

            # Intensity weighting: difference from threshold
            val = frame[i, j]
            weight = abs(val - threshold)

            # Use lat/lon directly for moments
            y = lat[i]
            x = lon[j]
            if spherical_coords:
                reference = reference_x[obj_id]
                x = reference + (x - reference + 180.0) % 360.0 - 180.0

            m00[obj_id] += weight
            m10[obj_id] += weight * x
            m01[obj_id] += weight * y
            m20[obj_id] += weight * x**2
            m02[obj_id] += weight * y**2
            m11[obj_id] += weight * x * y

    for obj_id in range(1, num_objects + 1):
        if m00[obj_id] == 0:
            continue

        # Centroid
        cx = m10[obj_id] / m00[obj_id]
        cy = m01[obj_id] / m00[obj_id]

        # Central moments
        mu20 = m20[obj_id] / m00[obj_id] - cx**2
        mu02 = m02[obj_id] / m00[obj_id] - cy**2
        mu11 = m11[obj_id] / m00[obj_id] - cx * cy

        if spherical_coords:
            km_per_deg_lon = KM_PER_DEG * np.cos(cy * DEG_TO_RAD)
            a = mu20 * km_per_deg_lon**2
            b = mu11 * km_per_deg_lon * KM_PER_DEG
            c = mu02 * KM_PER_DEG**2
        else:
            a = mu20
            b = mu11
            c = mu02

        term1 = (a + c) / 2.0
        term2 = np.sqrt(((a - c) / 2.0) ** 2 + b**2)

        lambda1 = term1 + term2
        lambda2 = term1 - term2

        major = 2.0 * np.sqrt(max(0.0, lambda1))
        minor = 2.0 * np.sqrt(max(0.0, lambda2))

        major_axes[obj_id] = major
        minor_axes[obj_id] = minor
        fitted_areas[obj_id] = np.pi * major * minor

        if abs(a - c) < 1e-10:
            orientations[obj_id] = 0.0
        else:
            orientations[obj_id] = 0.5 * np.arctan2(2.0 * b, a - c) * 180.0 / np.pi

    return raw_areas, fitted_areas, major_axes, minor_axes, orientations


@nb.njit(cache=True, nogil=True)
def _extract_centers(
    extrema: NDArray[np.float64],
    frame: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Extract grid indices and values of detected extrema."""
    idx = np.where(extrema > 0)
    r_idx = idx[0]
    c_idx = idx[1]
    vals = np.zeros(len(r_idx), dtype=np.float64)
    for i in range(len(r_idx)):
        vals[i] = frame[r_idx[i], c_idx[i]]
    return r_idx, c_idx, vals


@dataclass
class _TrackObjectPoint:
    y: int
    x: int
    val: float


@dataclass
class _TrackObject:
    id: int
    pts: list[_TrackObjectPoint] = field(default_factory=list)
    b_or_i: str = "i"


@dataclass
class _TrackExtremaPoint:
    ixy: int
    jxy: int
    val: float
    local_x: int
    local_y: int


def _detect_track_rectangular_candidates_reference(
    frame: NDArray[np.float64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    intensity_threshold: float,
    is_min: bool,
    min_grid_points: int,
    profile: dict[str, float | int] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
]:
    """Reproduce TRACK rectangular CCL, boundary merging, and extrema extraction."""
    profile_started = time.perf_counter() if profile is not None else 0.0
    # TRACK's rectangular object path is defined on a 0--360 degree cyclic
    # coordinate.  Normalize and reorder the physical field before extending
    # its periodic endpoint so signed and unsigned global inputs follow the
    # same object, candidate, spline, and GDFP coordinate path.
    normalized_lon = np.mod(lon.astype(np.float64, copy=False), 360.0)
    longitude_order = np.argsort(normalized_lon)
    lon = normalized_lon[longitude_order]
    frame = frame[:, longitude_order]
    ny, nx = frame.shape
    longitude_lower = float(lon[0])
    longitude_upper = longitude_lower + 360.0
    lon_ext = np.append(lon, longitude_upper)

    def obj_xreal(k: int) -> float:
        if k < 0:
            return float(lon_ext[k + nx] - 360.0)
        elif k >= nx + 1:
            return float(lon_ext[k - nx] + 360.0)
        return float(lon_ext[k])

    frame_ext = np.zeros((ny, nx + 1), dtype=np.float64)
    frame_ext[:, :nx] = frame
    frame_ext[:, nx] = frame[:, 0]

    mask = (
        frame_ext <= intensity_threshold if is_min else frame_ext >= intensity_threshold
    )
    labels = np.zeros((ny, nx + 1), dtype=np.int32)
    label_count = 0
    for i in range(ny):
        for j in range(nx + 1):
            if mask[i, j]:
                label_count += 1
                labels[i, j] = label_count

    if profile is not None:
        profile["thresholded_grid_points"] = label_count
        profile["threshold_seconds"] = time.perf_counter() - profile_started

    changed = True
    propagation_started = time.perf_counter() if profile is not None else 0.0
    propagation_passes = 0
    while changed:
        propagation_passes += 1
        changed = False
        for i in range(ny):
            for j in range(nx + 1):
                cur = labels[i, j]
                if cur == 0:
                    continue
                lowest = cur
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < ny and 0 <= nj < nx + 1:
                        nbr = labels[ni, nj]
                        if 0 < nbr < lowest:
                            lowest = nbr
                if lowest != cur:
                    labels[i, j] = lowest
                    changed = True
        for i in range(ny - 1, -1, -1):
            for j in range(nx, -1, -1):
                cur = labels[i, j]
                if cur == 0:
                    continue
                lowest = cur
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < ny and 0 <= nj < nx + 1:
                        nbr = labels[ni, nj]
                        if 0 < nbr < lowest:
                            lowest = nbr
                if lowest != cur:
                    labels[i, j] = lowest
                    changed = True

    if profile is not None:
        profile["propagation_passes"] = propagation_passes
        profile["propagation_seconds"] = time.perf_counter() - propagation_started

    materialization_started = time.perf_counter() if profile is not None else 0.0
    unique_labs = [int(u) for u in np.unique(labels) if u > 0]
    objects: list[_TrackObject] = []
    for lab in unique_labs:
        pts: list[_TrackObjectPoint] = []
        rows, cols = np.where(labels == lab)
        for idx_pt in range(len(rows)):
            r, c = rows[idx_pt], cols[idx_pt]
            pts.append(
                _TrackObjectPoint(
                    y=int(r + 1),
                    x=int(c + 1),
                    val=float(frame_ext[r, c]) * (-0.01 if is_min else 0.01),
                )
            )
        objects.append(_TrackObject(id=len(objects) + 1, pts=pts, b_or_i="i"))

    if profile is not None:
        profile["initial_object_count"] = len(objects)
        profile["materialized_points"] = sum(len(ob.pts) for ob in objects)
        profile["materialization_seconds"] = (
            time.perf_counter() - materialization_started
        )

    boundary_started = time.perf_counter() if profile is not None else 0.0
    for ob in objects:
        bi = 0
        for pt in ob.pts:
            px = pt.x
            py = pt.y
            if px == 1 or px == nx + 1:
                if bi == 0:
                    bi = 1
                    ob.b_or_i = "x"
                elif bi == 1 and ob.b_or_i == "y":
                    ob.b_or_i = "b"
            if py == 1 or py == ny:
                if bi == 0:
                    bi = 1
                    ob.b_or_i = "y"
                elif bi == 1 and ob.b_or_i == "x":
                    ob.b_or_i = "b"

    lb = np.zeros(ny, dtype=np.int32)
    rb = np.zeros(ny, dtype=np.int32)
    for idx, ob in enumerate(objects):
        ii = idx + 1
        if ob.b_or_i in ("x", "b"):
            ilb = 0
            for pt in ob.pts:
                ipx = pt.x
                ip = pt.y - 1
                if ipx == 1:
                    lb[ip] = ii
                    if not ilb:
                        ob.b_or_i = "l"
                        ilb = 1
                    elif ilb and ob.b_or_i == "r":
                        ob.b_or_i = "b"
                elif ipx == nx + 1:
                    rb[ip] = ii
                    if not ilb:
                        ob.b_or_i = "r"
                        ilb = 1
                    elif ilb and ob.b_or_i == "l":
                        ob.b_or_i = "b"

    boundary_merges = 0
    for i in range(ny):
        l1 = int(lb[i])
        r1 = int(rb[i])
        if l1 > 0 and r1 > 0 and l1 != r1:
            ob1 = objects[l1 - 1]
            ob2 = objects[r1 - 1]
            if len(ob1.pts) > len(ob2.pts):
                ob, cob = ob1, ob2
                lb2 = l1
                ll = rb
                inx = nx + 1
            else:
                ob, cob = ob2, ob1
                lb2 = r1
                ll = lb
                inx = 1

            kept_cob_pts: list[_TrackObjectPoint] = []
            cob_boi = cob.b_or_i
            for pt2 in cob.pts:
                ixx = pt2.x
                if ixx == inx:
                    ll[pt2.y - 1] = lb2
                else:
                    if cob_boi == "r":
                        pt2.x = pt2.x - nx
                    elif cob_boi == "l":
                        pt2.x = pt2.x + nx
                    kept_cob_pts.append(pt2)
            ob.pts.extend(kept_cob_pts)
            cob.pts = []
            cob.b_or_i = "n"
            boundary_merges += 1

    if profile is not None:
        profile["boundary_merges"] = boundary_merges
        profile["boundary_seconds"] = time.perf_counter() - boundary_started

    filtered_objects = [ob for ob in objects if len(ob.pts) >= min_grid_points]

    extrema_started = time.perf_counter() if profile is not None else 0.0
    extrema_count = 0
    cand_lats: list[float] = []
    cand_lons: list[float] = []
    cand_vals: list[float] = []
    cand_obj_ids: list[int] = []

    for ob_idx, ob in enumerate(filtered_objects):
        pxmn = min(pt.x for pt in ob.pts)
        pxmx = max(pt.x for pt in ob.pts)
        pymn = min(pt.y for pt in ob.pts)
        pymx = max(pt.y for pt in ob.pts)

        xdim = pxmx - pxmn + 3
        ydim = pymx - pymn + 3
        aa = np.full((ydim, xdim), -np.inf, dtype=np.float64)
        aa_mask = np.zeros((ydim, xdim), dtype=bool)
        for pt in ob.pts:
            aa[pt.y - pymn + 1, pt.x - pxmn + 1] = pt.val
            aa_mask[pt.y - pymn + 1, pt.x - pxmn + 1] = True

        extrema_pts: list[_TrackExtremaPoint] = []
        for j in range(1, ydim - 1):
            for k in range(1, xdim - 1):
                if aa_mask[j, k]:
                    val = aa[j, k]
                    is_local_max = True
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            if dj == 0 and dk == 0:
                                continue
                            if aa[j + dj, k + dk] >= val:
                                is_local_max = False
                                break
                        if not is_local_max:
                            break
                    if is_local_max:
                        extrema_pts.append(
                            _TrackExtremaPoint(
                                ixy=k + pxmn - 1,
                                jxy=j + pymn - 1,
                                val=val,
                                local_x=k,
                                local_y=j,
                            )
                        )

        if not extrema_pts:
            continue

        extrema_count += len(extrema_pts)
        if len(extrema_pts) == 1:
            fpt = extrema_pts[0]
            sx = obj_xreal(fpt.ixy - 1)
            sy = float(lat[fpt.jxy - 1])
            if sx < longitude_lower:
                sx += 360.0
            elif sx > longitude_upper:
                sx -= 360.0
            cand_lats.append(sy)
            cand_lons.append(sx)
            cand_vals.append(float(fpt.val) * (-100.0 if is_min else 100.0))
            cand_obj_ids.append(ob_idx + 1)
        else:
            n_ext = len(extrema_pts)
            ext_labels = list(range(1, n_ext + 1))
            for a in range(n_ext):
                for b in range(a + 1, n_ext):
                    if (
                        max(
                            abs(extrema_pts[a].local_x - extrema_pts[b].local_x),
                            abs(extrema_pts[a].local_y - extrema_pts[b].local_y),
                        )
                        <= 1
                    ):
                        la, lb_ = ext_labels[a], ext_labels[b]
                        lowest = min(la, lb_)
                        for idx_ in range(n_ext):
                            if ext_labels[idx_] in (la, lb_):
                                ext_labels[idx_] = lowest
            unique_ext_labs: list[int] = []
            for l_ in ext_labels:
                if l_ not in unique_ext_labs:
                    unique_ext_labs.append(l_)
            for ulab in unique_ext_labs:
                grp = [
                    extrema_pts[idx_]
                    for idx_ in range(n_ext)
                    if ext_labels[idx_] == ulab
                ]
                sx = sum(obj_xreal(pt.ixy - 1) for pt in grp) / len(grp)
                sy = sum(float(lat[pt.jxy - 1]) for pt in grp) / len(grp)
                if sx < longitude_lower:
                    sx += 360.0
                elif sx > longitude_upper:
                    sx -= 360.0
                first_val = float(grp[0].val)
                cand_lats.append(sy)
                cand_lons.append(sx)
                cand_vals.append(first_val * (-100.0 if is_min else 100.0))
                cand_obj_ids.append(ob_idx + 1)

    if profile is not None:
        profile["filtered_object_count"] = len(filtered_objects)
        profile["candidate_extrema_count"] = extrema_count
        profile["extrema_seconds"] = time.perf_counter() - extrema_started

    return (
        np.array(cand_lats, dtype=np.float64),
        np.array(cand_lons, dtype=np.float64),
        np.array(cand_vals, dtype=np.float64),
        np.array(cand_obj_ids, dtype=np.int32),
    )


@nb.njit(cache=True, nogil=True)
def _rectangular_object_longitude(
    index: int,
    longitude_extended: NDArray[np.float64],
    nx: int,
) -> float:
    """Return the TRACK-compatible real longitude for one grid index."""
    if index < 0:
        return float(longitude_extended[index + nx] - 360.0)
    if index >= nx + 1:
        return float(longitude_extended[index - nx] + 360.0)
    return float(longitude_extended[index])


@nb.njit(cache=True, nogil=True)
def _detect_rectangular_candidates_numba(
    frame: NDArray[np.float64],
    lat: NDArray[np.float64],
    longitude_extended: NDArray[np.float64],
    intensity_threshold: float,
    is_min: bool,
    min_grid_points: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
]:
    """Packed Numba implementation of the rectangular TRACK scan.

    This keeps the reference implementation above available for exact
    equivalence checks during the reconciliation work.  Labels, object points,
    boundary merges, local extrema, and adjacent-extrema grouping follow the same
    source order
    and tie rules; only transient Python objects are replaced by arrays and
    linked lists.
    """
    ny, nx = frame.shape
    nxe = nx + 1
    frame_ext = np.empty((ny, nxe), dtype=np.float64)
    for row in range(ny):
        for column in range(nx):
            frame_ext[row, column] = frame[row, column]
        frame_ext[row, nx] = frame[row, 0]

    labels = np.zeros((ny, nxe), dtype=np.int32)
    label_count = 0
    for row in range(ny):
        for column in range(nxe):
            value = frame_ext[row, column]
            eligible = (
                value <= intensity_threshold if is_min else value >= intensity_threshold
            )
            if eligible:
                label_count += 1
                labels[row, column] = label_count

    changed = True
    while changed:
        changed = False
        for row in range(ny):
            for column in range(nxe):
                current = labels[row, column]
                if current == 0:
                    continue
                lowest = current
                for direction in range(4):
                    if direction == 0:
                        neighbor_row = row - 1
                        neighbor_column = column
                    elif direction == 1:
                        neighbor_row = row + 1
                        neighbor_column = column
                    elif direction == 2:
                        neighbor_row = row
                        neighbor_column = column - 1
                    else:
                        neighbor_row = row
                        neighbor_column = column + 1
                    if 0 <= neighbor_row < ny and 0 <= neighbor_column < nxe:
                        neighbor = labels[neighbor_row, neighbor_column]
                        if 0 < neighbor < lowest:
                            lowest = neighbor
                if lowest != current:
                    labels[row, column] = lowest
                    changed = True

        for row in range(ny - 1, -1, -1):
            for column in range(nxe - 1, -1, -1):
                current = labels[row, column]
                if current == 0:
                    continue
                lowest = current
                for direction in range(4):
                    if direction == 0:
                        neighbor_row = row - 1
                        neighbor_column = column
                    elif direction == 1:
                        neighbor_row = row + 1
                        neighbor_column = column
                    elif direction == 2:
                        neighbor_row = row
                        neighbor_column = column - 1
                    else:
                        neighbor_row = row
                        neighbor_column = column + 1
                    if 0 <= neighbor_row < ny and 0 <= neighbor_column < nxe:
                        neighbor = labels[neighbor_row, neighbor_column]
                        if 0 < neighbor < lowest:
                            lowest = neighbor
                if lowest != current:
                    labels[row, column] = lowest
                    changed = True

    # np.unique(labels)[np.unique(labels) > 0] is sorted.  A root label is the
    # first row-major cell of its component, so first-seen compaction is the
    # same source ordering without constructing a temporary Python list.
    label_map = np.zeros(label_count + 1, dtype=np.int32)
    num_objects = 0
    for row in range(ny):
        for column in range(nxe):
            root = labels[row, column]
            if root > 0 and label_map[root] == 0:
                num_objects += 1
                label_map[root] = num_objects
    for row in range(ny):
        for column in range(nxe):
            root = labels[row, column]
            if root > 0:
                labels[row, column] = label_map[root]

    point_count = 0
    for row in range(ny):
        for column in range(nxe):
            if labels[row, column] > 0:
                point_count += 1

    point_y = np.empty(point_count, dtype=np.int32)
    point_x = np.empty(point_count, dtype=np.int32)
    point_values = np.empty(point_count, dtype=np.float64)
    point_next = np.full(point_count, -1, dtype=np.int64)
    object_head = np.full(num_objects + 1, -1, dtype=np.int64)
    object_tail = np.full(num_objects + 1, -1, dtype=np.int64)
    object_sizes = np.zeros(num_objects + 1, dtype=np.int64)
    point_index = 0
    value_scale = -0.01 if is_min else 0.01
    for row in range(ny):
        for column in range(nxe):
            object_id = labels[row, column]
            if object_id == 0:
                continue
            point_y[point_index] = row + 1
            point_x[point_index] = column + 1
            point_values[point_index] = frame_ext[row, column] * value_scale
            if object_head[object_id] < 0:
                object_head[object_id] = point_index
            else:
                point_next[object_tail[object_id]] = point_index
            object_tail[object_id] = point_index
            object_sizes[object_id] += 1
            point_index += 1

    # TRACK boundary state codes: i, x, y, b, l, r, n.
    state_x = 1
    state_y = 2
    state_b = 3
    state_l = 4
    state_r = 5
    state_n = 6
    object_state = np.zeros(num_objects + 1, dtype=np.int8)
    for object_id in range(1, num_objects + 1):
        boundary_indicator = 0
        point = object_head[object_id]
        while point >= 0:
            px = point_x[point]
            py = point_y[point]
            if px == 1 or px == nxe:
                if boundary_indicator == 0:
                    boundary_indicator = 1
                    object_state[object_id] = state_x
                elif boundary_indicator == 1 and object_state[object_id] == state_y:
                    object_state[object_id] = state_b
            if py == 1 or py == ny:
                if boundary_indicator == 0:
                    boundary_indicator = 1
                    object_state[object_id] = state_y
                elif boundary_indicator == 1 and object_state[object_id] == state_x:
                    object_state[object_id] = state_b
            point = point_next[point]

    left_boundary = np.zeros(ny, dtype=np.int32)
    right_boundary = np.zeros(ny, dtype=np.int32)
    for object_id in range(1, num_objects + 1):
        if object_state[object_id] not in (state_x, state_b):
            continue
        boundary_indicator = 0
        point = object_head[object_id]
        while point >= 0:
            px = point_x[point]
            row = point_y[point] - 1
            if px == 1:
                left_boundary[row] = object_id
                if boundary_indicator == 0:
                    object_state[object_id] = state_l
                    boundary_indicator = 1
                elif object_state[object_id] == state_r:
                    object_state[object_id] = state_b
            elif px == nxe:
                right_boundary[row] = object_id
                if boundary_indicator == 0:
                    object_state[object_id] = state_r
                    boundary_indicator = 1
                elif object_state[object_id] == state_l:
                    object_state[object_id] = state_b
            point = point_next[point]

    for row in range(ny):
        left_id = int(left_boundary[row])
        right_id = int(right_boundary[row])
        if left_id <= 0 or right_id <= 0 or left_id == right_id:
            continue
        if object_sizes[left_id] > object_sizes[right_id]:
            object_id = left_id
            combined_id = right_id
            boundary = right_boundary
            endpoint = nxe
        else:
            object_id = right_id
            combined_id = left_id
            boundary = left_boundary
            endpoint = 1

        combined_state = object_state[combined_id]
        point = object_head[combined_id]
        while point >= 0:
            next_point = point_next[point]
            if point_x[point] == endpoint:
                boundary[point_y[point] - 1] = object_id
            else:
                if combined_state == state_r:
                    point_x[point] -= nx
                elif combined_state == state_l:
                    point_x[point] += nx
                point_next[point] = -1
                if object_head[object_id] < 0:
                    object_head[object_id] = point
                else:
                    point_next[object_tail[object_id]] = point
                object_tail[object_id] = point
                object_sizes[object_id] += 1
            point = next_point
        object_head[combined_id] = -1
        object_tail[combined_id] = -1
        object_sizes[combined_id] = 0
        object_state[combined_id] = state_n

    filtered_ids = np.empty(num_objects, dtype=np.int32)
    filtered_count = 0
    for object_id in range(1, num_objects + 1):
        if object_sizes[object_id] >= min_grid_points:
            filtered_ids[filtered_count] = object_id
            filtered_count += 1

    candidate_lats = np.empty(point_count, dtype=np.float64)
    candidate_lons = np.empty(point_count, dtype=np.float64)
    candidate_values = np.empty(point_count, dtype=np.float64)
    candidate_object_ids = np.empty(point_count, dtype=np.int32)
    candidate_count = 0
    for filtered_index in range(filtered_count):
        object_id = filtered_ids[filtered_index]
        px_min = nx + 1
        px_max = 0
        py_min = ny + 1
        py_max = 0
        point = object_head[object_id]
        while point >= 0:
            px = point_x[point]
            py = point_y[point]
            px_min = min(px_min, px)
            px_max = max(px_max, px)
            py_min = min(py_min, py)
            py_max = max(py_max, py)
            point = point_next[point]

        xdim = px_max - px_min + 3
        ydim = py_max - py_min + 3
        values = np.full((ydim, xdim), -np.inf, dtype=np.float64)
        present = np.zeros((ydim, xdim), dtype=np.bool_)
        point = object_head[object_id]
        while point >= 0:
            local_x = point_x[point] - px_min + 1
            local_y = point_y[point] - py_min + 1
            values[local_y, local_x] = point_values[point]
            present[local_y, local_x] = True
            point = point_next[point]

        extrema_ix = np.empty(object_sizes[object_id], dtype=np.int32)
        extrema_iy = np.empty(object_sizes[object_id], dtype=np.int32)
        extrema_values = np.empty(object_sizes[object_id], dtype=np.float64)
        extrema_local_x = np.empty(object_sizes[object_id], dtype=np.int32)
        extrema_local_y = np.empty(object_sizes[object_id], dtype=np.int32)
        extrema_count = 0
        for local_y in range(1, ydim - 1):
            for local_x in range(1, xdim - 1):
                if not present[local_y, local_x]:
                    continue
                value = values[local_y, local_x]
                is_local_maximum = True
                for delta_y in range(-1, 2):
                    for delta_x in range(-1, 2):
                        if delta_y == 0 and delta_x == 0:
                            continue
                        if values[local_y + delta_y, local_x + delta_x] >= value:
                            is_local_maximum = False
                            break
                    if not is_local_maximum:
                        break
                if is_local_maximum:
                    extrema_ix[extrema_count] = local_x + px_min - 1
                    extrema_iy[extrema_count] = local_y + py_min - 1
                    extrema_values[extrema_count] = value
                    extrema_local_x[extrema_count] = local_x
                    extrema_local_y[extrema_count] = local_y
                    extrema_count += 1

        if extrema_count == 0:
            continue
        if extrema_count == 1:
            grid_x = extrema_ix[0] - 1
            longitude = _rectangular_object_longitude(grid_x, longitude_extended, nx)
            if longitude < longitude_extended[0]:
                longitude += 360.0
            elif longitude > longitude_extended[nx]:
                longitude -= 360.0
            candidate_lats[candidate_count] = lat[extrema_iy[0] - 1]
            candidate_lons[candidate_count] = longitude
            candidate_values[candidate_count] = extrema_values[0] * (
                -100.0 if is_min else 100.0
            )
            candidate_object_ids[candidate_count] = filtered_index + 1
            candidate_count += 1
            continue

        extrema_labels = np.arange(extrema_count, dtype=np.int32) + 1
        for first in range(extrema_count):
            for second in range(first + 1, extrema_count):
                if (
                    max(
                        abs(extrema_local_x[first] - extrema_local_x[second]),
                        abs(extrema_local_y[first] - extrema_local_y[second]),
                    )
                    <= 1
                ):
                    first_label = extrema_labels[first]
                    second_label = extrema_labels[second]
                    lowest = min(first_label, second_label)
                    for index in range(extrema_count):
                        if (
                            extrema_labels[index] == first_label
                            or extrema_labels[index] == second_label
                        ):
                            extrema_labels[index] = lowest

        unique_extrema_labels = np.empty(extrema_count, dtype=np.int32)
        unique_count = 0
        for index in range(extrema_count):
            label = extrema_labels[index]
            already_seen = False
            for unique_index in range(unique_count):
                if unique_extrema_labels[unique_index] == label:
                    already_seen = True
                    break
            if not already_seen:
                unique_extrema_labels[unique_count] = label
                unique_count += 1

        for unique_index in range(unique_count):
            group_label = unique_extrema_labels[unique_index]
            longitude_sum = 0.0
            latitude_sum = 0.0
            group_size = 0
            first_value = 0.0
            for index in range(extrema_count):
                if extrema_labels[index] != group_label:
                    continue
                if group_size == 0:
                    first_value = extrema_values[index]
                grid_x = extrema_ix[index] - 1
                longitude_sum += _rectangular_object_longitude(
                    grid_x, longitude_extended, nx
                )
                latitude_sum += lat[extrema_iy[index] - 1]
                group_size += 1
            longitude = longitude_sum / group_size
            if longitude < longitude_extended[0]:
                longitude += 360.0
            elif longitude > longitude_extended[nx]:
                longitude -= 360.0
            candidate_lats[candidate_count] = latitude_sum / group_size
            candidate_lons[candidate_count] = longitude
            candidate_values[candidate_count] = first_value * (
                -100.0 if is_min else 100.0
            )
            candidate_object_ids[candidate_count] = filtered_index + 1
            candidate_count += 1

    return (
        candidate_lats[:candidate_count],
        candidate_lons[:candidate_count],
        candidate_values[:candidate_count],
        candidate_object_ids[:candidate_count],
    )


def _detect_track_rectangular_candidates(
    frame: NDArray[np.float64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    intensity_threshold: float,
    is_min: bool,
    min_grid_points: int,
    grid: RectangularGridPreparation | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
]:
    """Detect rectangular TRACK candidates with packed native numerics."""
    prepared_grid = (
        grid
        if grid is not None
        else prepare_rectangular_grid(lat, lon, periodic_x=True)
    )
    if not prepared_grid.periodic_x:
        raise ValueError("rectangular TRACK candidates require periodic longitude")
    ordered_frame = np.ascontiguousarray(frame[:, prepared_grid.longitude_order])
    return _detect_rectangular_candidates_numba(
        ordered_frame,
        lat.astype(np.float64, copy=False),
        prepared_grid.extended_longitudes,
        intensity_threshold,
        is_min,
        min_grid_points,
    )


# ---------------------------------------------------------------------------
# Frame-Level Detection Orchestrator
# ---------------------------------------------------------------------------


def detect_hodges_frame(
    frame: NDArray[np.float64],
    time_val: TimeInput,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    *,
    intensity_threshold: float,
    mode: ResolvedDetectionMode = "min",
    search_window_size: int = DEFAULT_SEARCH_WINDOW_SIZE,
    min_grid_points: int = MIN_OBJECT_GRID_POINTS_DEFAULT,
    feature_refinement: HodgesFeatureRefinement = "bspline",
    group_adjacent_extrema: bool = False,
    exclude_boundary_extrema: bool = False,
    bspline_smoothing: float = 0.0,
    track_smoopy_optimization_scale: float = TRACK_SMOOPY_OPTIMIZATION_SCALE_DEFAULT,
    bspline_max_iterations: int = 100,
    bspline_gradient_tolerance: float = 1.0e-5,
    periodic_x: bool = True,
    projected_xy: bool = False,
    rectangular_grid: RectangularGridPreparation | None = None,
) -> tuple[HodgesCenterFrame, tuple[HodgesFeaturePointDiagnostic, ...]]:
    """Process a single 2D spatial frame for Hodges feature detection and refinement."""
    point_time = coerce_time_input(time_val)
    assert point_time is not None
    use_quadratic = feature_refinement == "quadratic"
    use_spherical_quadratic = feature_refinement == "spherical_quadratic"
    use_spherical_spline = feature_refinement == "spherical_bspline"
    use_rectangular_spline = feature_refinement == "bspline"
    use_grouped_extrema = group_adjacent_extrema or feature_refinement in (
        "bspline",
        "spherical_bspline",
        "quadratic",
    )
    is_min = mode == "min"

    raw_areas = np.empty(0, dtype=np.float64)
    fitted_areas = np.empty(0, dtype=np.float64)
    majors = np.empty(0, dtype=np.float64)
    minors = np.empty(0, dtype=np.float64)
    orientations = np.empty(0, dtype=np.float64)

    if use_rectangular_spline and periodic_x:
        (
            refined_lats,
            refined_lons,
            raw_vals,
            object_ids,
        ) = _detect_track_rectangular_candidates(
            frame,
            lat,
            lon,
            intensity_threshold=intensity_threshold,
            is_min=is_min,
            min_grid_points=min_grid_points,
            grid=rectangular_grid,
        )
        r_idx = np.empty(0, dtype=np.int64)
        c_idx = np.empty(0, dtype=np.int64)
        source_feature_order = np.arange(raw_vals.size, dtype=np.int64)
    else:
        # 1. Segment using Connected Component Labeling fused with threshold evaluation.
        labeled_mask, num_objects = _label_connected_components(
            frame,
            threshold=intensity_threshold,
            is_min=is_min,
            periodic_x=periodic_x,
        )
        (
            raw_areas,
            fitted_areas,
            majors,
            minors,
            orientations,
        ) = _compute_object_properties(
            frame,
            labeled_mask,
            num_objects,
            lat,
            lon,
            threshold=intensity_threshold,
            is_min=is_min,
            spherical_coords=not projected_xy,
        )
        object_first_rows, object_first_columns = _find_object_first_indices(
            labeled_mask,
            num_objects,
        )

        # 2. Find local extrema within each object.
        extrema = _find_object_extrema(
            frame,
            labeled_mask,
            num_objects,
            search_window_size,
            is_min,
            min_grid_points,
            periodic_x=periodic_x,
            exclude_boundary_extrema=exclude_boundary_extrema,
        )

        # 4. Extract grid extrema.
        if use_grouped_extrema:
            (
                refined_lats,
                refined_lons,
                raw_vals,
                object_ids,
            ) = _group_object_extrema(
                extrema,
                labeled_mask,
                frame,
                lat,
                lon,
                periodic_x=periodic_x,
            )
            r_idx = np.empty(0, dtype=np.int64)
            c_idx = np.empty(0, dtype=np.int64)
        else:
            r_idx, c_idx, raw_vals = _extract_centers(extrema, frame)
            refined_lats = lat[r_idx].astype(np.float64, copy=True)
            refined_lons = lon[c_idx].astype(np.float64, copy=True)
            object_ids = labeled_mask[r_idx, c_idx]

        source_feature_order = np.arange(raw_vals.size, dtype=np.int64)
        feature_order = np.lexsort(
            (
                np.asarray(refined_lons),
                np.asarray(refined_lats),
                object_first_rows[object_ids],
                object_first_columns[object_ids],
            )
        )
        refined_lats = refined_lats[feature_order]
        refined_lons = refined_lons[feature_order]
        raw_vals = raw_vals[feature_order]
        object_ids = object_ids[feature_order]
        source_feature_order = source_feature_order[feature_order]
        if r_idx.size > 0:
            r_idx = r_idx[feature_order]
            c_idx = c_idx[feature_order]
        if not use_grouped_extrema:
            r_idx = r_idx[feature_order]
            c_idx = c_idx[feature_order]

    # These are immutable candidate positions.  In particular, spherical
    # duplicate selection must not reconstruct an "initial" position from
    # coordinates that an earlier refinement pass has updated in place.
    initial_lats = refined_lats.copy()
    initial_lons = refined_lons.copy()

    n_feats = len(raw_vals)
    refined_values = np.zeros(n_feats)
    f_raw_size = np.zeros(n_feats)
    f_fit_size = np.zeros(n_feats)
    f_major = np.zeros(n_feats)
    f_minor = np.zeros(n_feats)
    f_orient = np.zeros(n_feats)

    diagnostics: list[HodgesFeaturePointDiagnostic] = []

    spline_build = (
        build_spherical_bspline_surface(
            frame,
            lat,
            lon,
            periodic_x=periodic_x,
            smoothing=bspline_smoothing,
        )
        if use_spherical_spline and n_feats > 0
        else None
    )
    if (
        use_spherical_spline
        and n_feats > 0
        and (spline_build is None or spline_build.surface is None)
    ):
        status = (
            spline_build.status
            if spline_build is not None
            else "spline_construction_failure"
        )
        raise RuntimeError(f"spherical_bspline surface construction failed: {status}")

    smoopy_build = (
        build_bspline_surface(
            frame,
            lat,
            lon,
            periodic_x=periodic_x,
            smoothing=bspline_smoothing,
            grid=rectangular_grid,
        )
        if use_rectangular_spline and n_feats > 0
        else None
    )
    if (
        use_rectangular_spline
        and n_feats > 0
        and (smoopy_build is None or smoopy_build.surface is None)
    ):
        status = (
            smoopy_build.status
            if smoopy_build is not None
            else "spline_construction_failure"
        )
        raise RuntimeError(f"bspline surface construction failed: {status}")

    if use_quadratic and n_feats > 0:
        # The existing index-space quadratic remains unchanged.
        q_lats, q_lons, q_vals = refine_quadratic_feature_coordinates(
            frame,
            refined_lats,
            refined_lons,
            lat,
            lon,
            periodic_x=periodic_x,
        )
        for i in range(n_feats):
            init_lat = float(refined_lats[i])
            init_lon = float(refined_lons[i])
            refined_lats[i] = q_lats[i]
            refined_lons[i] = q_lons[i]
            refined_values[i] = q_vals[i]
            diagnostics.append(
                HodgesFeaturePointDiagnostic(
                    init_lat,
                    init_lon,
                    "success",
                    None,
                )
            )
            obj_id = object_ids[i]
            f_raw_size[i] = raw_areas[obj_id]
            f_fit_size[i] = fitted_areas[obj_id]
            f_major[i] = majors[obj_id]
            f_minor[i] = minors[obj_id]
            f_orient[i] = orientations[obj_id]
    elif use_spherical_quadratic and n_feats > 0:
        # This method is defined on the immutable grid extrema identified by
        # the detector, rather than on a coordinate reconstructed from another
        # refinement method.
        spherical_quadratic = refine_spherical_quadratic_feature_points(
            frame,
            r_idx,
            c_idx,
            lat,
            lon,
            is_minimum=is_min,
            periodic_x=periodic_x,
        )
        for i in range(n_feats):
            refinement_status = spherical_quadratic_status_name(
                int(spherical_quadratic.status_codes[i])
            )
            refined_lats[i] = spherical_quadratic.latitudes[i]
            refined_lons[i] = spherical_quadratic.longitudes[i]
            refined_values[i] = spherical_quadratic.values[i]
            diagnostics.append(
                HodgesFeaturePointDiagnostic(
                    float(initial_lats[i]),
                    float(initial_lons[i]),
                    refinement_status,
                    None if refinement_status == "success" else refinement_status,
                )
            )
            obj_id = object_ids[i]
            if raw_areas.size > obj_id:
                f_raw_size[i] = raw_areas[obj_id]
                f_fit_size[i] = fitted_areas[obj_id]
                f_major[i] = majors[obj_id]
                f_minor[i] = minors[obj_id]
                f_orient[i] = orientations[obj_id]
    else:
        for i in range(n_feats):
            initial_latitude = float(initial_lats[i])
            initial_longitude = float(initial_lons[i])
            if use_spherical_spline:
                assert spline_build is not None
                assert spline_build.surface is not None
                refinement = refine_spherical_bspline_feature_point(
                    spline_build.surface,
                    initial_latitude,
                    initial_longitude,
                    is_minimum=is_min,
                    search_window_size=search_window_size,
                    max_iterations=bspline_max_iterations,
                    gradient_tolerance=bspline_gradient_tolerance,
                )
                if refinement.status == "success":
                    refined_lats[i] = refinement.latitude
                    refined_lons[i] = refinement.longitude
                    refined_values[i] = refinement.value
                    diagnostics.append(
                        HodgesFeaturePointDiagnostic(
                            initial_latitude,
                            initial_longitude,
                            "success",
                            None,
                        )
                    )
                else:
                    refined_values[i] = DUFF_FEATURE_VALUE
                    diagnostics.append(
                        HodgesFeaturePointDiagnostic(
                            initial_latitude,
                            initial_longitude,
                            refinement.status,
                            refinement.status,
                        )
                    )
            elif use_rectangular_spline:
                assert smoopy_build is not None
                assert smoopy_build.surface is not None
                refinement = refine_bspline_feature_point(
                    smoopy_build.surface,
                    initial_latitude,
                    initial_longitude,
                    is_minimum=is_min,
                    initial_value=float(raw_vals[i]),
                    optimization_scale=track_smoopy_optimization_scale,
                    max_iterations=bspline_max_iterations,
                    gradient_tolerance=bspline_gradient_tolerance,
                )
                if refinement.status == "success":
                    refined_lats[i] = refinement.latitude
                    refined_lons[i] = refinement.longitude
                    refined_values[i] = refinement.value
                    diagnostics.append(
                        HodgesFeaturePointDiagnostic(
                            initial_latitude,
                            initial_longitude,
                            "success",
                            None,
                        )
                    )
                else:
                    refined_values[i] = raw_vals[i]
                    diagnostics.append(
                        HodgesFeaturePointDiagnostic(
                            initial_latitude,
                            initial_longitude,
                            "optimizer_no_convergence",
                            refinement.status,
                        )
                    )
            else:  # grid
                refined_values[i] = raw_vals[i]
                diagnostics.append(
                    HodgesFeaturePointDiagnostic(
                        initial_latitude,
                        initial_longitude,
                        "success",
                        None,
                    )
                )

            obj_id = object_ids[i]
            if raw_areas.size > obj_id:
                f_raw_size[i] = raw_areas[obj_id]
                f_fit_size[i] = fitted_areas[obj_id]
                f_major[i] = majors[obj_id]
                f_minor[i] = minors[obj_id]
                f_orient[i] = orientations[obj_id]

    if use_rectangular_spline:
        for i in range(n_feats):
            for k in range(i):
                if object_ids[i] != object_ids[k]:
                    continue
                if (
                    geod_dist(
                        float(refined_lats[i]),
                        float(refined_lons[i]),
                        float(refined_lats[k]),
                        float(refined_lons[k]),
                    )
                    < FEATURE_DUPLICATE_TOLERANCE_RAD
                ):
                    refined_values[i] = DUFF_FEATURE_VALUE
                    break
    elif use_spherical_spline:
        for obj_id in np.unique(object_ids):
            cand_indices = np.flatnonzero(object_ids == obj_id)
            valid_cand = [
                int(idx)
                for idx in cand_indices
                if refined_values[idx] > DUFF_FEATURE_CUTOFF
            ]
            if len(valid_cand) <= 1:
                continue
            visited: set[int] = set()
            for c_idx_val in valid_cand:
                if c_idx_val in visited:
                    continue
                cluster: list[int] = [c_idx_val]
                visited.add(c_idx_val)
                for other in valid_cand:
                    if other in visited:
                        continue
                    d = geod_dist(
                        float(refined_lats[c_idx_val]),
                        float(refined_lons[c_idx_val]),
                        float(refined_lats[other]),
                        float(refined_lons[other]),
                    )
                    if d < FEATURE_DUPLICATE_TOLERANCE_RAD:
                        cluster.append(other)
                        visited.add(other)

                def rep_key(
                    idx: int,
                    _rvals: NDArray[np.float64] = refined_values,
                    _rlats: NDArray[np.float64] = refined_lats,
                    _rlons: NDArray[np.float64] = refined_lons,
                    _is_min: bool = is_min,
                    _initial_lats: NDArray[np.float64] = initial_lats,
                    _initial_lons: NDArray[np.float64] = initial_lons,
                ) -> tuple[float, float, float, float, int]:
                    val_rank = _rvals[idx] if _is_min else -_rvals[idx]
                    init_lat = float(_initial_lats[idx])
                    init_lon = float(_initial_lons[idx])
                    disp = geod_dist(
                        float(_rlats[idx]),
                        float(_rlons[idx]),
                        init_lat,
                        init_lon,
                    )
                    return (val_rank, disp, init_lat, init_lon % 360.0, idx)

                best_idx = min(cluster, key=rep_key)
                for other in cluster:
                    if other != best_idx:
                        refined_values[other] = DUFF_FEATURE_VALUE

    step = HodgesCenterFrame(
        point_time,
        refined_lats,
        refined_lons,
        refined_values,
        {
            "raw_value": raw_vals,
            "object_gridcell_area_km2": f_raw_size,
            "object_moment_fitted_area_km2": f_fit_size,
            "object_moment_major_axis_km": f_major,
            "object_moment_minor_axis_km": f_minor,
            "object_moment_orientation_degrees": f_orient,
        },
        {
            "raw_value": None,
            "object_gridcell_area_km2": "km2",
            "object_moment_fitted_area_km2": "km2",
            "object_moment_major_axis_km": "km",
            "object_moment_minor_axis_km": "km",
            "object_moment_orientation_degrees": "degrees",
        },
    )
    return step, tuple(diagnostics)


class HodgesDetector:
    """Feature detector based on Hodges scientific and TRACK lineage.

    Hodges (1994, 1995) establishes the threshold/object/feature-tracking
    methodology.  TRACK 1.5.4 establishes exact source-compatibility details
    such as threshold boundaries, scan order, endpoint handling, and feature
    prefiltering; the relevant source paths are ``src/threshold.c``,
    ``src/object_local_maxs.c``, and ``src/feature_pt_filter.c`` in the
    immutable source map:
    https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/src/threshold.c
    https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/src/object_local_maxs.c
    https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/src/feature_pt_filter.c

    Identifies local extrema (min/max) within thresholded objects.  The
    connected-component implementation and detector integration are PST
    engineering/numerical layers and are not attributed wholesale to Hodges.
    """

    def __init__(
        self,
        pathname: str | Path | None,
        variable_name: str,
        time_range: TimeRange | None = None,
        global_start_idx: int = 0,
        global_total_steps: int | None = None,
        engine: str | None = None,
    ) -> None:
        self.pathname = (
            Path(pathname)
            if pathname is not None
            and not (isinstance(pathname, str) and "://" in pathname)
            else pathname
        )
        self.requested_variable_name = variable_name
        self.time_range = time_range
        self.global_start_idx = global_start_idx
        self.global_total_steps = global_total_steps

        self._loader = DataLoader(self.pathname, engine=engine)
        self._data: xr.DataArray | None = None
        self.variable_name = variable_name
        self.last_refinement_diagnostics: tuple[HodgesFeaturePointDiagnostic, ...] = ()

    def _ensure_open(self) -> None:
        if self._data is None:
            ds = self._loader.ensure_open()
            actual_var = self._loader.resolve_variable_name(
                ds, self.requested_variable_name
            )
            if actual_var is None:
                raise KeyError(f"Variable '{self.requested_variable_name}' not found.")
            self.variable_name = actual_var
            self._data = ds[self.variable_name]

    @property
    def lat(self) -> NDArray[np.float64]:
        self._ensure_open()
        ds = self._loader.ensure_open()
        _, lat_name, _ = self._loader.get_coords()
        return np.asarray(ds[lat_name].values, dtype=np.float64)

    @property
    def lon(self) -> NDArray[np.float64]:
        self._ensure_open()
        ds = self._loader.ensure_open()
        _, _, lon_name = self._loader.get_coords()
        return np.asarray(ds[lon_name].values, dtype=np.float64)

    def get_variable(self, frame_idx: int | None = None) -> NDArray[np.float64]:
        self._ensure_open()
        assert self._data is not None
        time_dim, _, _ = self._loader.get_coords()

        if self.time_range:
            data = self._data.sel(
                {time_dim: slice(self.time_range.start, self.time_range.end)}
            )
        else:
            data = self._data

        if frame_idx is not None:
            data = data.isel({time_dim: frame_idx})
            return np.asarray(data.values, dtype=np.float64).reshape(
                (data.shape[-2], data.shape[-1])
            )

        return np.asarray(data.values, dtype=np.float64).reshape(
            (data.shape[0], data.shape[-2], data.shape[-1])
        )

    def get_time(self) -> np.ndarray:
        self._ensure_open()
        ds = self._loader.ensure_open()
        time_dim, _, _ = self._loader.get_coords()
        if self.time_range:
            times = ds[time_dim].sel(
                {time_dim: slice(self.time_range.start, self.time_range.end)}
            )
        else:
            times = ds[time_dim]
        return np.asarray(times.values)

    def get_xarray(
        self,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
    ) -> xr.DataArray:
        """Returns the requested data range as an xarray DataArray."""
        self._ensure_open()
        assert self._data is not None
        effective_start = (
            start_time
            if start_time is not None
            else self.time_range.start
            if self.time_range is not None
            else None
        )
        effective_end = (
            end_time
            if end_time is not None
            else self.time_range.end
            if self.time_range is not None
            else None
        )
        selected = select_time_range(
            self._data,
            start_time=effective_start,
            end_time=effective_end,
        )
        assert isinstance(selected, xr.DataArray)
        return selected

    @classmethod
    def from_xarray(
        cls, data: xr.DataArray, variable_name: str | None = None
    ) -> HodgesDetector:
        obj = cls.__new__(cls)
        obj.requested_variable_name = variable_name or (
            str(data.name) if data.name else "var"
        )
        obj._data = (
            data[obj.requested_variable_name] if isinstance(data, xr.Dataset) else data
        )
        obj._loader = DataLoader(obj._data)
        obj.pathname = None
        obj.time_range = None
        obj.global_start_idx = 0
        obj.global_total_steps = None
        obj.last_refinement_diagnostics = ()
        return obj

    def detect(
        self,
        search_window_size: int = DEFAULT_SEARCH_WINDOW_SIZE,
        intensity_threshold: float | None = None,
        detection_mode: ResolvedDetectionMode = "min",
        min_grid_points: int = MIN_OBJECT_GRID_POINTS_DEFAULT,
        feature_refinement: HodgesFeatureRefinement = "bspline",
        group_adjacent_extrema: bool = False,
        exclude_boundary_extrema: bool = False,
        bspline_smoothing: float = 0.0,
        track_smoopy_optimization_scale: float = (
            TRACK_SMOOPY_OPTIMIZATION_SCALE_DEFAULT
        ),
        bspline_max_iterations: int = 100,
        bspline_gradient_tolerance: float = 1.0e-5,
        **kwargs: object,
    ) -> list[HodgesCenterFrame]:
        if kwargs:
            unexpected = ", ".join(repr(k) for k in kwargs)
            raise TypeError(
                f"detect() got unexpected keyword argument(s): {unexpected}"
            )

        if search_window_size <= 0 or search_window_size % 2 == 0:
            raise ValueError("search_window_size must be a positive odd integer")
        if min_grid_points <= 0:
            raise ValueError("min_grid_points must be positive")
        if feature_refinement not in (
            "grid",
            "quadratic",
            "spherical_quadratic",
            "bspline",
            "spherical_bspline",
        ):
            raise ValueError(
                f"unsupported feature_refinement {feature_refinement!r}; expected "
                "'grid', 'quadratic', 'spherical_quadratic', "
                "'bspline', or 'spherical_bspline'"
            )
        if feature_refinement == "spherical_quadratic" and search_window_size != 3:
            raise ValueError(
                "spherical_quadratic uses the detector's fixed 3 by 3 grid neighborhood"
            )
        if group_adjacent_extrema and feature_refinement != "grid":
            raise ValueError(
                "group_adjacent_extrema requires feature_refinement='grid'"
            )
        if bspline_smoothing < 0.0 or not np.isfinite(bspline_smoothing):
            raise ValueError("bspline_smoothing must be finite and nonnegative")
        if track_smoopy_optimization_scale <= 0.0 or not np.isfinite(
            track_smoopy_optimization_scale
        ):
            raise ValueError(
                "track_smoopy_optimization_scale must be finite and positive"
            )
        if bspline_max_iterations <= 0:
            raise ValueError("bspline_max_iterations must be positive")
        if bspline_gradient_tolerance <= 0.0 or not np.isfinite(
            bspline_gradient_tolerance
        ):
            raise ValueError("bspline_gradient_tolerance must be finite and positive")

        if intensity_threshold is None:
            if self.requested_variable_name == "vo":
                intensity_threshold = DEFAULT_VO_OBJECT_THRESHOLD
            else:
                intensity_threshold = DEFAULT_MSL_OBJECT_THRESHOLD

        times = self.get_time()
        lat, lon = self.lat, self.lon
        _, _, lon_name = self._loader.get_coords()
        projected_xy = lon_name == "x"
        periodic_x = not projected_xy and self._loader.is_global_longitude()

        rectangular_grid = (
            prepare_rectangular_grid(lat, lon, periodic_x=periodic_x)
            if feature_refinement == "bspline"
            else None
        )

        if feature_refinement == "spherical_bspline" and (
            not periodic_x or not self._loader.is_global_longitude()
        ):
            raise ValueError(
                "spherical_bspline requires a global periodic longitude grid; "
                "use bspline or quadratic for regional data"
            )

        full_variable = self.get_variable()
        num_steps = len(times)
        LOGGER.debug(
            "Hodges detector configured for %d frames shape=%s refinement=%s",
            num_steps,
            full_variable.shape,
            feature_refinement,
        )

        raw_results: list[HodgesCenterFrame] = []
        diagnostics: list[HodgesFeaturePointDiagnostic] = []
        for it, t in enumerate(times):
            frame = full_variable[it]
            step, frame_diag = detect_hodges_frame(
                frame,
                t,
                lat,
                lon,
                intensity_threshold=intensity_threshold,
                mode=detection_mode,
                search_window_size=search_window_size,
                min_grid_points=min_grid_points,
                feature_refinement=feature_refinement,
                group_adjacent_extrema=group_adjacent_extrema,
                exclude_boundary_extrema=exclude_boundary_extrema,
                bspline_smoothing=bspline_smoothing,
                track_smoopy_optimization_scale=track_smoopy_optimization_scale,
                bspline_max_iterations=bspline_max_iterations,
                bspline_gradient_tolerance=bspline_gradient_tolerance,
                periodic_x=periodic_x,
                projected_xy=projected_xy,
                rectangular_grid=rectangular_grid,
            )
            raw_results.append(step)
            diagnostics.extend(frame_diag)

        self.last_refinement_diagnostics = tuple(diagnostics)
        return raw_results
