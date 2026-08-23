"""Local quadratic feature refinement on planar and spherical domains.

The intrinsic spherical quadratic is a PyStormTracker-specific local
feature-refinement formulation built from established Riemannian logarithm
and exponential maps and local least-squares quadratic fitting.  Its specific
combination of a sphere log map, centered tangent-plane quadratic, least
squares/SVD solve, Hessian rank and conditioning checks, curvature checks,
locality constraint, and sphere exp map is a PST extension; it is not a TRACK
algorithm.  The relevant mathematical background is standard Riemannian
optimization, including Huang, Gallivan, and Absil (2015), Edelman, Arias,
and Smith (1998), and Smith, *Optimization Techniques on Riemannian
Manifolds*:

https://doi.org/10.1137/140955483
https://doi.org/10.1137/S0895479895290954
https://doi.org/10.1090/fic/003/09
"""

from __future__ import annotations

from typing import Final, Literal, NamedTuple

import numba as nb
import numpy as np
from numpy.typing import NDArray

SphericalQuadraticRefinementStatus = Literal[
    "success",
    "invalid_neighborhood",
    "singular_or_ill_conditioned_fit",
    "wrong_curvature",
    "outside_locality",
    "nonfinite_failure",
]

_STATUS_BY_CODE: Final[dict[int, SphericalQuadraticRefinementStatus]] = {
    0: "success",
    1: "invalid_neighborhood",
    2: "singular_or_ill_conditioned_fit",
    3: "wrong_curvature",
    4: "outside_locality",
    5: "nonfinite_failure",
}

_SQ_SUCCESS = 0
_SQ_INVALID_NEIGHBORHOOD = 1
_SQ_ILL_CONDITIONED = 2
_SQ_WRONG_CURVATURE = 3
_SQ_OUTSIDE_LOCALITY = 4
_SQ_NONFINITE = 5
_SQ_MAX_CONDITION_NUMBER = 1.0e8
_SQ_MIN_NEIGHBOR_SAMPLES = 5


class SphericalQuadraticRefinementBatch(NamedTuple):
    """Results and compact scientific diagnostics for local spherical quadratics."""

    latitudes: NDArray[np.float64]
    longitudes: NDArray[np.float64]
    values: NDArray[np.float64]
    status_codes: NDArray[np.int8]
    hessian_eigenvalues: NDArray[np.float64]
    condition_numbers: NDArray[np.float64]
    normalized_displacements: NDArray[np.float64]


def spherical_quadratic_status_name(
    status_code: int | np.integer,
) -> SphericalQuadraticRefinementStatus:
    """Translate an internal integer status into its canonical descriptive name."""
    return _STATUS_BY_CODE.get(int(status_code), "nonfinite_failure")


# ---------------------------------------------------------------------------
# Compiled Quadratic Surface Interpolation
# ---------------------------------------------------------------------------


@nb.njit(cache=True, nogil=True)
def _refine_quadratic_point(
    frame: NDArray[np.float64],
    r: int,
    c: int,
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    periodic_x: bool = True,
    signed_longitudes: bool = False,
) -> tuple[float, float, float]:
    """Perform local quadratic surface interpolation around a single grid extremum.

    Returns the refined (latitude, longitude, value).
    """
    ny, nx = frame.shape

    if r < 1 or r >= ny - 1 or (not periodic_x and (c < 1 or c >= nx - 1)):
        return latitudes[r], longitudes[c], frame[r, c]

    cm = (c - 1) % nx if periodic_x else c - 1
    cp = (c + 1) % nx if periodic_x else c + 1

    z00 = frame[r - 1, cm]
    z01 = frame[r - 1, c]
    z02 = frame[r - 1, cp]
    z10 = frame[r, cm]
    z11 = frame[r, c]
    z12 = frame[r, cp]
    z20 = frame[r + 1, cm]
    z21 = frame[r + 1, c]
    z22 = frame[r + 1, cp]

    f_yy = z01 - 2.0 * z11 + z21
    f_xx = z10 - 2.0 * z11 + z12
    f_yx = 0.25 * (z22 - z20 - z02 + z00)
    f_y = 0.5 * (z21 - z01)
    f_x = 0.5 * (z12 - z10)

    det = f_yy * f_xx - f_yx**2
    if abs(det) < 1e-10:
        return latitudes[r], longitudes[c], frame[r, c]

    dy = (f_yx * f_x - f_xx * f_y) / det
    dx = (f_yx * f_y - f_yy * f_x) / det
    if abs(dy) > 1.0 or abs(dx) > 1.0:
        return latitudes[r], longitudes[c], frame[r, c]

    if dy > 0.0:
        refined_y = latitudes[r] + dy * (latitudes[r + 1] - latitudes[r])
    else:
        refined_y = latitudes[r] + abs(dy) * (latitudes[r - 1] - latitudes[r])

    if periodic_x:
        if dx > 0.0:
            delta_x = (longitudes[cp] - longitudes[c] + 180.0) % 360.0 - 180.0
            refined_x = longitudes[c] + dx * delta_x
        else:
            delta_x = (longitudes[cm] - longitudes[c] + 180.0) % 360.0 - 180.0
            refined_x = longitudes[c] + abs(dx) * delta_x

        if signed_longitudes:
            refined_x = (refined_x + 180.0) % 360.0 - 180.0
        else:
            refined_x = refined_x % 360.0
    elif dx > 0.0:
        refined_x = longitudes[c] + dx * (longitudes[c + 1] - longitudes[c])
    else:
        refined_x = longitudes[c] + abs(dx) * (longitudes[c - 1] - longitudes[c])

    refined_value = z11 + 0.5 * (f_y * dy + f_x * dx)
    return refined_y, refined_x, refined_value


@nb.njit(cache=True, nogil=True)
def refine_quadratic_feature_points(
    frame: NDArray[np.float64],
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    periodic_x: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Batch-refine detected feature points using quadratic surface interpolation."""
    n = rows.size
    refined_lats = np.empty(n, dtype=np.float64)
    refined_lons = np.empty(n, dtype=np.float64)
    refined_vals = np.empty(n, dtype=np.float64)
    signed_longitudes = bool(np.min(longitudes) < 0.0)

    for i in range(n):
        rlat, rlon, rval = _refine_quadratic_point(
            frame,
            rows[i],
            columns[i],
            latitudes,
            longitudes,
            periodic_x=periodic_x,
            signed_longitudes=signed_longitudes,
        )
        refined_lats[i] = rlat
        refined_lons[i] = rlon
        refined_vals[i] = rval

    return refined_lats, refined_lons, refined_vals


@nb.njit(cache=True, nogil=True)
def refine_quadratic_feature_coordinates(
    frame: NDArray[np.float64],
    initial_latitudes: NDArray[np.float64],
    initial_longitudes: NDArray[np.float64],
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    periodic_x: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Batch-refine feature coordinates by finding nearest grid rows and columns."""
    n = initial_latitudes.size
    refined_lats = np.empty(n, dtype=np.float64)
    refined_lons = np.empty(n, dtype=np.float64)
    refined_vals = np.empty(n, dtype=np.float64)
    signed_longitudes = bool(np.min(longitudes) < 0.0)

    ny = latitudes.size
    nx = longitudes.size

    for i in range(n):
        init_lat = initial_latitudes[i]
        init_lon = initial_longitudes[i]

        best_r = 0
        min_lat_dist = abs(latitudes[0] - init_lat)
        for r in range(1, ny):
            dist = abs(latitudes[r] - init_lat)
            if dist < min_lat_dist:
                min_lat_dist = dist
                best_r = r

        best_c = 0
        if periodic_x:
            min_lon_dist = abs((longitudes[0] - init_lon + 180.0) % 360.0 - 180.0)
            for c in range(1, nx):
                dist = abs((longitudes[c] - init_lon + 180.0) % 360.0 - 180.0)
                if dist < min_lon_dist:
                    min_lon_dist = dist
                    best_c = c
        else:
            min_lon_dist = abs(longitudes[0] - init_lon)
            for c in range(1, nx):
                dist = abs(longitudes[c] - init_lon)
                if dist < min_lon_dist:
                    min_lon_dist = dist
                    best_c = c

        rlat, rlon, rval = _refine_quadratic_point(
            frame,
            best_r,
            best_c,
            latitudes,
            longitudes,
            periodic_x=periodic_x,
            signed_longitudes=signed_longitudes,
        )
        refined_lats[i] = rlat
        refined_lons[i] = rlon
        refined_vals[i] = rval

    return refined_lats, refined_lons, refined_vals


def refine_quadratic_feature_point(
    frame: NDArray[np.float64],
    row: int,
    column: int,
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    *,
    periodic_x: bool = True,
) -> tuple[float, float, float]:
    """Perform local quadratic surface interpolation for a single grid point."""
    return _refine_quadratic_point(
        frame,
        int(row),
        int(column),
        latitudes,
        longitudes,
        periodic_x=periodic_x,
        signed_longitudes=bool(np.min(longitudes) < 0.0),
    )


# ---------------------------------------------------------------------------
# Local Spherical Quadratic Refinement
# ---------------------------------------------------------------------------


def _candidate_localization_scales(
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    *,
    periodic_x: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Return the shared spherical-spline detector-locality half-widths."""
    ny = latitudes.size
    nx = longitudes.size
    valid = (rows >= 1) & (rows < ny - 1)
    if not periodic_x:
        valid &= (columns >= 1) & (columns < nx - 1)

    stencil_rows = np.clip(rows, 1, ny - 2)
    if periodic_x:
        stencil_columns = np.clip(columns, 0, nx - 1)
    else:
        stencil_columns = np.clip(columns, 1, nx - 2)
    theta = np.deg2rad(90.0 - latitudes[stencil_rows])
    theta_neighbors = np.deg2rad(
        90.0 - latitudes[stencil_rows[:, np.newaxis] + np.array([-1, 1])]
    )
    theta_half_width = np.max(np.abs(theta_neighbors - theta[:, np.newaxis]), axis=1)

    offsets = np.array([-1, 1], dtype=np.int64)
    if periodic_x:
        longitude_indices = (stencil_columns[:, np.newaxis] + offsets) % nx
    else:
        longitude_indices = stencil_columns[:, np.newaxis] + offsets
    longitude_neighbors = longitudes[longitude_indices]
    longitude_center = longitudes[stencil_columns]
    longitude_differences = (
        longitude_neighbors - longitude_center[:, np.newaxis] + 180.0
    ) % 360.0 - 180.0
    phi_half_width = np.sin(theta) * np.deg2rad(
        np.max(np.abs(longitude_differences), axis=1)
    )
    valid &= np.isfinite(theta_half_width) & np.isfinite(phi_half_width)
    valid &= (theta_half_width > 1.0e-12) & (phi_half_width > 1.0e-12)
    return theta_half_width, phi_half_width, valid


def refine_spherical_quadratic_samples(
    center_latitudes: NDArray[np.float64],
    center_longitudes: NDArray[np.float64],
    center_values: NDArray[np.float64],
    neighbor_latitudes: NDArray[np.float64],
    neighbor_longitudes: NDArray[np.float64],
    neighbor_values: NDArray[np.float64],
    *,
    is_minimum: bool,
    theta_half_width: NDArray[np.float64] | None = None,
    phi_half_width: NDArray[np.float64] | None = None,
    neighbor_mask: NDArray[np.bool_] | None = None,
    require_all_neighbors: bool = False,
) -> SphericalQuadraticRefinementBatch:
    """Fit centre-anchored spherical quadratics to regular or irregular rings.

    The caller supplies the grid topology and candidate-local samples. Every
    sample is mapped with the exact sphere logarithmic map, fitted in a
    normalized physical tangent basis, and accepted only when the stationary
    point has the requested curvature and remains in the fixed local box. If
    no half-widths are supplied, the box is the smallest axis-aligned box
    containing the valid neighbour-ring coordinates.  This complete
    log-map/least-squares/curvature/locality/exp-map formulation is a
    PyStormTracker scientific extension, not a TRACK algorithm.
    """
    n_features = center_values.size
    output_latitudes = np.full(n_features, np.nan, dtype=np.float64)
    output_longitudes = np.full(n_features, np.nan, dtype=np.float64)
    output_values = np.full(n_features, np.nan, dtype=np.float64)
    status_codes = np.full(n_features, _SQ_INVALID_NEIGHBORHOOD, dtype=np.int8)
    hessian_eigenvalues = np.full((n_features, 2), np.nan, dtype=np.float64)
    condition_numbers = np.full(n_features, np.inf, dtype=np.float64)
    normalized_displacements = np.full((n_features, 2), np.nan, dtype=np.float64)

    valid_shapes = (
        center_latitudes.ndim == 1
        and center_longitudes.ndim == 1
        and center_values.ndim == 1
        and center_latitudes.size == n_features
        and center_longitudes.size == n_features
        and neighbor_latitudes.ndim == 2
        and neighbor_longitudes.shape == neighbor_latitudes.shape
        and neighbor_values.shape == neighbor_latitudes.shape
        and neighbor_latitudes.shape[0] == n_features
        and neighbor_latitudes.shape[1] >= _SQ_MIN_NEIGHBOR_SAMPLES
    )
    if not valid_shapes:
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    output_latitudes[:] = center_latitudes
    output_longitudes[:] = center_longitudes
    output_values[:] = center_values
    n_neighbors = neighbor_latitudes.shape[1]
    if neighbor_mask is None:
        neighbor_mask = np.ones((n_features, n_neighbors), dtype=np.bool_)
    if neighbor_mask.shape != (n_features, n_neighbors):
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    if theta_half_width is None:
        theta_scales = np.full(n_features, np.nan, dtype=np.float64)
    else:
        theta_scales = np.asarray(theta_half_width, dtype=np.float64)
    if phi_half_width is None:
        phi_scales = np.full(n_features, np.nan, dtype=np.float64)
    else:
        phi_scales = np.asarray(phi_half_width, dtype=np.float64)
    supplied_scales = theta_half_width is not None and phi_half_width is not None
    if supplied_scales and (
        theta_scales.shape != (n_features,) or phi_scales.shape != (n_features,)
    ):
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    finite_centers = (
        np.isfinite(center_latitudes)
        & np.isfinite(center_longitudes)
        & np.isfinite(center_values)
    )
    nonfinite_centers = (
        np.isfinite(center_latitudes)
        & np.isfinite(center_longitudes)
        & ~np.isfinite(center_values)
    )
    if np.any(nonfinite_centers):
        status_codes[nonfinite_centers] = _SQ_NONFINITE
    if supplied_scales:
        finite_centers &= (
            np.isfinite(theta_scales)
            & np.isfinite(phi_scales)
            & (theta_scales > 1.0e-12)
            & (phi_scales > 1.0e-12)
        )
    candidate_indices = np.flatnonzero(finite_centers)
    if candidate_indices.size == 0:
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    center_latitude_rad = np.deg2rad(center_latitudes[candidate_indices])
    center_longitude_rad = np.deg2rad(center_longitudes[candidate_indices])
    sin_latitude = np.sin(center_latitude_rad)
    cos_latitude = np.cos(center_latitude_rad)
    sin_longitude = np.sin(center_longitude_rad)
    cos_longitude = np.cos(center_longitude_rad)
    center_points = np.column_stack(
        (cos_latitude * cos_longitude, cos_latitude * sin_longitude, sin_latitude)
    )
    e_theta = np.column_stack(
        (sin_latitude * cos_longitude, sin_latitude * sin_longitude, -cos_latitude)
    )
    e_phi = np.column_stack(
        (-sin_longitude, cos_longitude, np.zeros(candidate_indices.size))
    )

    candidate_neighbor_latitudes = np.deg2rad(neighbor_latitudes[candidate_indices])
    candidate_neighbor_longitudes = np.deg2rad(neighbor_longitudes[candidate_indices])
    neighbor_points = np.stack(
        (
            np.cos(candidate_neighbor_latitudes)
            * np.cos(candidate_neighbor_longitudes),
            np.cos(candidate_neighbor_latitudes)
            * np.sin(candidate_neighbor_longitudes),
            np.sin(candidate_neighbor_latitudes),
        ),
        axis=-1,
    )
    dot = np.clip(
        np.sum(neighbor_points * center_points[:, np.newaxis, :], axis=2), -1.0, 1.0
    )
    alpha = np.arccos(dot)
    sin_alpha = np.sin(alpha)
    log_scale = np.divide(
        alpha,
        sin_alpha,
        out=np.ones_like(alpha),
        where=np.abs(sin_alpha) > 1.0e-14,
    )
    tangent_vectors = log_scale[:, :, np.newaxis] * (
        neighbor_points - dot[:, :, np.newaxis] * center_points[:, np.newaxis, :]
    )
    xi_theta = np.sum(tangent_vectors * e_theta[:, np.newaxis, :], axis=2)
    xi_phi = np.sum(tangent_vectors * e_phi[:, np.newaxis, :], axis=2)
    usable_neighbors = (
        neighbor_mask[candidate_indices]
        & np.isfinite(candidate_neighbor_latitudes)
        & np.isfinite(candidate_neighbor_longitudes)
        & np.isfinite(neighbor_values[candidate_indices])
    )

    if not supplied_scales:
        theta_scales[candidate_indices] = np.max(
            np.where(usable_neighbors, np.abs(xi_theta), 0.0), axis=1
        )
        phi_scales[candidate_indices] = np.max(
            np.where(usable_neighbors, np.abs(xi_phi), 0.0), axis=1
        )
    scale_valid = (
        np.isfinite(theta_scales[candidate_indices])
        & np.isfinite(phi_scales[candidate_indices])
        & (theta_scales[candidate_indices] > 1.0e-12)
        & (phi_scales[candidate_indices] > 1.0e-12)
    )
    topology_counts = neighbor_mask[candidate_indices].sum(axis=1)
    finite_counts = usable_neighbors.sum(axis=1)
    if require_all_neighbors:
        sample_valid = usable_neighbors.all(axis=1) & scale_valid
        if np.any(~sample_valid):
            status_codes[candidate_indices[~sample_valid]] = _SQ_NONFINITE
    else:
        sample_valid = (finite_counts >= _SQ_MIN_NEIGHBOR_SAMPLES) & scale_valid
        invalid_topology = (topology_counts < _SQ_MIN_NEIGHBOR_SAMPLES) | ~scale_valid
        insufficient_finite = (topology_counts >= _SQ_MIN_NEIGHBOR_SAMPLES) & (
            finite_counts < _SQ_MIN_NEIGHBOR_SAMPLES
        )
        if np.any(invalid_topology):
            status_codes[candidate_indices[invalid_topology]] = _SQ_INVALID_NEIGHBORHOOD
        if np.any(insufficient_finite):
            status_codes[candidate_indices[insufficient_finite]] = _SQ_NONFINITE

    sample_candidates = candidate_indices[sample_valid]
    if sample_candidates.size == 0:
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    sample_positions = np.flatnonzero(sample_valid)
    sample_mask = usable_neighbors[sample_positions]
    a = theta_scales[candidate_indices][sample_positions]
    b = phi_scales[candidate_indices][sample_positions]
    u = np.divide(
        xi_theta[sample_positions],
        a[:, np.newaxis],
    )
    v = np.divide(
        xi_phi[sample_positions],
        b[:, np.newaxis],
    )
    u = np.where(sample_mask, u, 0.0)
    v = np.where(sample_mask, v, 0.0)
    design = np.stack(
        (u, v, 0.5 * u * u, u * v, 0.5 * v * v),
        axis=2,
    )
    svd_u, singular_values, svd_vh = np.linalg.svd(design, full_matrices=False)
    fit_condition = np.divide(
        singular_values[:, 0],
        singular_values[:, -1],
        out=np.full(singular_values.shape[0], np.inf, dtype=np.float64),
        where=singular_values[:, -1] > 0.0,
    )
    condition_numbers[sample_candidates] = fit_condition
    fit_is_valid = (
        np.isfinite(design).all(axis=(1, 2))
        & np.isfinite(singular_values).all(axis=1)
        & (singular_values[:, -1] > 0.0)
        & np.isfinite(fit_condition)
        & (fit_condition <= _SQ_MAX_CONDITION_NUMBER)
    )
    if np.any(~fit_is_valid):
        status_codes[sample_candidates[~fit_is_valid]] = _SQ_ILL_CONDITIONED

    fit_candidates = sample_candidates[fit_is_valid]
    if fit_candidates.size == 0:
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    safe_neighbor_values = np.where(
        sample_mask,
        neighbor_values[candidate_indices][sample_positions],
        output_values[sample_candidates, np.newaxis],
    )
    fit_values = safe_neighbor_values[fit_is_valid]
    fit_centers = output_values[fit_candidates]
    fit_u = svd_u[fit_is_valid]
    fit_singular_values = singular_values[fit_is_valid]
    fit_vh = svd_vh[fit_is_valid]
    projected_values = np.einsum(
        "nki,nk->ni", fit_u, fit_values - fit_centers[:, np.newaxis]
    )
    coefficients = np.einsum(
        "nji,nj->ni", fit_vh, projected_values / fit_singular_values
    )

    hessian = np.empty((fit_candidates.size, 2, 2), dtype=np.float64)
    hessian[:, 0, 0] = coefficients[:, 2]
    hessian[:, 0, 1] = coefficients[:, 3]
    hessian[:, 1, 0] = coefficients[:, 3]
    hessian[:, 1, 1] = coefficients[:, 4]
    eigenvalues = np.linalg.eigvalsh(hessian)
    hessian_eigenvalues[fit_candidates] = eigenvalues
    absolute_eigenvalues = np.abs(eigenvalues)
    hessian_condition = np.divide(
        np.max(absolute_eigenvalues, axis=1),
        np.min(absolute_eigenvalues, axis=1),
        out=np.full(fit_candidates.size, np.inf, dtype=np.float64),
        where=np.min(absolute_eigenvalues, axis=1) > 0.0,
    )
    hessian_is_well_conditioned = (
        np.isfinite(coefficients).all(axis=1)
        & np.isfinite(eigenvalues).all(axis=1)
        & np.isfinite(hessian_condition)
        & (np.abs(eigenvalues[:, 0]) > 1.0e-12)
        & (hessian_condition <= _SQ_MAX_CONDITION_NUMBER)
    )
    if np.any(~hessian_is_well_conditioned):
        status_codes[fit_candidates[~hessian_is_well_conditioned]] = _SQ_ILL_CONDITIONED

    curvature_candidates = fit_candidates[hessian_is_well_conditioned]
    if curvature_candidates.size == 0:
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    curvature_eigenvalues = eigenvalues[hessian_is_well_conditioned]
    if is_minimum:
        correct_curvature = np.all(curvature_eigenvalues > 0.0, axis=1)
    else:
        correct_curvature = np.all(curvature_eigenvalues < 0.0, axis=1)
    if np.any(~correct_curvature):
        status_codes[curvature_candidates[~correct_curvature]] = _SQ_WRONG_CURVATURE

    stationary_candidates = curvature_candidates[correct_curvature]
    if stationary_candidates.size == 0:
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    stationary_coefficients = coefficients[hessian_is_well_conditioned][
        correct_curvature
    ]
    stationary_hessian = hessian[hessian_is_well_conditioned][correct_curvature]
    try:
        delta = -np.linalg.solve(
            stationary_hessian, stationary_coefficients[:, :2, np.newaxis]
        )[:, :, 0]
    except np.linalg.LinAlgError:
        status_codes[stationary_candidates] = _SQ_ILL_CONDITIONED
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    normalized_displacements[stationary_candidates] = delta
    inside_locality = np.isfinite(delta).all(axis=1) & (
        np.max(np.abs(delta), axis=1) <= 1.0 + 1.0e-12
    )
    if np.any(~inside_locality):
        status_codes[stationary_candidates[~inside_locality]] = _SQ_OUTSIDE_LOCALITY

    accepted_candidates = stationary_candidates[inside_locality]
    if accepted_candidates.size == 0:
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    accepted_positions = np.flatnonzero(inside_locality)
    accepted_delta = delta[accepted_positions]
    accepted_a = a[fit_is_valid][hessian_is_well_conditioned][correct_curvature][
        accepted_positions
    ]
    accepted_b = b[fit_is_valid][hessian_is_well_conditioned][correct_curvature][
        accepted_positions
    ]
    accepted_points = center_points[sample_positions][fit_is_valid][
        hessian_is_well_conditioned
    ][correct_curvature][accepted_positions]
    accepted_e_theta = e_theta[sample_positions][fit_is_valid][
        hessian_is_well_conditioned
    ][correct_curvature][accepted_positions]
    accepted_e_phi = e_phi[sample_positions][fit_is_valid][hessian_is_well_conditioned][
        correct_curvature
    ][accepted_positions]
    eta = (
        accepted_a[:, np.newaxis] * accepted_delta[:, 0, np.newaxis] * accepted_e_theta
        + accepted_b[:, np.newaxis] * accepted_delta[:, 1, np.newaxis] * accepted_e_phi
    )
    eta_norm = np.linalg.norm(eta, axis=1)
    mapped_points = (
        np.cos(eta_norm)[:, np.newaxis] * accepted_points
        + np.divide(
            np.sin(eta_norm),
            eta_norm,
            out=np.ones_like(eta_norm),
            where=eta_norm > 1.0e-14,
        )[:, np.newaxis]
        * eta
    )
    finite_mapped = np.isfinite(mapped_points).all(axis=1)
    if np.any(~finite_mapped):
        status_codes[accepted_candidates[~finite_mapped]] = _SQ_NONFINITE
    mapped_indices = accepted_candidates[finite_mapped]
    if mapped_indices.size:
        mapped = mapped_points[finite_mapped]
        output_latitudes[mapped_indices] = np.rad2deg(
            np.arcsin(np.clip(mapped[:, 2], -1.0, 1.0))
        )
        raw_longitudes = np.rad2deg(np.arctan2(mapped[:, 1], mapped[:, 0]))
        if np.min(center_longitudes) < 0.0:
            output_longitudes[mapped_indices] = (raw_longitudes + 180.0) % 360.0 - 180.0
        else:
            output_longitudes[mapped_indices] = raw_longitudes % 360.0
        accepted_coefficients = stationary_coefficients[accepted_positions][
            finite_mapped
        ]
        accepted_center_values = output_values[mapped_indices]
        accepted_displacements = accepted_delta[finite_mapped]
        output_values[mapped_indices] = accepted_center_values + (
            accepted_coefficients[:, 0] * accepted_displacements[:, 0]
            + accepted_coefficients[:, 1] * accepted_displacements[:, 1]
            + 0.5
            * accepted_coefficients[:, 2]
            * accepted_displacements[:, 0]
            * accepted_displacements[:, 0]
            + accepted_coefficients[:, 3]
            * accepted_displacements[:, 0]
            * accepted_displacements[:, 1]
            + 0.5
            * accepted_coefficients[:, 4]
            * accepted_displacements[:, 1]
            * accepted_displacements[:, 1]
        )
        finite_values = np.isfinite(output_values[mapped_indices])
        if np.any(~finite_values):
            status_codes[mapped_indices[~finite_values]] = _SQ_NONFINITE
        status_codes[mapped_indices[finite_values]] = _SQ_SUCCESS

    return SphericalQuadraticRefinementBatch(
        output_latitudes,
        output_longitudes,
        output_values,
        status_codes,
        hessian_eigenvalues,
        condition_numbers,
        normalized_displacements,
    )


def refine_spherical_quadratic_feature_points(
    frame: NDArray[np.float64],
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    *,
    is_minimum: bool,
    periodic_x: bool = True,
) -> SphericalQuadraticRefinementBatch:
    """Refine regular-grid candidates with the PST spherical quadratic core.

    The detector-local neighborhood is passed to the same PyStormTracker
    spherical extension used for irregular rings; the method is not claimed
    as a TRACK refinement algorithm.
    """
    n_features = rows.size
    output_latitudes = np.full(n_features, np.nan, dtype=np.float64)
    output_longitudes = np.full(n_features, np.nan, dtype=np.float64)
    output_values = np.full(n_features, np.nan, dtype=np.float64)
    status_codes = np.full(n_features, _SQ_INVALID_NEIGHBORHOOD, dtype=np.int8)
    hessian_eigenvalues = np.full((n_features, 2), np.nan, dtype=np.float64)
    condition_numbers = np.full(n_features, np.inf, dtype=np.float64)
    normalized_displacements = np.full((n_features, 2), np.nan, dtype=np.float64)

    if (
        frame.ndim != 2
        or frame.shape != (latitudes.size, longitudes.size)
        or rows.ndim != 1
        or columns.ndim != 1
        or columns.size != n_features
        or latitudes.size < 3
        or longitudes.size < 3
        or not np.isfinite(latitudes).all()
        or not np.isfinite(longitudes).all()
    ):
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    center_in_bounds = (
        (rows >= 0)
        & (rows < latitudes.size)
        & (columns >= 0)
        & (columns < longitudes.size)
    )
    if np.any(center_in_bounds):
        in_bounds_indices = np.flatnonzero(center_in_bounds)
        output_latitudes[in_bounds_indices] = latitudes[rows[in_bounds_indices]]
        output_longitudes[in_bounds_indices] = longitudes[columns[in_bounds_indices]]
        output_values[in_bounds_indices] = frame[
            rows[in_bounds_indices], columns[in_bounds_indices]
        ]

    theta_half_width, phi_half_width, valid = _candidate_localization_scales(
        latitudes,
        longitudes,
        rows,
        columns,
        periodic_x=periodic_x,
    )
    valid &= center_in_bounds
    candidate_indices = np.flatnonzero(valid)
    if candidate_indices.size == 0:
        return SphericalQuadraticRefinementBatch(
            output_latitudes,
            output_longitudes,
            output_values,
            status_codes,
            hessian_eigenvalues,
            condition_numbers,
            normalized_displacements,
        )

    neighbor_offsets = np.array(
        [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    candidate_rows = rows[candidate_indices]
    candidate_columns = columns[candidate_indices]
    neighbor_rows = candidate_rows[:, np.newaxis] + neighbor_offsets[np.newaxis, :, 0]
    if periodic_x:
        neighbor_columns = (
            candidate_columns[:, np.newaxis] + neighbor_offsets[np.newaxis, :, 1]
        ) % longitudes.size
    else:
        neighbor_columns = (
            candidate_columns[:, np.newaxis] + neighbor_offsets[np.newaxis, :, 1]
        )
    candidate_batch = refine_spherical_quadratic_samples(
        latitudes[candidate_rows],
        longitudes[candidate_columns],
        frame[candidate_rows, candidate_columns],
        latitudes[neighbor_rows],
        longitudes[neighbor_columns],
        frame[neighbor_rows, neighbor_columns],
        is_minimum=is_minimum,
        theta_half_width=theta_half_width[candidate_indices],
        phi_half_width=phi_half_width[candidate_indices],
        require_all_neighbors=True,
    )
    output_latitudes[candidate_indices] = candidate_batch.latitudes
    output_longitudes[candidate_indices] = candidate_batch.longitudes
    output_values[candidate_indices] = candidate_batch.values
    status_codes[candidate_indices] = candidate_batch.status_codes
    hessian_eigenvalues[candidate_indices] = candidate_batch.hessian_eigenvalues
    condition_numbers[candidate_indices] = candidate_batch.condition_numbers
    normalized_displacements[candidate_indices] = (
        candidate_batch.normalized_displacements
    )
    return SphericalQuadraticRefinementBatch(
        output_latitudes,
        output_longitudes,
        output_values,
        status_codes,
        hessian_eigenvalues,
        condition_numbers,
        normalized_displacements,
    )


__all__ = [
    "SphericalQuadraticRefinementBatch",
    "SphericalQuadraticRefinementStatus",
    "refine_quadratic_feature_coordinates",
    "refine_quadratic_feature_point",
    "refine_quadratic_feature_points",
    "refine_spherical_quadratic_feature_points",
    "refine_spherical_quadratic_samples",
    "spherical_quadratic_status_name",
]
