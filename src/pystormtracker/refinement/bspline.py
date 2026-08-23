"""B-spline surfaces and feature-point refinement.

The implementation has several distinct provenance layers:

* Hodges (1994, 1995) establishes the scientific feature-tracking and unit-
  sphere lineage.
* The rectangular ``bspline`` path preserves TRACK 1.5.4 SMOOPY/GDFP
  compatibility semantics.  The exact coordinate-space optimizer is in
  ``lib/src/gdfp_optimize.c`` and ``lib/src/update_h.c`` of the immutable
  TRACK source, not in the Hodges papers alone:
  https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/lib/src/gdfp_optimize.c
  https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/lib/src/update_h.c
* SciPy/FITPACK supplies the rectangular and spherical spline construction;
  see Dierckx (1993), *Curve and Surface Fitting with Splines*.
  https://doi.org/10.1093/oso/9780198534419.001.0001
  PST supplies coefficient extraction, compiled evaluation, and integration
  with the detector/refinement pipeline.
* ``spherical_bspline`` is a PyStormTracker extension: its intrinsic
  Riemannian quasi-Newton optimizer is not the TRACK optimizer.  Its
  mathematical background includes Huang, Gallivan, and Absil (2015),
  Edelman, Arias, and Smith (1998), and Smith's *Optimization Techniques on
  Riemannian Manifolds*:
  https://doi.org/10.1137/140955483
  https://doi.org/10.1137/S0895479895290954
  https://doi.org/10.1090/fic/003/09

The verified Hodges references are: K. I. Hodges (1994), “A General Method
for Tracking Analysis and Its Application to Meteorological Data,” *Monthly
Weather Review*, 122(11), 2573--2586,
https://doi.org/10.1175/1520-0493(1994)122<2573:AGMFTA>2.0.CO;2; and K. I.
Hodges (1995), “Feature Tracking on the Unit Sphere,” *Monthly Weather
Review*, 123(12), 3458--3465,
https://doi.org/10.1175/1520-0493(1995)123<3458:FTOTUS>2.0.CO;2.
"""

from __future__ import annotations

from typing import Final, Literal, NamedTuple

import numba as nb
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, RectBivariateSpline, RectSphereBivariateSpline

from ..preprocessing.spherical_geometry import (
    SphericalLocalizationRegion,
    spherical_localization_region,
)
from ..preprocessing.spherical_geometry import (
    inside_spherical_localization_region as _inside_spherical_localization_region,
)
from ..preprocessing.spherical_geometry import (
    sphere_point_and_basis as _sphere_point_and_basis,
)

_DEFAULT_SMOOTHING: Final[float] = 0.0
_TRACK_SMOOPY_OPTIMIZATION_SCALE: Final[float] = 1.0
_DEFAULT_MAX_ITERATIONS: Final[int] = 100
_DEFAULT_GRADIENT_TOLERANCE: Final[float] = 1.0e-5
_RECTANGULAR_SPLINE_DEGREE: Final[int] = 3
_RECTANGULAR_SPLINE_BANDWIDTH: Final[int] = _RECTANGULAR_SPLINE_DEGREE

BsplineRefinementStatus = Literal[
    "success",
    "invalid_neighborhood",
    "optimizer_no_convergence",
    "spline_construction_failure",
]


class BsplineRefinementResult(NamedTuple):
    """Result of a local feature-point search on a surface."""

    latitude: float
    longitude: float
    value: float
    status: BsplineRefinementStatus


_STATUS_BY_CODE: Final[dict[int, BsplineRefinementStatus]] = {
    0: "success",
    1: "invalid_neighborhood",
    2: "optimizer_no_convergence",
    3: "spline_construction_failure",
}


class SphericalBsplineSurface(NamedTuple):
    """Reusable spherical B-spline state for one frame."""

    theta_knots: NDArray[np.float64]
    phi_knots: NDArray[np.float64]
    coeffs: NDArray[np.float64]
    theta_lower: float
    theta_upper: float
    phi_lower: float
    phi_upper: float
    first_sample_phi: float
    last_sample_phi: float
    signed_longitudes: bool
    sample_latitudes: NDArray[np.float64]
    sample_longitudes: NDArray[np.float64]


class SphericalBsplineSurfaceResult(NamedTuple):
    """Outcome of constructing a frame-level spherical B-spline."""

    surface: SphericalBsplineSurface | None
    status: BsplineRefinementStatus


class BsplineSurface(NamedTuple):
    """Reusable rectangular B-spline state for one frame."""

    x_knots: NDArray[np.float64]
    y_knots: NDArray[np.float64]
    coeffs: NDArray[np.float64]
    x_lower: float
    x_upper: float
    y_lower: float
    y_upper: float
    first_sample_x: float
    last_sample_x: float
    periodic_x: bool = False


class BsplineSurfaceResult(NamedTuple):
    """Outcome of constructing a frame-level rectangular spline."""

    surface: BsplineSurface | None
    status: BsplineRefinementStatus


class RectangularGridPreparation(NamedTuple):
    """Immutable fixed-grid state reused by rectangular frame operations.

    The coordinate order, FITPACK-compatible cubic knot vectors, and their
    banded collocation factorizations depend only on the grid.  The factors
    never contain frame values and are read-only after construction.
    """

    sorted_longitudes: NDArray[np.float64]
    longitude_order: NDArray[np.int64]
    sorted_latitudes: NDArray[np.float64]
    latitude_order: NDArray[np.int64]
    extended_longitudes: NDArray[np.float64]
    periodic_x: bool
    x_knots: NDArray[np.float64]
    y_knots: NDArray[np.float64]
    x_factor: _BandedFactorization
    y_factor: _BandedFactorization


class _BandedFactorization(NamedTuple):
    """Read-only FITPACK QR factorization of a cubic collocation matrix."""

    upper: NDArray[np.float64]
    cosines: NDArray[np.float64]
    sines: NDArray[np.float64]
    active: NDArray[np.bool_]
    row_numbers: NDArray[np.int64]


@nb.njit(cache=True, nogil=True)
def _apply_cached_qr_rows(
    values: NDArray[np.float64],
    upper: NDArray[np.float64],
    cosines: NDArray[np.float64],
    sines: NDArray[np.float64],
    active: NDArray[np.bool_],
    row_numbers: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Apply a cached FITPACK rotation sequence to several right-hand sides."""
    transformed = np.zeros((upper.shape[0], values.shape[1]), dtype=np.float64)
    right = np.empty(values.shape[1], dtype=np.float64)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            right[column] = values[row, column]
        number = row_numbers[row]
        for offset in range(_RECTANGULAR_SPLINE_DEGREE + 1):
            if not active[row, offset]:
                continue
            target = number + offset
            cosine = cosines[row, offset]
            sine = sines[row, offset]
            for column in range(values.shape[1]):
                old_right = right[column]
                old_transformed = transformed[target, column]
                right[column] = cosine * old_right - sine * old_transformed
                transformed[target, column] = (
                    cosine * old_transformed + sine * old_right
                )
    return transformed


@nb.njit(cache=True, nogil=True)
def _back_substitute_cached_qr(
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve a cached upper-banded FITPACK factorization."""
    solution = rhs.copy()
    size = upper.shape[0]
    for row in range(size - 1, -1, -1):
        stop = min(size, row + _RECTANGULAR_SPLINE_BANDWIDTH + 1)
        for column in range(solution.shape[1]):
            value = solution[row, column]
            for trailing in range(row + 1, stop):
                value -= upper[row, trailing - row] * solution[trailing, column]
            solution[row, column] = value / upper[row, 0]
    return solution


def _cubic_interpolation_knots(
    coordinates: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the cubic knots used by FITPACK's ``s=0`` rectangular path."""
    return np.concatenate(
        (
            np.full(_RECTANGULAR_SPLINE_DEGREE + 1, coordinates[0]),
            coordinates[2:-2],
            np.full(_RECTANGULAR_SPLINE_DEGREE + 1, coordinates[-1]),
        )
    )


def _factor_cubic_collocation_matrix(
    coordinates: NDArray[np.float64],
    knots: NDArray[np.float64],
) -> _BandedFactorization:
    """Factor a cubic collocation matrix using FITPACK's Givens rotations."""
    design = BSpline.design_matrix(
        coordinates,
        knots,
        _RECTANGULAR_SPLINE_DEGREE,
        extrapolate=False,
    )
    size = coordinates.size
    degree = _RECTANGULAR_SPLINE_DEGREE
    upper = np.zeros((size, degree + 2), dtype=np.float64)
    cosines = np.zeros((size, degree + 1), dtype=np.float64)
    sines = np.zeros((size, degree + 1), dtype=np.float64)
    active = np.zeros((size, degree + 1), dtype=np.bool_)

    # This is the span scan used by FITPACK's fpgrre.  For the interpolation
    # knot vector, ``number`` is the first nonzero coefficient for a row.
    row_numbers = np.empty(size, dtype=np.int64)
    knot_span = degree + 1
    next_knot = degree + 2
    for row, coordinate in enumerate(coordinates):
        while not (coordinate < knots[next_knot - 1] or knot_span == size):
            knot_span = next_knot
            next_knot += 1
        row_numbers[row] = knot_span - (degree + 1)

    for row, number_value in enumerate(row_numbers):
        number = int(number_value)
        start = int(design.indptr[row])
        stop = int(design.indptr[row + 1])
        h = np.asarray(design.data[start:stop], dtype=np.float64).copy()
        for offset in range(degree + 1):
            target = number + offset
            pivot = float(h[offset])
            if pivot == 0.0:
                continue

            previous_diagonal = upper[target, 0]
            if abs(pivot) >= previous_diagonal:
                diagonal = abs(pivot) * np.sqrt(1.0 + (previous_diagonal / pivot) ** 2)
            else:
                diagonal = previous_diagonal * np.sqrt(
                    1.0 + (pivot / previous_diagonal) ** 2
                )
            cosine = previous_diagonal / diagonal
            sine = pivot / diagonal
            upper[target, 0] = diagonal
            cosines[row, offset] = cosine
            sines[row, offset] = sine
            active[row, offset] = True

            for trailing in range(offset + 1, degree + 1):
                old_h = h[trailing]
                old_upper = upper[target, trailing - offset]
                # FITPACK's fprota: [a', b'] = [c -s; s c] [a, b].
                h[trailing] = cosine * old_h - sine * old_upper
                upper[target, trailing - offset] = cosine * old_upper + sine * old_h

    arrays = (upper, cosines, sines, active, row_numbers)
    for array in arrays:
        array.setflags(write=False)
    return _BandedFactorization(*arrays)


def prepare_rectangular_grid(
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    *,
    periodic_x: bool,
) -> RectangularGridPreparation:
    """Prepare immutable coordinate orderings for repeated frame builds."""
    x = (
        np.mod(longitudes.astype(np.float64, copy=False), 360.0)
        if periodic_x
        else longitudes.astype(np.float64, copy=False)
    )
    longitude_order = np.argsort(x)
    sorted_longitudes = np.asarray(x[longitude_order], dtype=np.float64)
    latitude_order = np.argsort(latitudes)
    sorted_latitudes = np.asarray(latitudes[latitude_order], dtype=np.float64)
    extended_longitudes = (
        np.concatenate((sorted_longitudes, [sorted_longitudes[0] + 360.0]))
        if periodic_x
        else sorted_longitudes.copy()
    )
    x_knots = _cubic_interpolation_knots(extended_longitudes)
    y_knots = _cubic_interpolation_knots(sorted_latitudes)
    x_factor = _factor_cubic_collocation_matrix(extended_longitudes, x_knots)
    y_factor = _factor_cubic_collocation_matrix(sorted_latitudes, y_knots)

    coordinate_arrays = (
        sorted_longitudes,
        longitude_order,
        sorted_latitudes,
        latitude_order,
        extended_longitudes,
        x_knots,
        y_knots,
    )
    for array in coordinate_arrays:
        array.setflags(write=False)
    return RectangularGridPreparation(
        sorted_longitudes,
        longitude_order,
        sorted_latitudes,
        latitude_order,
        extended_longitudes,
        periodic_x,
        x_knots,
        y_knots,
        x_factor,
        y_factor,
    )


def _spline_scalar(value: NDArray[np.float64]) -> float:
    """Convert SciPy's scalar-shaped spline output to a Python float."""
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def build_spherical_bspline_surface(
    frame: NDArray[np.float64],
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    *,
    periodic_x: bool,
    smoothing: float = _DEFAULT_SMOOTHING,
) -> SphericalBsplineSurfaceResult:
    """Construct a global periodic bicubic spherical spline for one frame.

    Uses colatitude in radians, a longitude interval beginning at zero,
    cubic splines, periodic longitude continuity, and no polar-value or
    polar-derivative constraint by default.
    SciPy/FITPACK provides the spline construction; the global spherical
    coordinate convention and its use for feature tracking follow the Hodges
    lineage, while PST owns the surrounding periodic-coordinate handling.
    """
    if (
        not periodic_x
        or frame.ndim != 2
        or frame.shape != (latitudes.size, longitudes.size)
        or latitudes.size < 4
        or longitudes.size < 4
        or not np.isfinite(frame).all()
        or not np.isfinite(latitudes).all()
        or not np.isfinite(longitudes).all()
        or not np.isfinite(smoothing)
        or smoothing < 0.0
    ):
        return SphericalBsplineSurfaceResult(None, "spline_construction_failure")

    theta = np.deg2rad(90.0 - latitudes)
    theta_order = np.argsort(theta)
    theta_sorted = theta[theta_order]
    if np.any(np.diff(theta_sorted) <= 0.0) or theta_sorted.size < 4:
        return SphericalBsplineSurfaceResult(None, "spline_construction_failure")

    eps = 1.0e-7
    clamped_theta = np.clip(theta_sorted, eps, np.pi - eps)
    if np.any(np.diff(clamped_theta) <= 0.0):
        return SphericalBsplineSurfaceResult(None, "spline_construction_failure")

    phi_input = np.deg2rad(longitudes)
    phi_raw = phi_input % (2.0 * np.pi)
    phi_order = np.argsort(phi_raw)
    phi = phi_raw[phi_order]
    if np.any(np.diff(phi) <= 0.0) or phi.size < 4:
        return SphericalBsplineSurfaceResult(None, "spline_construction_failure")

    ordered_frame = frame[theta_order, :][:, phi_order]

    try:
        spline = RectSphereBivariateSpline(
            clamped_theta,
            phi,
            ordered_frame,
            s=smoothing,
            pole_continuity=False,
            pole_values=None,
            pole_exact=False,
        )
    except Exception:  # noqa: BLE001
        return SphericalBsplineSurfaceResult(None, "spline_construction_failure")

    fp_tuple = spline.tck
    theta_knots = np.asarray(fp_tuple[0], dtype=np.float64)
    phi_knots = np.asarray(fp_tuple[1], dtype=np.float64)
    coeffs_raw = np.asarray(fp_tuple[2], dtype=np.float64)

    nx_knots = len(theta_knots) - 4
    ny_knots = len(phi_knots) - 4
    coeffs_arr = coeffs_raw.reshape(nx_knots, ny_knots).copy()

    return SphericalBsplineSurfaceResult(
        SphericalBsplineSurface(
            theta_knots=theta_knots,
            phi_knots=phi_knots,
            coeffs=coeffs_arr,
            theta_lower=float(theta_knots[3]),
            theta_upper=float(theta_knots[-4]),
            phi_lower=float(phi_knots[3]),
            phi_upper=float(phi_knots[-4]),
            first_sample_phi=float(phi[0]),
            last_sample_phi=float(phi[-1]),
            signed_longitudes=bool(np.min(longitudes) < 0.0),
            sample_latitudes=latitudes.astype(np.float64, copy=True),
            sample_longitudes=longitudes.astype(np.float64, copy=True),
        ),
        "success",
    )


def _ordered_rectangular_frame(
    frame: NDArray[np.float64],
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    *,
    periodic_x: bool,
    grid: RectangularGridPreparation | None = None,
) -> (
    tuple[
        RectangularGridPreparation,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
    | None
):
    """Validate and order one rectangular frame in the cached grid space."""
    if (
        frame.ndim != 2
        or frame.shape != (latitudes.size, longitudes.size)
        or latitudes.size < 4
        or longitudes.size < 4
        or not np.isfinite(frame).all()
        or not np.isfinite(latitudes).all()
        or not np.isfinite(longitudes).all()
    ):
        return None

    # Rectangular TRACK compatibility is evaluated in one internal global
    # coordinate system.  Signed input is equivalent to unsigned input only
    # after its data columns are cyclically reordered with these coordinates.
    prepared_grid = (
        grid
        if grid is not None
        else prepare_rectangular_grid(latitudes, longitudes, periodic_x=periodic_x)
    )
    if prepared_grid.periodic_x != periodic_x:
        return None

    x = prepared_grid.sorted_longitudes
    x_order = prepared_grid.longitude_order
    if np.any(np.diff(x) <= 0.0) or x.size < 4:
        return None

    y = prepared_grid.sorted_latitudes
    y_order = prepared_grid.latitude_order
    if np.any(np.diff(y) <= 0.0) or y.size < 4:
        return None

    z = frame[y_order, :][:, x_order]
    extended_z = np.concatenate((z, z[:, :1]), axis=1) if periodic_x else z
    return prepared_grid, x, y, extended_z


def _rectangular_surface(
    grid: RectangularGridPreparation,
    coeffs: NDArray[np.float64],
    periodic_x: bool,
) -> BsplineSurface:
    """Build the evaluator state shared by cached and FITPACK paths."""
    return BsplineSurface(
        x_knots=grid.x_knots,
        y_knots=grid.y_knots,
        coeffs=coeffs,
        x_lower=float(grid.x_knots[3]),
        x_upper=float(grid.x_knots[-4]),
        y_lower=float(grid.y_knots[3]),
        y_upper=float(grid.y_knots[-4]),
        first_sample_x=float(grid.sorted_longitudes[0]),
        last_sample_x=float(grid.sorted_longitudes[-1]),
        periodic_x=periodic_x,
    )


def _solve_cached_rectangular_coefficients(
    grid: RectangularGridPreparation,
    extended_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the fixed tensor-product interpolation system for one frame."""
    # FITPACK applies the x rotations, then the y rotations, and solves y
    # before x.  The intermediate arrays retain FITPACK's x-major layout.
    x_transformed = _apply_cached_qr_rows(
        np.ascontiguousarray(extended_values.T, dtype=np.float64),
        grid.x_factor.upper,
        grid.x_factor.cosines,
        grid.x_factor.sines,
        grid.x_factor.active,
        grid.x_factor.row_numbers,
    )
    y_transformed = _apply_cached_qr_rows(
        x_transformed.T,
        grid.y_factor.upper,
        grid.y_factor.cosines,
        grid.y_factor.sines,
        grid.y_factor.active,
        grid.y_factor.row_numbers,
    ).T
    y_solved = _back_substitute_cached_qr(grid.y_factor.upper, y_transformed.T).T
    coefficients = _back_substitute_cached_qr(grid.x_factor.upper, y_solved)
    return np.ascontiguousarray(coefficients, dtype=np.float64)


def build_bspline_surface_reference(
    frame: NDArray[np.float64],
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    *,
    periodic_x: bool,
    smoothing: float = _DEFAULT_SMOOTHING,
    grid: RectangularGridPreparation | None = None,
) -> BsplineSurfaceResult:
    """Construct the rectangular surface through SciPy/FITPACK directly.

    This is retained as the numerical reference path for equivalence tests
    and validation.  Production uses :func:`build_bspline_surface` with the
    fixed-grid solve when ``smoothing == 0``.
    """
    if not np.isfinite(smoothing) or smoothing < 0.0:
        return BsplineSurfaceResult(None, "spline_construction_failure")
    ordered = _ordered_rectangular_frame(
        frame,
        latitudes,
        longitudes,
        periodic_x=periodic_x,
        grid=grid,
    )
    if ordered is None:
        return BsplineSurfaceResult(None, "spline_construction_failure")
    prepared_grid, x, y, extended_z = ordered
    extended_x = (
        prepared_grid.extended_longitudes
        if periodic_x
        else prepared_grid.sorted_longitudes
    )

    try:
        spline = RectBivariateSpline(
            extended_x,
            y,
            extended_z.T,
            kx=3,
            ky=3,
            s=smoothing,
        )
    except Exception:  # noqa: BLE001
        return BsplineSurfaceResult(None, "spline_construction_failure")

    fp_tuple = spline.tck
    x_knots = np.asarray(fp_tuple[0], dtype=np.float64)
    y_knots = np.asarray(fp_tuple[1], dtype=np.float64)
    coeffs_raw = np.asarray(fp_tuple[2], dtype=np.float64)

    nx_knots = len(x_knots) - 4
    ny_knots = len(y_knots) - 4
    coeffs_arr = coeffs_raw.reshape(nx_knots, ny_knots).copy()

    return BsplineSurfaceResult(
        BsplineSurface(
            x_knots=x_knots,
            y_knots=y_knots,
            coeffs=coeffs_arr,
            x_lower=float(x_knots[3]),
            x_upper=float(x_knots[-4]),
            y_lower=float(y_knots[3]),
            y_upper=float(y_knots[-4]),
            first_sample_x=float(x[0]),
            last_sample_x=float(x[-1]),
            periodic_x=periodic_x,
        ),
        "success",
    )


def build_bspline_surface(
    frame: NDArray[np.float64],
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    *,
    periodic_x: bool,
    smoothing: float = _DEFAULT_SMOOTHING,
    grid: RectangularGridPreparation | None = None,
) -> BsplineSurfaceResult:
    """Construct a rectangular bicubic spline for one frame.

    For the TRACK-compatible interpolation case (cubic degree and ``s=0``),
    the fixed-grid tensor-product system is solved using the immutable
    preparation.  Other smoothing values retain the FITPACK construction.
    The compiled evaluator and SMOOPY/GDFP layer are unchanged.
    """
    if not np.isfinite(smoothing) or smoothing < 0.0:
        return BsplineSurfaceResult(None, "spline_construction_failure")
    ordered = _ordered_rectangular_frame(
        frame,
        latitudes,
        longitudes,
        periodic_x=periodic_x,
        grid=grid,
    )
    if ordered is None:
        return BsplineSurfaceResult(None, "spline_construction_failure")
    prepared_grid, _x, _y, extended_z = ordered

    if smoothing != _DEFAULT_SMOOTHING:
        return build_bspline_surface_reference(
            frame,
            latitudes,
            longitudes,
            periodic_x=periodic_x,
            smoothing=smoothing,
            grid=prepared_grid,
        )

    try:
        coeffs_arr = _solve_cached_rectangular_coefficients(
            prepared_grid,
            extended_z,
        )
    except Exception:  # noqa: BLE001
        return BsplineSurfaceResult(None, "spline_construction_failure")
    return BsplineSurfaceResult(
        _rectangular_surface(prepared_grid, coeffs_arr, periodic_x),
        "success",
    )


# ---------------------------------------------------------------------------
# Numba-compiled B-spline evaluation and GDFP optimization kernels
# ---------------------------------------------------------------------------


@nb.njit(cache=True, nogil=True)
def _find_span(n: int, t: NDArray[np.float64], u: float) -> int:
    """Find the knot span index for parameter u."""
    if u >= t[n + 1]:
        return n
    if u <= t[3]:
        return 3
    low = 3
    high = n + 1
    mid = (low + high) // 2
    while u < t[mid] or u >= t[mid + 1]:
        if u < t[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


@nb.njit(cache=True, nogil=True)
def _bspline_basis_and_derivs(
    t: NDArray[np.float64], u: float, span: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute non-zero cubic B-spline basis functions and their first derivatives."""
    left = np.zeros(4, dtype=np.float64)
    right = np.zeros(4, dtype=np.float64)
    n_mat = np.zeros((4, 4), dtype=np.float64)
    n_mat[0, 0] = 1.0
    for j in range(1, 4):
        left[j] = u - t[span + 1 - j]
        right[j] = t[span + j] - u
        saved = 0.0
        for r in range(j):
            n_mat[j, r] = right[r + 1] + left[j - r]
            temp = n_mat[r, j - 1] / n_mat[j, r]
            n_mat[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        n_mat[j, j] = saved

    basis = np.array(
        [n_mat[0, 3], n_mat[1, 3], n_mat[2, 3], n_mat[3, 3]], dtype=np.float64
    )
    deriv = np.zeros(4, dtype=np.float64)
    for r in range(4):
        j = span - 3 + r
        term1 = 0.0
        if r > 0:
            d1 = t[j + 3] - t[j]
            if d1 > 0.0:
                term1 = n_mat[r - 1, 2] / d1
        term2 = 0.0
        if r < 3:
            d2 = t[j + 4] - t[j + 1]
            if d2 > 0.0:
                term2 = n_mat[r, 2] / d2
        deriv[r] = 3.0 * (term1 - term2)
    return basis, deriv


@nb.njit(cache=True, nogil=True)
def _eval_bspline_2d(
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
    c: NDArray[np.float64],
    x: float,
    y: float,
) -> tuple[float, float, float]:
    """Evaluate 2D bicubic B-spline value, dx, and dy in native nogil code."""
    nx = len(tx) - 4
    ny = len(ty) - 4
    span_x = _find_span(nx - 1, tx, x)
    span_y = _find_span(ny - 1, ty, y)
    bx, dbx = _bspline_basis_and_derivs(tx, x, span_x)
    by, dby = _bspline_basis_and_derivs(ty, y, span_y)

    val = 0.0
    dx = 0.0
    dy = 0.0
    for i in range(4):
        cx = span_x - 3 + i
        for j in range(4):
            cy = span_y - 3 + j
            coef = c[cx, cy]
            val += coef * bx[i] * by[j]
            dx += coef * dbx[i] * by[j]
            dy += coef * bx[i] * dby[j]
    return val, dx, dy


@nb.njit(cache=True, nogil=True)
def _eval_bspline_val(
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
    c: NDArray[np.float64],
    x: float,
    y: float,
) -> float:
    """Evaluate 2D bicubic B-spline value only in native nogil code."""
    nx = len(tx) - 4
    ny = len(ty) - 4
    span_x = _find_span(nx - 1, tx, x)
    span_y = _find_span(ny - 1, ty, y)
    bx, _ = _bspline_basis_and_derivs(tx, x, span_x)
    by, _ = _bspline_basis_and_derivs(ty, y, span_y)

    val = 0.0
    for i in range(4):
        cx = span_x - 3 + i
        for j in range(4):
            cy = span_y - 3 + j
            val += c[cx, cy] * bx[i] * by[j]
    return val


@nb.njit(cache=True, nogil=True)
def _box_line_limit(
    point: NDArray[np.float64],
    direction: NDArray[np.float64],
    bounds: NDArray[np.float64],
) -> float:
    """Find the positive step to the first bounding-box intersection."""
    min_limit = np.inf
    has_limit = False
    for k in range(2):
        coord = point[k]
        comp = direction[k]
        low = bounds[k, 0]
        high = bounds[k, 1]
        if comp < 0.0 and np.isfinite(low):
            step = (coord - low) / -comp
            if step < min_limit:
                min_limit = step
                has_limit = True
        elif comp > 0.0 and np.isfinite(high):
            step = (high - coord) / comp
            if step < min_limit:
                min_limit = step
                has_limit = True
    if not has_limit:
        return np.inf
    return max(0.0, min_limit)


@nb.njit(cache=True, nogil=True)
def _cubic_minimum_step(
    v0: float, g0: float, v1: float, g1: float, step1: float
) -> float:
    """Find minimum location of cubic Hermite interpolant on [0, step1]."""
    if step1 <= 0.0 or not (
        np.isfinite(v0) and np.isfinite(g0) and np.isfinite(v1) and np.isfinite(g1)
    ):
        return -1.0

    d1 = g0 + g1 - 3.0 * (v0 - v1) / step1
    d2_sq = d1 * d1 - g0 * g1
    if d2_sq < 0.0:
        return -1.0
    d2 = np.sqrt(d2_sq)

    denom = g1 - g0 + 2.0 * d2
    if abs(denom) < 1.0e-14:
        return -1.0
    gamma = (g1 + d2 - d1) / denom
    if gamma < 0.0 or gamma > 1.0:
        return -1.0
    return float(gamma * step1)


# --- Spherical Spline Line Search & GDFP in Numba ---


@nb.njit(cache=True, nogil=True)
def _eval_spherical_intrinsic(
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
    c: NDArray[np.float64],
    sign: float,
    r_vec: NDArray[np.float64],
) -> tuple[float, float, float, NDArray[np.float64], float, float]:
    """Evaluate objective value, 2D gradient, and 3D gradient vector at 3D point."""
    r_norm_sq = r_vec[0] ** 2 + r_vec[1] ** 2 + r_vec[2] ** 2
    inv_len = 1.0 / np.sqrt(r_norm_sq) if r_norm_sq > 1.0e-20 else 1.0
    rx = r_vec[0] * inv_len
    ry = r_vec[1] * inv_len
    rz = r_vec[2] * inv_len
    if rz > 1.0:
        rz = 1.0
    elif rz < -1.0:
        rz = -1.0

    theta = np.arccos(rz)
    phi = np.arctan2(ry, rx) % (2.0 * np.pi)

    eps_theta = 1.0e-7
    if theta < eps_theta:
        theta_eval = eps_theta
    elif theta > np.pi - eps_theta:
        theta_eval = np.pi - eps_theta
    else:
        theta_eval = theta

    val, dth, dph = _eval_bspline_2d(tx, ty, c, theta_eval, phi)
    f_val = sign * val
    st = np.sin(theta_eval)
    st = max(st, eps_theta)
    g_theta = sign * dth
    g_phi = (sign * dph) / st

    _, e_th, e_ph = _sphere_point_and_basis(theta, phi)
    g_3d = np.array(
        [
            g_theta * e_th[0] + g_phi * e_ph[0],
            g_theta * e_th[1] + g_phi * e_ph[1],
            g_theta * e_th[2] + g_phi * e_ph[2],
        ],
        dtype=np.float64,
    )
    return f_val, g_theta, g_phi, g_3d, theta, phi


@nb.njit(cache=True, nogil=True)
def _parallel_transport_sphere(
    v_3d: NDArray[np.float64],
    u_3d: NDArray[np.float64],
    best_u: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Parallel transport 3D tangent vector along great circle from u_3d to best_u."""
    v_dot_u = v_3d[0] * u_3d[0] + v_3d[1] * u_3d[1] + v_3d[2] * u_3d[2]
    return np.array(
        [
            v_dot_u * best_u[0] + (v_3d[0] - v_dot_u * u_3d[0]),
            v_dot_u * best_u[1] + (v_3d[1] - v_dot_u * u_3d[1]),
            v_dot_u * best_u[2] + (v_3d[2] - v_dot_u * u_3d[2]),
        ],
        dtype=np.float64,
    )


def _spherical_localization_region(
    surface: SphericalBsplineSurface,
    latitude: float,
    longitude: float,
    search_window_size: int,
) -> SphericalLocalizationRegion | None:
    """Derive a fixed local region from the candidate's detector footprint.

    For a 3 by 3 detector window, the region covers the candidate's immediate
    meridional and zonal input-grid neighbours.  Gaussian latitude spacing and
    the local physical zonal distance are retained rather than replacing them
    with a constant angular threshold.
    """
    return spherical_localization_region(
        surface.sample_latitudes,
        surface.sample_longitudes,
        latitude,
        longitude,
        search_window_size,
    )


@nb.njit(cache=True, nogil=True)
def _spherical_localization_boundary_distance(
    point: NDArray[np.float64],
    unit_tangent: NDArray[np.float64],
    origin: NDArray[np.float64],
    origin_e_theta: NDArray[np.float64],
    origin_e_phi: NDArray[np.float64],
    theta_half_width: float,
    phi_half_width: float,
) -> float:
    """Find the first geodesic intersection with the fixed local boundary."""
    if not _inside_spherical_localization_region(
        point,
        origin,
        origin_e_theta,
        origin_e_phi,
        theta_half_width,
        phi_half_width,
    ):
        return 0.0

    upper = np.pi - 1.0e-12
    outer = np.cos(upper) * point + np.sin(upper) * unit_tangent
    if _inside_spherical_localization_region(
        outer,
        origin,
        origin_e_theta,
        origin_e_phi,
        theta_half_width,
        phi_half_width,
    ):
        return upper

    lower = 0.0
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        trial = np.cos(middle) * point + np.sin(middle) * unit_tangent
        if _inside_spherical_localization_region(
            trial,
            origin,
            origin_e_theta,
            origin_e_phi,
            theta_half_width,
            phi_half_width,
        ):
            lower = middle
        else:
            upper = middle
    # Keep a small interior margin because the accepted point is converted
    # through latitude/longitude coordinates before the next iteration.
    # Without this guard, round-off can reconstruct a boundary point just
    # outside the predicate despite bisection having classified ``lower`` as
    # inside.
    return max(0.0, lower - 1.0e-8)


@nb.njit(cache=True, nogil=True)
def _inverse_hessian_is_well_conditioned(matrix: NDArray[np.float64]) -> bool:
    """Check the transported 2 by 2 inverse-Hessian approximation is SPD."""
    trace = matrix[0, 0] + matrix[1, 1]
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    discriminant = max(trace * trace - 4.0 * determinant, 0.0)
    root = np.sqrt(discriminant)
    eigenvalue_minimum = 0.5 * (trace - root)
    eigenvalue_maximum = 0.5 * (trace + root)
    if (
        not np.isfinite(eigenvalue_minimum)
        or not np.isfinite(eigenvalue_maximum)
        or eigenvalue_minimum <= 1.0e-14
    ):
        return False
    return bool(eigenvalue_maximum / eigenvalue_minimum <= 1.0e6)


@nb.njit(cache=True, nogil=True)
def _spherical_geodesic_gdfp_search(
    theta_0: float,
    phi_0: float,
    sign: float,
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
    c: NDArray[np.float64],
    max_iterations: int,
    gradient_tolerance: float,
    max_disp_rad: float = 0.08726646259971647,
) -> tuple[int, float, float, float]:
    """Intrinsic S^2 Riemannian quasi-Newton refinement (PyStormTracker extension).

    This is a PyStormTracker-specific extension, not part of the TRACK-1.5.4
    coordinate-space GDFP implementation.
    The great-circle geodesic search, parallel transport of gradients,
    tangent-space change-of-basis for the inverse-Hessian approximation, and
    tangent-space DFP updates follow standard Riemannian optimization
    principles; see Huang, Gallivan & Absil (2015), Edelman, Arias & Smith
    (1998), and Smith, *Optimization Techniques on Riemannian Manifolds*:
    https://doi.org/10.1137/140955483
    https://doi.org/10.1137/S0895479895290954
    https://doi.org/10.1090/fic/003/09
    """
    v0 = _eval_bspline_val(tx, ty, c, theta_0, phi_0)
    scale_f = abs(v0)
    scale_f = max(scale_f, 1.0e-12)

    r_cur, e_th_cur, e_ph_cur = _sphere_point_and_basis(theta_0, phi_0)
    f_cur, g_th, g_ph, g_3d_cur, theta_cur, phi_cur = _eval_spherical_intrinsic(
        tx, ty, c, sign, r_cur
    )
    previous_f = f_cur
    h_mat = np.eye(2, dtype=np.float64)

    for _iteration in range(max_iterations):
        g_norm = np.sqrt(g_th**2 + g_ph**2)
        dimless_grad = g_norm / scale_f
        if dimless_grad < gradient_tolerance:
            return 0, theta_cur, phi_cur, sign * f_cur

        g_2d = np.array([g_th, g_ph], dtype=np.float64)
        p_2d = -h_mat @ g_2d
        dir_deriv = p_2d[0] * g_th + p_2d[1] * g_ph
        if dir_deriv >= 0.0:
            h_mat = np.eye(2, dtype=np.float64)
            p_2d = np.array([-g_th, -g_ph], dtype=np.float64)
            dir_deriv = p_2d[0] * g_th + p_2d[1] * g_ph

        v_3d = np.array(
            [
                p_2d[0] * e_th_cur[0] + p_2d[1] * e_ph_cur[0],
                p_2d[0] * e_th_cur[1] + p_2d[1] * e_ph_cur[1],
                p_2d[0] * e_th_cur[2] + p_2d[1] * e_ph_cur[2],
            ],
            dtype=np.float64,
        )
        v_norm = np.sqrt(v_3d[0] ** 2 + v_3d[1] ** 2 + v_3d[2] ** 2)
        if v_norm < 1.0e-14:
            if dimless_grad < gradient_tolerance:
                return 0, theta_cur, phi_cur, sign * f_cur
            return 2, theta_cur, phi_cur, sign * f_cur

        u_3d = v_3d / v_norm
        slope_0 = g_3d_cur[0] * u_3d[0] + g_3d_cur[1] * u_3d[1] + g_3d_cur[2] * u_3d[2]

        limit_step = max_disp_rad
        delta_f = abs(f_cur - previous_f)
        if delta_f > 0.0 and abs(slope_0) > 0.0:
            init_step = min(limit_step, min(0.2, 2.0 * delta_f / abs(slope_0)))
        else:
            init_step = min(
                limit_step,
                min(0.05, 0.01 * scale_f / (g_norm + 1.0e-12 * scale_f)),
            )

        if init_step < 1.0e-6:
            init_step = 1.0e-4

        # Line search along geodesic circle
        step_a = 0.0
        slope_a = slope_0
        val_a = f_cur

        step_b = min(limit_step, init_step)
        r_b = np.cos(step_b) * r_cur + np.sin(step_b) * u_3d
        u_b = -np.sin(step_b) * r_cur + np.cos(step_b) * u_3d
        val_b, _, _, g_3d_b, th_b, ph_b = _eval_spherical_intrinsic(
            tx, ty, c, sign, r_b
        )
        slope_b = g_3d_b[0] * u_b[0] + g_3d_b[1] * u_b[1] + g_3d_b[2] * u_b[2]

        line_it = 0
        while slope_b < 0.0 and step_b < limit_step and line_it < 500:
            line_it += 1
            step_a = step_b
            slope_a = slope_b
            val_a = val_b
            step_b = min(limit_step, 2.0 * step_b)
            r_b = np.cos(step_b) * r_cur + np.sin(step_b) * u_3d
            u_b = -np.sin(step_b) * r_cur + np.cos(step_b) * u_3d
            val_b, _, _, g_3d_b, th_b, ph_b = _eval_spherical_intrinsic(
                tx, ty, c, sign, r_b
            )
            slope_b = g_3d_b[0] * u_b[0] + g_3d_b[1] * u_b[1] + g_3d_b[2] * u_b[2]

        best_step = step_b
        best_val = val_b
        best_u = u_b
        best_g_3d = g_3d_b
        best_th = th_b
        best_ph = ph_b

        if slope_b >= 0.0:
            while abs(step_b - step_a) > 1.0e-5 and line_it < 500:
                line_it += 1
                interval = step_b - step_a
                z = 3.0 * (val_a - val_b) / interval + slope_a + slope_b
                disc = z**2 - slope_a * slope_b
                if disc < 0.0:
                    step_mid = 0.5 * (step_a + step_b)
                else:
                    w = np.sqrt(disc)
                    denom = slope_b - slope_a + 2.0 * w
                    if abs(denom) < 1.0e-12:
                        step_mid = 0.5 * (step_a + step_b)
                    else:
                        step_ratio = 1.0 - (slope_b + w - z) / denom
                        if step_ratio < 0.0 or step_ratio > 1.0:
                            step_mid = 0.5 * (step_a + step_b)
                        else:
                            step_interp = step_a + step_ratio * interval
                            step_mid = min(
                                max(step_interp, step_a + 0.1 * interval),
                                step_b - 0.1 * interval,
                            )

                r_mid = np.cos(step_mid) * r_cur + np.sin(step_mid) * u_3d
                u_mid = -np.sin(step_mid) * r_cur + np.cos(step_mid) * u_3d
                val_mid, _, _, g_3d_mid, th_mid, ph_mid = _eval_spherical_intrinsic(
                    tx, ty, c, sign, r_mid
                )
                slope_mid = (
                    g_3d_mid[0] * u_mid[0]
                    + g_3d_mid[1] * u_mid[1]
                    + g_3d_mid[2] * u_mid[2]
                )

                best_step = step_mid
                best_val = val_mid
                best_u = u_mid
                best_g_3d = g_3d_mid
                best_th = th_mid
                best_ph = ph_mid

                if slope_mid >= 0.0:
                    step_b = step_mid
                    slope_b = slope_mid
                    val_b = val_mid
                else:
                    step_a = step_mid
                    slope_a = slope_mid
                    val_a = val_mid

        previous_f = f_cur
        f_cur = best_val
        if abs(f_cur - previous_f) / scale_f < 1.0e-6:
            return 0, best_th, best_ph, sign * f_cur

        # Parallel transport old tangent basis and old gradient along geodesic
        e_th_old_tr = _parallel_transport_sphere(e_th_cur, u_3d, best_u)
        e_ph_old_tr = _parallel_transport_sphere(e_ph_cur, u_3d, best_u)
        g_old_transported = _parallel_transport_sphere(g_3d_cur, u_3d, best_u)

        r_new, e_th_new, e_ph_new = _sphere_point_and_basis(best_th, best_ph)

        # 2x2 Orthogonal matrix Q mapping old tangent components to new basis
        q_mat = np.array(
            [
                [
                    e_th_new[0] * e_th_old_tr[0]
                    + e_th_new[1] * e_th_old_tr[1]
                    + e_th_new[2] * e_th_old_tr[2],
                    e_th_new[0] * e_ph_old_tr[0]
                    + e_th_new[1] * e_ph_old_tr[1]
                    + e_th_new[2] * e_ph_old_tr[2],
                ],
                [
                    e_ph_new[0] * e_th_old_tr[0]
                    + e_ph_new[1] * e_th_old_tr[1]
                    + e_ph_new[2] * e_th_old_tr[2],
                    e_ph_new[0] * e_ph_old_tr[0]
                    + e_ph_new[1] * e_ph_old_tr[1]
                    + e_ph_new[2] * e_ph_old_tr[2],
                ],
            ],
            dtype=np.float64,
        )

        # Transport the inverse-Hessian approximation into the new tangent basis
        h_transport = q_mat @ h_mat @ q_mat.T

        g_old_proj = np.array(
            [
                g_old_transported[0] * e_th_new[0]
                + g_old_transported[1] * e_th_new[1]
                + g_old_transported[2] * e_th_new[2],
                g_old_transported[0] * e_ph_new[0]
                + g_old_transported[1] * e_ph_new[1]
                + g_old_transported[2] * e_ph_new[2],
            ],
            dtype=np.float64,
        )
        g_new_proj = np.array(
            [
                best_g_3d[0] * e_th_new[0]
                + best_g_3d[1] * e_th_new[1]
                + best_g_3d[2] * e_th_new[2],
                best_g_3d[0] * e_ph_new[0]
                + best_g_3d[1] * e_ph_new[1]
                + best_g_3d[2] * e_ph_new[2],
            ],
            dtype=np.float64,
        )

        s_2d = best_step * np.array(
            [
                best_u[0] * e_th_new[0]
                + best_u[1] * e_th_new[1]
                + best_u[2] * e_th_new[2],
                best_u[0] * e_ph_new[0]
                + best_u[1] * e_ph_new[1]
                + best_u[2] * e_ph_new[2],
            ],
            dtype=np.float64,
        )
        y_2d = g_new_proj - g_old_proj

        gamma = s_2d[0] * y_2d[0] + s_2d[1] * y_2d[1]
        h_gamma = h_transport @ y_2d
        delta_h_delta = y_2d[0] * h_gamma[0] + y_2d[1] * h_gamma[1]

        if gamma > 1.0e-14 and delta_h_delta > 1.0e-14:
            h_mat = (
                h_transport
                + np.outer(s_2d, s_2d) / gamma
                - np.outer(h_gamma, h_gamma) / delta_h_delta
            )
        else:
            h_mat = np.eye(2, dtype=np.float64)

        theta_cur = best_th
        phi_cur = best_ph
        r_cur = r_new
        e_th_cur = e_th_new
        e_ph_cur = e_ph_new
        g_3d_cur = best_g_3d
        g_th = g_new_proj[0]
        g_ph = g_new_proj[1]

    return 2, theta_cur, phi_cur, sign * f_cur


@nb.njit(cache=True, nogil=True)
def _spherical_local_gdfp_search(
    theta_0: float,
    phi_0: float,
    sign: float,
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
    c: NDArray[np.float64],
    max_iterations: int,
    gradient_tolerance: float,
    origin: NDArray[np.float64],
    origin_e_theta: NDArray[np.float64],
    origin_e_phi: NDArray[np.float64],
    theta_half_width: float,
    phi_half_width: float,
) -> tuple[int, int, int, float, float, float]:
    """Refine on S² without leaving the original detector neighbourhood.

    Trial points use exact great-circle geodesics.  Every line search is
    clipped at the first intersection with the immutable candidate-centred
    tangent-space rectangle and accepts only finite Armijo descent.  Strong
    Wolfe curvature is used when it is available; a decreasing boundary step
    remains a valid constrained iteration but cannot itself be called success.
    """
    initial_value = _eval_bspline_val(tx, ty, c, theta_0, phi_0)
    scale_f = max(abs(initial_value), 1.0e-12)
    point, e_theta, e_phi = _sphere_point_and_basis(theta_0, phi_0)
    value, g_theta, g_phi, gradient_3d, theta, phi = _eval_spherical_intrinsic(
        tx, ty, c, sign, point
    )
    inverse_hessian = np.eye(2, dtype=np.float64)
    armijo_c1 = 1.0e-4
    wolfe_c2 = 0.9
    boundary_contacts = 0

    for _iteration in range(max_iterations):
        gradient_norm = np.sqrt(g_theta * g_theta + g_phi * g_phi)
        if gradient_norm / scale_f < gradient_tolerance:
            return 0, 0, boundary_contacts, theta, phi, sign * value

        gradient_2d = np.array([g_theta, g_phi], dtype=np.float64)
        if not _inverse_hessian_is_well_conditioned(inverse_hessian):
            inverse_hessian = np.eye(2, dtype=np.float64)
        direction_2d = -inverse_hessian @ gradient_2d
        directional_derivative = (
            direction_2d[0] * gradient_2d[0] + direction_2d[1] * gradient_2d[1]
        )
        if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
            inverse_hessian = np.eye(2, dtype=np.float64)
            direction_2d = -gradient_2d

        tangent_vector = np.array(
            [
                direction_2d[0] * e_theta[0] + direction_2d[1] * e_phi[0],
                direction_2d[0] * e_theta[1] + direction_2d[1] * e_phi[1],
                direction_2d[0] * e_theta[2] + direction_2d[1] * e_phi[2],
            ],
            dtype=np.float64,
        )
        tangent_norm = np.sqrt(
            tangent_vector[0] * tangent_vector[0]
            + tangent_vector[1] * tangent_vector[1]
            + tangent_vector[2] * tangent_vector[2]
        )
        if tangent_norm < 1.0e-14:
            return 2, 1, boundary_contacts, theta, phi, sign * value
        unit_tangent = tangent_vector / tangent_norm
        slope_start = (
            gradient_3d[0] * unit_tangent[0]
            + gradient_3d[1] * unit_tangent[1]
            + gradient_3d[2] * unit_tangent[2]
        )
        if not np.isfinite(slope_start) or slope_start >= 0.0:
            inverse_hessian = np.eye(2, dtype=np.float64)
            tangent_vector = -gradient_3d
            tangent_norm = np.sqrt(
                tangent_vector[0] * tangent_vector[0]
                + tangent_vector[1] * tangent_vector[1]
                + tangent_vector[2] * tangent_vector[2]
            )
            if tangent_norm < 1.0e-14:
                return 2, 1, boundary_contacts, theta, phi, sign * value
            unit_tangent = tangent_vector / tangent_norm
            slope_start = (
                gradient_3d[0] * unit_tangent[0]
                + gradient_3d[1] * unit_tangent[1]
                + gradient_3d[2] * unit_tangent[2]
            )

        max_step = _spherical_localization_boundary_distance(
            point,
            unit_tangent,
            origin,
            origin_e_theta,
            origin_e_phi,
            theta_half_width,
            phi_half_width,
        )
        if max_step <= 1.0e-10:
            return 2, 2, boundary_contacts, theta, phi, sign * value

        initial_step = min(
            max_step,
            min(0.05, 0.01 * scale_f / (gradient_norm + 1.0e-12 * scale_f)),
        )
        initial_step = max(initial_step, min(max_step, 1.0e-5))
        accepted = False
        best_step = 0.0
        best_value = value
        best_gradient_3d = gradient_3d
        best_theta = theta
        best_phi = phi
        best_tangent = unit_tangent
        lower_step = 0.0
        lower_value = value
        lower_gradient_3d = gradient_3d
        lower_theta = theta
        lower_phi = phi
        lower_tangent = unit_tangent
        step = initial_step
        upper_step = 0.0
        bracketed = False

        for _line_iteration in range(32):
            trial_point = np.cos(step) * point + np.sin(step) * unit_tangent
            if not _inside_spherical_localization_region(
                trial_point,
                origin,
                origin_e_theta,
                origin_e_phi,
                theta_half_width,
                phi_half_width,
            ):
                upper_step = step
                bracketed = True
                break
            trial_tangent = -np.sin(step) * point + np.cos(step) * unit_tangent
            (
                trial_value,
                _trial_g_theta,
                _trial_g_phi,
                trial_gradient_3d,
                trial_theta,
                trial_phi,
            ) = _eval_spherical_intrinsic(tx, ty, c, sign, trial_point)
            trial_slope = (
                trial_gradient_3d[0] * trial_tangent[0]
                + trial_gradient_3d[1] * trial_tangent[1]
                + trial_gradient_3d[2] * trial_tangent[2]
            )
            armijo = (
                np.isfinite(trial_value)
                and np.isfinite(trial_slope)
                and trial_value <= value + armijo_c1 * step * slope_start
            )
            if armijo:
                lower_step = step
                lower_value = trial_value
                lower_gradient_3d = trial_gradient_3d
                lower_theta = trial_theta
                lower_phi = trial_phi
                lower_tangent = trial_tangent
                if (
                    abs(trial_slope) <= -wolfe_c2 * slope_start
                    or step >= max_step - 1.0e-12
                ):
                    accepted = True
                    break
                if trial_slope < 0.0:
                    next_step = min(max_step, 2.0 * step)
                    if next_step <= step + 1.0e-12:
                        accepted = True
                        break
                    step = next_step
                    continue

            upper_step = step
            bracketed = True
            break

        if not accepted and bracketed:
            for _zoom_iteration in range(32):
                if upper_step - lower_step <= 1.0e-12:
                    break
                step = 0.5 * (lower_step + upper_step)
                trial_point = np.cos(step) * point + np.sin(step) * unit_tangent
                if not _inside_spherical_localization_region(
                    trial_point,
                    origin,
                    origin_e_theta,
                    origin_e_phi,
                    theta_half_width,
                    phi_half_width,
                ):
                    upper_step = step
                    continue
                trial_tangent = -np.sin(step) * point + np.cos(step) * unit_tangent
                (
                    trial_value,
                    _trial_g_theta,
                    _trial_g_phi,
                    trial_gradient_3d,
                    trial_theta,
                    trial_phi,
                ) = _eval_spherical_intrinsic(tx, ty, c, sign, trial_point)
                trial_slope = (
                    trial_gradient_3d[0] * trial_tangent[0]
                    + trial_gradient_3d[1] * trial_tangent[1]
                    + trial_gradient_3d[2] * trial_tangent[2]
                )
                armijo = (
                    np.isfinite(trial_value)
                    and np.isfinite(trial_slope)
                    and trial_value <= value + armijo_c1 * step * slope_start
                )
                if armijo and trial_value < lower_value:
                    lower_step = step
                    lower_value = trial_value
                    lower_gradient_3d = trial_gradient_3d
                    lower_theta = trial_theta
                    lower_phi = trial_phi
                    lower_tangent = trial_tangent
                    if abs(trial_slope) <= -wolfe_c2 * slope_start:
                        accepted = True
                        break
                    if trial_slope < 0.0:
                        continue
                upper_step = step

        if not accepted and lower_step > 0.0 and lower_value < value:
            accepted = True
        if not accepted:
            return 2, 3, boundary_contacts, theta, phi, sign * value

        best_step = lower_step
        best_value = lower_value
        best_gradient_3d = lower_gradient_3d
        best_theta = lower_theta
        best_phi = lower_phi
        best_tangent = lower_tangent
        if not np.isfinite(best_value) or best_value >= value:
            return 2, 3, boundary_contacts, theta, phi, sign * value
        if best_step >= max_step - 1.0e-12:
            boundary_contacts += 1

        transported_e_theta = _parallel_transport_sphere(
            e_theta, unit_tangent, best_tangent
        )
        transported_e_phi = _parallel_transport_sphere(
            e_phi, unit_tangent, best_tangent
        )
        transported_gradient = _parallel_transport_sphere(
            gradient_3d, unit_tangent, best_tangent
        )
        point_new, e_theta_new, e_phi_new = _sphere_point_and_basis(
            best_theta, best_phi
        )
        if not _inside_spherical_localization_region(
            point_new,
            origin,
            origin_e_theta,
            origin_e_phi,
            theta_half_width,
            phi_half_width,
        ):
            return 2, 2, boundary_contacts, theta, phi, sign * value
        q_matrix = np.array(
            [
                [
                    e_theta_new[0] * transported_e_theta[0]
                    + e_theta_new[1] * transported_e_theta[1]
                    + e_theta_new[2] * transported_e_theta[2],
                    e_theta_new[0] * transported_e_phi[0]
                    + e_theta_new[1] * transported_e_phi[1]
                    + e_theta_new[2] * transported_e_phi[2],
                ],
                [
                    e_phi_new[0] * transported_e_theta[0]
                    + e_phi_new[1] * transported_e_theta[1]
                    + e_phi_new[2] * transported_e_theta[2],
                    e_phi_new[0] * transported_e_phi[0]
                    + e_phi_new[1] * transported_e_phi[1]
                    + e_phi_new[2] * transported_e_phi[2],
                ],
            ],
            dtype=np.float64,
        )
        transported_hessian = q_matrix @ inverse_hessian @ q_matrix.T
        old_gradient = np.array(
            [
                transported_gradient[0] * e_theta_new[0]
                + transported_gradient[1] * e_theta_new[1]
                + transported_gradient[2] * e_theta_new[2],
                transported_gradient[0] * e_phi_new[0]
                + transported_gradient[1] * e_phi_new[1]
                + transported_gradient[2] * e_phi_new[2],
            ],
            dtype=np.float64,
        )
        new_gradient = np.array(
            [
                best_gradient_3d[0] * e_theta_new[0]
                + best_gradient_3d[1] * e_theta_new[1]
                + best_gradient_3d[2] * e_theta_new[2],
                best_gradient_3d[0] * e_phi_new[0]
                + best_gradient_3d[1] * e_phi_new[1]
                + best_gradient_3d[2] * e_phi_new[2],
            ],
            dtype=np.float64,
        )
        displacement = best_step * np.array(
            [
                best_tangent[0] * e_theta_new[0]
                + best_tangent[1] * e_theta_new[1]
                + best_tangent[2] * e_theta_new[2],
                best_tangent[0] * e_phi_new[0]
                + best_tangent[1] * e_phi_new[1]
                + best_tangent[2] * e_phi_new[2],
            ],
            dtype=np.float64,
        )
        gradient_change = new_gradient - old_gradient
        curvature = (
            displacement[0] * gradient_change[0] + displacement[1] * gradient_change[1]
        )
        h_gradient_change = transported_hessian @ gradient_change
        transformed_curvature = (
            gradient_change[0] * h_gradient_change[0]
            + gradient_change[1] * h_gradient_change[1]
        )
        if curvature > 1.0e-14 and transformed_curvature > 1.0e-14:
            updated_hessian = (
                transported_hessian
                + np.outer(displacement, displacement) / curvature
                - np.outer(h_gradient_change, h_gradient_change) / transformed_curvature
            )
            if _inverse_hessian_is_well_conditioned(updated_hessian):
                inverse_hessian = updated_hessian
            else:
                inverse_hessian = np.eye(2, dtype=np.float64)
        else:
            inverse_hessian = np.eye(2, dtype=np.float64)

        point = point_new
        e_theta = e_theta_new
        e_phi = e_phi_new
        gradient_3d = best_gradient_3d
        g_theta = new_gradient[0]
        g_phi = new_gradient[1]
        value = best_value
        theta = best_theta
        phi = best_phi

    return 2, 4, boundary_contacts, theta, phi, sign * value


# --- Rectangular Spline Line Search & GDFP in Numba ---


@nb.njit(cache=True, nogil=True)
def _eval_rectangular_line_point(
    point: NDArray[np.float64],
    direction: NDArray[np.float64],
    step: float,
    sign: float,
    opt_scale: float,
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
    c: NDArray[np.float64],
) -> tuple[bool, NDArray[np.float64], float, NDArray[np.float64]]:
    trial = point + step * direction
    val, dx, dy = _eval_bspline_2d(tx, ty, c, trial[0], trial[1])
    value = sign * opt_scale * val
    derivative = np.array(
        [sign * opt_scale * dx, sign * opt_scale * dy], dtype=np.float64
    )
    if not np.isfinite(value) or not (
        np.isfinite(derivative[0]) and np.isfinite(derivative[1])
    ):
        return False, trial, np.nan, derivative
    return True, trial, value, derivative


@nb.njit(cache=True, nogil=True)
def _line_search_rectangular(
    point: NDArray[np.float64],
    direction: NDArray[np.float64],
    sign: float,
    opt_scale: float,
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
    c: NDArray[np.float64],
    initial_slope: float,
    initial_step: float,
    maximum_step: float,
    interval_tolerance: float,
    maximum_iterations: int,
) -> tuple[int, float, NDArray[np.float64], float, NDArray[np.float64]]:
    """Line search along descent direction for rectangular spline."""
    step_a = 0.0
    slope_a = initial_slope
    ok_a, _point_a, value_a, _ = _eval_rectangular_line_point(
        point, direction, 0.0, sign, opt_scale, tx, ty, c
    )
    if not ok_a:
        return 2, 0.0, point, np.nan, np.empty(2, dtype=np.float64)
    step_b = min(maximum_step, max(1.0e-8, initial_step))
    ok, point_b, value_b, derivative_b = _eval_rectangular_line_point(
        point, direction, step_b, sign, opt_scale, tx, ty, c
    )
    if not ok:
        return 2, 0.0, point, np.nan, np.empty(2, dtype=np.float64)
    slope_b = np.dot(direction, derivative_b)

    iteration = 0
    while slope_b > 0.0 and step_b < maximum_step and iteration < maximum_iterations:
        if value_b < value_a:
            break
        iteration += 1
        step_a = step_b
        slope_a = slope_b
        value_a = value_b
        step_b = min(maximum_step, 2.0 * step_b)
        ok, point_b, value_b, derivative_b = _eval_rectangular_line_point(
            point, direction, step_b, sign, opt_scale, tx, ty, c
        )
        if not ok:
            return 2, 0.0, point, np.nan, np.empty(2, dtype=np.float64)
        slope_b = np.dot(direction, derivative_b)

    if slope_b > 0.0 and value_b >= value_a:
        return 0, step_b, point_b, value_b, derivative_b

    best_step = step_b
    best_point = point_b.copy()
    best_value = value_b
    best_derivative = derivative_b.copy()

    while abs(step_b - step_a) > interval_tolerance and iteration < maximum_iterations:
        iteration += 1
        ok_a, _point_a, value_a, _ = _eval_rectangular_line_point(
            point, direction, step_a, sign, opt_scale, tx, ty, c
        )
        ok_b, point_b, value_b, derivative_b = _eval_rectangular_line_point(
            point, direction, step_b, sign, opt_scale, tx, ty, c
        )
        if not ok_a or not ok_b:
            return 2, 0.0, point, np.nan, np.empty(2, dtype=np.float64)
        slope_b = np.dot(direction, derivative_b)

        has_interp = _cubic_minimum_step(
            value_a, slope_a, value_b, slope_b, step_b - step_a
        )
        if has_interp < 0.0:
            step_mid = 0.5 * (step_a + step_b)
        else:
            interpolated = step_a + has_interp
            step_mid = min(
                max(interpolated, step_a + 0.1 * (step_b - step_a)),
                step_b - 0.1 * (step_b - step_a),
            )

        (
            ok_mid,
            point_mid,
            value_mid,
            derivative_mid,
        ) = _eval_rectangular_line_point(
            point, direction, step_mid, sign, opt_scale, tx, ty, c
        )
        if not ok_mid:
            return 2, 0.0, point, np.nan, np.empty(2, dtype=np.float64)
        slope_mid = np.dot(direction, derivative_mid)

        best_step = step_mid
        best_point = point_mid.copy()
        best_value = value_mid
        best_derivative = derivative_mid.copy()

        if slope_mid <= 0.0:
            step_b = step_mid
            slope_b = slope_mid
        else:
            step_a = step_mid
            slope_a = slope_mid

    return 0, best_step, best_point, best_value, best_derivative


@nb.njit(cache=True, nogil=True)
def _smoopy_gdfp_search(
    start: NDArray[np.float64],
    sign: float,
    opt_scale: float,
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
    c: NDArray[np.float64],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    initial_value: float,
    max_iterations: int,
    gradient_tolerance: float,
) -> tuple[int, NDArray[np.float64]]:
    """TRACK-compatible rectangular SMOOPY GDFP search.

    Implements the coordinate-space GDFP variable-metric optimization and
    Goldfarb DFP update following TRACK 1.5.4 SMOOPY behavior.  The exact
    source references are ``lib/src/gdfp_optimize.c`` and ``lib/src/update_h.c``:
    https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/lib/src/gdfp_optimize.c
    https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/lib/src/update_h.c
    """
    line_interval_tolerance = 1.0e-5
    function_change_tolerance = 1.0e-6
    maximum_line_iterations = 500

    bounds = np.array(
        [[x_bounds[0], x_bounds[1]], [y_bounds[0], y_bounds[1]]],
        dtype=np.float64,
    )
    point = start.copy()
    value = initial_value
    _val, dx, dy = _eval_bspline_2d(tx, ty, c, point[0], point[1])
    derivative = np.array(
        [sign * opt_scale * dx, sign * opt_scale * dy], dtype=np.float64
    )
    if not np.isfinite(value) or not (
        np.isfinite(derivative[0]) and np.isfinite(derivative[1])
    ):
        return 2, point

    previous_value = value
    inverse_metric = np.eye(2, dtype=np.float64)
    iteration = 0
    while True:
        direction = inverse_metric @ derivative
        if max(abs(direction[0]), abs(direction[1])) < gradient_tolerance:
            return 0, point

        directional_derivative = np.dot(direction, derivative)
        if directional_derivative <= 0.0:
            inverse_metric = np.eye(2, dtype=np.float64)
            direction = derivative.copy()
            directional_derivative = np.dot(direction, derivative)

        limit_step = _box_line_limit(point, direction, bounds)
        if limit_step <= 0.0:
            return 2, point

        delta_value = abs(value - previous_value)
        if delta_value > 0.0 and directional_derivative > 0.0:
            initial_step = min(
                limit_step,
                min(0.5, 2.0 * delta_value / directional_derivative),
            )
        else:
            initial_step = min(limit_step, 0.01)

        (
            line_status,
            _step,
            trial_point,
            step_value,
            step_derivative,
        ) = _line_search_rectangular(
            point,
            direction,
            sign,
            opt_scale,
            tx,
            ty,
            c,
            directional_derivative,
            initial_step,
            limit_step,
            line_interval_tolerance,
            maximum_line_iterations,
        )
        if line_status != 0:
            return line_status, point

        iteration += 1
        previous_value = value
        value = step_value
        point_delta = trial_point - point
        derivative_delta = step_derivative - derivative
        point = trial_point.copy()
        derivative = step_derivative.copy()

        if max(abs(derivative[0]), abs(derivative[1])) < gradient_tolerance:
            return 0, point
        if iteration >= max_iterations:
            return 2, point
        if abs(value - previous_value) < function_change_tolerance:
            return 0, point

        # Goldfarb DFP update for maximization:
        gamma = np.dot(point_delta, derivative_delta)  # s^T y (< 0)
        h_gamma = inverse_metric @ derivative_delta
        delta_h_delta = np.dot(derivative_delta, h_gamma)
        if abs(gamma) < 1.0e-20 or delta_h_delta < 1.0e-20:
            inverse_metric = np.eye(2, dtype=np.float64)
            continue

        h_outer = np.outer(h_gamma, h_gamma) / delta_h_delta
        p_outer = np.outer(point_delta, point_delta) / gamma
        inverse_metric = inverse_metric - h_outer - p_outer


# ---------------------------------------------------------------------------
# Public feature point refinement entry points
# ---------------------------------------------------------------------------


def refine_spherical_bspline_feature_point(
    surface: SphericalBsplineSurface,
    latitude: float,
    longitude: float,
    *,
    is_minimum: bool,
    search_window_size: int = 3,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    gradient_tolerance: float = _DEFAULT_GRADIENT_TOLERANCE,
) -> BsplineRefinementResult:
    """Locate a stationary spline extremum in the detector's fixed locality."""
    if (
        not np.isfinite(latitude)
        or not np.isfinite(longitude)
        or search_window_size <= 0
        or search_window_size % 2 == 0
        or max_iterations <= 0
        or gradient_tolerance <= 0.0
    ):
        return BsplineRefinementResult(
            latitude, longitude, np.nan, "invalid_neighborhood"
        )

    theta = np.deg2rad(90.0 - latitude)
    phi = np.deg2rad(longitude) % (2.0 * np.pi)
    if theta < surface.theta_lower or theta > surface.theta_upper:
        return BsplineRefinementResult(
            latitude, longitude, np.nan, "invalid_neighborhood"
        )

    localization = _spherical_localization_region(
        surface,
        latitude,
        longitude,
        search_window_size,
    )
    if localization is None:
        return BsplineRefinementResult(
            latitude, longitude, np.nan, "invalid_neighborhood"
        )

    sign = 1.0 if is_minimum else -1.0
    status_code, _termination_code, _boundary_contacts, ref_th, ref_ph, ref_val = (
        _spherical_local_gdfp_search(
            theta,
            phi,
            sign,
            surface.theta_knots,
            surface.phi_knots,
            surface.coeffs,
            max_iterations,
            gradient_tolerance,
            localization.origin,
            localization.origin_e_theta,
            localization.origin_e_phi,
            localization.theta_half_width,
            localization.phi_half_width,
        )
    )
    status = _STATUS_BY_CODE.get(status_code, "optimizer_no_convergence")
    if status != "success":
        return BsplineRefinementResult(latitude, longitude, np.nan, status)

    refined_lat = float(np.rad2deg(np.pi / 2.0 - ref_th))
    refined_lon_deg = float(np.rad2deg(ref_ph))
    if surface.signed_longitudes:
        refined_lon = float((refined_lon_deg + 180.0) % 360.0 - 180.0)
    else:
        refined_lon = float(refined_lon_deg % 360.0)

    return BsplineRefinementResult(
        refined_lat,
        refined_lon,
        ref_val,
        "success",
    )


def refine_bspline_feature_point(
    surface: BsplineSurface,
    latitude: float,
    longitude: float,
    *,
    is_minimum: bool,
    initial_value: float | None = None,
    optimization_scale: float = _TRACK_SMOOPY_OPTIMIZATION_SCALE,
    max_iterations: int = 20,
    gradient_tolerance: float = _DEFAULT_GRADIENT_TOLERANCE,
) -> BsplineRefinementResult:
    """Locate a rectangular spline feature point using Goldfarb GDFP.

    SciPy/FITPACK supplies the surface; the coordinate-space optimizer and
    boundary semantics are the TRACK-compatible SMOOPY layer, integrated and
    evaluated through the PST refinement pipeline.
    """
    if (
        not np.isfinite(latitude)
        or not np.isfinite(longitude)
        or not np.isfinite(optimization_scale)
        or optimization_scale <= 0.0
        or max_iterations <= 0
        or gradient_tolerance <= 0.0
    ):
        return BsplineRefinementResult(
            latitude,
            longitude,
            np.nan,
            "invalid_neighborhood",
        )

    x = float(longitude)
    if surface.periodic_x:
        x %= 360.0
        if x < surface.x_lower:
            x += 360.0
        # The repeated endpoint is a distinct TRACK/GDFP starting boundary.
        # Preserve an explicit positive 360-degree input rather than folding
        # it to zero, while every signed/unsigned physical grid candidate is
        # already normalized before this call.
        if x == surface.x_lower and longitude > surface.x_lower:
            x = surface.x_upper
    if (
        x < surface.x_lower
        or x > surface.x_upper
        or latitude < surface.y_lower
        or latitude > surface.y_upper
    ):
        return BsplineRefinementResult(
            latitude,
            longitude,
            np.nan,
            "invalid_neighborhood",
        )

    sign = -1.0 if is_minimum else 1.0
    seam_tolerance = 1.0e-5
    x_bounds = (surface.x_lower, surface.x_upper)
    y_bounds = (surface.y_lower, surface.y_upper)

    # Check if starting on seam boundary and stepping left/right
    start_x = x
    if abs(start_x - surface.x_lower) < seam_tolerance:
        _, dx0, _ = _eval_bspline_2d(
            surface.x_knots, surface.y_knots, surface.coeffs, start_x, latitude
        )
        der_x0 = sign * optimization_scale * dx0
        if der_x0 < 0.0:
            start_x = surface.last_sample_x
    elif abs(start_x - surface.x_upper) < seam_tolerance:
        _, dx0, _ = _eval_bspline_2d(
            surface.x_knots, surface.y_knots, surface.coeffs, start_x, latitude
        )
        der_x0 = sign * optimization_scale * dx0
        if der_x0 > 0.0:
            start_x = surface.first_sample_x

    start = np.array([start_x, latitude], dtype=np.float64)
    init_val = (
        sign * optimization_scale * float(initial_value)
        if initial_value is not None and np.isfinite(initial_value)
        else (
            sign
            * optimization_scale
            * _eval_bspline_val(
                surface.x_knots,
                surface.y_knots,
                surface.coeffs,
                start_x,
                latitude,
            )
        )
    )

    status_code, result = _smoopy_gdfp_search(
        start,
        sign,
        optimization_scale,
        surface.x_knots,
        surface.y_knots,
        surface.coeffs,
        x_bounds,
        y_bounds,
        init_val,
        max_iterations,
        gradient_tolerance,
    )
    status = _STATUS_BY_CODE.get(status_code, "optimizer_no_convergence")
    if status != "success":
        fallback_val = (
            float(initial_value)
            if initial_value is not None and np.isfinite(initial_value)
            else _eval_bspline_val(
                surface.x_knots, surface.y_knots, surface.coeffs, start[0], start[1]
            )
        )
        return BsplineRefinementResult(latitude, longitude, fallback_val, status)

    tolper = 1.0e-6
    if result[0] - surface.x_lower < tolper:
        restart_x = surface.x_upper
    elif surface.x_upper - result[0] < tolper:
        restart_x = surface.x_lower
    else:
        restart_x = None

    if restart_x is not None:
        restart = np.array([restart_x, result[1]], dtype=np.float64)
        restart_init_val = (
            sign
            * optimization_scale
            * _eval_bspline_val(
                surface.x_knots,
                surface.y_knots,
                surface.coeffs,
                restart[0],
                restart[1],
            )
        )
        status_code2, result2 = _smoopy_gdfp_search(
            restart,
            sign,
            optimization_scale,
            surface.x_knots,
            surface.y_knots,
            surface.coeffs,
            x_bounds,
            y_bounds,
            restart_init_val,
            max_iterations,
            gradient_tolerance,
        )
        if status_code2 == 0:
            result = result2

    refined_lat = min(90.0, max(-90.0, float(result[1])))
    refined_longitude = float(result[0])
    if surface.periodic_x:
        refined_longitude %= 360.0

    refined_value = _eval_bspline_val(
        surface.x_knots, surface.y_knots, surface.coeffs, result[0], result[1]
    )
    return BsplineRefinementResult(
        refined_lat,
        refined_longitude,
        refined_value,
        "success",
    )


__all__ = [
    "BsplineRefinementResult",
    "BsplineRefinementStatus",
    "BsplineSurface",
    "BsplineSurfaceResult",
    "SphericalBsplineSurface",
    "SphericalBsplineSurfaceResult",
    "build_bspline_surface",
    "build_spherical_bspline_surface",
    "refine_bspline_feature_point",
    "refine_spherical_bspline_feature_point",
]
