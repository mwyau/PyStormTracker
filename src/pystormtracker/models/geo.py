from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ..models.constants import DEGTORAD, R_EARTH_KM

if TYPE_CHECKING:
    import xarray as xr

# Type alias for a map bounding box (xmin, xmax, ymin, ymax) in km
MapExtent: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SpatialBounds:
    """An optional declared geographic domain for a track collection.

    Longitude edges are interval boundaries and therefore allow ``180``.
    A west edge greater than the east edge denotes an antimeridian-crossing
    interval.  A west edge equal to an east edge is only valid for the full
    global interval ``[-180, 180]``.
    """

    south: float
    north: float
    west: float
    east: float

    def __post_init__(self) -> None:
        values = (self.south, self.north, self.west, self.east)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("spatial bounds must be finite")
        if not -90.0 <= self.south <= self.north <= 90.0:
            raise ValueError("bounds must satisfy -90 <= south <= north <= 90")
        if not -180.0 <= self.west <= 180.0:
            raise ValueError("bounds west must be in [-180, 180]")
        if not -180.0 <= self.east <= 180.0:
            raise ValueError("bounds east must be in [-180, 180]")
        if self.west == self.east and (self.west, self.east) != (-180.0, 180.0):
            raise ValueError(
                "bounds west == east is ambiguous; use west=-180 and east=180 "
                "for the global domain"
            )


def spatial_bounds_from_xarray(
    data: xr.DataArray | xr.Dataset,
) -> SpatialBounds | None:
    """Return reliable latitude/longitude bounds from geographic coordinates."""
    coordinate_map = data.coords
    latitude = next(
        (
            coordinate_map[name]
            for name in ("lat", "latitude")
            if name in coordinate_map
        ),
        None,
    )
    longitude = next(
        (
            coordinate_map[name]
            for name in ("lon", "longitude")
            if name in coordinate_map
        ),
        None,
    )
    if latitude is None or longitude is None:
        return None
    lat_values = np.asarray(latitude.values, dtype=np.float64).ravel()
    lon_values = np.asarray(longitude.values, dtype=np.float64).ravel()
    if (
        lat_values.size == 0
        or lon_values.size == 0
        or np.any(~np.isfinite(lat_values))
        or np.any(~np.isfinite(lon_values))
    ):
        return None
    latitude_edges = _coordinate_edges(data, latitude, lat_values)
    south = float(np.min(latitude_edges))
    north = float(np.max(latitude_edges))
    if not -90.0 <= south <= north <= 90.0:
        return None
    longitude_edges = _coordinate_edges(data, longitude, lon_values)
    if _looks_global(lon_values):
        west, east = -180.0, 180.0
    else:
        west, east, _ = minimal_longitude_interval(longitude_edges)
    try:
        return SpatialBounds(south, north, west, east)
    except ValueError:
        return None


def _coordinate_edges(
    data: xr.DataArray | xr.Dataset,
    coordinate: xr.DataArray,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Use CF coordinate bounds when present, otherwise coordinate values."""
    bounds_name = coordinate.attrs.get("bounds")
    if isinstance(bounds_name, str) and bounds_name in data:
        bound_values = np.asarray(data[bounds_name].values, dtype=np.float64).ravel()
        if bound_values.size and np.isfinite(bound_values).all():
            return bound_values
    return values


def _looks_global(values: NDArray[np.float64]) -> bool:
    normalized = np.sort(np.unique(np.mod(values, 360.0)))
    if normalized.size < 2:
        return False
    gaps = np.diff(np.concatenate((normalized, normalized[:1] + 360.0)))
    median_gap = float(np.median(gaps))
    return median_gap > 0.0 and float(np.max(gaps)) <= 1.5 * median_gap


def minimal_longitude_interval(
    values: NDArray[np.float64],
) -> tuple[float, float, bool]:
    """Return the smallest circular interval containing normalized longitudes."""
    raw = np.asarray(values, dtype=np.float64).ravel()
    if raw.size == 0 or np.any(~np.isfinite(raw)):
        raise ValueError("longitude values must be finite and nonempty")
    normalized = np.sort(np.unique(np.mod(raw, 360.0)))
    if normalized.size == 1:
        point = float(normalize_longitudes_signed(normalized)[0])
        return point, point, False
    gaps = np.diff(np.concatenate((normalized, normalized[:1] + 360.0)))
    largest_gap_index = int(np.argmax(gaps))
    start = float(normalized[(largest_gap_index + 1) % normalized.size])
    end = float(normalized[largest_gap_index])
    west = float(normalize_longitudes_signed(np.asarray([start]))[0])
    east = float(normalize_longitudes_signed(np.asarray([end]))[0])
    return west, east, west > east


def normalize_longitudes_signed(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize east-positive longitudes to the internal ``[-180, 180)`` range."""
    return np.remainder(values + 180.0, 360.0) - 180.0


def normalize_longitudes_360(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize east-positive longitudes to the ``[0, 360)`` range."""
    return np.remainder(values, 360.0)


def cyclic_longitude_delta(
    longitude: NDArray[np.float64] | float,
    center: float,
) -> NDArray[np.float64]:
    """Return signed shortest-arc differences from ``center`` in degrees."""
    values = np.asarray(longitude, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(center):
        raise ValueError("longitude values and center must be finite")
    return (values - center + 180.0) % 360.0 - 180.0


@nb.njit(cache=True, nogil=True)
def geod_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great circle distance (angular separation) in radians."""
    phi1 = lat1 * DEGTORAD
    phi2 = lat2 * DEGTORAD
    lam1 = lon1 * DEGTORAD
    lam2 = lon2 * DEGTORAD

    # Dot product of unit vectors
    dot = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(
        lam1 - lam2
    )

    # Clamp for precision
    if dot > 1.0:
        dot = 1.0
    if dot < -1.0:
        dot = -1.0

    return float(np.arccos(dot))


@nb.njit(cache=True, nogil=True)
def geod_dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great circle distance in kilometers."""
    return float(geod_dist(lat1, lon1, lat2, lon2) * R_EARTH_KM)


@nb.njit(cache=True, nogil=True)
def stereo_to_latlon(
    x: float, y: float, hemisphere: int, lon_0: float = 0.0
) -> tuple[float, float]:
    """
    Converts (x, y) coordinates on a polar stereographic projection (in km)
    back to (lat, lon) in degrees.

    Args:
        x: X coordinate in km.
        y: Y coordinate in km.
        hemisphere: 1 for Northern Hemisphere, -1 for Southern Hemisphere.
        lon_0: Central longitude in degrees.

    Returns:
        (lat, lon) in degrees.
    """
    rho = np.sqrt(x**2 + y**2)
    if rho == 0.0:
        return 90.0 * hemisphere, lon_0

    if hemisphere == 1:
        theta = 2.0 * np.arctan(rho / (2.0 * R_EARTH_KM))
        phi = (np.radians(lon_0) + np.arctan2(x, -y)) % (2 * np.pi)
    else:
        theta = np.pi - 2.0 * np.arctan(rho / (2.0 * R_EARTH_KM))
        phi = (np.radians(lon_0) + np.arctan2(x, y)) % (2 * np.pi)

    lat = 90.0 - np.degrees(theta)
    lon = np.degrees(phi) % 360.0
    return lat, lon
