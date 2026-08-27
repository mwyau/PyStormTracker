"""Shared preprocessing semantics for the tracker implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np
import xarray as xr

from ..backends import Backend
from ..io.data_loader import DataLoader
from ..models.tracks import ProcessingStep
from .regrid import SpectralRegridder
from .spectral import DCTFilter, SHTFilter
from .taper import BoundaryTaper

if TYPE_CHECKING:
    from ..models.geo import MapExtent

Projection = Literal["global", "nh_stereo", "sh_stereo", "healpix"]
FilterBounds = tuple[int, int]

_SPECTRAL_FILTER_OPERATION: Final[str] = "spectral_filter"
_SPATIAL_TAPER_OPERATION: Final[str] = "spatial_taper"
_REGRID_OPERATION: Final[str] = "regrid"

_HEALPIX_RESOLUTION_FACTOR: Final[float] = 58.6
_DEFAULT_POLAR_EXTENT_KM: Final[tuple[float, float, float, float]] = (
    -13000.0,
    13000.0,
    -13000.0,
    13000.0,
)

LOGGER = logging.getLogger(__name__)


def resolve_filter_bounds(
    lmin: int | None,
    lmax: int | None,
) -> FilterBounds | None:
    """Validate and normalize the optional user-requested filter band."""
    if (lmin is None) != (lmax is None):
        raise ValueError("lmin and lmax must be supplied together")
    if lmin is None or lmax is None:
        return None
    if isinstance(lmin, bool) or isinstance(lmax, bool):
        raise TypeError("lmin and lmax must be integers")
    if lmin < 0 or lmax < 0:
        raise ValueError("lmin and lmax must be nonnegative")
    if lmin > lmax:
        raise ValueError("lmin must be less than or equal to lmax")
    return int(lmin), int(lmax)


def _validate_taper_points(taper_points: int) -> None:
    if isinstance(taper_points, bool) or taper_points < 0:
        raise ValueError("taper_points must be nonnegative")


def _source_supported_lmax(data: xr.DataArray) -> int:
    """Return the finite harmonic bandwidth supported by the source grid."""
    loader = DataLoader(data)
    variable_name = str(data.name) if data.name is not None else None
    if loader.is_reduced_gaussian(variable_name):
        metadata = loader.get_grid_metadata(variable_name)
        nlat = len(metadata["theta"])
        nlon = int(np.max(metadata["nphi"]))
    else:
        _time_dim, lat_dim, lon_dim = loader.get_coords()
        if lat_dim not in data.sizes or lon_dim not in data.sizes:
            raise ValueError("source data must have latitude and longitude dimensions")
        nlat = int(data.sizes[lat_dim])
        nlon = int(data.sizes[lon_dim])
    return max(0, min(nlat - 1, nlon // 2 - 1))


def resolve_healpix_nside(
    data: xr.DataArray,
    *,
    nside: int | None,
) -> int:
    """Validate or derive a power-of-two HEALPix target resolution."""
    if nside is not None:
        if isinstance(nside, bool) or nside <= 0 or nside & (nside - 1):
            raise ValueError("nside must be a positive power of two")
        return int(nside)

    loader = DataLoader(data)
    _time_dim, lat_dim, lon_dim = loader.get_coords()
    if lat_dim not in data.coords or lon_dim not in data.coords:
        raise ValueError("cannot derive HEALPix nside without latitude and longitude")
    lat = np.asarray(data[lat_dim].values, dtype=np.float64)
    lon = np.mod(np.asarray(data[lon_dim].values, dtype=np.float64), 360.0)
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
        raise ValueError("cannot derive HEALPix nside from a degenerate grid")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("cannot derive HEALPix nside from non-finite coordinates")
    lon_unique = np.unique(lon)
    lon_gaps = np.diff(np.concatenate((lon_unique, lon_unique[:1] + 360.0)))
    resolution = max(
        float(np.median(np.abs(np.diff(lat)))),
        float(np.median(lon_gaps[lon_gaps > 0.0])),
    )
    estimate = max(1.0, _HEALPIX_RESOLUTION_FACTOR / resolution)
    exponent = int(np.ceil(np.log2(estimate)))
    return int(2 ** max(0, exponent))


def resolve_healpix_transform_lmax(
    data: xr.DataArray,
    *,
    nside: int,
    filter_bounds: FilterBounds | None,
) -> int:
    """Resolve the transform bandwidth from source, target, and filter limits."""
    source_lmax = _source_supported_lmax(data)
    target_lmax = 3 * nside - 1
    if filter_bounds is not None:
        requested_lmin, requested_lmax = filter_bounds
        if requested_lmax > source_lmax:
            raise ValueError(
                f"requested lmax={requested_lmax} exceeds source bandwidth "
                f"lmax={source_lmax}"
            )
        if requested_lmax > target_lmax:
            raise ValueError(
                f"requested lmax={requested_lmax} exceeds HEALPix nside={nside} "
                f"bandwidth lmax={target_lmax}"
            )
        transform_lmax = min(source_lmax, target_lmax, requested_lmax)
        if requested_lmin > transform_lmax:
            raise ValueError("requested filter band is empty at the target resolution")
        return transform_lmax
    return min(source_lmax, target_lmax)


def _resolve_projection_lmax(
    data: xr.DataArray,
    filter_bounds: FilterBounds | None,
) -> int:
    source_lmax = _source_supported_lmax(data)
    declared_lmax = data.attrs.get("spectral_lmax")
    if declared_lmax is not None:
        if (
            isinstance(declared_lmax, bool)
            or not isinstance(declared_lmax, (int, np.integer))
            or int(declared_lmax) < 0
        ):
            raise ValueError(
                "data spectral_lmax metadata must be a nonnegative integer"
            )
        source_lmax = min(source_lmax, int(declared_lmax))
    if filter_bounds is None:
        return source_lmax
    requested_lmin, requested_lmax = filter_bounds
    if requested_lmax > source_lmax:
        raise ValueError(
            f"requested lmax={requested_lmax} exceeds source bandwidth "
            f"lmax={source_lmax}"
        )
    if requested_lmin > requested_lmax:
        raise ValueError("requested filter band is empty")
    return requested_lmax


def _apply_optional_filter(
    data: xr.DataArray,
    *,
    filter_bounds: FilterBounds,
    filter_type: Literal["sht", "dct"],
    spectral_taper: float,
    backend: Backend = "serial",
    sht_threads: int | None = None,
) -> xr.DataArray:
    requested_lmin, requested_lmax = filter_bounds
    if data.attrs.get("grid_type") == "healpix" or "cell" in data.dims:
        raise ValueError("optional filtering of already-HEALPix data is unsupported")
    if filter_type == "sht":
        if sht_threads is None:
            filtered = SHTFilter(
                lmin=requested_lmin,
                lmax=requested_lmax,
                taper_val=spectral_taper,
            ).filter(data, backend=backend)
        else:
            filtered = SHTFilter(
                lmin=requested_lmin,
                lmax=requested_lmax,
                taper_val=spectral_taper,
                sht_threads=sht_threads,
            ).filter(data, backend=backend)
    else:
        filtered = DCTFilter(
            lmin=requested_lmin,
            lmax=requested_lmax,
            taper_val=spectral_taper,
        ).filter(data, backend=backend)
    filtered.name = data.name
    return filtered


def preprocess_tracking_data(
    data: xr.DataArray,
    *,
    lmin: int | None = None,
    lmax: int | None = None,
    taper_points: int = 0,
    spectral_taper: float = 0.1,
    projection: Projection = "global",
    nside: int | None = None,
    stereo_grid_spacing_km: float | None = 100.0,
    extent: MapExtent | None = None,
    filter_type: Literal["sht", "dct", "auto"] = "auto",
    backend: Backend = "serial",
    sht_threads: int | None = None,
) -> tuple[xr.DataArray, tuple[ProcessingStep, ...]]:
    """Apply one preprocessing configuration to all trackers."""
    filter_bounds = resolve_filter_bounds(lmin, lmax)
    _validate_taper_points(taper_points)
    if not 0.0 < spectral_taper <= 1.0:
        raise ValueError("spectral_taper must be in the interval (0, 1]")
    if stereo_grid_spacing_km is not None and stereo_grid_spacing_km <= 0.0:
        raise ValueError(
            "stereo_grid_spacing_km must be positive stereographic grid spacing "
            "in kilometres"
        )

    loader = DataLoader(data)
    if filter_type == "auto":
        filter_type = "sht" if loader.is_global_longitude() else "dct"

    LOGGER.debug(
        "Preprocessing geometry projection=%s filter_backend=%s bounds=%r "
        "spectral_taper=%g backend=%s dims=%s",
        projection,
        filter_type,
        filter_bounds,
        spectral_taper,
        backend,
        dict(data.sizes),
    )

    steps: list[ProcessingStep] = []
    if taper_points > 0:
        LOGGER.info("Spatial taper enabled with %d points", taper_points)
        data = cast(xr.DataArray, BoundaryTaper(n_points=taper_points).filter(data))
        steps.append(
            ProcessingStep(_SPATIAL_TAPER_OPERATION, True, {"points": taper_points})
        )

    if filter_bounds is not None:
        LOGGER.info(
            "Spectral filter enabled: %s lmin=%d lmax=%d taper=%g",
            filter_type,
            filter_bounds[0],
            filter_bounds[1],
            spectral_taper,
        )
        data = _apply_optional_filter(
            data,
            filter_bounds=filter_bounds,
            filter_type=filter_type,
            spectral_taper=spectral_taper,
            backend=backend,
            sht_threads=sht_threads,
        )
        requested_lmin, requested_lmax = filter_bounds
        parameters: dict[str, str | int | float | bool | None] = {
            "method": filter_type,
            "lmin": requested_lmin,
            "lmax": requested_lmax,
            "spectral_taper": spectral_taper,
        }
        steps.append(ProcessingStep(_SPECTRAL_FILTER_OPERATION, True, parameters))

    if projection == "global":
        LOGGER.debug(
            "Preprocessing output geometry remains global dims=%s", dict(data.sizes)
        )
        return data, tuple(steps)
    if projection == "healpix" and (
        data.attrs.get("grid_type") == "healpix" or "cell" in data.dims
    ):
        return data, tuple(steps)
    if data.ndim not in (2, 3):
        raise ValueError(
            "projection preprocessing requires time, latitude, and longitude"
        )

    lat_reverse = loader.is_lat_reversed()
    if projection == "healpix":
        target_nside = resolve_healpix_nside(data, nside=nside)
        transform_lmax = resolve_healpix_transform_lmax(
            data,
            nside=target_nside,
            filter_bounds=filter_bounds,
        )
        regridder = SpectralRegridder(lmax=transform_lmax)
        result = regridder.to_healpix(
            data,
            nside=target_nside,
            transform_lmax=transform_lmax,
            lat_reverse=lat_reverse,
            sht_threads=sht_threads,
            backend=backend,
        )
        parameters = {
            "projection": "healpix",
            "nside": target_nside,
            "transform_lmax": transform_lmax,
        }
    else:
        transform_lmax = _resolve_projection_lmax(data, filter_bounds)
        regridder = SpectralRegridder(lmax=transform_lmax)
        hemisphere: Literal["nh", "sh"] = "nh" if projection == "nh_stereo" else "sh"
        polar_extent = extent if extent is not None else _DEFAULT_POLAR_EXTENT_KM
        if stereo_grid_spacing_km is None:
            raise ValueError(
                "stereo_grid_spacing_km must be positive stereographic grid spacing "
                "in kilometres"
            )
        result = regridder.to_polar_stereo(
            data,
            hemisphere=hemisphere,
            transform_lmax=transform_lmax,
            lat_reverse=lat_reverse,
            stereo_grid_spacing_km=stereo_grid_spacing_km,
            extent=polar_extent,
            backend=backend,
            sht_threads=sht_threads,
        )
        parameters = {
            "projection": projection,
            "stereo_grid_spacing_km": stereo_grid_spacing_km,
            "transform_lmax": transform_lmax,
        }
        if extent is not None:
            parameters["extent"] = ",".join(str(value) for value in extent)

    result.attrs["projection"] = projection
    if projection == "healpix":
        result.attrs["grid_type"] = "healpix"
        result.attrs["nside"] = target_nside

    steps.append(ProcessingStep(_REGRID_OPERATION, True, parameters))
    LOGGER.info("Projection/regrid selected: %s", projection)
    LOGGER.debug(
        "Preprocessing output geometry dims=%s parameters=%s",
        dict(result.sizes),
        parameters,
    )
    return result, tuple(steps)
