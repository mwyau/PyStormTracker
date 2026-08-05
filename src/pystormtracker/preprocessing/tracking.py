"""Shared preprocessing semantics for the tracker implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import xarray as xr

from ..io.data_loader import DataLoader
from ..models.tracks import (
    REGRID_OPERATION,
    SPATIAL_TAPER_OPERATION,
    SPECTRAL_FILTER_OPERATION,
    ProcessingStep,
)
from .regrid import SpectralRegridder
from .spectral import DCTFilter, SHTFilter
from .taper import TaperFilter

if TYPE_CHECKING:
    from ..models.geo import MapExtent

Projection = Literal["global", "nh_stereo", "sh_stereo", "healpix"]
FilterBounds = tuple[int, int]

_HEALPIX_PIXEL_DIAMETER_DEGREES = 58.6
_DEFAULT_POLAR_EXTENT: tuple[float, float, float, float] = (
    -13000.0,
    13000.0,
    -13000.0,
    13000.0,
)


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
    estimate = max(1.0, _HEALPIX_PIXEL_DIAMETER_DEGREES / resolution)
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
) -> xr.DataArray:
    requested_lmin, requested_lmax = filter_bounds
    if data.attrs.get("grid_type") == "healpix" or "cell" in data.dims:
        raise ValueError("optional filtering of already-HEALPix data is unsupported")
    filter_class = SHTFilter if filter_type == "sht" else DCTFilter
    filtered = filter_class(lmin=requested_lmin, lmax=requested_lmax).filter(data)
    filtered.name = data.name
    return filtered


def preprocess_tracking_data(
    data: xr.DataArray,
    *,
    lmin: int | None,
    lmax: int | None,
    taper_points: int,
    projection: Projection,
    nside: int | None = None,
    resolution: float | None = 100.0,
    extent: MapExtent | None = None,
    filter_type: Literal["sht", "dct", "auto"] = "auto",
) -> tuple[xr.DataArray, tuple[ProcessingStep, ...]]:
    """Apply one consistent preprocessing policy for all trackers."""
    filter_bounds = resolve_filter_bounds(lmin, lmax)
    _validate_taper_points(taper_points)
    if resolution is not None and resolution <= 0.0:
        raise ValueError("resolution must be positive")

    loader = DataLoader(data)
    if filter_type == "auto":
        filter_type = "sht" if loader.is_global_longitude() else "dct"
    if data.chunks:
        data = data.compute()

    steps: list[ProcessingStep] = []
    if taper_points > 0:
        data = cast(xr.DataArray, TaperFilter(n_points=taper_points).filter(data))
        steps.append(
            ProcessingStep(SPATIAL_TAPER_OPERATION, True, {"points": taper_points})
        )

    if filter_bounds is not None:
        data = _apply_optional_filter(
            data,
            filter_bounds=filter_bounds,
            filter_type=filter_type,
        )
        requested_lmin, requested_lmax = filter_bounds
        parameters: dict[str, str | int | float | bool | None] = {
            "method": filter_type,
            "lmin": requested_lmin,
            "lmax": requested_lmax,
        }
        steps.append(ProcessingStep(SPECTRAL_FILTER_OPERATION, True, parameters))

    if projection == "global":
        return data, tuple(steps)
    if data.ndim != 3:
        if projection == "healpix" and (
            data.attrs.get("grid_type") == "healpix" or "cell" in data.dims
        ):
            return data, tuple(steps)
        raise ValueError(
            "projection preprocessing requires time, latitude, and longitude"
        )

    time_dim, _lat_dim, _lon_dim = loader.get_coords()
    lat_reverse = loader.is_lat_reversed()
    frames: list[xr.DataArray] = []
    if projection == "healpix":
        target_nside = resolve_healpix_nside(data, nside=nside)
        transform_lmax = resolve_healpix_transform_lmax(
            data,
            nside=target_nside,
            filter_bounds=filter_bounds,
        )
        regridder = SpectralRegridder(lmax=transform_lmax)
        for index in range(data.sizes[time_dim]):
            frames.append(
                regridder.to_healpix(
                    data.isel({time_dim: index}).squeeze(),
                    nside=target_nside,
                    transform_lmax=transform_lmax,
                    lat_reverse=lat_reverse,
                )
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
        polar_extent = extent if extent is not None else _DEFAULT_POLAR_EXTENT
        if resolution is None:
            raise ValueError("polar stereographic projection requires resolution")
        for index in range(data.sizes[time_dim]):
            frames.append(
                regridder.to_polar_stereo(
                    data.isel({time_dim: index}).squeeze(),
                    hemisphere=hemisphere,
                    transform_lmax=transform_lmax,
                    lat_reverse=lat_reverse,
                    resolution=resolution,
                    extent=polar_extent,
                )
            )
        parameters = {
            "projection": projection,
            "resolution": resolution,
            "transform_lmax": transform_lmax,
        }
        if extent is not None:
            parameters["extent"] = ",".join(str(value) for value in extent)

    result = xr.concat(frames, dim=data[time_dim])
    result.name = data.name
    result.attrs = dict(data.attrs)
    result.attrs["map_proj"] = projection
    if projection == "healpix":
        result.attrs["nside"] = parameters["nside"]
    steps.append(ProcessingStep(REGRID_OPERATION, True, parameters))
    return result, tuple(steps)
