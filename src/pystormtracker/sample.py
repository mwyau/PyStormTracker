from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Literal, cast

import numpy as np
import xarray as xr

from .models.geo import cyclic_longitude_delta, geod_dist_km
from .models.tracks import Tracks
from .models.units import normalize_variable_units
from .utils.cli import add_cli_observability_options, nonnegative_float

LOGGER = logging.getLogger(__name__)

type SamplingMethod = Literal["nearest", "bilinear", "mean", "max", "min"]


def _prepare_longitude_axis(
    da: xr.DataArray,
    *,
    lon_dim: str,
) -> tuple[xr.DataArray, np.ndarray, bool]:
    """Validate and prepare a reusable one-dimensional longitude axis."""
    coordinate = da[lon_dim]
    lons = np.asarray(coordinate.values, dtype=np.float64)
    if lons.ndim != 1 or lons.size == 0:
        raise ValueError("longitude coordinate must be one-dimensional and nonempty")
    if not np.isfinite(lons).all():
        raise ValueError("longitude coordinate must contain finite values")

    normalized = np.mod(lons, 360.0)
    unique_count = np.unique(normalized).size
    if unique_count != lons.size:
        endpoint_duplicate = np.isclose(
            float(cyclic_longitude_delta(np.asarray([lons[-1]]), lons[0])[0]),
            0.0,
        )
        if not endpoint_duplicate:
            raise ValueError("longitude coordinate contains duplicate cyclic values")
        first_values = np.asarray(da.isel({lon_dim: 0}).values)
        last_values = np.asarray(da.isel({lon_dim: -1}).values)
        if not np.allclose(first_values, last_values, equal_nan=True):
            raise ValueError(
                "duplicate cyclic longitude endpoints have conflicting data"
            )
        da = da.isel({lon_dim: slice(None, -1)})
        lons = lons[:-1]

    is_increasing = lons.size < 2 or bool(lons[1] > lons[0])
    return da, lons, is_increasing


def _nearest_sample(
    data: xr.DataArray,
    *,
    lat_dim: str,
    lon_dim: str,
    lats: np.ndarray,
    lons: np.ndarray,
    lat: float,
    lon: float,
) -> float:
    # Nearest grid point search using shortest cyclic longitude arc
    lat_index = int(np.argmin(np.abs(lats - lat)))
    lon_delta = cyclic_longitude_delta(lons, lon)
    lon_index = int(np.argmin(np.abs(lon_delta)))
    return float(data.isel({lat_dim: lat_index, lon_dim: lon_index}).values)


def _bilinear_sample(
    data: xr.DataArray,
    *,
    lat_dim: str,
    lon_dim: str,
    lons: np.ndarray,
    lat: float,
    lon: float,
) -> float:
    # Transform longitudes to relative cyclic offsets for antimeridian interpolation
    relative_lon = cyclic_longitude_delta(lons, lon)
    temporary = data.assign_coords({lon_dim: relative_lon}).sortby(lon_dim)
    value = temporary.interp({lon_dim: 0.0, lat_dim: lat}, method="linear").values
    scalar = np.asarray(value)
    if scalar.size != 1:
        raise ValueError("bilinear interpolation did not produce one value")
    return float(scalar.reshape(-1)[0])


def sample_tracks(
    tracks: Tracks,
    ds: xr.Dataset,
    variable_name: str,
    method: SamplingMethod = "nearest",
    radius_km: float = 0.0,
    output_variable_name: str | None = None,
) -> Tracks:
    """
    Samples a variable from a NetCDF dataset along storm tracks.

    Args:
        tracks: The Tracks object to update.
        ds: The xarray Dataset containing the variable to sample.
        variable_name: The name of the variable in the dataset.
        method: The sampling method ('nearest', 'bilinear', 'mean', 'max', 'min').
        radius_km: The radius in km for spatial operations (mean, max, min).
        output_variable_name: The name to store in the track's 'vars' dictionary.
                        Defaults to variable_name.

    Returns:
        The updated Tracks object.
    """
    if method not in ("nearest", "bilinear", "mean", "max", "min"):
        raise ValueError(f"Unsupported sampling method: {method}")
    if radius_km < 0.0:
        raise ValueError("Sampling radius must be nonnegative.")
    from .io.data_loader import DataLoader

    loader = DataLoader(ds)
    source_ds = loader.ensure_open()
    if variable_name not in source_ds:
        raise ValueError(f"Variable '{variable_name}' not found in dataset.")

    da = source_ds[variable_name]
    out_name = output_variable_name or variable_name
    da, _unused_threshold, sampled_unit = normalize_variable_units(
        da,
        variable=out_name,
        intensity_threshold=None,
    )

    sampled_values = np.full(len(tracks.times), np.nan, dtype=np.float64)

    # Identify and validate the spatial axes once for this sampling call.
    lat_dim = loader.find_coordinate_dimension(da, "latitude")
    lon_dim = loader.find_coordinate_dimension(da, "longitude")
    time_dim = loader.find_coordinate_dimension(da, "time")

    if not lat_dim or not lon_dim:
        raise ValueError("Could not identify latitude or longitude dimensions.")

    lats = np.asarray(da[lat_dim].values, dtype=np.float64)
    if lats.ndim != 1 or lats.size == 0 or not np.isfinite(lats).all():
        raise ValueError("latitude coordinate must be one-dimensional and finite")
    da, lons, _is_lon_increasing = _prepare_longitude_axis(da, lon_dim=lon_dim)

    for track in tracks:
        point_slice = track.point_slice
        for i, center in enumerate(track):
            global_idx = (point_slice.start or 0) + i

            # 1. Select the correct time slice
            if time_dim and center.time is not None:
                try:
                    timestamp = int(cast(int | np.integer, center.time))
                    source_time = np.asarray([timestamp], dtype="datetime64[ms]")[0]
                    da_step = da.sel({time_dim: source_time}, method="nearest")
                except KeyError:
                    # Time might be out of range
                    sampled_values[global_idx] = np.nan
                    continue
            else:
                da_step = da

            # 2. Perform sampling
            if method in ("nearest", "bilinear"):
                if method == "nearest":
                    sampled_values[global_idx] = _nearest_sample(
                        da_step,
                        lat_dim=lat_dim,
                        lon_dim=lon_dim,
                        lats=lats,
                        lons=lons,
                        lat=center.lat,
                        lon=center.lon,
                    )
                else:
                    sampled_values[global_idx] = _bilinear_sample(
                        da_step,
                        lat_dim=lat_dim,
                        lon_dim=lon_dim,
                        lons=lons,
                        lat=center.lat,
                        lon=center.lon,
                    )

            elif method in ("mean", "max", "min"):
                if radius_km <= 0:
                    # Fallback to nearest if radius is 0
                    sampled_values[global_idx] = _nearest_sample(
                        da_step,
                        lat_dim=lat_dim,
                        lon_dim=lon_dim,
                        lats=lats,
                        lons=lons,
                        lat=center.lat,
                        lon=center.lon,
                    )
                    continue

                # Conservative bounding box in degrees
                lat_buffer = (radius_km / 111.0) * 1.5
                cos_lat = abs(float(np.cos(np.radians(center.lat))))
                lon_buffer = (
                    180.0
                    if cos_lat < 1e-12
                    else min((radius_km / (111.0 * cos_lat)) * 1.5, 180.0)
                )

                lat_delta = np.abs(lats - center.lat)
                lat_indices = np.flatnonzero(lat_delta <= lat_buffer)
                lon_delta = np.abs(cyclic_longitude_delta(lons, center.lon))
                lon_indices = np.flatnonzero(lon_delta <= lon_buffer)
                if lat_indices.size == 0 or lon_indices.size == 0:
                    sampled_values[global_idx] = np.nan
                    continue
                subset = da_step.isel({lat_dim: lat_indices, lon_dim: lon_indices})

                if subset.size == 0:
                    sampled_values[global_idx] = np.nan
                    continue

                # Calculate distances to all points in subset
                sub_lats, sub_lons = xr.broadcast(subset[lat_dim], subset[lon_dim])

                dist_func = np.vectorize(geod_dist_km)
                dists = dist_func(
                    center.lat, center.lon, sub_lats.values, sub_lons.values
                )

                mask = dists <= radius_km
                valid_data = subset.values[mask]

                if valid_data.size == 0:
                    sampled_values[global_idx] = np.nan
                elif method == "mean":
                    sampled_values[global_idx] = float(np.nanmean(valid_data))
                elif method == "max":
                    sampled_values[global_idx] = float(np.nanmax(valid_data))
                elif method == "min":
                    sampled_values[global_idx] = float(np.nanmin(valid_data))

    variables = dict(tracks.variables)
    variables[out_name] = sampled_values
    units = dict(tracks.units)
    units[out_name] = sampled_unit
    metadata = replace(tracks.metadata, units=units)
    return tracks.with_variables(variables, metadata=metadata)


def setup_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Sets up the argument parser for the sample command."""
    parser = subparsers.add_parser(
        "sample",
        description="Sample variables from a NetCDF dataset along storm tracks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_cli_observability_options(parser)
    parser.add_argument("-i", "--input", required=True, help="Input track file (JSON).")
    parser.add_argument(
        "-d", "--data", required=True, help="Input NetCDF data file to sample from."
    )
    parser.add_argument(
        "--variable", required=True, help="Variable name in the NetCDF file."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output track file (JSON).",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["nearest", "bilinear", "mean", "max", "min"],
        default="nearest",
        help="Sampling method.",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=nonnegative_float,
        default=0.0,
        help="Radius in km for spatial methods (mean, max, min).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name to store in the tracks. Defaults to the variable name.",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=["h5netcdf", "netcdf4", "cfgrib"],
        default=None,
        help="Xarray engine for reading data.",
    )
    parser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the sample command.

    Samples a variable from a NetCDF dataset at track coordinates using
    interpolation or spatial aggregation (mean/max/min within a radius).
    """
    if args.method in ("mean", "max", "min") and args.radius <= 0.0:
        raise ValueError(f"sampling method '{args.method}' requires a positive radius")

    LOGGER.info("Reading tracks from %s", args.input)
    from .io.format import load_tracks, save_tracks

    tracks = load_tracks(args.input)

    LOGGER.info("Opening dataset %s", args.data)
    from .io.data_loader import DataLoader

    ds = DataLoader(args.data, engine=args.engine).ensure_open()

    LOGGER.info("Sampling %r using method %r", args.variable, args.method)
    if args.radius > 0:
        LOGGER.debug("Sampling radius: %g km", args.radius)

    tracks = sample_tracks(
        tracks=tracks,
        ds=ds,
        variable_name=args.variable,
        method=args.method,
        radius_km=args.radius,
        output_variable_name=args.name,
    )

    LOGGER.info("Writing updated tracks to %s", args.output)
    save_tracks(tracks, args.output)
    LOGGER.info("Sampling completed")
