from __future__ import annotations

import argparse
from typing import Literal

import numpy as np
import xarray as xr

from .io.json import read_json, write_json
from .models.geo import geod_dist_km
from .models.tracks import Tracks

SamplingMethod = Literal["nearest", "bilinear", "mean", "max", "min"]


def sample_tracks(
    tracks: Tracks,
    ds: xr.Dataset,
    varname: str,
    method: SamplingMethod = "nearest",
    radius_km: float = 0.0,
    output_varname: str | None = None,
) -> Tracks:
    """
    Samples a variable from a NetCDF dataset along storm tracks.

    Args:
        tracks: The Tracks object to update.
        ds: The xarray Dataset containing the variable to sample.
        varname: The name of the variable in the dataset.
        method: The sampling method ('nearest', 'bilinear', 'mean', 'max', 'min').
        radius_km: The radius in km for spatial operations (mean, max, min).
        output_varname: The name to store in the track's 'vars' dictionary.
                        Defaults to varname.

    Returns:
        The updated Tracks object.
    """
    if varname not in ds:
        raise ValueError(f"Variable '{varname}' not found in dataset.")

    da = ds[varname]
    out_name = output_varname or varname

    # Initialize the new variable in tracks.vars if not present
    if out_name not in tracks.vars:
        tracks.vars[out_name] = np.full(len(tracks.track_ids), np.nan)

    # Identify lat/lon dimensions
    from .io.data_loader import DataLoader

    lat_dim = next(
        (d for d in DataLoader.VAR_MAPPING["latitude"] if d in da.dims), None
    )
    lon_dim = next(
        (d for d in DataLoader.VAR_MAPPING["longitude"] if d in da.dims), None
    )
    time_dim = next((d for d in DataLoader.VAR_MAPPING["time"] if d in da.dims), None)

    if not lat_dim or not lon_dim:
        raise ValueError("Could not identify latitude or longitude dimensions.")

    lats = da[lat_dim].values
    lons = da[lon_dim].values
    is_lat_increasing = lats[1] > lats[0] if len(lats) > 1 else True
    is_lon_increasing = lons[1] > lons[0] if len(lons) > 1 else True

    for track in tracks:
        global_indices = track.indices
        for i, center in enumerate(track):
            global_idx = global_indices[i]

            # 1. Select the correct time slice
            if time_dim and center.time is not None:
                try:
                    da_step = da.sel({time_dim: center.time}, method="nearest")
                except (ValueError, KeyError):
                    # Time might be out of range
                    tracks.vars[out_name][global_idx] = np.nan
                    continue
            else:
                da_step = da

            # 2. Perform sampling
            if method in ("nearest", "bilinear"):
                interp_method: Literal["nearest", "linear"] = (
                    "nearest" if method == "nearest" else "linear"
                )
                try:
                    val = da_step.interp(
                        {lat_dim: center.lat, lon_dim: center.lon},
                        method=interp_method,
                    ).values
                    tracks.vars[out_name][global_idx] = float(val)
                except Exception:
                    tracks.vars[out_name][global_idx] = np.nan

            elif method in ("mean", "max", "min"):
                if radius_km <= 0:
                    # Fallback to nearest if radius is 0
                    val = da_step.sel(
                        {lat_dim: center.lat, lon_dim: center.lon}, method="nearest"
                    ).values
                    tracks.vars[out_name][global_idx] = float(val)
                    continue

                # Conservative bounding box in degrees
                lat_buffer = (radius_km / 111.0) * 1.5
                lon_buffer = (
                    radius_km / (111.0 * np.cos(np.radians(center.lat)))
                ) * 1.5
                lon_buffer = min(lon_buffer, 180.0)

                lat_min, lat_max = center.lat - lat_buffer, center.lat + lat_buffer
                lon_min, lon_max = center.lon - lon_buffer, center.lon + lon_buffer

                lat_slice = (
                    slice(lat_min, lat_max)
                    if is_lat_increasing
                    else slice(lat_max, lat_min)
                )
                lon_slice = (
                    slice(lon_min, lon_max)
                    if is_lon_increasing
                    else slice(lon_max, lon_min)
                )

                # Handle longitude wrapping
                if lon_min < 0 or lon_max > 360:
                    # For now, just slice latitude and filter longitude manually
                    subset = da_step.sel({lat_dim: lat_slice})
                else:
                    subset = da_step.sel({lat_dim: lat_slice, lon_dim: lon_slice})

                if subset.size == 0:
                    tracks.vars[out_name][global_idx] = np.nan
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
                    tracks.vars[out_name][global_idx] = np.nan
                elif method == "mean":
                    tracks.vars[out_name][global_idx] = float(np.nanmean(valid_data))
                elif method == "max":
                    tracks.vars[out_name][global_idx] = float(np.nanmax(valid_data))
                elif method == "min":
                    tracks.vars[out_name][global_idx] = float(np.nanmin(valid_data))

    return tracks


def setup_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Sets up the argument parser for the sample command."""
    parser = subparsers.add_parser(
        "sample",
        description="Sample variables from a NetCDF dataset along storm tracks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input track file (JSON).")
    parser.add_argument(
        "-d", "--data", required=True, help="Input NetCDF data file to sample from."
    )
    parser.add_argument(
        "-v", "--var", required=True, help="Variable name in the NetCDF file."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output track file (JSON)."
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
        type=float,
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
        "-e", "--engine", default=None, help="Xarray engine for reading data."
    )
    parser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the sample command.

    Samples a variable from a NetCDF dataset at track coordinates using
    interpolation or spatial aggregation (mean/max/min within a radius).
    """
    print(f"Reading tracks from {args.input}...")
    tracks = read_json(args.input)

    print(f"Opening dataset {args.data}...")
    ds = xr.open_dataset(args.data, engine=args.engine)

    print(f"Sampling '{args.var}' using method '{args.method}'...")
    if args.radius > 0:
        print(f"Radius: {args.radius} km")

    tracks = sample_tracks(
        tracks=tracks,
        ds=ds,
        varname=args.var,
        method=args.method,
        radius_km=args.radius,
        output_varname=args.name,
    )

    print(f"Writing updated tracks to {args.output}...")
    write_json(tracks, args.output)
    print("Done!")
