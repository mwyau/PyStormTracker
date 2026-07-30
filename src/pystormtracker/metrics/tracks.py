from __future__ import annotations

from typing import Literal

import numba as nb
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..models.constants import DEGTORAD
from ..models.geo import geod_dist_km
from ..models.tracks import Tracks
from .weighting import WeightType, calculate_spherical_weight


@nb.njit(cache=True, nogil=True)
def _compute_weighted_stats(
    grid_lat: NDArray[np.float64],
    grid_lon: NDArray[np.float64],
    track_ids: NDArray[np.int64],
    lats: NDArray[np.float64],
    lons: NDArray[np.float64],
    amps: NDArray[np.float64],
    radius_km: float,
    weight_type: int,
    kappa: float,
    is_min: bool,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    ny = len(grid_lat)
    nx = len(grid_lon)

    cyclone_frequency = np.zeros((ny, nx), dtype=np.float64)
    cyclone_amplitude = np.zeros((ny, nx), dtype=np.float64)
    track_frequency = np.zeros((ny, nx), dtype=np.float64)
    aca = np.zeros((ny, nx), dtype=np.float64)
    ata = np.zeros((ny, nx), dtype=np.float64)

    n_points = len(track_ids)
    if n_points == 0:
        return cyclone_amplitude, cyclone_frequency, track_frequency, aca, ata

    unique_ids = np.unique(track_ids)

    # For Fisher, we need a larger margin as it doesn't have a hard cutoff
    # but decays exponentially. Using 2500km for Fisher (~22 degrees).
    margin_km = radius_km if weight_type != 1 else 2500.0
    lat_margin = (margin_km / 111.0) + 1.0

    for tid in unique_ids:
        # Find points for this track
        idx = np.where(track_ids == tid)[0]
        t_lats = lats[idx]
        t_lons = lons[idx]
        t_amps = amps[idx]

        t_hits = np.zeros((ny, nx), dtype=np.float64)

        # Initialize max_amp array properly
        init_val = 1e9 if is_min else -1e9
        t_max_amp = np.full((ny, nx), init_val, dtype=np.float64)

        for p in range(len(idx)):
            plat = t_lats[p]
            plon = t_lons[p]
            pamp = t_amps[p]

            for i in range(ny):
                glat = grid_lat[i]
                if abs(glat - plat) > lat_margin:
                    continue

                # For points near the poles, we skip the dlon optimization
                # as the longitude margin becomes huge (entire circle)
                if abs(glat) < 80.0:
                    lon_margin = lat_margin / max(0.1, np.cos(glat * DEGTORAD))

                    for j in range(nx):
                        glon = grid_lon[j]
                        dlon = abs(glon - plon)
                        if dlon > 180.0:
                            dlon = 360.0 - dlon
                        if dlon > lon_margin:
                            continue

                        dist = geod_dist_km(glat, glon, plat, plon)
                        weight = calculate_spherical_weight(
                            dist, radius_km, weight_type, kappa
                        )

                        if weight > 0:
                            cyclone_frequency[i, j] += weight
                            aca[i, j] += pamp * weight

                            # Track stats: we take the weighted contribution of the
                            # peak intensity within the search window
                            t_hits[i, j] = max(t_hits[i, j], weight)

                            if is_min:
                                if pamp < t_max_amp[i, j]:
                                    t_max_amp[i, j] = pamp
                            else:
                                if pamp > t_max_amp[i, j]:
                                    t_max_amp[i, j] = pamp
                else:
                    # Polar handling: check all longitudes
                    for j in range(nx):
                        glon = grid_lon[j]
                        dist = geod_dist_km(glat, glon, plat, plon)
                        weight = calculate_spherical_weight(
                            dist, radius_km, weight_type, kappa
                        )

                        if weight > 0:
                            cyclone_frequency[i, j] += weight
                            aca[i, j] += pamp * weight
                            t_hits[i, j] = max(t_hits[i, j], weight)

                            if is_min:
                                if pamp < t_max_amp[i, j]:
                                    t_max_amp[i, j] = pamp
                            else:
                                if pamp > t_max_amp[i, j]:
                                    t_max_amp[i, j] = pamp

        for i in range(ny):
            for j in range(nx):
                if t_hits[i, j] > 0:
                    track_frequency[i, j] += t_hits[i, j]
                    ata[i, j] += t_max_amp[i, j] * t_hits[i, j]

    for i in range(ny):
        for j in range(nx):
            if cyclone_frequency[i, j] > 0:
                cyclone_amplitude[i, j] = aca[i, j] / cyclone_frequency[i, j]

    return cyclone_amplitude, cyclone_frequency, track_frequency, aca, ata


def compute_track_metrics(
    tracks: Tracks,
    grid_lat: NDArray[np.float64],
    grid_lon: NDArray[np.float64],
    radius_km: float = 500.0,
    kernel: Literal[
        "constant", "fisher", "cressman", "linear", "quadratic"
    ] = "constant",
    kappa: float = 20.0,
    varname: str | None = None,
    is_min: bool = False,
    monthly: bool = True,
) -> xr.Dataset:
    """
    Computes storm track metrics on a 2D spatial grid using weighted estimators.
    Supports 5 Lagrangian metrics (Yau and Chang 2020, Hodges 1999, Simmonds 2026):
    - cyclone_amplitude
    - cyclone_frequency (weighted)
    - track_frequency (weighted)
    - aca (Accumulated Cyclone Activity)
    - ata (Accumulated Track Activity)

    Args:
        tracks: Tracks object containing the storm tracks.
        grid_lat: 1D array of latitude coordinates.
        grid_lon: 1D array of longitude coordinates.
        radius_km: Radius of influence in km. Default 500km (Yau & Chang).
        kernel: Kernel type: 'constant', 'fisher', 'cressman', 'linear', 'quadratic'.
        kappa: Smoothing parameter for Fisher kernel (default 20.0).
        varname: Variable in tracks.vars to use as amplitude.
        is_min: If True, tracks are defined by minima (e.g., SLP).
        monthly: If True (default), metrics are aggregated into monthly values.

    Returns:
        xr.Dataset: Dataset containing the computed metrics.
    """
    kernel_map = {
        "constant": WeightType.CONSTANT,
        "fisher": WeightType.FISHER,
        "cressman": WeightType.CRESSMAN,
        "linear": WeightType.LINEAR,
        "quadratic": WeightType.QUADRATIC,
    }
    if kernel not in kernel_map:
        raise ValueError(f"Unknown kernel: {kernel}")

    wtype = kernel_map[kernel]

    if varname is None:
        if len(tracks.vars) > 0:
            varname = next(iter(tracks.vars.keys()))
        else:
            raise ValueError("Tracks object does not contain any variables.")

    if varname not in tracks.vars:
        raise ValueError(f"Variable '{varname}' not found in tracks.")

    if monthly:
        unique_times = tracks.times
        if len(unique_times) == 0:
            return xr.Dataset()

        all_months = np.unique(unique_times.astype("datetime64[M]"))
        ds_list = []

        for month in all_months:
            mask = unique_times.astype("datetime64[M]") == month
            if not np.any(mask):
                continue

            m_track_ids = tracks.track_ids[mask]
            m_lats = tracks.lats[mask]
            m_lons = tracks.lons[mask]
            m_amps = tracks.vars[varname][mask]

            ca, cf, tf, aca_val, ata_val = _compute_weighted_stats(
                np.asarray(grid_lat, dtype=np.float64),
                np.asarray(grid_lon, dtype=np.float64),
                m_track_ids,
                m_lats,
                m_lons,
                m_amps,
                float(radius_km),
                int(wtype),
                float(kappa),
                bool(is_min),
            )

            ds_month = xr.Dataset(
                {
                    "cyclone_amplitude": (("lat", "lon"), ca),
                    "cyclone_frequency": (("lat", "lon"), cf),
                    "track_frequency": (("lat", "lon"), tf),
                    "aca": (("lat", "lon"), aca_val),
                    "ata": (("lat", "lon"), ata_val),
                },
                coords={
                    "lat": grid_lat,
                    "lon": grid_lon,
                    "time": month.astype("datetime64[ns]"),
                },
            )
            ds_list.append(ds_month)

        if not ds_list:
            return xr.Dataset()

        ds = xr.concat(ds_list, dim="time")
    else:
        amps = tracks.vars[varname]
        ca, cf, tf, aca_val, ata_val = _compute_weighted_stats(
            np.asarray(grid_lat, dtype=np.float64),
            np.asarray(grid_lon, dtype=np.float64),
            tracks.track_ids,
            tracks.lats,
            tracks.lons,
            amps,
            float(radius_km),
            int(wtype),
            float(kappa),
            bool(is_min),
        )

        ds = xr.Dataset(
            {
                "cyclone_amplitude": (("lat", "lon"), ca),
                "cyclone_frequency": (("lat", "lon"), cf),
                "track_frequency": (("lat", "lon"), tf),
                "aca": (("lat", "lon"), aca_val),
                "ata": (("lat", "lon"), ata_val),
            },
            coords={
                "lat": grid_lat,
                "lon": grid_lon,
            },
        )

    ds.attrs.update(
        {
            "description": "Storm track metrics (Weighted Spherical Estimator)",
            "radius_km": radius_km,
            "kernel": kernel,
            "amplitude_variable": varname,
        }
    )
    if kernel == "fisher":
        ds.attrs["kappa"] = kappa

    return ds
