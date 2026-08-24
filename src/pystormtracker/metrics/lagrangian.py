from __future__ import annotations

from datetime import datetime
from itertools import pairwise
from typing import Final, Literal

import numba as nb
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from ..models.geo import DEG_TO_RAD, geod_dist_km, normalize_longitudes_signed
from ..models.time import decode_time_values
from ..models.tracks import Tracks
from .weighting import _WeightType, calculate_spherical_weight

type Kernel = Literal["constant", "fisher", "cressman", "linear", "quadratic"]
type ATAInterpolation = Literal["linear", "linear_pchip"]

_ATA_HOUR_MS: Final[int] = 3_600_000


def _hourly_times(times: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return one-hour samples through a strictly increasing time sequence."""
    if times.ndim != 1:
        raise ValueError("track times must be one-dimensional")
    if len(times) == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("track times must be strictly increasing")

    result: list[int] = [int(times[0])]
    for start, stop in pairwise(times):
        next_time = int(start) + _ATA_HOUR_MS
        stop_time = int(stop)
        while next_time < stop_time:
            result.append(next_time)
            next_time += _ATA_HOUR_MS
        result.append(stop_time)
    return np.asarray(result, dtype=np.int64)


def _segment_fractions(
    times: NDArray[np.int64],
    sample_times: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return the source segment and elapsed fraction for each sample time."""
    if len(times) < 2:
        raise ValueError("at least two source times are required for segments")
    segment = np.searchsorted(times, sample_times, side="right") - 1
    segment = np.minimum(segment, len(times) - 2).astype(np.int64, copy=False)
    start = times[segment].astype(np.float64)
    duration = (times[segment + 1] - times[segment]).astype(np.float64)
    fraction = (sample_times.astype(np.float64) - start) / duration
    return segment, fraction


def _interpolate_linear_position(
    lat0: NDArray[np.float64],
    lon0: NDArray[np.float64],
    lat1: NDArray[np.float64],
    lon1: NDArray[np.float64],
    fraction: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate latitude and shortest-wrapped longitude in coordinate space."""
    latitudes = lat0 + fraction * (lat1 - lat0)
    longitude_delta = (lon1 - lon0 + 180.0) % 360.0 - 180.0
    longitudes = normalize_longitudes_signed(lon0 + fraction * longitude_delta)
    return latitudes, longitudes


def _interpolate_linear_amplitude(
    amp0: NDArray[np.float64],
    amp1: NDArray[np.float64],
    fraction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate amplitude linearly in time."""
    return amp0 + fraction * (amp1 - amp0)


def _interpolate_pchip_amplitude(
    times: NDArray[np.int64],
    amplitudes: NDArray[np.float64],
    sample_times: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Interpolate one track's amplitude history with SciPy's PCHIP."""
    if len(amplitudes) == 1:
        return np.full(len(sample_times), amplitudes[0], dtype=np.float64)

    time_hours = (times.astype(np.float64) - float(times[0])) / _ATA_HOUR_MS
    sample_hours = (sample_times.astype(np.float64) - float(times[0])) / _ATA_HOUR_MS
    interpolator = PchipInterpolator(
        time_hours,
        amplitudes,
        extrapolate=False,
    )
    result = np.asarray(interpolator(sample_hours), dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise ValueError("PCHIP amplitude interpolation produced a non-finite value")
    return result


def _interpolate_ata_track(
    times: NDArray[np.int64],
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    interpolation: ATAInterpolation,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Resample one packed track for ATA without mutating its source arrays."""
    if not (len(times) == len(latitudes) == len(longitudes) == len(amplitudes)):
        raise ValueError("track time, position, and amplitude arrays must be aligned")
    if len(times) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    if np.any(np.diff(times) <= 0):
        raise ValueError("track times must be strictly increasing")
    if interpolation not in ("linear", "linear_pchip"):
        raise ValueError(f"Unknown ATA interpolation: {interpolation}")

    sample_times = _hourly_times(times)
    if len(times) == 1:
        return (
            sample_times,
            latitudes.copy(),
            normalize_longitudes_signed(longitudes.copy()),
            amplitudes.copy(),
        )

    segment, fraction = _segment_fractions(times, sample_times)
    lat0 = latitudes[segment]
    lon0 = longitudes[segment]
    lat1 = latitudes[segment + 1]
    lon1 = longitudes[segment + 1]
    output_latitudes, output_longitudes = _interpolate_linear_position(
        lat0, lon0, lat1, lon1, fraction
    )

    if interpolation == "linear_pchip":
        output_amplitudes = _interpolate_pchip_amplitude(
            times, amplitudes, sample_times
        )
    else:
        output_amplitudes = _interpolate_linear_amplitude(
            amplitudes[segment], amplitudes[segment + 1], fraction
        )

    # Every observed knot is in the hourly output. Restore the source values
    # explicitly so source topology and exact knot values survive conversions.
    knot_indices = np.searchsorted(sample_times, times)
    output_latitudes[knot_indices] = latitudes
    output_longitudes[knot_indices] = normalize_longitudes_signed(longitudes)
    output_amplitudes[knot_indices] = amplitudes
    return sample_times, output_latitudes, output_longitudes, output_amplitudes


def _interpolate_tracks_for_ata(
    tracks: Tracks,
    variable_name: str,
    interpolation: ATAInterpolation,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Interpolate all packed tracks once and retain their packed offsets."""
    time_parts: list[NDArray[np.int64]] = []
    latitude_parts: list[NDArray[np.float64]] = []
    longitude_parts: list[NDArray[np.float64]] = []
    amplitude_parts: list[NDArray[np.float64]] = []
    output_offsets = [0]

    for track_index in range(len(tracks)):
        point_slice = slice(
            int(tracks.offsets[track_index]), int(tracks.offsets[track_index + 1])
        )
        interpolated = _interpolate_ata_track(
            tracks.times[point_slice],
            tracks.lats[point_slice],
            tracks.lons[point_slice],
            tracks.variables[variable_name][point_slice],
            interpolation,
        )
        track_times, track_lats, track_lons, track_amplitudes = interpolated
        time_parts.append(track_times)
        latitude_parts.append(track_lats)
        longitude_parts.append(track_lons)
        amplitude_parts.append(track_amplitudes)
        output_offsets.append(output_offsets[-1] + len(track_times))

    if time_parts:
        output_times = np.concatenate(time_parts).astype(np.int64, copy=False)
        output_latitudes = np.concatenate(latitude_parts).astype(np.float64, copy=False)
        output_longitudes = np.concatenate(longitude_parts).astype(
            np.float64, copy=False
        )
        output_amplitudes = np.concatenate(amplitude_parts).astype(
            np.float64, copy=False
        )
    else:
        output_times = np.empty(0, dtype=np.int64)
        output_latitudes = np.empty(0, dtype=np.float64)
        output_longitudes = np.empty(0, dtype=np.float64)
        output_amplitudes = np.empty(0, dtype=np.float64)

    return (
        np.asarray(output_offsets, dtype=np.int64),
        output_times,
        output_latitudes,
        output_longitudes,
        output_amplitudes,
    )


@nb.njit(cache=True, nogil=True)
def _compute_weighted_stats(
    grid_lat: NDArray[np.float64],
    grid_lon: NDArray[np.float64],
    offsets: NDArray[np.int64],
    lats: NDArray[np.float64],
    lons: NDArray[np.float64],
    amps: NDArray[np.float64],
    radius_km: float,
    weight_type: int,
    kappa: float,
    is_min: bool,
    active_points: NDArray[np.bool_],
    ata_offsets: NDArray[np.int64],
    ata_lats: NDArray[np.float64],
    ata_lons: NDArray[np.float64],
    ata_amps: NDArray[np.float64],
    ata_active_points: NDArray[np.bool_],
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

    n_points = len(lats)
    if n_points == 0:
        return cyclone_amplitude, cyclone_frequency, track_frequency, aca, ata

    # For Fisher, we need a larger margin as it doesn't have a hard cutoff
    # but decays exponentially. Using 2500km for Fisher (~22 degrees).
    margin_km = radius_km if weight_type != 1 else 2500.0
    lat_margin = (margin_km / 111.0) + 1.0

    for track_index in range(len(offsets) - 1):
        start = offsets[track_index]
        stop = offsets[track_index + 1]
        track_hits = np.zeros((ny, nx), dtype=np.float64)

        for point_index in range(start, stop):
            if not active_points[point_index]:
                continue
            plat = lats[point_index]
            plon = lons[point_index]
            pamp = amps[point_index]

            for i in range(ny):
                glat = grid_lat[i]
                if abs(glat - plat) > lat_margin:
                    continue

                # For points near the poles, we skip the dlon optimization
                # as the longitude margin becomes huge (entire circle)
                if abs(glat) < 80.0:
                    lon_margin = lat_margin / max(0.1, np.cos(glat * DEG_TO_RAD))

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
                            track_hits[i, j] = max(track_hits[i, j], weight)
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
                            track_hits[i, j] = max(track_hits[i, j], weight)

        for i in range(ny):
            for j in range(nx):
                if track_hits[i, j] > 0:
                    track_frequency[i, j] += track_hits[i, j]

        ata_start = ata_offsets[track_index]
        ata_stop = ata_offsets[track_index + 1]
        t_hits = np.zeros((ny, nx), dtype=np.float64)
        init_val = 1e9 if is_min else -1e9
        t_max_amp = np.full((ny, nx), init_val, dtype=np.float64)

        for point_index in range(ata_start, ata_stop):
            if not ata_active_points[point_index]:
                continue
            plat = ata_lats[point_index]
            plon = ata_lons[point_index]
            pamp = ata_amps[point_index]

            for i in range(ny):
                glat = grid_lat[i]
                if abs(glat - plat) > lat_margin:
                    continue

                if abs(glat) < 80.0:
                    lon_margin = lat_margin / max(0.1, np.cos(glat * DEG_TO_RAD))

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
                            t_hits[i, j] = max(t_hits[i, j], weight)
                            if is_min:
                                t_max_amp[i, j] = min(t_max_amp[i, j], pamp)
                            else:
                                t_max_amp[i, j] = max(t_max_amp[i, j], pamp)
                else:
                    for j in range(nx):
                        glon = grid_lon[j]
                        dist = geod_dist_km(glat, glon, plat, plon)
                        weight = calculate_spherical_weight(
                            dist, radius_km, weight_type, kappa
                        )

                        if weight > 0:
                            t_hits[i, j] = max(t_hits[i, j], weight)
                            if is_min:
                                t_max_amp[i, j] = min(t_max_amp[i, j], pamp)
                            else:
                                t_max_amp[i, j] = max(t_max_amp[i, j], pamp)

        for i in range(ny):
            for j in range(nx):
                if t_hits[i, j] > 0:
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
    kernel: Kernel = "constant",
    kappa: float = 20.0,
    variable_name: str | None = None,
    is_min: bool = False,
    monthly: bool = True,
    interpolation: ATAInterpolation = "linear",
) -> xr.Dataset:
    """
    Compute established and study-derived storm-track statistics on a grid.

    Cyclone frequency, cyclone amplitude, and track frequency are established
    track statistics evaluated by Yau and Chang (2020).  ACA combines cyclone
    frequency and amplitude following Guo et al. (2017), as discussed by Yau
    and Chang.  ATA is the novel metric introduced by Yau and Chang: each
    track contributes once at its applicable maximum amplitude while within
    the accumulation radius.  The published ATA procedure linearly
    interpolates 6-hourly cyclone positions and amplitudes to hourly values.

    ``interpolation="linear"`` is the closest literal implementation of that
    published temporal step.  Latitude is interpolated linearly, longitude
    follows the shortest wrapped coordinate difference, and amplitude is
    interpolated linearly.  Yau and Chang specify linear interpolation of
    track positions but do not state the coordinate geometry, so this
    latitude/longitude implementation is not presented as uniquely implied by
    the paper.

    ``interpolation="linear_pchip"`` is a PyStormTracker extension using the
    same piecewise coordinate-linear position interpolation and
    shape-preserving piecewise cubic Hermite interpolation (PCHIP; Fritsch
    and Butland, 1984), provided by
    :class:`scipy.interpolate.PchipInterpolator`, for amplitude only.  PCHIP
    is not part of the Yau and Chang (2020) definition.

    The weighting choices are layered separately: the 500-km constant-radius
    rule corresponds directly to the Yau and Chang accumulation window;
    Fisher/von Mises--Fisher-style and Cressman weights have their own
    numerical/statistical lineages, while linear and quadratic compact kernels
    are PyStormTracker generalizations.

    The five returned metrics are:
    - cyclone_amplitude
    - cyclone_frequency (weighted)
    - track_frequency (weighted)
    - aca (Accumulated Cyclone Activity)
    - ata (Accumulated Track Activity)

    Args:
        tracks: Tracks object containing the storm tracks.
        grid_lat: 1D array of latitude coordinates.
        grid_lon: 1D array of longitude coordinates.
        radius_km: Radius of influence in km. The 500-km default corresponds
            to the accumulation window used by Yau and Chang (2020).
        kernel: Kernel type: 'constant', 'fisher', 'cressman', 'linear', 'quadratic'.
        kappa: Smoothing parameter for Fisher kernel (default 20.0).
        variable_name: Variable in tracks.variables to use as amplitude.
        is_min: If True, tracks are defined by minima (e.g., SLP).
        monthly: If True (default), metrics are aggregated into monthly values.
        interpolation: ATA temporal interpolation method: ``"linear"``
            (default) or ``"linear_pchip"``.

    Returns:
        xr.Dataset: Dataset containing the computed metrics.
    """
    kernel_map = {
        "constant": _WeightType.CONSTANT,
        "fisher": _WeightType.FISHER,
        "cressman": _WeightType.CRESSMAN,
        "linear": _WeightType.LINEAR,
        "quadratic": _WeightType.QUADRATIC,
    }
    if kernel not in kernel_map:
        raise ValueError(f"Unknown kernel: {kernel}")

    wtype = kernel_map[kernel]

    if variable_name is None:
        if len(tracks.variables) > 0:
            variable_name = next(iter(tracks.variables.keys()))
        else:
            raise ValueError("Tracks object does not contain any variables.")

    if variable_name not in tracks.variables:
        raise ValueError(f"Variable '{variable_name}' not found in tracks.")

    if interpolation not in ("linear", "linear_pchip"):
        raise ValueError(f"Unknown ATA interpolation: {interpolation}")

    (
        ata_offsets,
        ata_times,
        ata_lats,
        ata_lons,
        ata_amps,
    ) = _interpolate_tracks_for_ata(tracks, variable_name, interpolation)

    if monthly:
        unique_times = tracks.times
        if len(unique_times) == 0:
            return xr.Dataset()

        decoded_times = decode_time_values(unique_times)
        month_keys = np.asarray(
            [(value.year, value.month) for value in decoded_times], dtype=np.int64
        )
        ata_month_keys = np.asarray(
            [(value.year, value.month) for value in decode_time_values(ata_times)],
            dtype=np.int64,
        )
        all_months = sorted(
            set(map(tuple, month_keys.tolist()))
            | set(map(tuple, ata_month_keys.tolist()))
        )
        ds_list = []

        for year, month_number in all_months:
            mask = (month_keys[:, 0] == year) & (month_keys[:, 1] == month_number)
            ata_mask = (ata_month_keys[:, 0] == year) & (
                ata_month_keys[:, 1] == month_number
            )
            if not np.any(mask) and not np.any(ata_mask):
                continue

            ca, cf, tf, aca_val, ata_val = _compute_weighted_stats(
                np.asarray(grid_lat, dtype=np.float64),
                np.asarray(grid_lon, dtype=np.float64),
                tracks.offsets,
                tracks.lats,
                tracks.lons,
                tracks.variables[variable_name],
                float(radius_km),
                int(wtype),
                float(kappa),
                bool(is_min),
                mask,
                ata_offsets,
                ata_lats,
                ata_lons,
                ata_amps,
                ata_mask,
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
                    "time": datetime(
                        int(year),
                        int(month_number),
                        1,
                    ),
                },
            )
            ds_list.append(ds_month)

        if not ds_list:
            return xr.Dataset()

        ds = xr.concat(ds_list, dim="time")
    else:
        ca, cf, tf, aca_val, ata_val = _compute_weighted_stats(
            np.asarray(grid_lat, dtype=np.float64),
            np.asarray(grid_lon, dtype=np.float64),
            tracks.offsets,
            tracks.lats,
            tracks.lons,
            tracks.variables[variable_name],
            float(radius_km),
            int(wtype),
            float(kappa),
            bool(is_min),
            np.ones(len(tracks.times), dtype=np.bool_),
            ata_offsets,
            ata_lats,
            ata_lons,
            ata_amps,
            np.ones(len(ata_times), dtype=np.bool_),
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
            "amplitude_variable": variable_name,
        }
    )
    if kernel == "fisher":
        ds.attrs["kappa"] = kappa

    return ds
