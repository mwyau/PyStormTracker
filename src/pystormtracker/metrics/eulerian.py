from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import xarray as xr

from ..models.constants import EULERIAN_FILTER_HOURS, WIND_PERCENTILE_DEFAULT


def compute_high_wind_index(
    ds: xr.Dataset,
    u_var: str,
    v_var: str,
    freq: str = "MS",
    percentile: float = WIND_PERCENTILE_DEFAULT,
    outfile: str | Path | None = None,
) -> xr.DataArray:
    """
    Computes the high-wind index (e.g., 95th percentile of wind speed).
    Used as a weather impact metric (Yau and Chang 2020).

    Args:
        ds: Dataset containing u and v wind components.
        u_var: Variable name for zonal wind.
        v_var: Variable name for meridional wind.
        freq: Resampling frequency (default "MS" for monthly start).
        percentile: Quantile to calculate (default 0.95).
        outfile: Optional path to save the result directly to disk.

    Returns:
        xr.DataArray: The computed high-wind index.
    """
    u = ds[u_var]
    v = ds[v_var]

    # Calculate wind speed
    ws = cast(xr.DataArray, np.sqrt(u**2 + v**2))
    ws.name = "high_wind_index"
    ws.attrs = {"description": f"{percentile * 100}th percentile wind speed ({freq})"}

    # Resample and calculate percentile
    resampled = ws.resample(time=freq).quantile(percentile, dim="time")

    if outfile:
        # Evaluate lazily and stream to disk
        resampled.to_netcdf(outfile)
        return xr.open_dataarray(outfile)

    return resampled.compute()


def compute_variance_metric(
    da: xr.DataArray,
    freq: str = "MS",
    outfile: str | Path | None = None,
) -> xr.DataArray:
    """
    Computes the Eulerian variance metric using a 24-hour difference filter.
    e.g., Var(SLP) = [SLP(t + 24h) - SLP(t)]^2 averaged over the period.

    Args:
        da: Input DataArray (e.g., SLP or z500).
        freq: Resampling frequency (default "MS" for monthly start).
        outfile: Optional path to save the result directly to disk.

    Returns:
        xr.DataArray: The computed variance metric.
    """
    # Create time shift
    # Assuming the data is sorted by time and has a regular interval,
    # we can use shift, but the safest way is to use xarray's shift if we
    # know the frequency, or interpolate if it's irregular.
    # For simplicity, assuming standard 6-hourly data:
    # 24 hours / 6 hours = 4 steps.

    # Calculate time step in hours to determine shift amount
    time_diffs = da.time.diff("time").values
    if len(time_diffs) == 0:
        raise ValueError("DataArray must have a time dimension with >1 steps.")

    # Assuming regular time steps, take the median diff in hours
    dt_hours = np.median(time_diffs).astype("timedelta64[h]").astype(int)

    if dt_hours <= 0:
        raise ValueError("Time step must be > 0 hours.")

    shift_steps = EULERIAN_FILTER_HOURS // dt_hours

    if shift_steps <= 0:
        msg = f"Filter ({EULERIAN_FILTER_HOURS}h) must be >= time step ({dt_hours}h)."
        raise ValueError(msg)

    # [X(t + 24h) - X(t)]^2
    diff = da.shift(time=-shift_steps) - da
    var = diff**2
    var.name = f"var_{da.name}"
    var.attrs = {
        "description": f"{EULERIAN_FILTER_HOURS}-h difference variance ({freq})"
    }

    # Average over the resampling period
    resampled = var.resample(time=freq).mean(dim="time")

    if outfile:
        resampled.to_netcdf(outfile)
        return xr.open_dataarray(outfile)

    return resampled.compute()


def compute_eke(
    u: xr.DataArray,
    v: xr.DataArray,
    freq: str = "MS",
    outfile: str | Path | None = None,
) -> xr.DataArray:
    """
    Computes Eddy Kinetic Energy (EKE) using a 24-hour difference filter.
    EKE = 1/2 * [Var(u) + Var(v)]

    Args:
        u: Zonal wind DataArray.
        v: Meridional wind DataArray.
        freq: Resampling frequency (default "MS" for monthly start).
        outfile: Optional path to save the result directly to disk.

    Returns:
        xr.DataArray: The computed EKE.
    """
    var_u = compute_variance_metric(u, freq=freq)
    var_v = compute_variance_metric(v, freq=freq)

    eke = 0.5 * (var_u + var_v)
    eke.name = "eke"
    eke.attrs = {
        "description": f"Eddy Kinetic Energy ({EULERIAN_FILTER_HOURS}-h filter, {freq})"
    }

    if outfile:
        eke.to_netcdf(outfile)
        return xr.open_dataarray(outfile)

    return eke.compute()
