from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import numpy as np
import xarray as xr

DIFFERENCE_FILTER_HOURS: Final[int] = 24


def compute_high_wind_index(
    ds: xr.Dataset,
    u_variable: str,
    v_variable: str,
    *,
    percentile: float,
    freq: str = "MS",
    outfile: str | Path | None = None,
) -> xr.DataArray:
    """
    Compute a percentile wind-speed index.

    Yau and Chang (2020) use a 95th-percentile 10-m wind index as a study
    configuration for weather impacts.  This function accepts an arbitrary
    percentile and therefore generalizes that configuration; the percentile
    is not a universal PyStormTracker scientific constant.

    Args:
        ds: Dataset containing u and v wind components.
        u_variable: Variable name for zonal wind.
        v_variable: Variable name for meridional wind.
        percentile: Quantile to calculate.
        freq: Resampling frequency (default "MS" for monthly start).
        outfile: Optional path to save the result directly to disk.

    Returns:
        xr.DataArray: The computed high-wind index.
    """
    u = ds[u_variable]
    v = ds[v_variable]

    # Calculate wind speed
    ws = cast(xr.DataArray, np.sqrt(u**2 + v**2))
    ws.name = "high_wind_index"
    ws.attrs = {"description": f"{percentile * 100}th percentile wind speed ({freq})"}

    # Resample and calculate percentile
    resampled = ws.resample(time=freq).quantile(percentile, dim="time")

    if outfile:
        # Evaluate lazily and stream to disk
        resampled.to_netcdf(outfile, engine="h5netcdf")
        return xr.open_dataarray(outfile, engine="h5netcdf")

    return resampled.compute()


def compute_variance_metric(
    da: xr.DataArray,
    freq: str = "MS",
    outfile: str | Path | None = None,
) -> xr.DataArray:
    """
    Compute the Eulerian variance metric using a 24-hour difference filter.

    The simple difference-filter construction is credited to the storm-track
    analysis lineage of Wallace, Lim, and Blackmon (1988), and is also used as
    an evaluated Eulerian metric by Yau and Chang (2020).  The implementation
    accepts the input cadence and derives the shift in samples.

    Reference:
        Wallace, J. M., G.-H. Lim, and M. L. Blackmon (1988). Relationship
        between Cyclone Tracks, Anticyclone Tracks and Baroclinic Waveguides.
        *Journal of the Atmospheric Sciences*, 45(3), 439--462.
        https://doi.org/10.1175/1520-0469(1988)045<0439:RBCTAT>2.0.CO;2

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

    shift_steps = DIFFERENCE_FILTER_HOURS // dt_hours

    if shift_steps <= 0:
        msg = f"Filter ({DIFFERENCE_FILTER_HOURS}h) must be >= time step ({dt_hours}h)."
        raise ValueError(msg)

    # [X(t + 24h) - X(t)]^2
    diff = da.shift(time=-shift_steps) - da
    variance = diff**2
    variance.name = f"var_{da.name}"
    variance.attrs = {
        "description": f"{DIFFERENCE_FILTER_HOURS}-h difference variance ({freq})"
    }

    # Average over the resampling period
    resampled = variance.resample(time=freq).mean(dim="time")

    if outfile:
        resampled.to_netcdf(outfile, engine="h5netcdf")
        return xr.open_dataarray(outfile, engine="h5netcdf")

    return resampled.compute()


def compute_eke(
    u: xr.DataArray,
    v: xr.DataArray,
    freq: str = "MS",
    outfile: str | Path | None = None,
) -> xr.DataArray:
    """
    Compute standard eddy kinetic energy from 24-hour difference variances.

    EKE is standard mathematics used by Yau and Chang (2020), not a metric
    invented by that study.  The current PST implementation uses
    ``1/2 * [Var(u) + Var(v)]`` with the shared simple difference filter.

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
        "description": (
            f"Eddy Kinetic Energy ({DIFFERENCE_FILTER_HOURS}-h filter, {freq})"
        )
    }

    if outfile:
        eke.to_netcdf(outfile, engine="h5netcdf")
        return xr.open_dataarray(outfile, engine="h5netcdf")

    return eke.compute()
