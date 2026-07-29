from __future__ import annotations

import numpy as np
import xarray as xr

try:
    import xeofs as xe
except ImportError:
    xe = None


def find_best_cca_truncation(
    X: xr.DataArray,
    Y: xr.DataArray,
    max_modes: int = 15,
    leave_n_out: int = 3,
    pca: bool = True,
) -> xr.Dataset:
    """
    Evaluates CCA truncation numbers (M) to find the best hyperparameter.

    Args:
        X: Predictor field, dimensions (time, lat, lon).
        Y: Predictand field, dimensions (time, lat, lon).
        max_modes: Maximum number of PCA/CCA modes to test.
        leave_n_out: Samples to hold out (default 3 for seasonal).
        pca: If True, performs PCA pre-filtering before CCA.

    Returns:
        xr.Dataset: Dataset containing 'acc' and 'fve' scores.
    """
    if xe is None:
        raise ImportError(
            "The 'xeofs' library is required for cross-validation. "
            "Install it with 'pip install PyStormTracker[metrics]'"
        )

    # Ensure time dimensions are aligned
    X, Y = xr.align(X, Y)
    n_samples = len(X.time)

    m_values = np.arange(1, max_modes + 1)
    acc_scores = np.zeros(len(m_values))
    fve_scores = np.zeros(len(m_values))

    for idx, m in enumerate(m_values):
        all_preds = []

        # Cross-validation loop stepping by leave_n_out
        for start in range(0, n_samples, leave_n_out):
            end = min(start + leave_n_out, n_samples)

            # Training indices
            t_pre = np.arange(0, start)
            t_post = np.arange(end, n_samples)
            train_idx = np.concatenate([t_pre, t_post])
            test_idx = np.arange(start, end)

            X_train, X_test = X.isel(time=train_idx), X.isel(time=test_idx)
            Y_train = Y.isel(time=train_idx)

            # Initialize CCA model
            model = xe.models.CCA(n_modes=m, use_coslat=True, pca=pca)
            model.fit(X_train, Y_train, dim="time")

            # Predict the held-out samples
            Y_pred = model.predict(X_test)
            all_preds.append(Y_pred)

        # Reconstruct full predicted field
        full_pred = xr.concat(all_preds, dim="time")

        # Calculate ACC (Anomaly Correlation Coefficient)
        acc = xr.corr(Y, full_pred, dim="time")
        acc_scores[idx] = float(acc.mean())

        # Calculate FVE (Fraction of Variance Explained)
        mse = ((Y - full_pred) ** 2).mean(dim="time")
        var = Y.var(dim="time")
        fve = 1 - (mse / var)
        fve_scores[idx] = float(fve.mean())

    ds = xr.Dataset(
        {
            "acc": (("M"), acc_scores),
            "fve": (("M"), fve_scores),
        },
        coords={"M": m_values},
        attrs={
            "description": "CCA Truncation Sensitivity Analysis",
            "leave_n_out": leave_n_out,
        },
    )

    return ds


def train_cca_model(
    X: xr.DataArray,
    Y: xr.DataArray,
    n_modes: int,
) -> xe.models.CCA:
    """
    Trains a CCA model with the given number of modes on the full dataset.

    Args:
        X: Predictor field, dimensions (time, lat, lon).
        Y: Predictand field, dimensions (time, lat, lon).
        n_modes: Number of PCA/CCA modes to use (usually the best M).
        pca: If True, performs PCA pre-filtering before CCA.

    Returns:
        The trained xeofs CCA model.
    """
    if xe is None:
        raise ImportError(
            "The 'xeofs' library is required for cross-validation. "
            "Install it with 'pip install PyStormTracker[metrics]'"
        )

    # Ensure time dimensions are aligned
    X, Y = xr.align(X, Y)

    # Hardcoded pca=True as per the evaluation framework in Yau and Chang 2020
    model = xe.models.CCA(n_modes=n_modes, use_coslat=True, pca=True)
    model.fit(X, Y, dim="time")

    return model


def compute_cormax(
    impact_da: xr.DataArray,
    metric_da: xr.DataArray,
    search_lon: float = 60.0,
    search_lat: float = 20.0,
) -> xr.DataArray:
    """
    Computes the CORMAX score: maximum one-point correlation within a local region.
    For each grid point in impact_da, find the max correlation with metric_da
    within a search_lon x search_lat window (Yau and Chang 2020).

    Args:
        impact_da: Weather impact anomalies (time, lat, lon).
        metric_da: Storm track metric anomalies (time, lat, lon).
        search_lon: Longitude window width in degrees (default 60).
        search_lat: Latitude window width in degrees (default 20).

    Returns:
        xr.DataArray: CORMAX scores (lat, lon).
    """
    # Ensure time dimensions are aligned
    impact_da, metric_da = xr.align(impact_da, metric_da)

    # Calculate grid spacing to determine point shifts
    dlat = float(abs(impact_da.lat[1] - impact_da.lat[0]))
    dlon = float(abs(impact_da.lon[1] - impact_da.lon[0]))

    n_lat_half = int(np.round(search_lat / 2.0 / dlat))
    n_lon_half = int(np.round(search_lon / 2.0 / dlon))

    # Initialize cormax with lowest possible correlation
    cormax = xr.full_like(impact_da.isel(time=0), -1.0, dtype=float).drop_vars("time")
    cormax.name = "cormax"
    cormax.attrs = {
        "description": "Maximum local one-point correlation (CORMAX)",
        "search_window": f"{search_lon}x{search_lat} deg",
    }

    # Optimization: Iterate over local window shifts and take max correlation
    # Using xarray's vectorized correlation is more efficient than a grid loop
    for j in range(-n_lat_half, n_lat_half + 1):
        for i in range(-n_lon_half, n_lon_half + 1):
            # Shift lat (non-periodic, fills NaN at poles)
            # Roll lon (periodic wrapping)
            shifted = metric_da.shift(lat=j).roll(lon=i, roll_coords=False)

            # Compute correlation field
            r = xr.corr(impact_da, shifted, dim="time")

            # Update maximum (xr.where handles NaNs by keeping existing values)
            cormax = xr.where(r > cormax, r, cormax)

    return cormax
