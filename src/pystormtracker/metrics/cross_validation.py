from __future__ import annotations

import warnings
from typing import Protocol, cast

import numpy as np
import xarray as xr

try:
    from xeofs.cross import CCA as XeofsCCA
except ImportError:
    XeofsCCA = None


class CCAModel(Protocol):
    """Operations used from a fitted xeofs CCA model."""

    def fit(self, X: xr.DataArray, Y: xr.DataArray, dim: str) -> CCAModel: ...

    def predict(self, X: xr.DataArray) -> xr.DataArray: ...

    def inverse_transform(
        self, X: xr.DataArray | None = None, Y: xr.DataArray | None = None
    ) -> object: ...


def _require_cca() -> None:
    if XeofsCCA is None:
        raise ImportError(
            "The 'xeofs' library is required for cross-validation. "
            "Install it with 'uv sync --extra metrics'."
        )


def _align_cca_inputs(
    X: xr.DataArray, Y: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    if "time" not in X.dims or "time" not in Y.dims:
        raise ValueError("CCA inputs must both contain a 'time' dimension")
    X, Y = xr.align(X, Y, join="inner")
    if X.sizes["time"] == 0:
        raise ValueError("CCA inputs have no overlapping time coordinates")
    return X, Y


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
    _require_cca()
    if max_modes <= 0:
        raise ValueError("max_modes must be greater than zero")
    if leave_n_out <= 0:
        raise ValueError("leave_n_out must be greater than zero")

    # Ensure time dimensions are aligned
    X, Y = _align_cca_inputs(X, Y)
    n_samples = X.sizes["time"]
    if leave_n_out >= n_samples:
        raise ValueError("leave_n_out must be smaller than the sample count")
    min_training_samples = n_samples - leave_n_out
    feature_count = min(
        int(np.prod([X.sizes[dim] for dim in X.dims if dim != "time"])),
        int(np.prod([Y.sizes[dim] for dim in Y.dims if dim != "time"])),
    )
    if max_modes > min(min_training_samples, feature_count):
        raise ValueError("max_modes exceeds the available training rank")

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
            assert XeofsCCA is not None
            mode_count = int(m)
            model = XeofsCCA(
                n_modes=mode_count,
                use_coslat=True,
                use_pca=pca,
                n_pca_modes=mode_count if pca else 0.999,
            )
            model.fit(X_train, Y_train, dim="time")

            # Predict the held-out samples
            predicted_scores = model.predict(X_test)
            Y_pred = model.inverse_transform(Y=predicted_scores)
            if not isinstance(Y_pred, xr.DataArray):
                raise RuntimeError("xeofs CCA returned a non-DataArray prediction")
            all_preds.append(Y_pred)

        # Reconstruct full predicted field
        full_pred = xr.concat(all_preds, dim="time").sortby("time")
        Y_eval, full_pred = xr.align(Y, full_pred, join="exact")

        # Calculate ACC (Anomaly Correlation Coefficient)
        acc = xr.corr(Y_eval, full_pred, dim="time")
        acc_scores[idx] = float(acc.mean())

        # Calculate FVE (Fraction of Variance Explained)
        mse = ((Y_eval - full_pred) ** 2).mean(dim="time")
        var = Y_eval.var(dim="time")
        fve = xr.where(var > 0.0, 1 - (mse / var), np.nan)
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
) -> CCAModel:
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
    _require_cca()
    if n_modes <= 0:
        raise ValueError("n_modes must be greater than zero")

    # Ensure time dimensions are aligned
    X, Y = _align_cca_inputs(X, Y)
    if n_modes > X.sizes["time"]:
        raise ValueError("n_modes exceeds the available sample count")

    # Hardcoded pca=True as per the evaluation framework in Yau and Chang 2020
    assert XeofsCCA is not None
    model = XeofsCCA(
        n_modes=n_modes,
        use_coslat=True,
        use_pca=True,
        n_pca_modes=n_modes,
    )
    model.fit(X, Y, dim="time")

    return cast(CCAModel, model)


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
    if search_lon <= 0.0 or search_lat <= 0.0:
        raise ValueError("CORMAX search dimensions must be greater than zero")
    required_dims = {"time", "lat", "lon"}
    if not required_dims.issubset(impact_da.dims) or not required_dims.issubset(
        metric_da.dims
    ):
        raise ValueError("CORMAX inputs must have time, lat, and lon dimensions")

    # Ensure time and grid coordinates are aligned
    impact_da, metric_da = xr.align(impact_da, metric_da, join="inner")
    if impact_da.sizes["time"] < 2:
        raise ValueError("CORMAX requires at least two overlapping time steps")
    if impact_da.sizes["lat"] < 2 or impact_da.sizes["lon"] < 2:
        raise ValueError("CORMAX requires at least two latitude and longitude points")

    # Calculate grid spacing to determine point shifts
    dlat = float(abs(impact_da.lat[1] - impact_da.lat[0]))
    dlon = float(abs(impact_da.lon[1] - impact_da.lon[0]))

    n_lat_half = int(np.round(search_lat / 2.0 / dlat))
    n_lon_half = int(np.round(search_lon / 2.0 / dlon))

    cormax = xr.full_like(impact_da.isel(time=0, drop=True), np.nan, dtype=float)

    # Optimization: Iterate over local window shifts and take max correlation
    # Using xarray's vectorized correlation is more efficient than a grid loop
    for j in range(-n_lat_half, n_lat_half + 1):
        for i in range(-n_lon_half, n_lon_half + 1):
            # Shift lat (non-periodic, fills NaN at poles)
            # Roll lon (periodic wrapping)
            shifted = metric_da.shift(lat=j).roll(lon=i, roll_coords=False)

            # Compute correlation field
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Degrees of freedom <= 0 for slice",
                    category=RuntimeWarning,
                )
                r = xr.corr(impact_da, shifted, dim="time")

            # Preserve missing correlations while accumulating finite maxima.
            cormax = xr.where(cormax.isnull() | (r > cormax), r, cormax)

    cormax.name = "cormax"
    cormax.attrs = {
        "description": "Maximum local one-point correlation (CORMAX)",
        "search_window": f"{search_lon}x{search_lat} deg",
    }
    return cormax
