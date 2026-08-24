from __future__ import annotations

import warnings
from typing import Protocol, cast

import numpy as np
import xarray as xr


class CCAModel(Protocol):
    """Operations used from a fitted xeofs CCA model."""

    def fit(self, X: xr.DataArray, Y: xr.DataArray, dim: str) -> CCAModel: ...

    def predict(self, X: xr.DataArray) -> xr.DataArray: ...

    def inverse_transform(
        self, X: xr.DataArray | None = None, Y: xr.DataArray | None = None
    ) -> object: ...


class CCAConstructor(Protocol):
    """Constructor signature used from the optional xeofs CCA class."""

    def __call__(
        self,
        *,
        n_modes: int,
        use_coslat: bool,
        use_pca: bool,
        n_pca_modes: float,
    ) -> CCAModel: ...


XeofsCCA: CCAConstructor | None
try:
    from xeofs.cross import CCA as _XeofsCCA
except ImportError:
    XeofsCCA = None
else:
    XeofsCCA = cast(CCAConstructor, _XeofsCCA)


def _require_cca() -> CCAConstructor:
    if XeofsCCA is None:
        raise ImportError(
            "The 'xeofs' library is required for cross-validation. "
            "Install it with 'uv sync --extra eof'."
        )
    return XeofsCCA


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
    Evaluate CCA truncation numbers (M) with leave-n-out cross-validation.

    CCA, EOF/PCA, SVD, ACC, and Pearson correlation are established
    statistical methods; ``xeofs`` supplies the numerical CCA/EOF
    implementation.  The evaluation workflow is the framework of Yau and
    Chang (2020), whose study used 108 winter months, three withheld months
    per fold, 105 training months, 35 folds, and ``0 < M < 36``.  The PST
    parameters are configurable: its current defaults (including
    ``max_modes=15``) are implementation defaults, not paper-defined
    constants.  The study selected its best model by the domain mean of
    ``(ACC + FVE) / 2``; this function returns the per-mode scores and leaves
    any final selection to its caller.

    Reference:
        Yau, A. M.-W., and E. K.-M. Chang (2020). Finding Storm Track
        Activity Metrics That Are Highly Correlated with Weather Impacts.
        Part I. *Journal of Climate*, 33(23), 10169--10186.
        https://doi.org/10.1175/JCLI-D-20-0393.1

    Args:
        X: Predictor field, dimensions (time, lat, lon).
        Y: Predictand field, dimensions (time, lat, lon).
        max_modes: Maximum number of PCA/CCA modes to test.
        leave_n_out: Samples to hold out. Three is the study's seasonal
            configuration, but this parameter is configurable.
        pca: If True, performs PCA pre-filtering before CCA.

    Returns:
        xr.Dataset: Dataset containing 'acc' and 'fve' scores.
    """
    cca_class = _require_cca()
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
            mode_count = int(m)
            model = cca_class(
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
                raise TypeError("xeofs CCA returned a non-DataArray prediction")
            all_preds.append(Y_pred)

        # Reconstruct full predicted field
        full_pred = xr.concat(all_preds, dim="time").sortby("time")
        Y_eval, full_pred = xr.align(Y, full_pred, join="exact")

        # Calculate ACC (Anomaly Correlation Coefficient)
        acc = xr.corr(Y_eval, full_pred, dim="time")
        acc_scores[idx] = float(acc.mean())

        # Calculate FVE (Fraction of Variance Explained).  This retains the
        # current PST local aggregation, mean(1 - MSE / VAR).  Yau and Chang
        # define domain FVE as 1 - mean(MSE) / mean(VAR); the two formulas are
        # not equivalent in general, so this is not an exact reproduction of
        # the paper's domain-FVE calculation.
        mse = ((Y_eval - full_pred) ** 2).mean(dim="time")
        var = Y_eval.var(dim="time")
        valid_var = var.where(var > 0.0)
        fve = 1.0 - (mse / valid_var)
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

    CCA and the PCA/EOF preprocessing are established statistical methods;
    ``xeofs`` provides the numerical implementation.  The use of cosine
    latitude weighting and PCA before CCA follows the Yau and Chang (2020)
    evaluation framework, while the requested mode count remains a PST API
    choice.

    Args:
        X: Predictor field, dimensions (time, lat, lon).
        Y: Predictand field, dimensions (time, lat, lon).
        n_modes: Number of PCA/CCA modes to use (usually the best M).
    Returns:
        The trained xeofs CCA model.
    """
    cca_class = _require_cca()
    if n_modes <= 0:
        raise ValueError("n_modes must be greater than zero")

    # Ensure time dimensions are aligned
    X, Y = _align_cca_inputs(X, Y)
    if n_modes > X.sizes["time"]:
        raise ValueError("n_modes exceeds the available sample count")

    # The full-data model follows the study's PCA-before-CCA configuration;
    # this is an explicit PST training choice, not a new CCA method.
    model = cca_class(
        n_modes=n_modes,
        use_coslat=True,
        use_pca=True,
        n_pca_modes=n_modes,
    )
    model.fit(X, Y, dim="time")

    return model


def compute_cormax(
    impact_da: xr.DataArray,
    metric_da: xr.DataArray,
    search_lon: float = 60.0,
    search_lat: float = 20.0,
) -> xr.DataArray:
    """
    Compute the CORMAX evaluation score.

    CORMAX is the Yau and Chang (2020) evaluation construction: for each
    impact point, their study sought the maximum positive one-point Pearson
    correlation with a metric within a local search window.  The current PST
    implementation computes the maximum available correlation in that window;
    the search arguments generalize the study's 60-degree longitude by
    20-degree latitude configuration.  Pearson correlation is standard
    statistics.

    Reference:
        Yau, A. M.-W., and E. K.-M. Chang (2020). Finding Storm Track
        Activity Metrics That Are Highly Correlated with Weather Impacts.
        Part I. *Journal of Climate*, 33(23), 10169--10186.
        https://doi.org/10.1175/JCLI-D-20-0393.1

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
            cormax = r.where(cormax.isnull() | (r > cormax), cormax)

    cormax.name = "cormax"
    cormax.attrs = {
        "description": "Maximum local one-point correlation (CORMAX)",
        "search_window": f"{search_lon}x{search_lat} deg",
    }
    return cormax
