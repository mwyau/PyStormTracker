from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pystormtracker.metrics.cross_validation import (
    _align_cca_inputs,
    _domain_fve,
    compute_cormax,
    find_best_cca_truncation,
    train_cca_model,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:Deleting a single level.*:FutureWarning"),
    pytest.mark.filterwarnings(
        "ignore:The `dims` argument.*:PendingDeprecationWarning"
    ),
]


@pytest.fixture
def related_fields() -> tuple[xr.DataArray, xr.DataArray]:
    rng = np.random.default_rng(4)
    time = np.arange("2000-01", "2002-01", dtype="datetime64[M]")
    lat = np.array([-30.0, 0.0, 30.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    signal = rng.normal(size=(len(time), 1, 1))
    noise = rng.normal(scale=0.1, size=(len(time), len(lat), len(lon)))
    x_values = signal + noise
    y_values = 2.0 * signal + noise
    coords = {"time": time, "lat": lat, "lon": lon}
    X = xr.DataArray(x_values, dims=("time", "lat", "lon"), coords=coords)
    Y = xr.DataArray(y_values, dims=("time", "lat", "lon"), coords=coords)
    return X, Y


def test_find_best_cca_truncation(
    related_fields: tuple[xr.DataArray, xr.DataArray],
) -> None:
    pytest.importorskip("xeofs")

    X, Y = related_fields
    result = find_best_cca_truncation(X, Y, max_modes=2, leave_n_out=3)

    assert result.sizes == {"M": 2}
    assert np.isfinite(result.acc).all()
    assert np.isfinite(result.fve).all()
    assert float(result.acc.sel(M=1)) > 0.9


def test_train_cca_model(
    related_fields: tuple[xr.DataArray, xr.DataArray],
) -> None:
    pytest.importorskip("xeofs")

    X, Y = related_fields
    model = train_cca_model(X, Y, n_modes=1)
    scores = model.predict(X.isel(time=slice(0, 2)))
    prediction = model.inverse_transform(Y=scores)

    assert isinstance(prediction, xr.DataArray)
    assert prediction.dims == Y.dims
    assert prediction.sizes["time"] == 2


@pytest.mark.parametrize(
    ("max_modes", "leave_n_out", "message"),
    [(0, 3, "max_modes"), (1, 0, "leave_n_out"), (22, 3, "training rank")],
)
def test_find_best_cca_truncation_validates_parameters(
    related_fields: tuple[xr.DataArray, xr.DataArray],
    max_modes: int,
    leave_n_out: int,
    message: str,
) -> None:
    pytest.importorskip("xeofs")

    X, Y = related_fields
    with pytest.raises(ValueError, match=message):
        find_best_cca_truncation(X, Y, max_modes=max_modes, leave_n_out=leave_n_out)


def test_compute_cormax_finds_local_correlation(
    related_fields: tuple[xr.DataArray, xr.DataArray],
) -> None:
    X, Y = related_fields
    result = compute_cormax(X, Y, search_lon=180.0, search_lat=60.0)

    assert result.dims == ("lat", "lon")
    assert result.name == "cormax"
    assert float(result.min()) > 0.9


def test_compute_cormax_preserves_missing_correlations() -> None:
    field = xr.DataArray(
        np.ones((3, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1, 2], "lat": [0.0, 10.0], "lon": [0.0, 10.0]},
    )

    result = compute_cormax(field, field, search_lon=10.0, search_lat=10.0)

    assert result.isnull().all()


def test_domain_fve_averages_mse_and_variance_separately() -> None:
    observed = xr.DataArray(
        [[0.0, 0.0], [1.0, 10.0]],
        dims=("time", "point"),
    )
    predicted = xr.DataArray(
        [[0.0, 0.0], [1.0, 0.0]],
        dims=("time", "point"),
    )

    # Gridpoint MSE is [0, 50] and observed variance is [0.25, 25].
    expected = 1.0 - np.mean([0.0, 50.0]) / np.mean([0.25, 25.0])
    old_pointwise_average = np.mean([1.0, 1.0 - 50.0 / 25.0])

    assert _domain_fve(observed, predicted) == pytest.approx(expected)
    assert abs(expected - old_pointwise_average) > 0.9


def test_domain_fve_includes_zero_variance_points_with_finite_mse() -> None:
    observed = xr.DataArray(
        [[0.0, 0.0], [0.0, 2.0]],
        dims=("time", "point"),
    )
    predicted = xr.DataArray(
        [[1.0, 0.0], [1.0, 2.0]],
        dims=("time", "point"),
    )

    # Gridpoint MSE is [1, 0] and observed variance is [0, 1].
    # The zero-variance point must still contribute its nonzero MSE.
    expected = 1.0 - np.mean([1.0, 0.0]) / np.mean([0.0, 1.0])

    assert _domain_fve(observed, predicted) == pytest.approx(expected)


def test_align_cca_inputs_requires_and_normalizes_a_common_grid() -> None:
    time = np.arange(3)
    lat = np.array([-10.0, 10.0])
    lon = np.array([0.0, 20.0, 40.0])
    X = xr.DataArray(
        np.arange(18, dtype=float).reshape(3, 2, 3),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
    )
    Y = X.transpose("lon", "time", "lat") + 1.0

    aligned_X, aligned_Y = _align_cca_inputs(X, Y)

    assert aligned_X.dims == ("time", "lat", "lon")
    assert aligned_Y.dims == ("time", "lat", "lon")
    xr.testing.assert_equal(aligned_X, X)
    xr.testing.assert_equal(aligned_Y, X + 1.0)


def test_align_cca_inputs_rejects_inner_joinable_but_mismatched_times() -> None:
    X = xr.DataArray(
        np.zeros((3, 2)),
        dims=("time", "point"),
        coords={"time": [0, 1, 2], "point": [0, 1]},
    )
    Y = X.assign_coords(time=[1, 2, 3])

    with pytest.raises(ValueError, match="same spatial grid and time coordinates"):
        _align_cca_inputs(X, Y)


def test_compute_cormax_uses_the_maximum_positive_correlation() -> None:
    signal = np.array([-1.0, 0.0, 1.0, 2.0])
    coords = {
        "time": [0, 1, 2, 3],
        "lat": [0.0, 10.0],
        "lon": [0.0, 10.0, 20.0],
    }
    impact = xr.DataArray(
        np.broadcast_to(signal[:, None, None], (4, 2, 3)),
        dims=("time", "lat", "lon"),
        coords=coords,
    )
    metric = xr.DataArray(
        np.stack(
            [
                -signal,
                signal + np.array([1.0, -1.0, -1.0, 1.0]),
                signal + 0.5 * np.array([1.0, -1.0, -1.0, 1.0]),
            ],
            axis=1,
        )[:, None, :].repeat(2, axis=1),
        dims=("time", "lat", "lon"),
        coords=coords,
    )

    result = compute_cormax(impact, metric, search_lon=20.0, search_lat=0.1)

    np.testing.assert_allclose(result, np.sqrt(5.0 / 6.0))


def test_compute_cormax_wraps_the_longitude_search_window() -> None:
    signal = np.array([-1.0, 0.0, 1.0, 2.0])
    coords = {"time": [0, 1, 2, 3], "lat": [0.0, 10.0], "lon": [0.0, 10.0, 20.0]}
    impact = xr.DataArray(
        np.broadcast_to(signal[:, None, None], (4, 2, 3)),
        dims=("time", "lat", "lon"),
        coords=coords,
    )
    metric = xr.DataArray(
        np.stack([-signal, -signal, signal], axis=1)[:, None, :].repeat(2, axis=1),
        dims=("time", "lat", "lon"),
        coords=coords,
    )

    result = compute_cormax(impact, metric, search_lon=20.0, search_lat=0.1)

    assert float(result.sel(lon=0.0).min()) > 0.9
    assert float(result.min()) > 0.9


def test_compute_cormax_keeps_nan_correlations_missing() -> None:
    signal = np.array([-1.0, 0.0, 1.0, 2.0])
    coords = {"time": [0, 1, 2, 3], "lat": [0.0, 10.0], "lon": [0.0, 10.0]}
    impact = xr.DataArray(
        np.broadcast_to(signal[:, None, None], (4, 2, 2)),
        dims=("time", "lat", "lon"),
        coords=coords,
    )
    metric = impact.where(impact.lon == 0.0)

    result = compute_cormax(impact, metric, search_lon=0.1, search_lat=0.1)

    assert float(result.sel(lon=0.0).min()) > 0.9
    assert result.sel(lon=10.0).isnull().all()


def test_compute_cormax_returns_nan_when_window_has_no_positive_correlation() -> None:
    signal = np.array([-1.0, 0.0, 1.0, 2.0])
    coords = {"time": [0, 1, 2, 3], "lat": [0.0, 10.0], "lon": [0.0, 10.0]}
    impact = xr.DataArray(
        np.broadcast_to(signal[:, None, None], (4, 2, 2)),
        dims=("time", "lat", "lon"),
        coords=coords,
    )
    metric = -impact

    result = compute_cormax(impact, metric, search_lon=0.1, search_lat=0.1)

    assert result.isnull().all()


@pytest.mark.parametrize(
    "mismatch",
    ["different-time", "shifted-latitude", "shifted-longitude", "disjoint-grid"],
)
def test_compute_cormax_rejects_mismatched_time_or_grid(
    related_fields: tuple[xr.DataArray, xr.DataArray],
    mismatch: str,
) -> None:
    X, Y = related_fields
    if mismatch == "different-time":
        mutated = Y.assign_coords(time=Y.time + np.timedelta64(1, "D"))
    elif mismatch == "shifted-latitude":
        mutated = Y.assign_coords(lat=Y.lat + 1.0)
    elif mismatch == "shifted-longitude":
        mutated = Y.assign_coords(lon=Y.lon + 1.0)
    else:
        mutated = Y.assign_coords(lat=[100.0, 110.0, 120.0])

    with pytest.raises(ValueError, match="same spatial grid and time coordinates"):
        compute_cormax(X, mutated, search_lon=180.0, search_lat=60.0)


def test_compute_cormax_accepts_harmless_dimension_order_difference(
    related_fields: tuple[xr.DataArray, xr.DataArray],
) -> None:
    X, Y = related_fields
    result = compute_cormax(
        X.transpose("lon", "time", "lat"),
        Y.transpose("lon", "time", "lat"),
        search_lon=180.0,
        search_lat=60.0,
    )

    assert result.dims == ("lat", "lon")
    assert float(result.min()) > 0.9
