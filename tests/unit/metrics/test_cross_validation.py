from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pystormtracker.metrics.cross_validation import (
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
