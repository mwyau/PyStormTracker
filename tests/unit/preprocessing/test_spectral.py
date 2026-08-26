from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from pystormtracker.preprocessing import DCTFilter, SHTFilter


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_spectral_filter_serial(ny: int, nx: int) -> None:
    # Test with both 2.5-deg and 0.25-deg
    data: NDArray[np.float64] = np.random.default_rng().random((2, ny, nx))
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": [0, 1],
            "lat": np.linspace(90, -90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="msl",
    )

    filt = SHTFilter(lmin=5, lmax=42)
    filtered = filt.filter(da, backend="serial")

    assert filtered.shape == (2, ny, nx)
    assert filtered.dims == ("time", "lat", "lon")
    assert filtered.name == "msl"


def test_spectral_filter_invalid_shape() -> None:
    # 10 x 15 is invalid for SHT
    data: NDArray[np.float64] = np.random.default_rng().random((1, 10, 15))
    da = xr.DataArray(data, dims=["time", "lat", "lon"])

    filt = SHTFilter(lmin=0, lmax=42)
    with pytest.raises(ValueError, match="Unsupported shape for spectral filter"):
        filt.filter(da, backend="serial")


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_spectral_filter_lat_reverse(ny: int, nx: int) -> None:
    # Test latitude South to North (lat_reverse=False)
    data: NDArray[np.float64] = np.random.default_rng().random((1, ny, nx))
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": [0],
            "lat": np.linspace(-90, 90, ny),  # S->N
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="msl",
    )

    filt = SHTFilter(lmin=5, lmax=42, lat_reverse=False)
    filtered = filt.filter(da, backend="serial")

    assert filtered.shape == (1, ny, nx)
    assert filtered.lat[0] == -90


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_spectral_filter_lat_descending(ny: int, nx: int) -> None:
    # Test latitude North to South (lat_reverse=True)
    data: NDArray[np.float64] = np.random.default_rng().random((1, ny, nx))
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": [0],
            "lat": np.linspace(90, -90, ny),  # N->S
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="msl",
    )

    filt = SHTFilter(lmin=5, lmax=42, lat_reverse=True)
    filtered = filt.filter(da, backend="serial")

    assert filtered.shape == (1, ny, nx)
    assert filtered.lat[0] == 90


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_spectral_filter_numpy_ndarray(ny: int, nx: int) -> None:
    # Test passing a raw numpy array
    data: NDArray[np.float64] = np.random.default_rng().random((ny, nx))

    filt = SHTFilter(lmin=5, lmax=42)
    filtered = filt.filter(data)

    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == (ny, nx)


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_spectral_filter_numpy_ndarray_3d(ny: int, nx: int) -> None:
    # Test passing a 3D numpy array (T, ny, nx)
    data: NDArray[np.float64] = np.random.default_rng().random((3, ny, nx))

    filt = SHTFilter(lmin=5, lmax=42)
    filtered = filt.filter(data)

    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == (3, ny, nx)


def test_dct_filter_regional_dataarray() -> None:
    latitudes = np.linspace(40.0, 50.0, 5)
    longitudes = np.linspace(-10.0, 10.0, 6)
    data = xr.DataArray(
        np.arange(30.0).reshape(5, 6),
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": longitudes},
        name="msl",
    )

    filtered = DCTFilter(lmin=0, lmax=3, taper_val=1.0).filter(data)

    assert isinstance(filtered, xr.DataArray)
    assert filtered.dims == data.dims
    assert filtered.shape == data.shape
    assert filtered.name == data.name
    assert np.isfinite(filtered.values).all()


def test_dct_filter_rejects_numpy_array() -> None:
    data = np.ones((5, 6), dtype=np.float64)

    with pytest.raises(TypeError, match="requires xarray.DataArray"):
        DCTFilter(lmin=0, lmax=3).filter(
            data  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )


def test_spectral_filter_passes_explicit_sht_threads_to_ducc_wrapper() -> None:
    data = np.ones((8, 16), dtype=np.float64)
    with (
        patch(
            "pystormtracker.preprocessing.spectral._filter_sht_frame",
            return_value=data,
        ) as filter_frame,
        patch(
            "pystormtracker.preprocessing.spectral.configure_sht_threads"
        ) as configure,
    ):
        filtered = SHTFilter(lmin=0, lmax=3, sht_threads=4).filter(data)

    assert filtered.shape == data.shape
    assert filter_frame.call_args is not None
    assert filter_frame.call_args.kwargs["nthreads"] == 4
    configure.assert_called_once_with(4)


def test_sht_regridding_preserves_ascending_latitude_orientation() -> None:
    latitudes = np.linspace(-90.0, 90.0, 17)
    longitudes = np.linspace(0.0, 360.0, 36, endpoint=False)
    field = np.sin(np.deg2rad(latitudes))[:, None] * np.ones_like(longitudes)
    data = xr.DataArray(
        field[None, :, :],
        dims=("time", "latitude", "longitude"),
        coords={"time": [0], "latitude": latitudes, "longitude": longitudes},
        name="msl",
    )

    filtered = SHTFilter(
        lmin=0,
        lmax=3,
        taper_val=1.0,
        geometry="CC",
        out_geometry="GL",
        out_ntheta=8,
        out_nphi=16,
    ).filter(data)

    output_latitudes = np.asarray(filtered.latitude.values)
    assert np.all(np.diff(output_latitudes) > 0.0)
    expected = np.sin(np.deg2rad(output_latitudes))
    np.testing.assert_allclose(
        np.asarray(filtered.isel(time=0).mean("longitude").values),
        expected,
        atol=1.0e-6,
    )


def test_sht_regridding_dask_declares_output_sizes() -> None:
    latitudes = np.linspace(90.0, -90.0, 73)
    longitudes = np.linspace(0.0, 360.0, 144, endpoint=False)
    field = np.sin(np.deg2rad(latitudes))[:, None] * np.ones_like(longitudes)
    data = xr.DataArray(
        np.stack((field, field)),
        dims=("time", "latitude", "longitude"),
        coords={"time": [0, 1], "latitude": latitudes, "longitude": longitudes},
        name="msl",
    ).chunk({"time": 1, "latitude": -1, "longitude": -1})

    filtered = SHTFilter(
        lmin=0,
        lmax=3,
        taper_val=1.0,
        geometry="CC",
        out_geometry="GL",
        out_ntheta=8,
        out_nphi=16,
    ).filter(data, backend="dask")

    assert filtered.dims == ("time", "latitude", "longitude")
    assert filtered.shape == (2, 8, 16)
    assert hasattr(filtered.data, "dask")
    computed = filtered.compute()
    assert np.isfinite(computed.values).all()
