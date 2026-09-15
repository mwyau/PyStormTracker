from __future__ import annotations

import sys
from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import spharmgrid as sg
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


def test_spectral_filter_invalid_rectangular_grid_propagates_validation_error() -> None:
    latitudes = np.linspace(-80.0, 80.0, 10)
    longitudes = np.linspace(0.0, 360.0, 20, endpoint=False)
    da = xr.DataArray(
        np.random.default_rng(3).random((10, 20)),
        dims=("lat", "lon"),
        coords={"lat": latitudes, "lon": longitudes},
    )

    with pytest.raises(ValueError, match=r".+"):
        SHTFilter(lmin=0, lmax=3).filter(da)


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


def test_spectral_filter_numpy_uses_spharmgrid_adapter() -> None:
    data = np.ones((8, 16), dtype=np.float64)
    expected_input = xr.DataArray(
        data,
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.linspace(-90.0, 90.0, 8),
            "longitude": np.linspace(0.0, 360.0, 16, endpoint=False),
        },
    )
    expected = sg.filter(
        expected_input,
        lmin=0,
        lmax=3,
        taper=0.1,
        sht_threads=4,
    )

    with patch(
        "pystormtracker.preprocessing.spectral.sg.filter",
        wraps=sg.filter,
    ) as filter_call:
        filtered = SHTFilter(lmin=0, lmax=3, sht_threads=4).filter(data)

    assert filtered.shape == data.shape
    np.testing.assert_allclose(filtered, expected.values)
    assert filter_call.call_args is not None
    wrapped = filter_call.call_args.args[0]
    assert isinstance(wrapped, xr.DataArray)
    assert wrapped.dims == ("latitude", "longitude")
    assert filter_call.call_args.kwargs == {
        "lmin": 0,
        "lmax": 3,
        "taper": 0.1,
        "sht_threads": 4,
    }


def test_spectral_filter_rejects_dh_geometry() -> None:
    with pytest.raises(ValueError, match="geometry"):
        SHTFilter(
            lmin=0,
            lmax=3,
            geometry="DH",  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    ("geometry", "latitude_order"),
    [
        ("CC", "ascending"),
        ("CC", "descending"),
        ("GL", "ascending"),
        ("GL", "descending"),
    ],
)
def test_sht_filter_rectangular_dataarray_delegates_to_spharmgrid(
    geometry: Literal["CC", "GL"],
    latitude_order: Literal["ascending", "descending"],
) -> None:
    if geometry == "CC":
        grid = sg.clenshaw_curtis_grid(9, 18, latitude_order=latitude_order)
    else:
        grid = sg.gaussian_grid(8, 16, latitude_order=latitude_order)
    data = xr.DataArray(
        np.random.default_rng(4).random((grid.latitude.size, grid.longitude.size)),
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name="msl",
    )
    expected = sg.filter(
        data,
        lmin=1,
        lmax=3,
        taper=0.7,
        sht_threads=2,
    )

    with (
        patch(
            "pystormtracker.preprocessing.spectral.sg.filter",
            wraps=sg.filter,
        ) as filter_call,
        patch(
            "pystormtracker.preprocessing.spectral.configure_sht_threads"
        ) as configure,
    ):
        actual = SHTFilter(
            lmin=1,
            lmax=3,
            taper_val=0.7,
            geometry=geometry,
            sht_threads=2,
        ).filter(data)

    np.testing.assert_allclose(actual.values, expected.values)
    assert filter_call.call_args is not None
    assert filter_call.call_args.kwargs == {
        "lmin": 1,
        "lmax": 3,
        "taper": 0.7,
        "sht_threads": 2,
    }
    configure.assert_not_called()


@pytest.mark.parametrize("geometry", ["CC", "GL"])
@pytest.mark.parametrize("latitude_order", ["ascending", "descending"])
def test_sht_filter_numpy_matches_equivalent_xarray(
    geometry: Literal["CC", "GL"],
    latitude_order: Literal["ascending", "descending"],
) -> None:
    grid = (
        sg.clenshaw_curtis_grid(9, 18, latitude_order=latitude_order)
        if geometry == "CC"
        else sg.gaussian_grid(8, 16, latitude_order=latitude_order)
    )
    values = np.random.default_rng(12).normal(
        size=(grid.latitude.size, grid.longitude.size)
    )
    data = xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": grid.latitude, "longitude": grid.longitude},
    )
    expected = SHTFilter(
        lmin=1,
        lmax=3,
        taper_val=0.7,
        geometry=geometry,
    ).filter(data)
    actual = SHTFilter(
        lmin=1,
        lmax=3,
        taper_val=0.7,
        geometry=geometry,
        lat_reverse=latitude_order == "descending",
    ).filter(values)
    np.testing.assert_allclose(actual, expected.values)


def test_sht_filter_numpy_regridding_matches_spharmgrid() -> None:
    source = sg.clenshaw_curtis_grid(9, 18, latitude_order="ascending")
    target = sg.gaussian_grid(8, 16, latitude_order="ascending")
    values = np.random.default_rng(15).normal(
        size=(source.latitude.size, source.longitude.size)
    )
    data = xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": source.latitude, "longitude": source.longitude},
    )
    expected = sg.regrid(
        data,
        target,
        lmin=1,
        lmax=3,
        taper=0.7,
        sht_threads=None,
    )

    with patch(
        "pystormtracker.preprocessing.spectral.sg.regrid",
        wraps=sg.regrid,
    ) as regrid_call:
        actual = SHTFilter(
            lmin=1,
            lmax=3,
            taper_val=0.7,
            geometry="CC",
            out_geometry="GL",
            out_ntheta=8,
            out_nphi=16,
        ).filter(values)

    np.testing.assert_allclose(actual, expected.values)
    assert regrid_call.call_args is not None
    assert regrid_call.call_args.args[0].dims == data.dims
    assert regrid_call.call_args.kwargs == {
        "lmin": 1,
        "lmax": 3,
        "taper": 0.7,
        "sht_threads": None,
    }


def test_sht_filter_numpy_mpi_preserves_all_frames() -> None:
    data = np.random.default_rng(16).normal(size=(8, 9, 18))
    communicator = MagicMock()
    communicator.Get_rank.return_value = 1
    communicator.Get_size.return_value = 4
    mpi_module = MagicMock()
    mpi_module.MPI.COMM_WORLD = communicator

    with patch.dict(sys.modules, {"mpi4py": mpi_module}):
        mpi_result = SHTFilter(lmin=0, lmax=3).filter(data, backend="mpi")
    serial_result = SHTFilter(lmin=0, lmax=3).filter(data, backend="serial")

    assert mpi_result.shape == data.shape
    np.testing.assert_allclose(mpi_result, serial_result)


def test_sht_filter_mpi_partitions_frames_before_spharmgrid() -> None:
    data = xr.DataArray(
        np.zeros((5, 9, 18), dtype=np.float64),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(5),
            "lat": np.linspace(-90.0, 90.0, 9),
            "lon": np.linspace(0.0, 360.0, 18, endpoint=False),
        },
    )
    communicator = MagicMock()
    communicator.Get_rank.return_value = 2
    communicator.Get_size.return_value = 4
    mpi_module = MagicMock()
    mpi_module.MPI.COMM_WORLD = communicator

    with (
        patch.dict(sys.modules, {"mpi4py": mpi_module}),
        patch(
            "pystormtracker.preprocessing.spectral.sg.filter",
            wraps=sg.filter,
        ) as filter_call,
    ):
        SHTFilter(lmin=0, lmax=3).filter(data, backend="mpi")

    assert filter_call.call_args is not None
    local_data = filter_call.call_args.args[0]
    assert local_data.sizes["time"] == 1
    np.testing.assert_array_equal(local_data.time.values, np.array([3]))


@pytest.mark.parametrize(
    ("source_geometry", "source_order", "out_geometry"),
    [
        ("CC", "ascending", "CC"),
        ("CC", "descending", "CC"),
        ("CC", "ascending", "GL"),
        ("GL", "descending", "CC"),
    ],
)
def test_sht_regridding_preserves_coordinate_data_association(
    source_geometry: Literal["CC", "GL"],
    source_order: Literal["ascending", "descending"],
    out_geometry: Literal["CC", "GL"],
) -> None:
    if source_geometry == "CC":
        source = sg.clenshaw_curtis_grid(17, 36, latitude_order=source_order)
    else:
        source = sg.gaussian_grid(16, 32, latitude_order=source_order)
    field = np.sin(np.deg2rad(source.latitude))[:, None] * np.ones(
        source.longitude.size
    )
    data = xr.DataArray(
        field,
        dims=("latitude", "longitude"),
        coords={"latitude": source.latitude, "longitude": source.longitude},
        name="msl",
    )

    target_order: Literal["ascending", "descending"] = (
        "ascending" if out_geometry == "CC" else source_order
    )
    target = (
        sg.clenshaw_curtis_grid(8, 16, latitude_order=target_order)
        if out_geometry == "CC"
        else sg.gaussian_grid(8, 16, latitude_order=target_order)
    )
    expected = sg.regrid(
        data,
        target,
        lmin=0,
        lmax=3,
        taper=1.0,
        sht_threads=None,
    )
    filtered = SHTFilter(
        lmin=0,
        lmax=3,
        taper_val=1.0,
        geometry=source_geometry,
        out_geometry=out_geometry,
        out_ntheta=8,
        out_nphi=16,
    ).filter(data)

    xr.testing.assert_allclose(filtered, expected)
    output_latitudes = np.asarray(filtered.latitude.values)
    assert np.all(np.diff(output_latitudes) != 0.0)
    np.testing.assert_allclose(
        np.asarray(filtered.mean("longitude").values),
        np.sin(np.deg2rad(output_latitudes)),
        atol=1.0e-10,
    )


def test_sht_regridding_ascending_and_descending_inputs_have_same_field() -> None:
    ascending = sg.clenshaw_curtis_grid(17, 36, latitude_order="ascending")
    descending = sg.clenshaw_curtis_grid(17, 36, latitude_order="descending")
    target = sg.clenshaw_curtis_grid(8, 16, latitude_order="ascending")

    def make_field(grid: sg.Grid) -> xr.DataArray:
        values = np.sin(np.deg2rad(grid.latitude))[:, None] * np.ones(
            grid.longitude.size
        )
        return xr.DataArray(
            values,
            dims=("latitude", "longitude"),
            coords={"latitude": grid.latitude, "longitude": grid.longitude},
            name="msl",
        )

    ascending_result = SHTFilter(
        lmin=0,
        lmax=3,
        taper_val=1.0,
        geometry="CC",
        out_geometry="CC",
        out_ntheta=8,
        out_nphi=16,
    ).filter(make_field(ascending))
    descending_result = SHTFilter(
        lmin=0,
        lmax=3,
        taper_val=1.0,
        geometry="CC",
        out_geometry="CC",
        out_ntheta=8,
        out_nphi=16,
    ).filter(make_field(descending))

    expected = sg.regrid(
        make_field(ascending), target, lmin=0, lmax=3, taper=1.0, sht_threads=None
    )
    xr.testing.assert_allclose(ascending_result, expected)
    xr.testing.assert_allclose(descending_result, expected)


def test_sht_filter_regular_dask_path_stays_lazy() -> None:
    source = sg.clenshaw_curtis_grid(9, 18, latitude_order="ascending")
    data = xr.DataArray(
        np.stack(
            [
                np.sin(np.deg2rad(source.latitude))[:, None]
                * np.ones(source.longitude.size)
                for _ in range(2)
            ]
        ),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [0, 1],
            "latitude": source.latitude,
            "longitude": source.longitude,
        },
        name="msl",
    ).chunk({"time": 1, "latitude": -1, "longitude": -1})

    filtered = SHTFilter(lmin=0, lmax=3, taper_val=1.0).filter(data, backend="dask")

    assert hasattr(filtered.data, "dask")
    expected = sg.filter(data, lmin=0, lmax=3, taper=1.0, sht_threads=None)
    xr.testing.assert_allclose(filtered.compute(), expected.compute())


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
