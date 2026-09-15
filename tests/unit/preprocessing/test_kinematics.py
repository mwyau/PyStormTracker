from __future__ import annotations

from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
import spharmgrid as sg
import xarray as xr
from numpy.typing import NDArray

from pystormtracker.models.geo import R_EARTH_M
from pystormtracker.preprocessing.kinematics import (
    Kinematics,
    compute_vorticity_divergence,
)


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_compute_vort_div_shapes(ny: int, nx: int) -> None:
    rng = np.random.default_rng()
    u = rng.random((ny, nx))
    v = rng.random((ny, nx))

    div, vort = compute_vorticity_divergence(u, v)

    assert div.shape == (ny, nx)
    assert vort.shape == (ny, nx)

    # Check that they aren't all zeros
    assert np.any(div != 0.0)
    assert np.any(vort != 0.0)


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_compute_vorticity_divergence_xarray(ny: int, nx: int) -> None:
    rng = np.random.default_rng()
    u = xr.DataArray(
        rng.random((ny, nx)),
        coords={
            "lat": np.linspace(90, -90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        dims=["lat", "lon"],
    )
    v = xr.DataArray(
        rng.random((ny, nx)),
        coords={
            "lat": np.linspace(90, -90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        dims=["lat", "lon"],
    )

    div, vort = compute_vorticity_divergence(u, v)

    assert div.dims == ("lat", "lon")
    assert vort.dims == ("lat", "lon")
    assert div.name == "divergence"
    assert vort.name == "relative_vorticity"
    assert np.array_equal(div.lat, u.lat)
    assert np.array_equal(vort.lon, u.lon)


@pytest.mark.parametrize(
    ("geometry", "latitude_order"),
    [
        ("CC", "ascending"),
        ("CC", "descending"),
        ("GL", "ascending"),
        ("GL", "descending"),
    ],
)
def test_xarray_kinematics_delegates_to_spharmgrid(
    geometry: Literal["CC", "GL"],
    latitude_order: Literal["ascending", "descending"],
) -> None:
    grid = (
        sg.clenshaw_curtis_grid(9, 18, latitude_order=latitude_order)
        if geometry == "CC"
        else sg.gaussian_grid(8, 16, latitude_order=latitude_order)
    )
    rng = np.random.default_rng(6)
    u = xr.DataArray(
        rng.random((grid.latitude.size, grid.longitude.size)),
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )
    v = xr.DataArray(
        rng.random((grid.latitude.size, grid.longitude.size)),
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )
    expected = sg.kinematics(u, v, radius=R_EARTH_M, sht_threads=None)

    with patch(
        "pystormtracker.preprocessing.kinematics.sg.kinematics",
        wraps=sg.kinematics,
    ) as kinematics_call:
        divergence, vorticity = compute_vorticity_divergence(u, v)

    np.testing.assert_allclose(divergence.values, expected["d"].values)
    np.testing.assert_allclose(vorticity.values, expected["vo"].values)
    assert divergence.name == "divergence"
    assert vorticity.name == "relative_vorticity"
    assert kinematics_call.call_args is not None
    assert kinematics_call.call_args.args[0] is u
    assert kinematics_call.call_args.args[1] is v
    assert kinematics_call.call_args.kwargs == {
        "radius": R_EARTH_M,
        "sht_threads": None,
    }

    class_divergence, class_vorticity = Kinematics(geometry=geometry).compute(u, v)
    xr.testing.assert_allclose(divergence, class_divergence)
    xr.testing.assert_allclose(vorticity, class_vorticity)


def test_xarray_kinematics_preserves_physical_field_across_latitude_order() -> None:
    def make_winds(
        latitude_order: Literal["ascending", "descending"],
    ) -> tuple[xr.DataArray, xr.DataArray]:
        grid = sg.clenshaw_curtis_grid(17, 36, latitude_order=latitude_order)
        latitude = np.deg2rad(grid.latitude)[:, None]
        longitude = np.deg2rad(grid.longitude)[None, :]
        u = np.cos(latitude) * np.cos(longitude) + 0.2 * np.sin(2.0 * latitude)
        v = np.sin(latitude) * np.sin(longitude) + 0.1 * np.cos(3.0 * longitude)
        return (
            xr.DataArray(
                u,
                dims=("lat", "lon"),
                coords={"lat": grid.latitude, "lon": grid.longitude},
            ),
            xr.DataArray(
                v,
                dims=("lat", "lon"),
                coords={"lat": grid.latitude, "lon": grid.longitude},
            ),
        )

    ascending_u, ascending_v = make_winds("ascending")
    descending_u, descending_v = make_winds("descending")
    ascending_divergence, ascending_vorticity = compute_vorticity_divergence(
        ascending_u, ascending_v
    )
    descending_divergence, descending_vorticity = compute_vorticity_divergence(
        descending_u, descending_v
    )

    xr.testing.assert_allclose(
        ascending_divergence,
        descending_divergence.sortby("lat"),
    )
    xr.testing.assert_allclose(
        ascending_vorticity,
        descending_vorticity.sortby("lat"),
    )


@pytest.mark.parametrize("geometry", ["CC", "GL"])
@pytest.mark.parametrize("latitude_order", ["ascending", "descending"])
def test_numpy_kinematics_matches_equivalent_xarray(
    geometry: Literal["CC", "GL"],
    latitude_order: Literal["ascending", "descending"],
) -> None:
    grid = (
        sg.clenshaw_curtis_grid(9, 18, latitude_order=latitude_order)
        if geometry == "CC"
        else sg.gaussian_grid(8, 16, latitude_order=latitude_order)
    )
    rng = np.random.default_rng(13)
    u_values = rng.normal(size=(grid.latitude.size, grid.longitude.size))
    v_values = rng.normal(size=u_values.shape)
    u = xr.DataArray(
        u_values,
        dims=("latitude", "longitude"),
        coords={"latitude": grid.latitude, "longitude": grid.longitude},
    )
    v = xr.DataArray(
        v_values,
        dims=("latitude", "longitude"),
        coords={"latitude": grid.latitude, "longitude": grid.longitude},
    )

    expected_divergence, expected_vorticity = compute_vorticity_divergence(u, v)
    actual_divergence, actual_vorticity = compute_vorticity_divergence(
        u_values,
        v_values,
        geometry=geometry,
        lat_reverse=latitude_order == "descending",
    )
    np.testing.assert_allclose(actual_divergence, expected_divergence.values)
    np.testing.assert_allclose(actual_vorticity, expected_vorticity.values)


def test_kinematics_rejects_dh_geometry() -> None:
    with pytest.raises(ValueError, match="geometry"):
        compute_vorticity_divergence(  # type: ignore[call-overload]  # ty: ignore[no-matching-overload]
            np.ones((9, 18)),
            np.ones((9, 18)),
            geometry="DH",
        )


@pytest.mark.parametrize("geometry", ["CC", "GL"])
def test_explicit_lmax_preserves_latitude_orientation(
    geometry: Literal["CC", "GL"],
) -> None:
    grid_ascending = (
        sg.clenshaw_curtis_grid(17, 36, latitude_order="ascending")
        if geometry == "CC"
        else sg.gaussian_grid(16, 32, latitude_order="ascending")
    )
    grid_descending = (
        sg.clenshaw_curtis_grid(17, 36, latitude_order="descending")
        if geometry == "CC"
        else sg.gaussian_grid(16, 32, latitude_order="descending")
    )
    rng = np.random.default_rng(14)
    u_values = rng.normal(
        size=(grid_ascending.latitude.size, grid_ascending.longitude.size)
    )
    v_values = rng.normal(size=u_values.shape)
    u_ascending = xr.DataArray(
        u_values,
        dims=("latitude", "longitude"),
        coords={
            "latitude": grid_ascending.latitude,
            "longitude": grid_ascending.longitude,
        },
    )
    v_ascending = xr.DataArray(
        v_values,
        dims=u_ascending.dims,
        coords=u_ascending.coords,
    )
    u_descending = xr.DataArray(
        u_values[::-1],
        dims=("latitude", "longitude"),
        coords={
            "latitude": grid_descending.latitude,
            "longitude": grid_descending.longitude,
        },
    )
    v_descending = xr.DataArray(
        v_values[::-1],
        dims=u_descending.dims,
        coords=u_descending.coords,
    )

    ascending = compute_vorticity_divergence(
        u_ascending, v_ascending, geometry=geometry, lmax=5
    )
    descending = compute_vorticity_divergence(
        u_descending, v_descending, geometry=geometry, lmax=5
    )
    xr.testing.assert_allclose(ascending[0], descending[0].sortby("latitude"))
    xr.testing.assert_allclose(ascending[1], descending[1].sortby("latitude"))


def test_xarray_kinematics_dask_path_stays_lazy() -> None:
    grid = sg.clenshaw_curtis_grid(9, 18, latitude_order="descending")
    u = xr.DataArray(
        np.stack(
            [
                np.cos(np.deg2rad(grid.latitude))[:, None]
                * np.ones(grid.longitude.size)
                for _ in range(2)
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": grid.latitude, "lon": grid.longitude},
    ).chunk({"time": 1, "lat": -1, "lon": -1})
    v = xr.zeros_like(u)

    divergence, vorticity = compute_vorticity_divergence(u, v, backend="dask")

    assert hasattr(divergence.data, "dask")
    assert hasattr(vorticity.data, "dask")
    assert np.isfinite(divergence.compute().values).all()
    assert np.isfinite(vorticity.compute().values).all()


def test_xarray_kinematics_passes_sht_threads_to_spharmgrid() -> None:
    grid = sg.clenshaw_curtis_grid(9, 18)
    u = xr.DataArray(
        np.ones((9, 18)),
        dims=("latitude", "longitude"),
        coords={"latitude": grid.latitude, "longitude": grid.longitude},
    )
    v = xr.zeros_like(u)
    with patch(
        "pystormtracker.preprocessing.kinematics.sg.kinematics",
        wraps=sg.kinematics,
    ) as kinematics_call:
        compute_vorticity_divergence(u, v, nthreads=3)

    assert kinematics_call.call_args is not None
    assert kinematics_call.call_args.kwargs["sht_threads"] == 3


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_compute_vorticity_divergence_lat_reverse(ny: int, nx: int) -> None:
    # Test latitude South to North (lat_reverse=False)
    data: NDArray[np.float64] = np.random.default_rng().random((1, ny, nx))
    u = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": [0],
            "lat": np.linspace(-90, 90, ny),  # S->N
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="msl",
    )

    # compute_vorticity_divergence should handle it automatically.
    div, vort = compute_vorticity_divergence(u, u)

    assert div.shape == (1, ny, nx)
    assert div.lat[0] == -90
    assert vort.shape == (1, ny, nx)
    assert vort.lat[0] == -90


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_kinematics_class(ny: int, nx: int) -> None:
    rng = np.random.default_rng()
    u_np = rng.random((ny, nx))
    v_np = rng.random((ny, nx))

    # lat_reverse remains meaningful for the coordinate-free NumPy path.
    calc = Kinematics(lat_reverse=True)
    div_np, vort_np = calc.compute(u_np, v_np)

    assert div_np.shape == (ny, nx)
    assert vort_np.shape == (ny, nx)

    u_xr = xr.DataArray(
        u_np,
        coords={
            "lat": np.linspace(90, -90, ny),  # N->S
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        dims=["lat", "lon"],
    )
    v_xr = xr.DataArray(
        v_np,
        coords={
            "lat": np.linspace(90, -90, ny),  # N->S
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        dims=["lat", "lon"],
    )

    div_xr, _vort_xr = calc.compute(u_xr, v_xr)
    assert isinstance(div_xr, xr.DataArray)
    shared_div, shared_vort = compute_vorticity_divergence(u_xr, v_xr)
    xr.testing.assert_allclose(div_xr, shared_div)
    xr.testing.assert_allclose(_vort_xr, shared_vort)
    assert np.isfinite(div_np).all()
    assert np.isfinite(vort_np).all()


def test_solid_body_rotation() -> None:
    # Solid body rotation: u = U0 * cos(lat)
    ntheta, nphi = 73, 144
    lat = np.linspace(np.pi / 2, -np.pi / 2, ntheta)
    lon = np.linspace(0, 2 * np.pi, nphi, endpoint=False)

    _lon_grid, lat_grid = np.meshgrid(lon, lat)

    u = np.cos(lat_grid) * 10.0
    v = np.zeros_like(u)

    div, vort = compute_vorticity_divergence(u, v, nthreads=1)

    # Divergence of solid body rotation should be very close to zero
    np.testing.assert_allclose(div, 0, atol=1e-12)

    # Vorticity is non-zero
    assert np.max(np.abs(vort)) > 0
