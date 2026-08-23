from __future__ import annotations

import ducc0
import numpy as np
import pytest
import xarray as xr

from pystormtracker.healpix.detector import (
    HealpixDetector,
    _extract_healpix_centers,
    _find_healpix_object_extrema,
    _label_healpix_connected_components,
    _refine_healpix_quadratic_batch,
)
from pystormtracker.refinement import spherical_quadratic_status_name


def _unit_vector(latitude: float, longitude: float) -> np.ndarray:
    latitude_rad, longitude_rad = np.deg2rad([latitude, longitude])
    return np.array(
        [
            np.cos(latitude_rad) * np.cos(longitude_rad),
            np.cos(latitude_rad) * np.sin(longitude_rad),
            np.sin(latitude_rad),
        ]
    )


def _healpix_geometry(
    nside: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
    pixels = np.arange(12 * nside**2, dtype=np.int64)
    angles = hp_base.pix2ang(pixels)
    latitudes = np.rad2deg(0.5 * np.pi - angles[:, 0])
    longitudes = np.rad2deg(angles[:, 1]) % 360.0
    points = np.array(
        [
            _unit_vector(float(lat), float(lon))
            for lat, lon in zip(latitudes, longitudes, strict=True)
        ],
        dtype=np.float64,
    )
    neighbor_table = np.asarray(hp_base.neighbors(pixels).T, dtype=np.int64)
    return pixels, latitudes, longitudes, points, neighbor_table


def _geodesic_error(
    latitude: float,
    longitude: float,
    target: tuple[float, float],
) -> float:
    dot = np.dot(_unit_vector(latitude, longitude), _unit_vector(*target))
    return float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))


def test_healpix_detector_init() -> None:
    # 1D xarray data
    nside = 4
    npix = 12 * nside**2
    data = np.ones((1, npix))
    da = xr.DataArray(
        data,
        dims=["time", "cell"],
        coords={"time": [0], "cell": np.arange(npix)},
        name="msl",
    )

    detector = HealpixDetector.from_xarray(da)
    assert detector.requested_variable_name == "msl"
    assert detector.nside == nside
    assert detector._neighbor_table is not None
    assert detector._neighbor_table.shape == (8, npix)


def test_healpix_default_object_threshold_is_independent_of_simple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    from typing import cast

    detector_module = importlib.import_module("pystormtracker.healpix.detector")
    captured: list[float] = []

    def fake_detect(
        _frame: np.ndarray,
        time_val: object,
        _neighbor_table: np.ndarray,
        _lat: np.ndarray,
        _lon: np.ndarray,
        **kwargs: object,
    ) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
        del time_val
        captured.append(float(cast(float, kwargs["object_threshold"])))
        return (0, np.empty(0), np.empty(0), np.empty(0))

    monkeypatch.setattr(
        "pystormtracker.simple.constants.DEFAULT_MSL_FEATURE_THRESHOLD", 999.0
    )
    monkeypatch.setattr(detector_module, "detect_healpix_frame", fake_detect)

    nside = 4
    npix = 12 * nside**2
    data = xr.DataArray(
        np.ones((1, npix)),
        dims=["time", "cell"],
        coords={"time": [0], "cell": np.arange(npix)},
        name="msl",
    )
    HealpixDetector.from_xarray(data).detect()

    assert captured == [0.0]


def test_label_healpix_connected_components() -> None:
    nside = 4
    npix = 12 * nside**2
    data = np.zeros(npix, dtype=np.float64)
    # Create two disjoint objects
    data[10] = 100.0
    data[11] = 100.0  # neighbor
    data[50] = 100.0

    hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
    all_pix = np.arange(npix, dtype=np.int64)
    nbors = np.asarray(hp_base.neighbors(all_pix).T, dtype=np.int64)

    labels, num_objects = _label_healpix_connected_components(
        data, nbors, threshold=50.0, is_min=False
    )

    assert num_objects == 2
    assert labels[10] == labels[11]
    assert labels[10] != labels[50]
    assert labels[10] > 0
    assert labels[50] > 0


def test_find_healpix_object_extrema() -> None:
    nside = 4
    npix = 12 * nside**2
    data = np.zeros(npix, dtype=np.float64)
    data[10] = 100.0
    data[11] = 90.0  # center of max
    data[12] = 80.0

    hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
    all_pix = np.arange(npix, dtype=np.int64)
    nbors = np.asarray(hp_base.neighbors(all_pix).T, dtype=np.int64)

    labels, num_objects = _label_healpix_connected_components(
        data, nbors, threshold=50.0, is_min=False
    )
    extrema = _find_healpix_object_extrema(
        data, labels, nbors, num_objects, is_min=False, min_grid_points=1
    )

    assert extrema[10] == 1.0
    assert np.sum(extrema) == 1.0


@pytest.mark.parametrize(
    ("case", "target", "mode"),
    [
        ("center", None, "min"),
        ("off_center_minimum", (32.3, 179.2), "min"),
        ("off_center_maximum", (45.0, -179.0), "max"),
        ("high_latitude", (82.0, 30.0), "min"),
    ],
)
def test_healpix_quadratic_refines_analytic_spherical_extrema(
    case: str,
    target: tuple[float, float] | None,
    mode: str,
) -> None:
    pixels, lats, lons, points, neighbors = _healpix_geometry()
    if target is None:
        center_pixel = 100
        target = (float(lats[center_pixel]), float(lons[center_pixel]))
    target_vector = _unit_vector(*target)
    dot_products = points @ target_vector
    data = 1.0 - dot_products if mode == "min" else dot_products
    center_pixel = int(np.argmin(data) if mode == "min" else np.argmax(data))

    refined = _refine_healpix_quadratic_batch(
        data,
        neighbors,
        np.array([pixels[center_pixel]], dtype=np.int64),
        lats,
        lons,
        is_minimum=mode == "min",
    )

    assert spherical_quadratic_status_name(int(refined.status_codes[0])) == "success", (
        case
    )
    assert (
        _geodesic_error(
            float(refined.latitudes[0]), float(refined.longitudes[0]), target
        )
        < 0.05
    )


def test_healpix_quadratic_is_invariant_to_signed_longitude_representation() -> None:
    pixels, lats, lons, points, neighbors = _healpix_geometry()
    target = (32.3, 179.2)
    data = 1.0 - points @ _unit_vector(*target)
    center_pixel = int(np.argmin(data))
    signed_lons = np.where(lons >= 180.0, lons - 360.0, lons)

    unsigned = _refine_healpix_quadratic_batch(
        data,
        neighbors,
        np.array([pixels[center_pixel]], dtype=np.int64),
        lats,
        lons,
        is_minimum=True,
    )
    signed = _refine_healpix_quadratic_batch(
        data,
        neighbors,
        np.array([pixels[center_pixel]], dtype=np.int64),
        lats,
        signed_lons,
        is_minimum=True,
    )

    assert unsigned.status_codes.tolist() == signed.status_codes.tolist()
    assert (
        _geodesic_error(
            float(unsigned.latitudes[0]), float(unsigned.longitudes[0]), target
        )
        < 0.05
    )
    assert (
        _geodesic_error(float(signed.latitudes[0]), float(signed.longitudes[0]), target)
        < 0.05
    )


def test_healpix_quadratic_rejects_a_spherical_saddle() -> None:
    pixels, lats, lons, points, neighbors = _healpix_geometry()
    center_pixel = int(np.argmin(lats**2 + ((lons + 180.0) % 360.0 - 180.0) ** 2))
    center = points[center_pixel]
    latitude_rad, longitude_rad = np.deg2rad([lats[center_pixel], lons[center_pixel]])
    e_theta = np.array(
        [
            np.sin(latitude_rad) * np.cos(longitude_rad),
            np.sin(latitude_rad) * np.sin(longitude_rad),
            -np.cos(latitude_rad),
        ]
    )
    e_phi = np.array([-np.sin(longitude_rad), np.cos(longitude_rad), 0.0])
    data = (points @ e_theta) * (points @ e_phi)

    refined = _refine_healpix_quadratic_batch(
        data,
        neighbors,
        np.array([pixels[center_pixel]], dtype=np.int64),
        lats,
        lons,
        is_minimum=True,
    )

    assert spherical_quadratic_status_name(int(refined.status_codes[0])) == (
        "wrong_curvature"
    )
    assert refined.latitudes[0] == pytest.approx(lats[center_pixel])
    assert refined.longitudes[0] == pytest.approx(lons[center_pixel])
    assert np.isfinite(center).all()


def test_healpix_quadratic_rejects_an_ill_conditioned_ring() -> None:
    pixels, lats, lons, points, neighbors = _healpix_geometry()
    center_pixel = 100
    data = 1.0 - points @ points[center_pixel]
    ill_conditioned_neighbors = neighbors.copy()
    ill_conditioned_neighbors[:, center_pixel] = neighbors[0, center_pixel]

    refined = _refine_healpix_quadratic_batch(
        data,
        ill_conditioned_neighbors,
        np.array([pixels[center_pixel]], dtype=np.int64),
        lats,
        lons,
        is_minimum=True,
    )

    assert spherical_quadratic_status_name(int(refined.status_codes[0])) == (
        "singular_or_ill_conditioned_fit"
    )
    assert np.isinf(refined.condition_numbers[0])
    assert refined.latitudes[0] == pytest.approx(lats[center_pixel])
    assert refined.longitudes[0] == pytest.approx(lons[center_pixel])


def test_healpix_quadratic_rejects_a_stationary_point_outside_the_ring() -> None:
    pixels, lats, lons, points, neighbors = _healpix_geometry()
    target = (10.0, 0.0)
    data = 1.0 - points @ _unit_vector(*target)
    center_pixel = int(np.argmin(lats**2 + ((lons + 180.0) % 360.0 - 180.0) ** 2))

    refined = _refine_healpix_quadratic_batch(
        data,
        neighbors,
        np.array([pixels[center_pixel]], dtype=np.int64),
        lats,
        lons,
        is_minimum=True,
    )

    assert spherical_quadratic_status_name(int(refined.status_codes[0])) == (
        "outside_locality"
    )
    assert refined.latitudes[0] == pytest.approx(lats[center_pixel])
    assert refined.longitudes[0] == pytest.approx(lons[center_pixel])


def test_healpix_quadratic_uses_fewer_than_eight_valid_neighbors() -> None:
    pixels, lats, lons, points, neighbors = _healpix_geometry()
    center_pixel = 100
    data = 1.0 - points @ points[center_pixel]
    reduced_neighbors = neighbors.copy()
    valid_slot = int(np.flatnonzero(reduced_neighbors[:, center_pixel] >= 0)[0])
    reduced_neighbors[valid_slot, center_pixel] = -1

    refined = _refine_healpix_quadratic_batch(
        data,
        reduced_neighbors,
        np.array([pixels[center_pixel]], dtype=np.int64),
        lats,
        lons,
        is_minimum=True,
    )

    assert np.count_nonzero(reduced_neighbors[:, center_pixel] >= 0) == 7
    assert spherical_quadratic_status_name(int(refined.status_codes[0])) == "success"
    assert (
        _geodesic_error(
            float(refined.latitudes[0]),
            float(refined.longitudes[0]),
            (float(lats[center_pixel]), float(lons[center_pixel])),
        )
        < 0.05
    )


def test_extract_healpix_centers() -> None:
    data = np.zeros(192, dtype=np.float64)
    data[10] = 950.0
    data[20] = 960.0

    mask = np.zeros(192, dtype=np.float64)
    mask[10] = 1.0
    mask[20] = 1.0

    p_idx, vals = _extract_healpix_centers(mask, data)
    assert len(p_idx) == 2
    assert p_idx[0] == 10
    assert p_idx[1] == 20
    assert vals[0] == 950.0


def test_healpix_detector_detect() -> None:
    pixels, lats, lons, points, _neighbors = _healpix_geometry()
    npix = pixels.size
    p0 = 100
    data_1d = 980.0 + 1000.0 * (1.0 - points @ points[p0])

    data = data_1d.reshape(1, npix)

    da = xr.DataArray(
        data,
        dims=["time", "cell"],
        coords={
            "time": np.array(["2025-01-01"], dtype="datetime64[ns]"),
            "cell": np.arange(npix),
        },
        name="msl",
    )

    detector = HealpixDetector.from_xarray(da)
    # Threshold 1000.0
    raw_results = detector.detect(
        object_threshold=1000.0,
        detection_mode="min",
        min_object_grid_points=1,
    )

    assert len(raw_results) == 1
    _time_val, lats_out, lons_out, values = raw_results[0]
    assert len(lats_out) >= 1
    best = int(np.argmin(values))
    assert (
        _geodesic_error(
            float(lats_out[best]),
            float(lons_out[best]),
            (float(lats[p0]), float(lons[p0])),
        )
        < 0.05
    )
    assert values[best] < 980.01
