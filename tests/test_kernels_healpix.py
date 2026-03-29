from __future__ import annotations

import ducc0
import numpy as np

from pystormtracker.healpix.kernels import (
    _numba_get_healpix_centers,
    _numba_healpix_ccl,
    _numba_healpix_object_extrema,
    subgrid_refine_healpix,
)


def test_numba_healpix_ccl() -> None:
    nside = 4
    npix = 12 * nside**2
    data = np.zeros(npix, dtype=np.float64)
    # Create two disjoint objects
    data[10] = 100.0
    data[11] = 100.0  # neighbor
    data[50] = 100.0

    hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
    all_pix = np.arange(npix, dtype=np.int64)
    nbors = hp_base.neighbors(all_pix).T

    labels, num_objects = _numba_healpix_ccl(data, nbors, threshold=50.0, is_min=False)

    assert num_objects == 2
    assert labels[10] == labels[11]
    assert labels[10] != labels[50]
    assert labels[10] > 0
    assert labels[50] > 0


def test_numba_healpix_object_extrema() -> None:
    nside = 4
    npix = 12 * nside**2
    data = np.zeros(npix, dtype=np.float64)
    data[10] = 100.0
    data[11] = 90.0  # center of max
    data[12] = 80.0

    hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
    all_pix = np.arange(npix, dtype=np.int64)
    nbors = hp_base.neighbors(all_pix).T

    labels, num_objects = _numba_healpix_ccl(data, nbors, threshold=50.0, is_min=False)
    extrema = _numba_healpix_object_extrema(
        data, nbors, labels, num_objects, is_min=False, min_points=1
    )

    assert extrema[10] == 1.0

    assert np.sum(extrema) == 1.0


def test_subgrid_refine_healpix() -> None:
    nside = 16
    npix = 12 * nside**2
    data = np.zeros(npix, dtype=np.float64)

    hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
    all_pix = np.arange(npix, dtype=np.int64)
    nbors = hp_base.neighbors(all_pix).T

    # Get coordinates
    ang = hp_base.pix2ang(all_pix)
    lats = 90.0 - np.degrees(ang[:, 0])
    lons = np.degrees(ang[:, 1])

    # Center pixel
    p0 = 100
    lat0_rad = np.radians(lats[p0])
    lon0_rad = np.radians(lons[p0])

    # Create a smooth peak in RADIAN space around p0
    for i in range(npix):
        lat_i = np.radians(lats[i])
        lon_i = np.radians(lons[i])
        dlat = lat_i - lat0_rad
        dlon = (lon_i - lon0_rad) * np.cos(lat0_rad)
        dist_sq = dlat**2 + dlon**2
        if dist_sq < 0.01:  # ~5 degrees
            data[i] = 100.0 - 1000.0 * dist_sq

    ref_lat, ref_lon, ref_val = subgrid_refine_healpix(data, p0, nbors, lats, lons)

    # Should be very close to original since it's the exact peak
    assert abs(ref_lat - lats[p0]) < 0.1
    assert abs(ref_lon - lons[p0]) < 0.1
    assert ref_val >= 99.9


def test_numba_get_healpix_centers() -> None:
    data = np.zeros(192, dtype=np.float64)
    data[10] = 950.0
    data[20] = 960.0

    mask = np.zeros(192, dtype=np.float64)
    mask[10] = 1.0
    mask[20] = 1.0

    p_idx, vals = _numba_get_healpix_centers(mask, data)
    assert len(p_idx) == 2
    assert p_idx[0] == 10
    assert p_idx[1] == 20
    assert vals[0] == 950.0
    assert vals[1] == 960.0
