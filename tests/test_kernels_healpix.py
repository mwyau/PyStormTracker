from __future__ import annotations

import ducc0
import numpy as np

from pystormtracker.healpix.kernels import (
    _numba_get_healpix_centers,
    _numba_healpix_extrema,
)


def test_numba_healpix_extrema() -> None:
    # Nside = 4 -> 192 pixels
    nside = 4
    npix = 12 * nside**2
    # Perturbed background to avoid plateaus
    data = np.ones(npix, dtype=np.float64) * 100.0 + np.arange(npix) * 0.001
    # Minimum at pixel 50
    data[50] = 50.0

    # Neighbor lookup
    hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
    # ducc0 neighbors(pix) returns (N, 8)
    all_pix = np.arange(npix, dtype=np.int64)
    nbors = hp_base.neighbors(all_pix).T # Shape (8, N)

    # Test for local minimum, threshold 95.0 (so only the 50.0 point passes)
    out = _numba_healpix_extrema(data, nbors, threshold=95.0, is_min=True)
    assert out[50] == 1.0
    assert np.sum(out) == 1.0

    # Test for local maximum (50 will not be a maximum, but some other point will be)
    out_max = _numba_healpix_extrema(data, nbors, threshold=50.0, is_min=False)
    # Background is increasing, so the last pixel or some boundary pixels will be maxima
    assert np.sum(out_max) > 0.0
    assert out_max[50] == 0.0

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
