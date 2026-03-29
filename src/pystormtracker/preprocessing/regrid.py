from __future__ import annotations

from typing import Literal, cast

import ducc0
import numpy as np
import xarray as xr
from numpy.typing import NDArray


class SpectralRegridder:
    """
    Spectral regridder for transforming data between Clenshaw-Curtis (CC),
    Gauss-Legendre (GL), and HEALPix grids using ducc0 SHT.
    """

    def __init__(
        self,
        lmax: int | None = None,
        mmax: int | None = None,
    ) -> None:
        """
        Initialize the regridder.

        Args:
            lmax: Maximum total wave number for spectral transform. If None,
                  it will be inferred from the input grid resolution.
            mmax: Maximum zonal wave number. If None, defaults to lmax.
        """
        self.lmax = lmax
        self.mmax = mmax

    def _get_lmax_mmax(self, nlon: int) -> tuple[int, int]:
        """Infer lmax and mmax from grid dimensions if not provided."""
        lmax = self.lmax if self.lmax is not None else nlon // 2 - 1
        mmax = self.mmax if self.mmax is not None else lmax
        return lmax, min(mmax, lmax)

    def to_grid(
        self,
        data: xr.DataArray,
        nlat: int,
        nlon: int,
        in_geometry: Literal["CC", "GL"] = "CC",
        out_geometry: Literal["CC", "GL"] = "CC",
        lat_reverse: bool = False,
        nthreads: int = 1,
    ) -> xr.DataArray:
        """
        Spectrally regrid to a regular 2D grid (CC or GL).
        """
        frame = data.values
        if data.ndim != 2:
            raise ValueError(
                "Only 2D (lat, lon) data is currently supported for regridding."
            )

        if lat_reverse:
            frame = frame[::-1, :]

        _in_nlat, in_nlon = frame.shape
        lmax, mmax = self._get_lmax_mmax(in_nlon)

        # 1. Analyze (Forward SHT)
        alm = ducc0.sht.analysis_2d(
            map=np.expand_dims(frame, axis=0),
            spin=0,
            lmax=lmax,
            mmax=mmax,
            geometry=in_geometry,
            nthreads=nthreads,
        )

        # 2. Synthesize (Inverse SHT to target grid)
        out_map = cast(
            NDArray[np.float64],
            ducc0.sht.synthesis_2d(
                alm=alm,
                spin=0,
                lmax=lmax,
                mmax=mmax,
                ntheta=nlat,
                nphi=nlon,
                geometry=out_geometry,
                nthreads=nthreads,
            )[0],
        )

        if lat_reverse:
            out_map = out_map[::-1, :]

        # 3. Reconstruct DataArray
        if out_geometry == "CC":
            # ducc0 CC internally works North to South (0 to pi)
            if lat_reverse:
                # We flipped it back to South to North
                lat = np.linspace(-90, 90, nlat)
            else:
                # It remains North to South
                lat = np.linspace(90, -90, nlat)
        elif out_geometry == "GL":
            # For GL we don't easily have lat bounds in numpy without ducc0 help
            lat = np.arange(nlat, dtype=np.float64)
        else:
            lat = np.arange(nlat, dtype=np.float64)

        lon = np.linspace(0, 360, nlon, endpoint=False)

        return xr.DataArray(
            out_map,
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
            name=data.name,
        )

    def to_healpix(
        self,
        data: xr.DataArray,
        nside: int,
        in_geometry: Literal["CC", "GL"] = "CC",
        lat_reverse: bool = False,
        nthreads: int = 1,
    ) -> xr.DataArray:
        """
        Spectrally regrid to a 1D HEALPix grid.
        """
        frame = data.values
        if data.ndim != 2:
            raise ValueError(
                "Only 2D (lat, lon) data is currently supported for regridding."
            )

        if lat_reverse:
            frame = frame[::-1, :]

        _, in_nlon = frame.shape
        lmax, mmax = self._get_lmax_mmax(in_nlon)

        # 1. Analyze
        alm = ducc0.sht.analysis_2d(
            map=np.expand_dims(frame, axis=0),
            spin=0,
            lmax=lmax,
            mmax=mmax,
            geometry=in_geometry,
            nthreads=nthreads,
        )

        # 2. Synthesize to HEALPix
        hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
        sht_kwargs = hp_base.sht_info()

        # synthesis returns shape (nmaps, npix)
        out_map = cast(
            NDArray[np.float64],
            ducc0.sht.synthesis(
                alm=alm, spin=0, lmax=lmax, mmax=mmax, nthreads=nthreads, **sht_kwargs
            )[0],
        )

        # 3. Reconstruct DataArray
        cells = np.arange(hp_base.npix())

        return xr.DataArray(
            out_map, dims=["cell"], coords={"cell": cells}, name=data.name
        )
