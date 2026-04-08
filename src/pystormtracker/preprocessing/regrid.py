from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import ducc0
import numpy as np
import xarray as xr
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..models.geo import MapExtent


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

        if not lat_reverse:
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

        if not lat_reverse:
            out_map = out_map[::-1, :]

        # 3. Reconstruct DataArray
        if out_geometry == "CC":
            # ducc0 CC internally works North to South (0 to pi)
            if lat_reverse:
                # We kept it North to South
                lat = np.linspace(90, -90, nlat)
            else:
                # We flipped it back to South to North
                lat = np.linspace(-90, 90, nlat)
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

        if not lat_reverse:
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

    def to_polar_stereo(
        self,
        data: xr.DataArray,
        hemisphere: Literal["nh", "sh"] = "nh",
        extent: MapExtent = (-13000.0, 13000.0, -13000.0, 13000.0),
        resolution: float = 100.0,
        lon_0: float = 0.0,
        filter_lmin: int | None = None,
        in_geometry: Literal["CC", "GL"] = "CC",
        lat_reverse: bool = False,
        nthreads: int = 1,
    ) -> xr.DataArray:
        """
        Spectrally regrid to a Polar Stereographic grid.

        Args:
            extent: Bounding box from pole in km (xmin, xmax, ymin, ymax).
            resolution: Grid spacing in km.
        """
        from ..models.constants import R_EARTH_KM
        from .spectral import apply_bandpass_mask_to_alm

        frame = data.values
        if data.ndim != 2:
            raise ValueError(
                "Only 2D (lat, lon) data is currently supported for regridding."
            )

        if not lat_reverse:
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

        if filter_lmin is not None:
            apply_bandpass_mask_to_alm(alm, filter_lmin, lmax, mmax)

        # 2. Coordinate Generation
        xmin, xmax, ymin, ymax = extent
        # We need the number of points. To match extent precisely, use linspace
        # or calculate n_points based on extent and resolution.
        # Let's use linspace for robustness if extent does not perfectly divide.
        nx = int(np.round((xmax - xmin) / resolution)) + 1
        ny = int(np.round((ymax - ymin) / resolution)) + 1

        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)

        # Note: matrix 'ij' indexing vs 'xy'. Usually map is (y, x)
        X, Y = np.meshgrid(x, y)

        rho = np.sqrt(X**2 + Y**2)

        if hemisphere == "nh":
            theta = 2.0 * np.arctan(rho / (2.0 * R_EARTH_KM))
            phi = (np.radians(lon_0) + np.arctan2(X, -Y)) % (2 * np.pi)
        else:
            theta = np.pi - 2.0 * np.arctan(rho / (2.0 * R_EARTH_KM))
            phi = (np.radians(lon_0) + np.arctan2(X, Y)) % (2 * np.pi)

        # 3. Synthesize directly to these arbitrary points
        # ducc0 synthesis_general expects loc array of shape (N, 2)
        loc = np.stack([theta.ravel(), phi.ravel()], axis=-1)
        out_map = cast(
            NDArray[np.float64],
            ducc0.sht.synthesis_general(
                alm=alm,
                loc=loc,
                lmax=lmax,
                mmax=mmax,
                spin=0,
                epsilon=1e-6,
                nthreads=nthreads,
            )[0],
        )

        # Reshape back to 2D
        out_map = out_map.reshape(ny, nx)

        return xr.DataArray(
            out_map,
            dims=["y", "x"],
            coords={"y": y, "x": x},
            name=data.name,
            attrs={"projection": f"{hemisphere}_stereo", "resolution_km": resolution},
        )
