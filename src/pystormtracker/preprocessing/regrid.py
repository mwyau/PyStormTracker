from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..backends import Backend, configure_sht_threads, resolve_sht_threads

if TYPE_CHECKING:
    from ..models.geo import MapExtent


class SpectralRegridder:
    """Spectrally regrid among global and regional spherical grids.

    ``ducc0`` supplies the numerical spherical-harmonic transforms and grid
    geometry operations.  Its numerical lineage includes Reinecke and
    Seljebotn (2013), https://doi.org/10.1051/0004-6361/201321494, and Ishioka
    (2018), https://doi.org/10.2151/jmsj.2018-019.  The HEALPix target grid is
    the grid of Górski et al. (2005), https://doi.org/10.1086/427976; PST's
    tracking detector adaptation is documented separately.
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

    def _get_lmax_mmax(
        self, nlon: int, lmax_override: int | None = None
    ) -> tuple[int, int]:
        """Infer lmax and mmax from grid dimensions if not provided."""
        lmax = (
            lmax_override
            if lmax_override is not None
            else self.lmax
            if self.lmax is not None
            else nlon // 2 - 1
        )
        if lmax < 0:
            raise ValueError("lmax must be nonnegative")
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
        sht_threads: int | None = None,
        pl: NDArray[np.int32] | None = None,
    ) -> xr.DataArray:
        """
        Spectrally regrid to a regular 2D grid (CC or GL).
        Supports regular 2D and reduced Gaussian 1D inputs.
        """
        from ..io.data_loader import DataLoader

        variable_name = str(data.name) if data.name is not None else ""
        frame = data.values
        loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
        is_reduced = loader.is_reduced_gaussian(variable_name) or pl is not None

        if not is_reduced and data.ndim != 2:
            raise ValueError("Input must be 2D (lat, lon) or reduced Gaussian 1D grid.")

        if not is_reduced and not lat_reverse:
            frame = frame[::-1, :]

        # Determine input dimensions
        in_nlon: int
        if is_reduced:
            if pl is None:
                pl = loader.get_reduced_grid_pl(variable_name)
            if pl is None:
                raise ValueError("pl array required for reduced grid.")
            in_nlon = int(np.max(pl))
        else:
            in_nlon = frame.shape[1]

        lmax, mmax = self._get_lmax_mmax(in_nlon)
        import ducc0

        nthreads = resolve_sht_threads(sht_threads, "serial")
        configure_sht_threads(nthreads)

        # 1. Analyze (Forward SHT)
        alm: NDArray[np.complex128]
        if is_reduced:
            meta = loader.get_grid_metadata(variable_name)
            alm, _, _, _, _ = ducc0.sht.pseudo_analysis(
                map=np.expand_dims(frame, axis=0),
                spin=0,
                lmax=lmax,
                mmax=mmax,
                theta=meta["theta"],
                nphi=meta["nphi"],
                phi0=meta["phi0"],
                ringstart=meta["ringstart"],
                nthreads=nthreads,
                maxiter=100,
                epsilon=1e-6,
            )
        else:
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
            lat = (
                np.linspace(90, -90, nlat)
                if lat_reverse
                else np.linspace(-90, 90, nlat)
            )
        elif out_geometry == "GL":
            lats_gl = 90.0 - np.degrees(ducc0.misc.GL_thetas(nlat))
            lat = lats_gl if lat_reverse else lats_gl[::-1]
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
        sht_threads: int | None = None,
        pl: NDArray[np.int32] | None = None,
        transform_lmax: int | None = None,
        backend: Backend = "serial",
    ) -> xr.DataArray:
        """
        Spectrally regrid to a 1D HEALPix grid.
        Supports regular 2D/3D and reduced Gaussian inputs, with lazy Dask execution.
        """
        from ..io.data_loader import DataLoader

        variable_name = str(data.name) if data.name is not None else ""
        loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
        is_reduced = loader.is_reduced_gaussian(variable_name) or pl is not None

        # Determine input dimensions
        if is_reduced:
            if pl is None:
                pl = loader.get_reduced_grid_pl(variable_name)
            if pl is None:
                raise ValueError("pl array required for reduced grid.")
            in_nlon = int(np.max(pl))
            spatial_dim = "values" if "values" in data.dims else str(data.dims[-1])
            input_core_dims = [[spatial_dim]]
            meta = loader.get_grid_metadata(variable_name)
            theta_arr = meta["theta"]
            nphi_arr = meta["nphi"]
            phi0_arr = meta["phi0"]
            ringstart_arr = meta["ringstart"]
        else:
            _time_name, lat_dim, lon_dim = loader.get_coords()
            if lat_dim not in data.dims or lon_dim not in data.dims:
                raise ValueError(
                    "Input must have latitude and longitude dimensions or reduced grid."
                )
            in_nlon = int(data.sizes[lon_dim])
            input_core_dims = [[lat_dim, lon_dim]]

        lmax, mmax = self._get_lmax_mmax(in_nlon, transform_lmax)
        import ducc0

        hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
        sht_kwargs = hp_base.sht_info()
        eff_nthreads = resolve_sht_threads(sht_threads, backend)
        configure_sht_threads(eff_nthreads)

        def _healpix_frame(frame: NDArray[np.float64]) -> NDArray[np.float64]:
            if not is_reduced and not lat_reverse:
                frame = frame[::-1, :]
            if is_reduced:
                alm, _, _, _, _ = ducc0.sht.pseudo_analysis(
                    map=np.expand_dims(frame, axis=0),
                    spin=0,
                    lmax=lmax,
                    mmax=mmax,
                    theta=theta_arr,
                    nphi=nphi_arr,
                    phi0=phi0_arr,
                    ringstart=ringstart_arr,
                    nthreads=eff_nthreads,
                    maxiter=100,
                    epsilon=1e-6,
                )
            else:
                alm = ducc0.sht.analysis_2d(
                    map=np.expand_dims(frame, axis=0),
                    spin=0,
                    lmax=lmax,
                    mmax=mmax,
                    geometry=in_geometry,
                    nthreads=eff_nthreads,
                )
            out_map = cast(
                NDArray[np.float64],
                ducc0.sht.synthesis(
                    alm=alm,
                    spin=0,
                    lmax=lmax,
                    mmax=mmax,
                    nthreads=eff_nthreads,
                    **sht_kwargs,
                )[0],
            )
            return out_map

        dask_mode: Literal["forbidden", "allowed", "parallelized"] = (
            "parallelized" if data.chunks and backend == "dask" else "allowed"
        )

        res = cast(
            xr.DataArray,
            xr.apply_ufunc(
                _healpix_frame,
                data,
                input_core_dims=input_core_dims,
                output_core_dims=[["cell"]],
                vectorize=True,
                dask=dask_mode,
                output_dtypes=[np.float64],
                dask_gufunc_kwargs={"output_sizes": {"cell": hp_base.npix()}},
            ),
        )

        cells = np.arange(hp_base.npix())
        res = res.assign_coords(cell=cells)
        res.name = data.name
        attrs = dict(data.attrs)
        attrs["grid_type"] = "healpix"
        attrs["nside"] = nside
        res.attrs = attrs
        return res

    def to_polar_stereo(
        self,
        data: xr.DataArray,
        hemisphere: Literal["nh", "sh"] = "nh",
        extent: MapExtent = (-13000.0, 13000.0, -13000.0, 13000.0),
        stereo_grid_spacing_km: float = 100.0,
        lon_0: float = 0.0,
        transform_lmax: int | None = None,
        in_geometry: Literal["CC", "GL"] = "CC",
        lat_reverse: bool = False,
        sht_threads: int | None = None,
        backend: Literal["serial", "mpi", "dask"] = "serial",
    ) -> xr.DataArray:
        """
        Spectrally regrid to a Polar Stereographic grid with lazy Dask support.

        Args:
            extent: Bounding box from pole in km (xmin, xmax, ymin, ymax).
            stereo_grid_spacing_km: Grid spacing in km.
            transform_lmax: Maximum total wave number for the transform.
        """
        from ..io.data_loader import DataLoader
        from ..models.geo import R_EARTH_KM

        loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
        _time_name, lat_dim, lon_dim = loader.get_coords()
        if lat_dim not in data.dims or lon_dim not in data.dims:
            raise ValueError(
                "Input must have latitude and longitude dimensions for "
                "polar stereo regridding."
            )

        in_nlon = int(data.sizes[lon_dim])
        lmax, mmax = self._get_lmax_mmax(in_nlon, transform_lmax)
        eff_nthreads = resolve_sht_threads(sht_threads, backend)
        configure_sht_threads(eff_nthreads)

        # Coordinate generation once
        xmin, xmax, ymin, ymax = extent
        nx = int(np.round((xmax - xmin) / stereo_grid_spacing_km)) + 1
        ny = int(np.round((ymax - ymin) / stereo_grid_spacing_km)) + 1

        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)

        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2 + Y**2)

        if hemisphere == "nh":
            theta = 2.0 * np.arctan(rho / (2.0 * R_EARTH_KM))
            phi = (np.radians(lon_0) + np.arctan2(X, -Y)) % (2 * np.pi)
        else:
            theta = np.pi - 2.0 * np.arctan(rho / (2.0 * R_EARTH_KM))
            phi = (np.radians(lon_0) + np.arctan2(X, Y)) % (2 * np.pi)

        valid = rho <= 2.0 * R_EARTH_KM
        latitude_names = ("lat", "latitude")
        latitude_coord = next(
            (data.coords[name] for name in latitude_names if name in data.coords),
            None,
        )
        if latitude_coord is not None and latitude_coord.size:
            latitude_values = np.asarray(latitude_coord.values, dtype=np.float64)
            valid &= np.degrees(theta) >= 90.0 - float(np.max(latitude_values))
            valid &= np.degrees(theta) <= 90.0 - float(np.min(latitude_values))

        loc = np.stack([theta.ravel(), phi.ravel()], axis=-1)

        def _polar_stereo_frame(frame: NDArray[np.float64]) -> NDArray[np.float64]:
            import ducc0

            if not lat_reverse:
                frame = frame[::-1, :]
            alm = ducc0.sht.analysis_2d(
                map=np.expand_dims(frame, axis=0),
                spin=0,
                lmax=lmax,
                mmax=mmax,
                geometry=in_geometry,
                nthreads=eff_nthreads,
            )
            out_map = cast(
                NDArray[np.float64],
                ducc0.sht.synthesis_general(
                    alm=alm,
                    loc=loc,
                    lmax=lmax,
                    mmax=mmax,
                    spin=0,
                    epsilon=1e-6,
                    nthreads=eff_nthreads,
                )[0],
            )
            out_map = out_map.reshape(ny, nx)
            out_map[~valid] = 0.0
            return out_map

        dask_mode: Literal["forbidden", "allowed", "parallelized"] = (
            "parallelized" if data.chunks and backend == "dask" else "allowed"
        )

        res = cast(
            xr.DataArray,
            xr.apply_ufunc(
                _polar_stereo_frame,
                data,
                input_core_dims=[[lat_dim, lon_dim]],
                output_core_dims=[["y", "x"]],
                vectorize=True,
                dask=dask_mode,
                output_dtypes=[np.float64],
                dask_gufunc_kwargs={"output_sizes": {"y": ny, "x": nx}},
            ),
        )

        res = res.assign_coords(y=y, x=x)
        res.name = data.name
        attrs = dict(data.attrs)
        attrs.update(
            {
                "projection": f"{hemisphere}_stereo",
                "stereo_grid_spacing_km": stereo_grid_spacing_km,
                "lmax": lmax,
                "source_domain_mask": "rho <= 2R and native latitude bounds",
            }
        )
        res.attrs = attrs
        return res
