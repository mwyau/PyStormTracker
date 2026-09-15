from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import spharmgrid as sg
import xarray as xr
from numpy.typing import NDArray

from ..backends import Backend, configure_sht_threads, resolve_sht_threads

if TYPE_CHECKING:
    from ..models.geo import MapExtent


def _spharmgrid_target_grid(
    geometry: Literal["CC", "GL"],
    nlat: int,
    nlon: int,
    *,
    latitude_order: Literal["ascending", "descending"],
) -> sg.Grid:
    """Construct a public spharmgrid grid for ``SpectralRegridder.to_grid``."""
    if geometry == "CC":
        return sg.clenshaw_curtis_grid(
            nlat,
            nlon,
            latitude_order=latitude_order,
        )
    if geometry == "GL":
        return sg.gaussian_grid(
            nlat,
            nlon,
            latitude_order=latitude_order,
        )
    raise ValueError(f"unsupported regular output geometry {geometry!r}")


def _spharmgrid_threads(sht_threads: int | None, data: xr.DataArray) -> int | None:
    """Map the legacy serial default to spharmgrid's eager-operation default."""
    resolved = resolve_sht_threads(
        sht_threads,
        "dask" if data.chunks is not None else "serial",
    )
    return None if resolved == 0 else resolved


def _to_grid_spharmgrid(
    data: xr.DataArray,
    *,
    nlat: int,
    nlon: int,
    out_geometry: Literal["CC", "GL"],
    lat_reverse: bool,
    lmax: int | None,
    sht_threads: int | None,
) -> xr.DataArray:
    """Adapt ``SpectralRegridder.to_grid`` to public spharmgrid regridding."""
    target = _spharmgrid_target_grid(
        out_geometry,
        nlat,
        nlon,
        latitude_order="descending" if lat_reverse else "ascending",
    )
    result = sg.regrid(
        data,
        target,
        truncation=None if lmax is None else f"T{lmax}",
        sht_threads=_spharmgrid_threads(sht_threads, data),
    )

    # ``to_grid`` historically returns a two-dimensional DataArray with the
    # canonical ``lat``/``lon`` dimensions and no inherited field attributes.
    latitude_name = _coordinate_name(data, "latitude")
    longitude_name = _coordinate_name(data, "longitude")
    return xr.DataArray(
        result.data,
        dims=("lat", "lon"),
        coords={
            "lat": np.asarray(result[latitude_name].values),
            "lon": np.asarray(result[longitude_name].values),
        },
        name=data.name,
    )


def _coordinate_name(
    data: xr.DataArray,
    axis: Literal["latitude", "longitude"],
) -> str:
    """Find the coordinate name used by spharmgrid for a supported field."""
    aliases = (axis, "lat") if axis == "latitude" else (axis, "lon")
    for name in aliases:
        if name in data.coords and data[name].ndim == 1:
            return name
    for coordinate_name, coordinate in data.coords.items():
        if coordinate.ndim == 1 and coordinate.attrs.get("standard_name") == axis:
            return str(coordinate_name)
    raise ValueError(f"could not identify {axis} coordinate")


class SpectralRegridder:
    """Spectrally regrid among global and regional spherical grids.

    Public ``spharmgrid`` supplies supported rectangular spherical-harmonic
    regridding. Reduced-Gaussian, HEALPix, and polar paths use direct
    ``ducc0``. Its numerical lineage includes Reinecke and Seljebotn (2013),
    https://doi.org/10.1051/0004-6361/201321494, and Ishioka (2018),
    https://doi.org/10.2151/jmsj.2018-019. The HEALPix target grid is
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
                  regular CC/GL regridding uses ``spharmgrid``'s supported
                  transform bandwidth for the source and target grids; special
                  direct-DUCC paths infer bandwidth as appropriate for those
                  paths.
            mmax: Maximum zonal wave number for special direct-grid paths. A
                  non-triangular value is unsupported for regular CC/GL grids.
        """
        if lmax is not None and lmax < 0:
            raise ValueError("lmax must be nonnegative")
        if mmax is not None and mmax < 0:
            raise ValueError("mmax must be nonnegative")
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
        if mmax > lmax:
            raise ValueError("mmax cannot exceed lmax")
        return lmax, mmax

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
        loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
        is_reduced = loader.is_reduced_gaussian(variable_name) or pl is not None

        if not is_reduced and data.ndim == 2:
            if self.mmax is not None:
                if self.lmax is None:
                    raise ValueError(
                        "lmax is required when mmax is specified for "
                        "regular CC/GL regridding"
                    )
                if self.mmax != self.lmax:
                    raise ValueError(
                        "mmax must equal lmax for regular CC/GL regridding; "
                        "non-triangular selections are not supported by the "
                        "public spharmgrid API"
                    )
            grid = sg.detect_grid(data)
            if in_geometry.lower() != grid.kind:
                raise ValueError(
                    f"in_geometry={in_geometry!r} does not match the "
                    f"coordinate-defined {grid.kind.upper()} grid"
                )
            return _to_grid_spharmgrid(
                data,
                nlat=nlat,
                nlon=nlon,
                out_geometry=out_geometry,
                lat_reverse=lat_reverse,
                lmax=self.lmax,
                sht_threads=sht_threads,
            )

        if not is_reduced and data.ndim != 2:
            raise ValueError("Input must be 2D (lat, lon) or reduced Gaussian 1D grid.")

        if pl is None:
            pl = loader.get_reduced_grid_pl(variable_name)
        if pl is None:
            raise ValueError("pl array required for reduced grid.")
        frame = data.values
        in_nlon = int(np.max(pl))

        lmax, mmax = self._get_lmax_mmax(in_nlon)
        import ducc0

        nthreads = resolve_sht_threads(sht_threads, "serial")
        configure_sht_threads(nthreads)

        # 1. Analyze (Forward SHT)
        alm: NDArray[np.complex128]
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

        # 3. Reconstruct DataArray with the public grid coordinate convention.
        target = _spharmgrid_target_grid(
            out_geometry,
            nlat,
            nlon,
            latitude_order="descending" if lat_reverse else "ascending",
        )

        return xr.DataArray(
            out_map,
            dims=["lat", "lon"],
            coords={"lat": target.latitude, "lon": target.longitude},
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
