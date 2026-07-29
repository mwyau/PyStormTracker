from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import ClassVar, cast

import ducc0
import numpy as np
import xarray as xr


class DataLoader:
    """
    Handles optimized xarray loading for local and remote datasets.
    Supports NetCDF, GRIB, and Zarr formats, with thread-safe caching.
    """

    _ds_cache: ClassVar[dict[str, xr.Dataset]] = {}
    _ds_lock: ClassVar[threading.Lock] = threading.Lock()

    # Common variable and coordinate name aliases
    VAR_MAPPING: ClassVar[dict[str, list[str]]] = {
        "msl": ["msl", "slp"],
        "vo": ["vo"],
        "latitude": ["latitude", "lat", "y"],
        "longitude": ["longitude", "lon", "x"],
        "time": ["time", "valid_time"],
    }

    def __init__(
        self,
        pathname: str | Path | xr.DataArray | xr.Dataset | None = None,
        engine: str | None = None,
    ) -> None:
        self.engine = engine
        self._ds: xr.Dataset | None = None
        self.pathname: str | Path | None

        if isinstance(pathname, (xr.DataArray, xr.Dataset)):
            self.pathname = None
            if isinstance(pathname, xr.DataArray):
                if pathname.name is None:
                    pathname = pathname.rename("data")
                self._ds = pathname.to_dataset()
            else:
                self._ds = pathname
        elif pathname is None:
            self.pathname = None
        elif isinstance(pathname, str) and "://" in pathname:
            self.pathname = pathname
        else:
            self.pathname = Path(pathname)

    def ensure_open(self) -> xr.Dataset:
        """Ensures the xarray dataset is open and returns it."""
        if self._ds is None:
            if self.pathname is None:
                raise ValueError(
                    "Cannot open dataset without a valid pathname or data object."
                )
            with self._ds_lock:
                cache_key = str(self.pathname)
                if cache_key not in self._ds_cache:
                    engine = self.engine
                    storage_options: dict[str, bool] = {}
                    is_remote = isinstance(self.pathname, str) and (
                        "://" in self.pathname
                    )

                    if is_remote and str(self.pathname).startswith(
                        ("http://", "https://")
                    ):
                        # fsspec handles anon HTTP by default; no special 'anon'
                        # key needed.
                        pass
                    elif is_remote and str(self.pathname).startswith(
                        ("s3://", "gs://")
                    ):
                        storage_options = {"anon": True}

                    if engine is None:
                        if is_remote:
                            pathname_str = str(self.pathname)
                            if pathname_str.endswith(".zarr"):
                                if importlib.util.find_spec("zarr") is None:
                                    raise ValueError(
                                        "zarr is required to open Zarr datasets. "
                                        "Please install it with: `uv pip install "
                                        "'pystormtracker[zarr]'`"
                                    ) from None
                                engine = "zarr"
                            elif pathname_str.endswith((".grib", ".grib2", ".grb")):
                                if importlib.util.find_spec("cfgrib") is None:
                                    raise ValueError(
                                        "cfgrib is required to open GRIB files. "
                                        "Please install it with: `uv pip install "
                                        "'pystormtracker[grib]'`"
                                    ) from None
                                engine = "cfgrib"
                            else:
                                # Default for remote that aren't zarr or grib
                                # (e.g., .nc)
                                engine = "h5netcdf"
                        else:
                            # Handle local paths
                            local_path = Path(self.pathname)
                            ext = local_path.suffix.lower()
                            if ext in [".grib", ".grib2", ".grb"]:
                                if importlib.util.find_spec("cfgrib") is None:
                                    raise ValueError(
                                        "cfgrib is required to open GRIB files. "
                                        "Please install it with: `uv pip install "
                                        "'pystormtracker[grib]'`"
                                    ) from None
                                engine = "cfgrib"
                            elif ext == ".zarr" or (
                                local_path.is_dir()
                                and (local_path / ".zmetadata").exists()
                            ):
                                if importlib.util.find_spec("zarr") is None:
                                    raise ValueError(
                                        "zarr is required to open Zarr datasets. "
                                        "Please install it with: `uv pip install "
                                        "'pystormtracker[zarr]'`"
                                    ) from None
                                engine = "zarr"
                            else:
                                # Standard xarray detection for everything else
                                engine = None

                    if engine == "zarr" and is_remote and storage_options:
                        self._ds_cache[cache_key] = xr.open_dataset(
                            self.pathname,
                            engine=engine,
                            chunks={},
                            storage_options=storage_options,
                        )
                    else:
                        self._ds_cache[cache_key] = xr.open_dataset(
                            self.pathname,
                            engine=engine,
                            chunks={},
                        )

                self._ds = self._ds_cache[cache_key]
        return self._ds

    def get_coords(self) -> tuple[str, str, str]:
        """Returns the mapped names for (time, lat, lon)."""
        ds = self.ensure_open()
        coords = ds.coords

        time_name = next((c for c in self.VAR_MAPPING["time"] if c in coords), "time")
        lat_name = next(
            (c for c in self.VAR_MAPPING["latitude"] if c in coords), "latitude"
        )
        lon_name = next(
            (c for c in self.VAR_MAPPING["longitude"] if c in coords), "longitude"
        )

        return time_name, lat_name, lon_name

    def is_lat_reversed(self) -> bool:
        """
        Detects if the latitude coordinate is North-to-South (reversed).
        Returns True if lat[0] > lat[-1].
        """
        ds = self.ensure_open()
        _, lat_name, _ = self.get_coords()
        if lat_name in ds.coords and len(ds[lat_name]) > 1:
            return bool(ds[lat_name][0] > ds[lat_name][-1])
        return False

    def is_reduced_gaussian(self, varname: str | None = None) -> bool:
        """Detects if the dataset represents a reduced Gaussian grid."""
        ds = self.ensure_open()
        # If varname not provided, check the first data variable
        if varname is None:
            varname = cast(str, next(iter(ds.data_vars))) if ds.data_vars else None

        if varname and varname in ds:
            da = ds[varname]
            # cfgrib tags reduced Gaussian grids with this attribute
            if da.attrs.get("GRIB_gridType") == "reduced_gg":
                return True
            # Alternative: check if latitude/longitude are 1D coordinates of a
            # non-spatial dimension
            if "values" in da.dims and da.ndim == 2:  # (time, values)
                return True
        return False

    def get_reduced_grid_pl(self, varname: str | None = None) -> np.ndarray | None:
        """Returns the 'pl' array (points per latitude) for a reduced grid."""
        ds = self.ensure_open()
        if varname is None:
            varname = cast(str, next(iter(ds.data_vars))) if ds.data_vars else None

        if varname and varname in ds:
            da = ds[varname]
            pl = da.attrs.get("GRIB_pl")
            if pl is not None:
                return np.array(pl, dtype=np.int32)
        return None

    def _get_theta(self, ntheta: int, geometry: str) -> np.ndarray:
        """Calculates colatitudes (theta) for a given geometry and resolution."""
        if geometry == "GL":
            # ducc0.misc.GL_thetas returns North-to-South (0 to pi)
            return cast(np.ndarray, ducc0.misc.GL_thetas(ntheta))
        if geometry == "CC":
            return np.linspace(0, np.pi, ntheta)
        # Default to equidistant
        return np.linspace(0, np.pi, ntheta)

    def get_grid_metadata(self, varname: str | None = None) -> dict[str, np.ndarray]:
        """
        Returns grid metadata (theta, nphi, phi0, ringstart) for SHT.
        Works for reduced Gaussian and HEALPix grids.
        """
        ds = self.ensure_open()
        if varname is None:
            varname = cast(str, next(iter(ds.data_vars))) if ds.data_vars else None

        da = ds[varname] if varname else next(iter(ds.data_vars.values()))

        # 1. Check for HEALPix
        if da.attrs.get("grid_type") == "healpix" or "cell" in da.dims:
            npix = da.sizes.get("cell", da.sizes.get("values", 0))
            nside = int(np.sqrt(npix / 12))
            hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
            return cast(dict[str, np.ndarray], hp_base.sht_info())

        # 2. Check for Reduced Gaussian
        if self.is_reduced_gaussian(varname):
            pl = self.get_reduced_grid_pl(varname)
            if pl is not None:
                # Gaussian latitudes for N rings
                ntheta = len(pl)
                theta = self._get_theta(ntheta, "GL")
                phi0 = np.zeros(ntheta, dtype=np.float64)
                ringstart = np.concatenate(([0], np.cumsum(pl)[:-1])).astype(np.uint64)
                return {
                    "theta": theta,
                    "nphi": pl.astype(np.uint64),
                    "phi0": phi0,
                    "ringstart": ringstart,
                }

        # 3. Default: regular grid (handled by analysis_2d, but provide here too)
        _time_name, lat_name, lon_name = self.get_coords()
        lat = da[lat_name].values
        lon = da[lon_name].values

        if self.is_lat_reversed():
            theta = np.radians(90.0 - lat)
        else:
            theta = np.radians(90.0 - lat[::-1])

        ntheta, nphi_val = len(lat), len(lon)
        nphi = np.full(ntheta, nphi_val, dtype=np.uint64)
        phi0 = np.zeros(ntheta, dtype=np.float64)
        ringstart = (np.arange(ntheta) * nphi_val).astype(np.uint64)

        return {
            "theta": theta,
            "nphi": nphi,
            "phi0": phi0,
            "ringstart": ringstart,
        }
