from __future__ import annotations

import logging
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import ClassVar, Literal, TypedDict, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..models.time import (
    TimeInput,
    encode_numeric_time_values,
    encode_time_values,
    infer_calendar,
    select_time_range,
)

DataLoaderChunks = int | str | tuple[int, ...] | dict[str, int | str | None] | None

LOGGER = logging.getLogger(__name__)


def _normalize_chunks_key(
    chunks: object,
) -> tuple[tuple[str, int], ...] | tuple[object, ...] | str | None:
    """Normalize chunk configurations for deterministic cache keys."""
    if chunks is None:
        return None
    if isinstance(chunks, dict):
        return tuple(
            sorted(
                (
                    str(k),
                    int(v) if isinstance(v, (int, np.integer)) else str(v),
                )
                for k, v in chunks.items()
            )
        )
    if isinstance(chunks, (tuple, list)):
        return tuple(
            int(x) if isinstance(x, (int, np.integer)) else str(x) for x in chunks
        )
    if isinstance(chunks, (int, str)):
        return (chunks,)
    return str(chunks)


class GridMetadata(TypedDict):
    theta: NDArray[np.float64]
    nphi: NDArray[np.uint64]
    phi0: NDArray[np.float64]
    ringstart: NDArray[np.uint64]


class DataLoader:
    """
    Handles optimized xarray loading for local and remote datasets.
    Supports NetCDF, GRIB, and Zarr formats, with thread-safe caching.

    NetCDF uses ``h5netcdf`` by default. Legacy NetCDF3 inputs must be opened
    with ``engine="netcdf4"`` explicitly because ``h5netcdf`` reads HDF5-based
    NetCDF4 files only.
    """

    _ds_cache: ClassVar[dict[tuple[str, str | None, object], xr.Dataset]] = {}
    _ds_lock: ClassVar[threading.Lock] = threading.Lock()

    # Common variable and coordinate name aliases
    _NAME_ALIASES: ClassVar[dict[str, list[str]]] = {
        "msl": ["msl", "slp"],
        "vo": ["vo"],
        "latitude": ["latitude", "lat", "y"],
        "longitude": ["longitude", "lon", "x"],
        "time": ["time", "valid_time"],
    }

    def find_coordinate_dimension(
        self,
        ds: xr.Dataset | xr.DataArray,
        dim_type: Literal["time", "latitude", "longitude"],
    ) -> str | None:
        """Find the coordinate dimension name for a given dimension type."""
        aliases = self._NAME_ALIASES.get(dim_type, [dim_type])
        for alias in aliases:
            if alias in ds.coords or alias in ds.dims:
                return alias
        return None

    def find_variable_name(self, ds: xr.Dataset, canonical_name: str) -> str | None:
        """Find the variable name in a dataset given a canonical name."""
        aliases = self._NAME_ALIASES.get(canonical_name, [canonical_name])
        for alias in aliases:
            if alias in ds.data_vars:
                return alias
        return None

    def resolve_variable_name(self, ds: xr.Dataset, requested_name: str) -> str | None:
        """Resolve a requested name to the actual variable name in a dataset."""
        if requested_name in ds.data_vars:
            return requested_name
        return self.find_variable_name(ds, requested_name)

    def __init__(
        self,
        pathname: str | Path | xr.DataArray | xr.Dataset | None = None,
        engine: str | None = None,
        chunks: DataLoaderChunks = None,
    ) -> None:
        self.engine = engine
        self.chunks = chunks
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
            self._ds = self._normalize_time_coordinate(self._ds)
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
                engine = self.engine
                if engine == "netcdf4" and find_spec("netCDF4") is None:
                    raise ValueError(
                        "netCDF4 is required for engine='netcdf4'. "
                        "Please install it with: `uv pip install "
                        "'pystormtracker[netcdf4]'`"
                    )
                is_remote = isinstance(self.pathname, str) and ("://" in self.pathname)
                storage_options: dict[str, bool] = {}
                if is_remote and str(self.pathname).startswith(("s3://", "gs://")):
                    storage_options = {"anon": True}

                if engine is None:
                    if is_remote:
                        pathname_str = str(self.pathname)
                        if pathname_str.endswith(".zarr"):
                            if find_spec("zarr") is None:
                                raise ValueError(
                                    "zarr is required to open Zarr datasets. "
                                    "Please install it with: `uv pip install "
                                    "'pystormtracker[zarr]'`"
                                ) from None
                            engine = "zarr"
                        elif pathname_str.endswith((".grib", ".grib2", ".grb")):
                            if find_spec("cfgrib") is None:
                                raise ValueError(
                                    "cfgrib is required to open GRIB files. "
                                    "Please install it with: `uv pip install "
                                    "'pystormtracker[grib]'`"
                                ) from None
                            engine = "cfgrib"
                        else:
                            engine = "h5netcdf"
                    else:
                        local_path = Path(self.pathname)
                        ext = local_path.suffix.lower()
                        if ext in [".grib", ".grib2", ".grb"]:
                            if find_spec("cfgrib") is None:
                                raise ValueError(
                                    "cfgrib is required to open GRIB files. "
                                    "Please install it with: `uv pip install "
                                    "'pystormtracker[grib]'`"
                                ) from None
                            engine = "cfgrib"
                        elif ext == ".zarr" or (
                            local_path.is_dir() and (local_path / ".zmetadata").exists()
                        ):
                            if find_spec("zarr") is None:
                                raise ValueError(
                                    "zarr is required to open Zarr datasets. "
                                    "Please install it with: `uv pip install "
                                    "'pystormtracker[zarr]'`"
                                ) from None
                            engine = "zarr"
                        else:
                            engine = "h5netcdf"

                cache_key = (
                    str(self.pathname),
                    str(engine),
                    _normalize_chunks_key(self.chunks),
                )
                if cache_key not in self._ds_cache:
                    if engine == "zarr":
                        if is_remote and storage_options:
                            ds = xr.open_dataset(
                                self.pathname,
                                engine=engine,
                                decode_times=False,
                                chunks=self.chunks,
                                storage_options=storage_options,
                                backend_kwargs={"consolidated": False},
                            )
                        elif self.chunks is not None:
                            ds = xr.open_dataset(
                                self.pathname,
                                engine=engine,
                                decode_times=False,
                                chunks=self.chunks,
                                backend_kwargs={"consolidated": False},
                            )
                        else:
                            ds = xr.open_dataset(
                                self.pathname,
                                engine=engine,
                                decode_times=False,
                                backend_kwargs={"consolidated": False},
                            )
                    elif self.chunks is not None:
                        ds = xr.open_dataset(
                            self.pathname,
                            engine=engine,
                            decode_times=False,
                            chunks=self.chunks,
                        )
                    else:
                        ds = xr.open_dataset(
                            self.pathname,
                            engine=engine,
                            decode_times=False,
                        )
                    self._ds_cache[cache_key] = self._normalize_time_coordinate(ds)

                self._ds = self._ds_cache[cache_key]
                LOGGER.debug(
                    "Opened input %r with engine=%s chunks=%r dims=%s",
                    self.pathname,
                    engine,
                    self.chunks,
                    dict(self._ds.sizes),
                )
        return self._ds

    @classmethod
    def _normalize_time_coordinate(cls, dataset: xr.Dataset) -> xr.Dataset:
        """Normalize supported source time coordinates to datetime64[ms]."""
        time_name = next(
            (name for name in cls._NAME_ALIASES["time"] if name in dataset.coords),
            None,
        )
        if time_name is None:
            return dataset
        coordinate = dataset[time_name]
        source_units = coordinate.attrs.get("units") or coordinate.encoding.get("units")
        source_calendar = coordinate.attrs.get("calendar") or coordinate.encoding.get(
            "calendar"
        )
        dtype_kind = getattr(coordinate.dtype, "kind", None)
        if not isinstance(dtype_kind, str):
            return dataset
        if dtype_kind in "iuf":
            if isinstance(source_units, str):
                values = encode_numeric_time_values(
                    coordinate.values,
                    units=source_units,
                    calendar=None if source_calendar is None else str(source_calendar),
                )
            else:
                values = encode_time_values(coordinate.values)
            decoded_values = values.astype("datetime64[ms]")
        elif dtype_kind == "M" or dtype_kind == "O":
            if source_calendar is not None:
                infer_calendar(
                    coordinate.values,
                    attrs={"calendar": str(source_calendar)},
                )
            decoded_values = encode_time_values(coordinate.values).astype(
                "datetime64[ms]"
            )
        else:
            return dataset
        decoded_coordinate = coordinate.copy(data=decoded_values)
        return dataset.assign_coords({time_name: decoded_coordinate})

    def get_coords(self) -> tuple[str, str, str]:
        """Returns the mapped names for (time, lat, lon)."""
        ds = self.ensure_open()
        coords = ds.coords

        time_name = next((c for c in self._NAME_ALIASES["time"] if c in coords), "time")
        lat_name = next(
            (c for c in self._NAME_ALIASES["latitude"] if c in coords), "latitude"
        )
        lon_name = next(
            (c for c in self._NAME_ALIASES["longitude"] if c in coords), "longitude"
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

    def is_global_longitude(self) -> bool:
        """Return whether a 1D longitude coordinate covers a periodic globe."""
        ds = self.ensure_open()
        _, _, lon_name = self.get_coords()
        if lon_name == "x" or lon_name not in ds.coords:
            return False

        lon = np.asarray(ds[lon_name].values, dtype=np.float64)
        if lon.ndim != 1 or lon.size < 2 or not np.isfinite(lon).all():
            return False

        normalized = np.unique(np.mod(lon, 360.0))
        if normalized.size < 2:
            return False

        cyclic = np.concatenate((normalized, normalized[:1] + 360.0))
        gaps = np.diff(cyclic)
        typical_gap = float(np.median(gaps))
        return typical_gap > 0.0 and float(np.max(gaps)) <= 1.5 * typical_gap

    def is_reduced_gaussian(self, variable_name: str | None = None) -> bool:
        """Detects if the dataset represents a reduced Gaussian grid."""
        ds = self.ensure_open()
        # If variable_name not provided, check the first data variable
        if variable_name is None:
            variable_name = (
                cast(str, next(iter(ds.data_vars))) if ds.data_vars else None
            )

        if variable_name and variable_name in ds:
            da = ds[variable_name]
            # cfgrib tags reduced Gaussian grids with this attribute
            if da.attrs.get("GRIB_gridType") == "reduced_gg":
                return True
            # Alternative: check if latitude/longitude are 1D coordinates of a
            # non-spatial dimension
            if "values" in da.dims and da.ndim == 2:  # (time, values)
                return True
        return False

    def get_reduced_grid_pl(
        self, variable_name: str | None = None
    ) -> np.ndarray | None:
        """Returns the 'pl' array (points per latitude) for a reduced grid."""
        ds = self.ensure_open()
        if variable_name is None:
            variable_name = (
                cast(str, next(iter(ds.data_vars))) if ds.data_vars else None
            )

        if variable_name and variable_name in ds:
            da = ds[variable_name]
            pl = da.attrs.get("GRIB_pl")
            if pl is not None:
                return np.array(pl, dtype=np.int32)
        return None

    def _get_theta(self, ntheta: int, geometry: str) -> NDArray[np.float64]:
        """Calculates colatitudes (theta) for a given geometry and resolution."""
        if geometry == "GL":
            import ducc0

            # ducc0.misc.GL_thetas returns North-to-South (0 to pi)
            return np.asarray(
                ducc0.misc.GL_thetas(ntheta),
                dtype=np.float64,
            )
        # Default to equidistant if geometry == "CC"
        return np.linspace(0, np.pi, ntheta, dtype=np.float64)

    def get_grid_metadata(self, variable_name: str | None = None) -> GridMetadata:
        """
        Returns grid metadata (theta, nphi, phi0, ringstart) for SHT.
        Works for reduced Gaussian and HEALPix grids.
        """
        ds = self.ensure_open()
        if variable_name is None:
            variable_name = (
                cast(str, next(iter(ds.data_vars))) if ds.data_vars else None
            )

        da = ds[variable_name] if variable_name else next(iter(ds.data_vars.values()))

        # 1. Check for HEALPix
        if da.attrs.get("grid_type") == "healpix" or "cell" in da.dims:
            import ducc0

            npix = da.sizes.get("cell", da.sizes.get("values", 0))
            nside = int(np.sqrt(npix / 12))
            hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
            info = hp_base.sht_info()
            return {
                "theta": np.asarray(info["theta"], dtype=np.float64),
                "nphi": np.asarray(info["nphi"], dtype=np.uint64),
                "phi0": np.asarray(info["phi0"], dtype=np.float64),
                "ringstart": np.asarray(info["ringstart"], dtype=np.uint64),
            }

        # 2. Check for Reduced Gaussian
        if self.is_reduced_gaussian(variable_name):
            pl = self.get_reduced_grid_pl(variable_name)
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


def normalize_tracking_data(
    source: str | Path | xr.DataArray | xr.Dataset,
    variable_name: str,
    *,
    start_time: TimeInput | None = None,
    end_time: TimeInput | None = None,
    engine: str | None = None,
    chunks: DataLoaderChunks = None,
    backend: Literal["serial", "mpi", "dask"] = "serial",
) -> xr.DataArray:
    """Normalize one public tracking input to one selected DataArray."""
    effective_chunks = chunks
    if (
        backend == "dask"
        and effective_chunks is None
        and not isinstance(source, (xr.DataArray, xr.Dataset))
    ):
        effective_chunks = "auto"
    loader = DataLoader(source, engine=engine, chunks=effective_chunks)
    dataset = loader.ensure_open()
    actual_name = loader.resolve_variable_name(dataset, variable_name)
    if (
        actual_name is None
        and isinstance(source, xr.DataArray)
        and len(dataset.data_vars) == 1
    ):
        actual_name = cast(str, next(iter(dataset.data_vars)))
    if actual_name is None:
        raise KeyError(
            f"Variable {variable_name!r} not found. Available: "
            f"{list(dataset.data_vars)}"
        )
    selected = select_time_range(
        dataset[actual_name],
        start_time=start_time,
        end_time=end_time,
    )
    if not isinstance(selected, xr.DataArray):
        raise TypeError("normalized tracking input must be a DataArray")
    if backend == "dask":
        coords = DataLoader(selected).get_coords()
        time_dim = coords[0]
        if time_dim in selected.dims:
            spatial_chunks = {d: -1 for d in selected.dims if d != time_dim}
            selected = selected.chunk({time_dim: 1, **spatial_chunks})
    selected_time_dim = DataLoader(selected).get_coords()[0]
    LOGGER.debug(
        "Selected input variable=%s requested=%s frames=%d dims=%s chunks=%r",
        actual_name,
        variable_name,
        selected.sizes.get(selected_time_dim, 0),
        dict(selected.sizes),
        selected.chunks,
    )
    return selected
