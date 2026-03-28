from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, ClassVar

import xarray as xr


class DataLoader:
    """
    Handles optimized xarray loading for local and remote datasets.
    Supports NetCDF, GRIB, and Zarr formats, with thread-safe caching.
    """

    _ds_cache: ClassVar[dict[str | Path, xr.Dataset]] = {}
    _ds_lock: ClassVar[threading.Lock] = threading.Lock()

    # Common variable and coordinate name aliases
    VAR_MAPPING: ClassVar[dict[str, list[str]]] = {
        "msl": ["msl", "slp"],
        "vo": ["vo"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon"],
        "time": ["time", "valid_time"],
    }

    def __init__(
        self,
        pathname: str | Path,
        engine: str | None = None,
        data: xr.DataArray | None = None,
    ) -> None:
        if isinstance(pathname, str) and "://" in pathname:
            self.pathname: str | Path = pathname
        else:
            self.pathname = Path(pathname)

        self.engine = engine
        self._ds: xr.Dataset | None = None
        if data is not None:
            self._ds = data.to_dataset() if isinstance(data, xr.DataArray) else data

    def ensure_open(self) -> xr.Dataset:
        """Ensures the xarray dataset is open and returns it."""
        if self._ds is None:
            with self._ds_lock:
                if self.pathname not in self._ds_cache:
                    engine = self.engine
                    storage_options: dict[str, Any] = {}
                    is_remote = isinstance(self.pathname, str) and (
                        "://" in str(self.pathname)
                    )

                    if is_remote and str(self.pathname).startswith(
                        ("http://", "https://")
                    ):
                        # fsspec handles anon HTTP by default; no special 'anon'
                        # key needed.
                        pass
                    elif is_remote and str(self.pathname).startswith(("s3://", "gs://")):
                        storage_options = {"anon": True}

                    if engine is None:
                        if is_remote:
                            if str(self.pathname).endswith(".zarr"):
                                engine = "zarr"
                            elif str(self.pathname).endswith(
                                (".grib", ".grib2", ".grb")
                            ):
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
                                engine = "cfgrib"
                            elif ext == ".zarr" or (
                                local_path.is_dir()
                                and (local_path / ".zmetadata").exists()
                            ):
                                engine = "zarr"
                            else:
                                engine = "h5netcdf"

                    open_kwargs: dict[str, Any] = {"chunks": {}}
                    if engine == "zarr" and is_remote and storage_options:
                        open_kwargs["storage_options"] = storage_options

                    self._ds_cache[self.pathname] = xr.open_dataset(
                        self.pathname,
                        engine=engine,
                        **open_kwargs,
                    )

                self._ds = self._ds_cache[self.pathname]
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
