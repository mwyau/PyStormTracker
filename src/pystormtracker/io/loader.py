from __future__ import annotations

import threading
from pathlib import Path
from typing import ClassVar

import xarray as xr


class DataLoader:
    """
    Handles optimized xarray loading for NetCDF and GRIB files.
    Provides robust coordinate mapping and thread-safe dataset caching.
    """

    _ds_cache: ClassVar[dict[Path, xr.Dataset]] = {}
    _ds_lock: ClassVar[threading.Lock] = threading.Lock()

    # Common variable and coordinate name aliases
    VAR_MAPPING: ClassVar[dict[str, list[str]]] = {
        "msl": ["msl", "slp", "prmsl", "mean_sea_level_pressure"],
        "vo": ["vo", "rv", "relative_vorticity", "vorticity"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon"],
        "time": ["time", "valid_time"],
    }

    def __init__(self, pathname: str | Path, engine: str | None = None) -> None:
        self.pathname = Path(pathname)
        self.engine = engine
        self._ds: xr.Dataset | None = None

    def ensure_open(self) -> xr.Dataset:
        """Ensures the xarray dataset is open and returns it."""
        if self._ds is None:
            with self._ds_lock:
                if self.pathname not in self._ds_cache:
                    # Automatic engine detection
                    engine = self.engine
                    if engine is None:
                        ext = self.pathname.suffix.lower()
                        if ext in [".grib", ".grib2", ".grb"]:
                            engine = "cfgrib"
                        else:
                            engine = "h5netcdf"

                    try:
                        self._ds_cache[self.pathname] = xr.open_dataset(
                            self.pathname, engine=engine, chunks={}
                        )
                    except Exception:
                        # Generic fallback if engine is missing or fails
                        self._ds_cache[self.pathname] = xr.open_dataset(
                            self.pathname, chunks={}
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
