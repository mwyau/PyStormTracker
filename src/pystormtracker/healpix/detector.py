from __future__ import annotations

import threading
from pathlib import Path
from typing import ClassVar, Literal

import ducc0
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.data_loader import DataLoader
from ..models import TimeRange
from ..models import constants as model_constants
from ..models.tracker import RawDetectionStep
from .kernels import (
    _numba_get_healpix_centers,
    _numba_healpix_ccl,
    _numba_healpix_object_extrema,
    subgrid_refine_healpix,
)


class HealpixDetector:
    """
    A meteorological feature detector that treats fields as 1D HEALPix maps.
    Uses xarray for lazy-loading and ducc0 for HEALPix grid math.
    """

    _ds_cache: ClassVar[dict[Path, xr.Dataset]] = {}
    _ds_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        pathname: str | Path,
        varname: str,
        time_range: TimeRange | None = None,
        global_start_idx: int = 0,
        global_total_steps: int | None = None,
        engine: str | None = None,
    ) -> None:
        self.pathname = Path(pathname)
        self.requested_varname = varname
        self.time_range = time_range
        self.global_start_idx = global_start_idx
        self.global_total_steps = global_total_steps

        self._loader = DataLoader(self.pathname, engine=engine)
        self._data: xr.DataArray | None = None
        self.varname = varname
        self._hp_base: ducc0.healpix.Healpix_Base | None = None
        self._neighbor_table: NDArray[np.int64] | None = None
        self._lat_lon_map: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None

    def _ensure_open(self) -> None:
        if self._data is None:
            ds = self._loader.ensure_open()

            actual_var = None
            possible_names = DataLoader.VAR_MAPPING.get(
                self.requested_varname, [self.requested_varname]
            )
            for name in possible_names:
                if name in ds.data_vars:
                    actual_var = name
                    break

            if actual_var is None:
                actual_var = self.requested_varname

            self.varname = actual_var
            self._data = ds[self.varname]

        if self._hp_base is None:
            # Enforce 1D spatial dimension (time, cell)
            # Find the spatial dimension name (not time)
            time_dim, _, _ = self._loader.get_coords()
            spatial_dims = [d for d in self._data.dims if d != time_dim]
            if len(spatial_dims) != 1:
                raise ValueError(
                    "HealpixDetector requires exactly 1 spatial dimension, "
                    f"got: {spatial_dims}"
                )

            self._cell_dim = spatial_dims[0]
            npix = self._data.sizes[self._cell_dim]

            # Calculate nside
            nside = int(np.sqrt(npix / 12))
            if 12 * nside * nside != npix:
                raise ValueError(
                    f"Number of pixels {npix} is not a valid HEALPix size (12*Nside^2)."
                )

            self._hp_base = ducc0.healpix.Healpix_Base(nside, "RING")

            # Precompute neighbor table (shape: 8, npix)
            all_pix = np.arange(npix, dtype=np.int64)
            nbors = self._hp_base.neighbors(all_pix)  # Shape should be (N, 8)
            self._neighbor_table = np.ascontiguousarray(nbors.T)  # Shape (8, N)

            # Precompute lat/lon
            ang = self._hp_base.pix2ang(all_pix)  # Shape (N, 2)
            colat = ang[:, 0]
            lon_rad = ang[:, 1]
            self._lat = 90.0 - np.degrees(colat)
            self._lon = np.degrees(lon_rad)

    def get_var(
        self, frame: int | tuple[int, int] | None = None
    ) -> NDArray[np.float64] | None:
        self._ensure_open()
        assert self._data is not None

        time_dim, _, _ = self._loader.get_coords()

        if self.time_range:
            start, end = self.time_range.start, self.time_range.end
            if not np.isnat(start) and not np.isnat(end):
                data_range = self._data.sel({time_dim: slice(start, end)})
            elif not np.isnat(start):
                data_range = self._data.where(self._data[time_dim] >= start, drop=True)
            elif not np.isnat(end):
                data_range = self._data.where(self._data[time_dim] <= end, drop=True)
            else:
                data_range = self._data
        else:
            data_range = self._data

        match frame:
            case int(idx):
                data = data_range.isel({time_dim: idx})
                return np.asarray(data.values)
            case (int(s_off), int(e_off)):
                data = data_range.isel({time_dim: slice(s_off, e_off)})
                return np.asarray(data.values)
            case None:
                return np.asarray(data_range.values)
            case _:
                raise TypeError("frame must be an int, tuple[int, int], or None")

    def get_time(self) -> NDArray[np.datetime64] | None:
        self._ensure_open()
        ds = self._loader.ensure_open()
        time_dim, _, _ = self._loader.get_coords()

        if self.time_range:
            start, end = self.time_range.start, self.time_range.end
            time_coord = ds[time_dim]
            if not np.isnat(start) and not np.isnat(end):
                times = time_coord.sel({time_dim: slice(start, end)})
            elif not np.isnat(start):
                times = time_coord.where(time_coord >= start, drop=True)
            elif not np.isnat(end):
                times = time_coord.where(time_coord <= end, drop=True)
            else:
                times = time_coord
        else:
            times = ds[time_dim]

        return np.asarray(times.values, dtype="datetime64[ns]")

    def detect(
        self,
        threshold: float | None = None,
        minmaxmode: Literal["min", "max"] = "min",
        min_points: int = 1,
    ) -> list[RawDetectionStep]:
        self._ensure_open()
        times = self.get_time()
        if times is None:
            return []

        # Set variable specific thresholds if not provided
        if threshold is None:
            if self.requested_varname == "vo":
                threshold = model_constants.DEFAULT_VO_THRESHOLD
            else:
                threshold = model_constants.DEFAULT_MSL_THRESHOLD

        raw_steps: list[RawDetectionStep] = []
        is_min = minmaxmode == "min"

        for i in range(len(times)):
            current_time = times[i]
            frame = self.get_var(i)
            if frame is None:
                continue

            # 1. Connected Component Labeling
            assert self._neighbor_table is not None
            labels, num_objects = _numba_healpix_ccl(
                frame, self._neighbor_table, threshold, is_min
            )

            # 2. Find Extrema within objects
            extrema = _numba_healpix_object_extrema(
                frame,
                self._neighbor_table,
                labels,
                num_objects,
                is_min,
                min_points,
            )

            # 3. Extract and Refine
            p_idx, _ = _numba_get_healpix_centers(extrema, frame)

            refined_lats = np.zeros(len(p_idx))
            refined_lons = np.zeros(len(p_idx))
            refined_vals = np.zeros(len(p_idx))

            for j in range(len(p_idx)):
                ref_lat, ref_lon, ref_val = subgrid_refine_healpix(
                    frame,
                    int(p_idx[j]),
                    self._neighbor_table,
                    self._lat,
                    self._lon,
                )
                refined_lats[j] = ref_lat
                refined_lons[j] = ref_lon
                refined_vals[j] = ref_val

            raw_step = (
                current_time,
                refined_lats,
                refined_lons,
                {self.varname: refined_vals},
            )
            raw_steps.append(raw_step)

        return raw_steps

    @classmethod
    def from_xarray(
        cls,
        data: xr.DataArray,
        time_range: TimeRange | None = None,
        global_start_idx: int = 0,
        global_total_steps: int | None = None,
    ) -> HealpixDetector:
        import uuid

        dummy_path = Path(f"/tmp/dummy_{uuid.uuid4().hex}.nc")
        detector = cls(
            pathname=dummy_path,
            varname=str(data.name) if data.name is not None else "var",
            time_range=time_range,
            global_start_idx=global_start_idx,
            global_total_steps=global_total_steps,
        )

        detector._data = data
        ds = xr.Dataset({str(data.name) if data.name is not None else "var": data})
        detector._loader._ds = ds

        # Force init of hp_base and neighbor table
        detector._ensure_open()

        return detector

    def get_xarray(self) -> xr.DataArray:
        self._ensure_open()
        assert self._data is not None
        return self._data

    def split(self, n: int) -> list[HealpixDetector]:
        """Splits the detector into n smaller detectors with disjoint time ranges."""
        self._ensure_open()
        times = self.get_time()
        if times is None:
            return [self]

        indices = np.array_split(np.arange(len(times)), n)
        detectors = []

        for chunk_indices in indices:
            if len(chunk_indices) == 0:
                continue

            start_idx = chunk_indices[0]
            end_idx = chunk_indices[-1]

            # Use TimeRange for splitting
            st = times[start_idx]
            et = times[end_idx]
            chunk_time_range = TimeRange(start=st, end=et)

            # Create a shallow copy with the new time range and global index tracking
            detector = HealpixDetector(
                pathname=self.pathname,
                varname=self.requested_varname,
                time_range=chunk_time_range,
                global_start_idx=self.global_start_idx + start_idx,
                global_total_steps=self.global_total_steps or len(times),
            )
            # Link to the already-open dataset to avoid re-opening
            detector._data = self._data
            detector._loader._ds = self._loader._ds
            detector._ensure_open()

            detectors.append(detector)

        return detectors
