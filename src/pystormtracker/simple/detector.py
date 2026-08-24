from __future__ import annotations

import threading
from pathlib import Path
from typing import ClassVar, Literal

import numba as nb
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.data_loader import DataLoader
from ..models.time import (
    TimeInput,
    TimeRange,
    coerce_time_input,
    is_missing_time,
    select_time_range,
)
from ..models.tracker import CenterFrame
from ..models.tracks import ResolvedDetectionMode
from ..refinement.quadratic import refine_quadratic_feature_points
from .constants import (
    DEFAULT_MSL_FEATURE_THRESHOLD,
    DEFAULT_SEARCH_WINDOW_SIZE,
    DEFAULT_VO_FEATURE_THRESHOLD,
)

type SimpleFeatureRefinement = Literal["grid", "quadratic"]

# ---------------------------------------------------------------------------
# Compiled Simple Detection Numerics
# ---------------------------------------------------------------------------


@nb.njit(nogil=True, cache=True)
def _filter_extrema(
    data: NDArray[np.float64],
    size: int,
    threshold: float,
    is_min: bool,
    periodic_x: bool = True,
) -> NDArray[np.float64]:
    """Filter local extrema by search window comparison and contrast threshold."""
    rows, cols = data.shape
    out = np.zeros_like(data)
    half_size = size // 2

    for r in range(half_size, rows - half_size):
        c_start = 0 if periodic_x else half_size
        c_end = cols if periodic_x else cols - half_size
        for c in range(c_start, c_end):
            center_val = data[r, c]
            if np.isnan(center_val) or np.isinf(center_val):
                continue

            is_extrema = True
            for i in range(-half_size, half_size + 1):
                rr = r + i
                for j in range(-half_size, half_size + 1):
                    cc = (c + j) % cols if periodic_x else c + j
                    if is_min:
                        if data[rr, cc] < center_val:
                            is_extrema = False
                            break
                    else:
                        if data[rr, cc] > center_val:
                            is_extrema = False
                            break
                if not is_extrema:
                    break

            if is_extrema:
                # Run rank filtering to remove flat plateaus
                rank = (size * size) // 3

                window = np.empty(size * size, dtype=data.dtype)
                idx = 0
                for i in range(-half_size, half_size + 1):
                    rr = r + i
                    for j in range(-half_size, half_size + 1):
                        cc = (c + j) % cols if periodic_x else c + j
                        window[idx] = data[rr, cc]
                        idx += 1
                window.sort()

                if is_min:
                    if window[rank] - center_val > threshold:
                        out[r, c] = 1.0
                else:
                    if window[size * size - 1 - rank] - center_val < -threshold:
                        out[r, c] = 1.0

    return out


@nb.njit(nogil=True, cache=True)
def _compute_masked_laplacian(
    data: NDArray[np.float64],
    mask: NDArray[np.float64],
    is_min: bool,
    periodic_x: bool = True,
) -> NDArray[np.float64]:
    """Compute 5-point stencil discrete Laplacian on masked candidate extrema."""
    rows, cols = data.shape
    out = np.zeros_like(data)
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] != 0:
                if r == 0 or r == rows - 1:
                    continue
                if not periodic_x and (c == 0 or c == cols - 1):
                    continue
                up = data[r - 1, c]
                down = data[r + 1, c]
                left_col = (c - 1) % cols if periodic_x else c - 1
                right_col = (c + 1) % cols if periodic_x else c + 1
                left = data[r, left_col]
                right = data[r, right_col]
                center = data[r, c]
                if is_min:
                    val = up + down + left + right - 4.0 * center
                else:
                    val = 4.0 * center - (up + down + left + right)
                out[r, c] = val * mask[r, c]
    return out


@nb.njit(nogil=True, cache=True)
def _remove_duplicate_extrema(
    laplacian: NDArray[np.float64], size: int, periodic_x: bool = True
) -> NDArray[np.float64]:
    """Suppress non-maximal Laplacian responses with deterministic tie-breaking."""
    rows, cols = laplacian.shape
    out = np.zeros_like(laplacian)
    half_size = size // 2

    for r in range(rows):
        for c in range(cols):
            center_val = laplacian[r, c]
            if center_val != 0:
                is_most_intense = True
                abs_center = abs(center_val)
                for i in range(-half_size, half_size + 1):
                    rr = r + i
                    if rr < 0 or rr >= rows:
                        continue
                    for j in range(-half_size, half_size + 1):
                        cc = c + j
                        if periodic_x:
                            cc %= cols
                        elif cc < 0 or cc >= cols:
                            continue
                        val = abs(laplacian[rr, cc])
                        if val > abs_center:
                            is_most_intense = False
                            break
                        elif val == abs_center:
                            # Tie-breaking: lower row-major index wins
                            if rr < r or (rr == r and cc < c):
                                is_most_intense = False
                                break
                    if not is_most_intense:
                        break
                if is_most_intense:
                    out[r, c] = 1.0
    return out


@nb.njit(nogil=True, cache=True)
def _extract_centers(
    extrema: NDArray[np.float64], frame: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Extract grid coordinates and raw field values for detected extrema."""
    rows, cols = extrema.shape

    count = 0
    for r in range(rows):
        for c in range(cols):
            if extrema[r, c] != 0.0:
                count += 1

    r_idx = np.empty(count, dtype=np.int64)
    c_idx = np.empty(count, dtype=np.int64)
    vals = np.empty(count, dtype=np.float64)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if extrema[r, c] != 0.0:
                r_idx[idx] = r
                c_idx[idx] = c
                vals[idx] = frame[r, c]
                idx += 1

    return r_idx, c_idx, vals


# ---------------------------------------------------------------------------
# Single-Frame Detection Orchestrator
# ---------------------------------------------------------------------------


def detect_simple_frame(
    frame: NDArray[np.float64],
    time_val: TimeInput,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    *,
    intensity_threshold: float,
    mode: ResolvedDetectionMode = "min",
    search_window_size: int = DEFAULT_SEARCH_WINDOW_SIZE,
    feature_refinement: SimpleFeatureRefinement = "quadratic",
    periodic_x: bool = True,
) -> CenterFrame:
    """Detect local extrema and optionally refine feature points for a single frame."""
    point_time = coerce_time_input(time_val)
    assert point_time is not None
    is_min = mode == "min"
    fill = np.inf if is_min else -np.inf
    filled_frame = np.where(np.isnan(frame), fill, frame)

    extrema = _filter_extrema(
        filled_frame,
        search_window_size,
        intensity_threshold,
        is_min,
        periodic_x,
    )

    if np.isnan(frame).any():
        extrema[np.isnan(frame)] = 0

    laplacian = _compute_masked_laplacian(filled_frame, extrema, is_min, periodic_x)
    extrema = _remove_duplicate_extrema(laplacian, size=5, periodic_x=periodic_x)

    r_idx, c_idx, vals = _extract_centers(extrema, frame)

    if feature_refinement == "quadratic" and len(r_idx) > 0:
        refined_lats, refined_lons, refined_vals = refine_quadratic_feature_points(
            frame,
            r_idx,
            c_idx,
            lat,
            lon,
            periodic_x=periodic_x,
        )
        return CenterFrame(point_time, refined_lats, refined_lons, refined_vals)

    return CenterFrame(point_time, lat[r_idx], lon[c_idx], vals)


class SimpleDetector:
    """A meteorological feature detector that treats fields as 2D images.

    Uses xarray for robust coordinate handling and lazy-loading.

    Yau and Chang (2020) support the Simple Tracker concept at the level of
    linking the closest features within 500 km over consecutive 6-hourly
    periods.  The current PST moving-window plateau-rank rule, five-point
    Laplacian strength, duplicate suppression, and tie ordering are current
    implementation behavior; they are not attributed to the verified main
    paper or its supplement here.
    """

    _ds_cache: ClassVar[dict[Path, xr.Dataset]] = {}
    _ds_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        pathname: str | Path | None,
        variable_name: str,
        time_range: TimeRange | None = None,
        global_start_idx: int = 0,
        global_total_steps: int | None = None,
        engine: str | None = None,
    ) -> None:
        self.pathname = (
            Path(pathname)
            if pathname is not None
            and not (isinstance(pathname, str) and "://" in pathname)
            else pathname
        )
        self.requested_variable_name = variable_name
        self.time_range = time_range
        self.global_start_idx = global_start_idx
        self.global_total_steps = global_total_steps

        self._loader = DataLoader(self.pathname, engine=engine)
        self._data: xr.DataArray | None = None
        self.variable_name = variable_name

    def _ensure_open(self) -> None:
        """Ensure the xarray dataset is open and basic variables are mapped."""
        if self._data is None:
            ds = self._loader.ensure_open()
            actual_var = self._loader.resolve_variable_name(
                ds, self.requested_variable_name
            )
            if actual_var is None:
                raise KeyError(
                    f"Variable '{self.requested_variable_name}' not found. "
                    f"Available: {list(ds.data_vars.keys())}"
                )

            self.variable_name = actual_var
            self._data = ds[self.variable_name]

    @property
    def lat(self) -> NDArray[np.float64]:
        self._ensure_open()
        ds = self._loader.ensure_open()
        _, lat_name, _ = self._loader.get_coords()
        return np.asarray(ds[lat_name].values)

    @property
    def lon(self) -> NDArray[np.float64]:
        self._ensure_open()
        ds = self._loader.ensure_open()
        _, _, lon_name = self._loader.get_coords()
        return np.asarray(ds[lon_name].values)

    def get_variable(self, frame_idx: int | None = None) -> NDArray[np.float64]:
        self._ensure_open()
        assert self._data is not None
        time_dim, _, _ = self._loader.get_coords()

        if self.time_range:
            data = self._data.sel(
                {time_dim: slice(self.time_range.start, self.time_range.end)}
            )
        else:
            data = self._data

        if frame_idx is not None:
            data = data.isel({time_dim: frame_idx})
            return np.asarray(data.values).reshape((data.shape[-2], data.shape[-1]))

        return np.asarray(data.values).reshape(
            (data.shape[0], data.shape[-2], data.shape[-1])
        )

    def get_time(self) -> np.ndarray:
        self._ensure_open()
        ds = self._loader.ensure_open()
        time_dim, _, _ = self._loader.get_coords()
        if self.time_range:
            times = ds[time_dim].sel(
                {time_dim: slice(self.time_range.start, self.time_range.end)}
            )
        else:
            times = ds[time_dim]
        return np.asarray(times.values)

    def get_xarray(
        self,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
    ) -> xr.DataArray:
        """Return the requested data range as an xarray DataArray."""
        self._ensure_open()
        assert self._data is not None
        effective_start = (
            start_time
            if start_time is not None
            else self.time_range.start
            if self.time_range is not None
            else None
        )
        effective_end = (
            end_time
            if end_time is not None
            else self.time_range.end
            if self.time_range is not None
            else None
        )
        selected = select_time_range(
            self._data,
            start_time=effective_start,
            end_time=effective_end,
        )
        assert isinstance(selected, xr.DataArray)
        return selected

    @classmethod
    def from_xarray(
        cls, data: xr.DataArray, variable_name: str | None = None
    ) -> SimpleDetector:
        """Create a detector from an existing xarray DataArray."""
        obj = cls.__new__(cls)
        obj.requested_variable_name = variable_name or (
            str(data.name) if data.name else "var"
        )
        obj.variable_name = obj.requested_variable_name
        obj._data = data
        obj._loader = DataLoader(data)
        obj.pathname = None
        obj.time_range = None
        obj.global_start_idx = 0
        obj.global_total_steps = None
        return obj

    def split(self, num: int) -> list[SimpleDetector]:
        self._ensure_open()
        time_name, _, _ = self._loader.get_coords()
        time_coord = self._loader.ensure_open()[time_name]

        if self.time_range:
            start, end = self.time_range.start, self.time_range.end
            if not is_missing_time(start) and not is_missing_time(end):
                active_times = time_coord.sel({time_name: slice(start, end)})
            elif not is_missing_time(start):
                active_times = time_coord.sel({time_name: slice(start, None)})
            elif not is_missing_time(end):
                active_times = time_coord.sel({time_name: slice(None, end)})
            else:
                active_times = time_coord
        else:
            active_times = time_coord
        time_values = np.asarray(active_times.values)
        total_len = len(time_values)

        range_size = total_len // num
        remainder = total_len % num

        detectors: list[SimpleDetector] = []
        for i in range(num):
            s_idx = i * range_size + min(i, remainder)
            e_idx = (i + 1) * range_size + min(i + 1, remainder)

            if s_idx >= e_idx:
                continue

            if self.pathname is None and self._data is not None:
                new_obj = SimpleDetector.from_xarray(self._data)
                new_obj.time_range = TimeRange(
                    start=time_values[s_idx], end=time_values[e_idx - 1]
                )
                new_obj.global_start_idx = s_idx
                new_obj.global_total_steps = total_len
                detectors.append(new_obj)
            else:
                detectors.append(
                    SimpleDetector(
                        self.pathname,
                        self.requested_variable_name,
                        time_range=TimeRange(
                            start=time_values[s_idx], end=time_values[e_idx - 1]
                        ),
                        global_start_idx=s_idx,
                        global_total_steps=total_len,
                        engine=self._loader.engine,
                    )
                )
        return detectors

    def detect(
        self,
        search_window_size: int = DEFAULT_SEARCH_WINDOW_SIZE,
        intensity_threshold: float | None = None,
        detection_mode: ResolvedDetectionMode = "min",
        feature_refinement: SimpleFeatureRefinement = "quadratic",
        **kwargs: object,
    ) -> list[CenterFrame]:
        if kwargs:
            unexpected = ", ".join(repr(k) for k in kwargs)
            raise TypeError(
                f"detect() got unexpected keyword argument(s): {unexpected}"
            )

        if search_window_size <= 0 or search_window_size % 2 == 0:
            raise ValueError("search_window_size must be a positive odd integer")
        if feature_refinement not in ("grid", "quadratic"):
            raise ValueError(
                f"unsupported feature_refinement {feature_refinement!r}; "
                "expected 'grid' or 'quadratic'"
            )

        if intensity_threshold is None:
            if self.requested_variable_name == "vo":
                intensity_threshold = DEFAULT_VO_FEATURE_THRESHOLD
            else:
                intensity_threshold = DEFAULT_MSL_FEATURE_THRESHOLD

        time_array = self.get_time()
        lat, lon = self.lat, self.lon
        _, _, lon_name = self._loader.get_coords()
        periodic_x = lon_name != "x" and self._loader.is_global_longitude()
        assert time_array is not None
        full_variable = self.get_variable()
        assert full_variable is not None

        raw_results: list[CenterFrame] = []

        for it, t in enumerate(time_array):
            frame = full_variable[it, :, :]
            step = detect_simple_frame(
                frame,
                t,
                lat,
                lon,
                intensity_threshold=intensity_threshold,
                mode=detection_mode,
                search_window_size=search_window_size,
                feature_refinement=feature_refinement,
                periodic_x=periodic_x,
            )
            raw_results.append(step)

        return raw_results
