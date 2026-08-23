"""HEALPix-native detection and local feature refinement.

The underlying spherical grid is HEALPix (Górski et al. (2005), “HEALPix: A
Framework for High-Resolution Discretization and Fast Analysis of Data
Distributed on the Sphere,” *The Astrophysical Journal*, 622(2), 759--771,
https://doi.org/10.1086/427976).  The threshold -> connected-object -> local
feature -> refinement pipeline is a PyStormTracker adaptation of Hodges-style
feature detection to HEALPix topology.  The topology adaptation and shared
intrinsic spherical quadratic refinement are not claims about the HEALPix
paper or exact TRACK behavior.  ``ducc0`` supplies the numerical HEALPix
indexing and neighbor operations.
"""

from __future__ import annotations

from typing import Literal

import numba as nb
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.data_loader import DataLoader
from ..models.time import TimeInput, TimeRange, coerce_time_input, is_missing_time
from ..models.tracker import CenterFrame
from ..models.tracks import ResolvedDetectionMode
from ..refinement.quadratic import (
    SphericalQuadraticRefinementBatch,
    refine_spherical_quadratic_samples,
)
from .constants import DEFAULT_MSL_OBJECT_THRESHOLD, DEFAULT_VO_OBJECT_THRESHOLD

type HealpixFeatureRefinement = Literal["grid", "quadratic"]

# ---------------------------------------------------------------------------
# Compiled HEALPix Detection and Quadratic Refinement Numerics
# ---------------------------------------------------------------------------


def _build_healpix_neighbor_table(nside: int) -> NDArray[np.int64]:
    """Build an (8, npix) array of neighbor pixel indices for a RING HEALPix grid."""
    import ducc0.healpix  # ty: ignore[unresolved-import]

    hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
    npix = 12 * nside * nside
    pixels = np.arange(npix, dtype=np.int64)
    nbors = hp_base.neighbors(pixels)
    return np.asarray(nbors.T, dtype=np.int64)


@nb.njit(nogil=True, cache=True)
def _label_healpix_connected_components(
    data: NDArray[np.float64],
    neighbor_table: NDArray[np.int64],
    threshold: float,
    is_min: bool,
) -> tuple[NDArray[np.int32], int]:
    """Graph-based connected-component labeling for a HEALPix neighbor graph.

    Args:
        data: 1D array of shape (N_pixels,).
        neighbor_table: 2D array of shape (8, N_pixels).
        threshold: Intensity threshold.
        is_min: If True, group pixels BELOW threshold. If False, ABOVE.

    Returns:
        (labels, num_objects)

    This is the PST topology adaptation of the Hodges-style threshold/object
    stage; it is not asserted to be the TRACK quad-tree implementation.
    """
    npix = len(data)
    labels = np.zeros(npix, dtype=np.int32)

    # 1. Initialize labels
    for p in range(npix):
        val = data[p]
        if is_min:
            if val <= threshold:
                labels[p] = p + 1
        else:
            if val >= threshold:
                labels[p] = p + 1

    # 2. Iterative label propagation
    changed = True
    while changed:
        changed = False
        for p in range(npix):
            if labels[p] == 0:
                continue

            cur_label = labels[p]
            min_label = cur_label

            for i in range(8):
                n_idx = neighbor_table[i, p]
                if n_idx != -1 and labels[n_idx] > 0 and labels[n_idx] < min_label:
                    min_label = labels[n_idx]

            if min_label < cur_label:
                labels[p] = min_label
                changed = True

    # 3. Compact labels to 1..N
    unique_labels = np.unique(labels)
    label_map = {0: 0}
    next_label = 1
    for ul in unique_labels:
        if ul != 0:
            label_map[ul] = next_label
            next_label += 1

    num_objects = next_label - 1
    for p in range(npix):
        labels[p] = label_map[labels[p]]

    return labels, num_objects


@nb.njit(nogil=True, cache=True)
def _find_healpix_object_extrema(
    data: NDArray[np.float64],
    labels: NDArray[np.int32],
    neighbor_table: NDArray[np.int64],
    num_objects: int,
    is_min: bool,
    min_grid_points: int = 1,
) -> NDArray[np.float64]:
    """Find the single absolute extremum for each distinct connected component."""
    npix = len(data)
    extrema_mask = np.zeros(npix, dtype=np.float64)

    if num_objects == 0:
        return extrema_mask

    # 1. Filter by minimum object size if requested
    if min_grid_points > 1:
        sizes = np.zeros(num_objects + 1, dtype=np.int32)
        for p in range(npix):
            lbl = labels[p]
            if lbl > 0:
                sizes[lbl] += 1
    else:
        sizes = np.ones(num_objects + 1, dtype=np.int32)

    # 2. Track best pixel per object label
    best_pixel = np.full(num_objects + 1, -1, dtype=np.int64)
    best_val = (
        np.full(num_objects + 1, np.inf, dtype=np.float64)
        if is_min
        else np.full(num_objects + 1, -np.inf, dtype=np.float64)
    )

    for p in range(npix):
        lbl = labels[p]
        if lbl == 0 or sizes[lbl] < min_grid_points:
            continue

        val = data[p]
        if is_min:
            if val < best_val[lbl]:
                best_val[lbl] = val
                best_pixel[lbl] = p
        else:
            if val > best_val[lbl]:
                best_val[lbl] = val
                best_pixel[lbl] = p

    # 3. Mark detected extrema
    for lbl in range(1, num_objects + 1):
        p = best_pixel[lbl]
        if p != -1:
            extrema_mask[p] = 1.0

    return extrema_mask


def _refine_healpix_quadratic_batch(
    data: NDArray[np.float64],
    neighbor_table: NDArray[np.int64],
    pixel_indices: NDArray[np.int64],
    pixel_lats: NDArray[np.float64],
    pixel_lons: NDArray[np.float64],
    *,
    is_minimum: bool,
) -> SphericalQuadraticRefinementBatch:
    """Refine an irregular HEALPix neighbour ring with the shared core."""
    n_features = pixel_indices.size
    n_neighbors = neighbor_table.shape[0]
    neighbor_indices = neighbor_table[:, pixel_indices].T
    neighbor_mask = (neighbor_indices >= 0) & (neighbor_indices < data.size)
    neighbor_lats = np.full((n_features, n_neighbors), np.nan, dtype=np.float64)
    neighbor_lons = np.full((n_features, n_neighbors), np.nan, dtype=np.float64)
    neighbor_values = np.full((n_features, n_neighbors), np.nan, dtype=np.float64)
    valid_rows, valid_columns = np.nonzero(neighbor_mask)
    valid_pixels = neighbor_indices[valid_rows, valid_columns]
    neighbor_lats[valid_rows, valid_columns] = pixel_lats[valid_pixels]
    neighbor_lons[valid_rows, valid_columns] = pixel_lons[valid_pixels]
    neighbor_values[valid_rows, valid_columns] = data[valid_pixels]
    return refine_spherical_quadratic_samples(
        pixel_lats[pixel_indices],
        pixel_lons[pixel_indices],
        data[pixel_indices],
        neighbor_lats,
        neighbor_lons,
        neighbor_values,
        is_minimum=is_minimum,
        neighbor_mask=neighbor_mask,
    )


def _refine_healpix_quadratic_point(
    data: NDArray[np.float64],
    neighbor_table: NDArray[np.int64],
    p_idx: int,
    pixel_lats: NDArray[np.float64],
    pixel_lons: NDArray[np.float64],
    *,
    is_minimum: bool = False,
) -> tuple[float, float, float]:
    """Refine one HEALPix candidate, retaining its grid fallback on failure."""
    batch = _refine_healpix_quadratic_batch(
        data,
        neighbor_table,
        np.asarray([p_idx], dtype=np.int64),
        pixel_lats,
        pixel_lons,
        is_minimum=is_minimum,
    )
    return batch.latitudes[0], batch.longitudes[0], batch.values[0]


def _refine_healpix_quadratic_points(
    data: NDArray[np.float64],
    neighbor_table: NDArray[np.int64],
    pixel_indices: NDArray[np.int64],
    pixel_lats: NDArray[np.float64],
    pixel_lons: NDArray[np.float64],
    *,
    is_minimum: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Batch-refine HEALPix candidates using the shared spherical core."""
    batch = _refine_healpix_quadratic_batch(
        data,
        neighbor_table,
        pixel_indices,
        pixel_lats,
        pixel_lons,
        is_minimum=is_minimum,
    )
    return batch.latitudes, batch.longitudes, batch.values


@nb.njit(nogil=True, cache=True)
def _extract_healpix_centers(
    extrema_mask: NDArray[np.float64],
    data: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Extract the pixel indices and values of detected extrema."""
    idx = np.where(extrema_mask > 0)[0]
    vals = np.empty(len(idx), dtype=np.float64)
    for i in range(len(idx)):
        vals[i] = data[idx[i]]
    return idx, vals


# ---------------------------------------------------------------------------
# Single-Frame Detection Orchestrator
# ---------------------------------------------------------------------------


def detect_healpix_frame(
    frame_values: NDArray[np.float64],
    time_val: TimeInput,
    neighbor_table: NDArray[np.int64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    *,
    object_threshold: float,
    mode: ResolvedDetectionMode = "min",
    min_object_grid_points: int = 1,
    feature_refinement: HealpixFeatureRefinement = "quadratic",
) -> CenterFrame:
    """Detect and refine one frame using the PST HEALPix adaptation.

    HEALPix supplies the underlying grid (Górski et al., 2005).  PST applies
    the Hodges-style threshold, connected-object, local-feature, and shared
    refinement stages on that topology.
    """
    point_time = coerce_time_input(time_val)
    assert point_time is not None
    is_min = mode == "min"
    labels, num_objects = _label_healpix_connected_components(
        frame_values,
        neighbor_table,
        threshold=object_threshold,
        is_min=is_min,
    )

    extrema = _find_healpix_object_extrema(
        frame_values,
        labels,
        neighbor_table,
        num_objects,
        is_min=is_min,
        min_grid_points=min_object_grid_points,
    )

    pixel_indices, vals = _extract_healpix_centers(extrema, frame_values)

    if feature_refinement == "quadratic" and len(pixel_indices) > 0:
        refined = _refine_healpix_quadratic_batch(
            frame_values,
            neighbor_table,
            pixel_indices,
            lat,
            lon,
            is_minimum=is_min,
        )
        return CenterFrame(
            point_time,
            refined.latitudes,
            refined.longitudes,
            refined.values,
        )

    return CenterFrame(
        point_time,
        lat[pixel_indices],
        lon[pixel_indices],
        vals,
    )


class HealpixDetector:
    """Feature detector operating natively on HEALPix spherical grids.

    HEALPix grid lineage: Górski et al. (2005), *The Astrophysical Journal*,
    622(2), 759--771. https://doi.org/10.1086/427976

    The detector is a PyStormTracker adaptation of the Hodges-style
    threshold -> connected object -> local feature -> refinement pipeline to
    HEALPix topology.  ``ducc0`` supplies efficient RING indexing and neighbor
    calculation; these engineering/numerical layers are separate from the
    HEALPix grid reference.
    """

    def __init__(
        self,
        pathname: str | None = None,
        variable_name: str = "msl",
        nside: int | None = None,
        time_range: TimeRange | None = None,
        engine: str | None = None,
    ) -> None:
        self.pathname = pathname
        self.requested_variable_name = variable_name
        self.nside = nside
        self.time_range = time_range
        self._loader = (
            DataLoader(pathname, engine=engine) if pathname is not None else None
        )
        self._data: xr.DataArray | None = None
        self._neighbor_table: NDArray[np.int64] | None = None
        self._lat: NDArray[np.float64] | None = None
        self._lon: NDArray[np.float64] | None = None

    @classmethod
    def from_xarray(
        cls,
        data: xr.DataArray,
        variable_name: str | None = None,
    ) -> HealpixDetector:
        """Create a HealpixDetector from an in-memory xarray DataArray."""
        obj = cls.__new__(cls)
        obj.pathname = None
        obj.requested_variable_name = variable_name or str(data.name)
        obj.time_range = None
        obj._loader = None
        obj._data = data

        nside = data.attrs.get("nside")
        if nside is None:
            npix = (
                data.shape[-1]
                if data.ndim == 2
                else len(data["cell"])
                if "cell" in data.dims
                else None
            )
            if npix is not None:
                nside = int(np.sqrt(npix / 12))
            else:
                raise ValueError(
                    "Cannot determine HEALPix nside from DataArray dimensions"
                )

        obj.nside = int(nside)
        obj._init_healpix_geometry()
        return obj

    def _init_healpix_geometry(self) -> None:
        """Precompute HEALPix pixel coordinates and neighbor table."""
        assert self.nside is not None
        import ducc0.healpix  # ty: ignore[unresolved-import]

        hp_base = ducc0.healpix.Healpix_Base(self.nside, "RING")
        npix = 12 * self.nside * self.nside
        pixels = np.arange(npix, dtype=np.int64)

        # 1. Neighbor table
        nbors = hp_base.neighbors(pixels)
        self._neighbor_table = np.asarray(nbors.T, dtype=np.int64)

        # 2. Center coordinates (lat, lon)
        ptg = hp_base.pix2ang(pixels)
        theta = ptg[:, 0]  # colatitude [0, pi]
        phi = ptg[:, 1]  # longitude [0, 2pi]
        self._lat = np.degrees(0.5 * np.pi - theta)
        self._lon = np.degrees(phi) % 360.0

    def _ensure_open(self) -> None:
        """Ensure the underlying xarray Dataset is open and mapped."""
        if self._data is None:
            assert self._loader is not None
            ds = self._loader.ensure_open()

            actual_var = self._loader.resolve_variable_name(
                ds, self.requested_variable_name
            )
            if actual_var is None:
                raise KeyError(f"Variable '{self.requested_variable_name}' not found.")

            self._data = ds[actual_var]

            if self.nside is None:
                nside = self._data.attrs.get("nside")
                if nside is None:
                    npix = self._data.shape[-1]
                    nside = int(np.sqrt(npix / 12))
                self.nside = int(nside)

            self._init_healpix_geometry()

    def get_variable(self, frame_idx: int | None = None) -> xr.DataArray | None:
        """Get the variable array, optionally slice by time index."""
        self._ensure_open()
        assert self._data is not None

        time_dim = "time"
        if self._loader:
            time_dim, _, _ = self._loader.get_coords()
        elif "time" not in self._data.dims and len(self._data.dims) > 1:
            time_dim = str(self._data.dims[0])

        if self.time_range and time_dim in self._data.dims:
            data = self._data.sel(
                {time_dim: slice(self.time_range.start, self.time_range.end)}
            )
        else:
            data = self._data

        if frame_idx is not None:
            if time_dim in data.dims:
                return data.isel({time_dim: frame_idx})
            return data

        return data

    def get_time(self) -> np.ndarray:
        """Get the array of timestamps."""
        if self._loader is None:
            assert self._data is not None
            if "time" in self._data.coords:
                return np.asarray(self._data["time"].values)
            return np.array([])

        ds = self._loader.ensure_open()
        time_dim, _, _ = self._loader.get_coords()

        if self.time_range:
            effective_start = self.time_range.start
            effective_end = self.time_range.end
            time_coord = ds[time_dim]

            if not is_missing_time(effective_start) and not is_missing_time(
                effective_end
            ):
                times = time_coord.sel(
                    {time_dim: slice(effective_start, effective_end)}
                )
            elif not is_missing_time(effective_start):
                times = time_coord.sel({time_dim: slice(effective_start, None)})
            elif not is_missing_time(effective_end):
                times = time_coord.sel({time_dim: slice(None, effective_end)})
            else:
                times = time_coord
        else:
            times = ds[time_dim]

        return np.asarray(times.values)

    def detect(
        self,
        object_threshold: float | None = None,
        detection_mode: ResolvedDetectionMode = "min",
        min_object_grid_points: int = 1,
        feature_refinement: HealpixFeatureRefinement = "quadratic",
        **kwargs: object,
    ) -> list[CenterFrame]:
        if kwargs:
            unexpected = ", ".join(repr(k) for k in kwargs)
            raise TypeError(
                f"detect() got unexpected keyword argument(s): {unexpected}"
            )

        if min_object_grid_points <= 0:
            raise ValueError("min_object_grid_points must be positive")
        if feature_refinement not in ("grid", "quadratic"):
            raise ValueError(
                f"unsupported feature_refinement {feature_refinement!r}; "
                "expected 'grid' or 'quadratic'"
            )

        self._ensure_open()
        times = self.get_time()
        if times is None:
            return []

        if object_threshold is None:
            if self.requested_variable_name == "vo":
                object_threshold = DEFAULT_VO_OBJECT_THRESHOLD
            else:
                object_threshold = DEFAULT_MSL_OBJECT_THRESHOLD

        assert self._neighbor_table is not None
        assert self._lat is not None
        assert self._lon is not None

        raw_steps: list[CenterFrame] = []
        for i in range(len(times)):
            current_time = times[i]
            frame = self.get_variable(i)
            if frame is None:
                continue

            raw_step = detect_healpix_frame(
                frame.values,
                current_time,
                self._neighbor_table,
                self._lat,
                self._lon,
                object_threshold=object_threshold,
                mode=detection_mode,
                min_object_grid_points=min_object_grid_points,
                feature_refinement=feature_refinement,
            )
            raw_steps.append(raw_step)

        return raw_steps
