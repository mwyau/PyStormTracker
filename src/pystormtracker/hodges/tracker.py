from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.data_loader import normalize_tracking_data
from ..models import constants as model_constants
from ..models.geo import SpatialBounds, spatial_bounds_from_xarray
from ..models.tracker import RawDetectionStep, Tracker, get_int_option
from ..models.tracks import (
    ProcessingStep,
    Tracks,
)
from ..models.units import Mode, ModeOption, normalize_variable_units, resolve_mode
from ..preprocessing.tracking import Projection, preprocess_tracking_data
from ..time import TimeInput
from . import constants
from .detector import HodgesDetector
from .linker import HodgesLinker

if TYPE_CHECKING:
    from ..models.geo import MapExtent


class HodgesTracker(Tracker):
    """
    A tracker implementing the Hodges (TRACK) algorithm with adaptive constraints.
    """

    def __init__(
        self,
        w1: float = constants.W1_DEFAULT,
        w2: float = constants.W2_DEFAULT,
        dmax: float = constants.DMAX_DEFAULT,
        phimax: float = constants.PHIMAX_DEFAULT,
        n_iterations: int = constants.ITERATIONS_DEFAULT,
        min_lifetime: int = constants.LIFETIME_DEFAULT,
        max_missing: int = constants.MISSING_DEFAULT,
        zones: NDArray[np.float64] | None = None,
        adapt_params: NDArray[np.float64] | None = None,
        use_standard_constraints: bool = True,
    ) -> None:
        """
        Initialize the Hodges Tracker.

        Args:
            w1: Weight for direction in cost function.
            w2: Weight for speed in cost function.
            dmax: Default maximum displacement in degrees.
            phimax: Penalty for phantom points (static cost).
            n_iterations: Number of MGE iterations (forward + backward).
            min_lifetime: Minimum number of steps for a valid track.
            max_missing: Maximum consecutive missing frames allowed.
            zones: Regional dmax zones [lon_min, lon_max, lat_min, lat_max, dmax].
            adapt_params: Adaptive smoothness parameters (2x4 array).
            use_standard_constraints: If True, use legacy standard zones/adaptive
                values if None provided.
        """
        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.n_iterations = n_iterations
        self.min_lifetime = min_lifetime
        self.max_missing = max_missing

        if zones is None:
            if use_standard_constraints:
                self.zones = constants.TRACK_ZONES
            else:
                self.zones = np.zeros((0, 5), dtype=np.float64)
        else:
            self.zones = zones

        if adapt_params is None:
            if self.phimax > 0:
                self.adapt_params = constants.ADAPT_PARAMS
            else:
                self.adapt_params = np.zeros((2, 0), dtype=np.float64)
        else:
            self.adapt_params = adapt_params

    def preprocess_standard_track(
        self,
        data: xr.DataArray,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        map_proj: Projection = "global",
        nside: int | None = None,
        resolution: float | None = 100.0,
        extent: MapExtent | None = None,
        filter_type: Literal["sht", "dct", "auto"] = "auto",
    ) -> tuple[xr.DataArray, tuple[ProcessingStep, ...]]:
        return preprocess_tracking_data(
            data,
            lmin=lmin,
            lmax=lmax,
            taper_points=taper_points,
            projection=map_proj,
            nside=nside,
            resolution=resolution,
            extent=extent,
            filter_type=filter_type,
        )

    def track(
        self,
        infile: str | Path | xr.DataArray | xr.Dataset,
        variable_name: str,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
        mode: ModeOption | None = "auto",
        map_proj: Literal["global", "nh_stereo", "sh_stereo", "healpix"] = "global",
        resolution: float = 100.0,
        extent: MapExtent | None = None,
        backend: Literal["serial", "mpi", "dask"] = "serial",
        n_workers: int | None = None,
        max_chunk_size: int | None = None,
        threshold: float | None = None,
        engine: str | None = None,
        overlap: int = model_constants.OVERLAP_DEFAULT,
        min_points: int = constants.MIN_POINTS_DEFAULT,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        nside: int | None = None,
        subgrid_refine: bool = True,
        **kwargs: float | str | None,
    ) -> Tracks:
        """
        Runs the Hodges tracking algorithm.
        Supports chunked detection if max_chunk_size is provided. Detections are
        gathered before a single linking pass so chunk boundaries do not affect
        the result.

        Args:
            infile: Path to the input data file.
            variable_name: Variable name to track.
            start_time, end_time: Time range for tracking.
            mode: Search for 'min' or 'max' extrema.
            backend: Processing backend (serial, mpi, dask).
            n_workers: Number of parallel workers.
            max_chunk_size: Number of steps per time chunk.
            threshold: Intensity threshold for detection.
            engine: Data loading engine (netcdf4, h5netcdf, etc).
            overlap: Retained for cross-tracker API compatibility. Hodges gathers
                detections before linking and does not require overlap.
            min_points: Minimum grid points per object.
            lmin, lmax: Optional spectral filter bounds.
            taper_points: Independent boundary tapering points.
        """
        import timeit

        resolved_mode = resolve_mode(variable_name, mode)

        if backend != "serial":
            raise NotImplementedError(
                "HodgesTracker currently supports only the serial backend."
            )
        if max_chunk_size is not None and max_chunk_size < 1:
            raise ValueError("max_chunk_size must be positive")

        t_total_start = timeit.default_timer()

        # 1. Load and optionally filter data
        t0 = timeit.default_timer()
        data_xr = normalize_tracking_data(
            infile,
            variable_name,
            start_time=start_time,
            end_time=end_time,
            engine=engine,
        )
        data_xr, threshold, stored_unit = normalize_variable_units(
            data_xr,
            variable_name=variable_name,
            threshold=threshold,
        )

        bounds = spatial_bounds_from_xarray(data_xr)
        processing: tuple[ProcessingStep, ...] = ()

        data_xr, processing = self.preprocess_standard_track(
            data_xr,
            lmin=lmin,
            lmax=lmax,
            taper_points=taper_points,
            map_proj=map_proj,
            nside=nside,
            resolution=resolution,
            extent=extent,
        )
        t1 = timeit.default_timer()
        print(f"    [Serial] Preprocessing time: {t1 - t0:.4f}s")

        if max_chunk_size is None:
            tracks = self._track_single_chunk_from_data(
                data_xr,
                primary_var=variable_name,
                mode=resolved_mode,
                bounds=bounds,
                threshold=threshold,
                unit=stored_unit,
                min_points=min_points,
                subgrid_refine=subgrid_refine,
                processing=processing,
                **kwargs,
            )
        else:
            # Detection can be partitioned, but linking must see the full series.
            from ..io.data_loader import DataLoader

            time_dim = DataLoader(data_xr).get_coords()[0]
            n_steps = data_xr.sizes[time_dim]
            detections: list[RawDetectionStep] = []

            for start_idx in range(0, n_steps, max_chunk_size):
                end_idx = min(start_idx + max_chunk_size, n_steps)
                chunk_data = data_xr.isel({time_dim: slice(start_idx, end_idx)})
                detections.extend(
                    self._detect_single_chunk_from_data(
                        chunk_data,
                        primary_var=variable_name,
                        mode=resolved_mode,
                        threshold=threshold,
                        min_points=min_points,
                        subgrid_refine=subgrid_refine,
                        **kwargs,
                    )
                )
            tracks = self._link_detections(
                detections,
                primary_var=variable_name,
                mode=resolved_mode,
                bounds=bounds,
                unit=stored_unit,
                processing=processing,
            )

        t_total_end = timeit.default_timer()
        print(f"Tracking time: {t_total_end - t_total_start:.4f}s")
        return tracks

    def _track_single_chunk_from_data(
        self,
        data: xr.DataArray,
        *,
        primary_var: str,
        mode: Mode,
        bounds: SpatialBounds | None,
        unit: str | None,
        processing: tuple[ProcessingStep, ...],
        threshold: float | None = None,
        min_points: int = constants.MIN_POINTS_DEFAULT,
        subgrid_refine: bool = True,
        **kwargs: float | str | None,
    ) -> Tracks:
        detections = self._detect_single_chunk_from_data(
            data,
            primary_var=primary_var,
            mode=mode,
            threshold=threshold,
            min_points=min_points,
            subgrid_refine=subgrid_refine,
            **kwargs,
        )
        return self._link_detections(
            detections,
            primary_var=primary_var,
            mode=mode,
            bounds=bounds,
            unit=unit,
            processing=processing,
        )

    def _detect_single_chunk_from_data(
        self,
        data: xr.DataArray,
        *,
        primary_var: str,
        mode: Mode,
        threshold: float | None = None,
        min_points: int = constants.MIN_POINTS_DEFAULT,
        subgrid_refine: bool = True,
        **kwargs: float | str | None,
    ) -> list[RawDetectionStep]:
        import timeit

        # 1. Detection
        t_detect_start = timeit.default_timer()
        detector = HodgesDetector.from_xarray(data, variable_name=primary_var)

        size = get_int_option(kwargs, "size", 5)

        detections = detector.detect(
            size=size,
            threshold=threshold,
            minmaxmode=mode,
            min_points=min_points,
            subgrid_refine=subgrid_refine,
        )

        map_proj = data.attrs.get("map_proj", "global")
        if map_proj in ("nh_stereo", "sh_stereo"):
            from ..models.geo import stereo_to_latlon

            hemi = 1 if map_proj == "nh_stereo" else -1
            converted_detections = []
            for dt, lats, lons, values in detections:
                new_lats = np.zeros_like(lats)
                new_lons = np.zeros_like(lons)
                for i in range(len(lats)):
                    # Note: lons[i] is x, lats[i] is y
                    lat, lon = stereo_to_latlon(lons[i], lats[i], hemi)
                    new_lats[i] = lat
                    new_lons[i] = lon
                converted_detections.append((dt, new_lats, new_lons, values))
            detections = converted_detections

        t_detect_end = timeit.default_timer()
        print(f"    [Serial] Detection time: {t_detect_end - t_detect_start:.4f}s")

        return detections

    def _link_detections(
        self,
        detections: list[RawDetectionStep],
        *,
        primary_var: str,
        mode: Mode,
        bounds: SpatialBounds | None = None,
        unit: str | None = None,
        processing: tuple[ProcessingStep, ...] = (),
    ) -> Tracks:
        import timeit

        # Linking uses the MGE cost function with adaptive constraints.
        # Cost = w1 * (1 - cos(theta)) + w2 * (1 - 2*sqrt(d1*d2)/(d1+d2))
        # This penalizes both changes in direction and changes in speed.
        t_link_start = timeit.default_timer()
        linker = HodgesLinker(
            w1=self.w1,
            w2=self.w2,
            dmax=self.dmax,
            phimax=self.phimax,
            n_iterations=self.n_iterations,
            max_missing=self.max_missing,
            zones=self.zones,
            adapt_params=self.adapt_params,
        )

        tracks = linker.link(
            detections,
            primary_var=primary_var,
            mode=mode,
            bounds=bounds,
            unit=unit,
            processing=processing,
        )
        t_link_end = timeit.default_timer()
        print(f"    [Serial] Linking time: {t_link_end - t_link_start:.4f}s")

        # Pruning
        valid_indices = np.asarray(
            [index for index, tr in enumerate(tracks) if len(tr) >= self.min_lifetime],
            dtype=np.int64,
        )
        return tracks.subset(valid_indices)
