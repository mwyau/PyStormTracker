from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from ..io.data_loader import normalize_tracking_data
from ..models.geo import SpatialBounds, spatial_bounds_from_xarray
from ..models.time import (
    TimeInput,
    TimeRange,
    coerce_time_input,
    encode_time_values,
)
from ..models.tracker import (
    Backend,
    FeaturePointMethod,
    RawDetectionStep,
    Tracker,
    TrackingInput,
)
from ..models.tracks import ProcessingStep, Tracks, TracksMetadata, _TracksBuilder
from ..models.units import (
    DetectionMode,
    ResolvedDetectionMode,
    canonical_unit_for,
    normalize_variable_units,
    resolve_mode,
)
from ..preprocessing.tracking import (
    Projection,
    preprocess_tracking_data,
    resolve_filter_bounds,
)
from .detector import SimpleDetector
from .linker import SimpleLinker

if TYPE_CHECKING:
    from ..models.geo import MapExtent


def _link_centers(
    raw_steps: list[RawDetectionStep],
    *,
    primary_var: str,
    mode: ResolvedDetectionMode,
    bounds: SpatialBounds | None = None,
    unit: str | None = None,
    processing: tuple[ProcessingStep, ...] = (),
) -> Tracks:
    """Sequentially links raw detection steps into a global Tracks object."""
    units = {primary_var: unit or canonical_unit_for(primary_var) or "1"}
    numeric_steps: list[RawDetectionStep] = [
        RawDetectionStep(
            int(encode_time_values([step[0]])[0]),
            step[1],
            step[2],
            step[3],
        )
        for step in raw_steps
    ]
    builder = _TracksBuilder(
        TracksMetadata(primary_var, mode, units, bounds, processing)
    )
    linker = SimpleLinker()
    for step_data in numeric_steps:
        linker.append(builder, step_data)
    return builder.finish()


def _detect_and_link(
    detector: SimpleDetector,
    search_window_size: int,
    intensity_threshold: float | None,
    detection_mode: ResolvedDetectionMode,
    feature_point_method: FeaturePointMethod = "quadratic",
) -> list[RawDetectionStep]:
    """Worker task: Detects centers and returns raw results for central linking."""
    return detector.detect(
        search_window_size=search_window_size,
        intensity_threshold=intensity_threshold,
        detection_mode=detection_mode,
        feature_point_method=feature_point_method,
    )


def _convert_stereo_steps(
    raw_steps: list[RawDetectionStep],
    projection: Projection,
) -> list[RawDetectionStep]:
    """Convert projected detection coordinates back to latitude and longitude."""
    if projection not in ("nh_stereo", "sh_stereo"):
        return raw_steps

    from ..models.geo import stereo_to_latlon

    hemisphere = 1 if projection == "nh_stereo" else -1
    converted: list[RawDetectionStep] = []
    for time, y, x, values in raw_steps:
        lats = np.empty_like(y)
        lons = np.empty_like(x)
        for i in range(len(y)):
            lats[i], lons[i] = stereo_to_latlon(x[i], y[i], hemisphere)
        converted.append(RawDetectionStep(time, lats, lons, values))
    return converted


class SimpleTracker(Tracker):
    """
    A tracker implementing the PyStormTracker simple parallel algorithm.
    """

    def __init__(
        self,
        *,
        projection: Projection = "global",
        stereo_grid_spacing_km: float = 100.0,
        extent: MapExtent | None = None,
        filter_lmin: int | None = None,
        filter_lmax: int | None = None,
        taper_points: int = 0,
        search_window_size: int = 5,
        feature_point_method: FeaturePointMethod = "grid",
        backend: Backend = "serial",
        workers: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        if search_window_size <= 0 or search_window_size % 2 == 0:
            raise ValueError("search_window_size must be a positive odd integer")
        if stereo_grid_spacing_km <= 0.0:
            raise ValueError(
                "stereo_grid_spacing_km must be positive stereographic grid spacing "
                "in kilometres"
            )
        if projection not in ("global", "nh_stereo", "sh_stereo", "healpix"):
            raise ValueError(f"unsupported projection {projection!r}")
        if backend not in ("serial", "mpi", "dask"):
            raise ValueError(f"unsupported backend {backend!r}")
        if workers is not None and workers <= 0:
            raise ValueError("workers must be positive")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if feature_point_method not in ("grid", "quadratic"):
            raise ValueError(
                f"unsupported feature_point_method {feature_point_method!r}; "
                "expected 'grid' or 'quadratic'"
            )
        resolve_filter_bounds(filter_lmin, filter_lmax)
        if taper_points < 0:
            raise ValueError("taper_points must be nonnegative")

        self.projection = projection
        self.stereo_grid_spacing_km = stereo_grid_spacing_km
        self.extent = extent
        self.filter_lmin = filter_lmin
        self.filter_lmax = filter_lmax
        self.taper_points = taper_points
        self.search_window_size = search_window_size
        self.feature_point_method = feature_point_method
        self.backend = backend
        self.workers = workers
        self.chunk_size = chunk_size

    def preprocess_standard_track(
        self,
        data: xr.DataArray,
        filter_lmin: int | None = None,
        filter_lmax: int | None = None,
        taper_points: int = 0,
        projection: Projection = "global",
        nside: int | None = None,
        stereo_grid_spacing_km: float | None = 100.0,
        extent: MapExtent | None = None,
        filter_type: Literal["sht", "dct", "auto"] = "auto",
    ) -> tuple[xr.DataArray, tuple[ProcessingStep, ...]]:
        return preprocess_tracking_data(
            data,
            filter_lmin=filter_lmin,
            filter_lmax=filter_lmax,
            taper_points=taper_points,
            projection=projection,
            nside=nside,
            stereo_grid_spacing_km=stereo_grid_spacing_km,
            extent=extent,
            filter_type=filter_type,
        )

    def _detect_serial(
        self,
        data: TrackingInput,
        variable: str,
        time_range: TimeRange | None,
        detection_mode: ResolvedDetectionMode,
        intensity_threshold: float | None = None,
        engine: str | None = None,
    ) -> Tracks:
        import timeit

        t0 = timeit.default_timer()
        data_xr = normalize_tracking_data(
            data,
            variable,
            start_time=time_range.start if time_range else None,
            end_time=time_range.end if time_range else None,
            engine=engine,
        )
        data_xr, intensity_threshold, variable_unit = normalize_variable_units(
            data_xr,
            variable=variable,
            intensity_threshold=intensity_threshold,
        )
        bounds = spatial_bounds_from_xarray(data_xr)

        data_xr, processing = self.preprocess_standard_track(
            data_xr,
            filter_lmin=self.filter_lmin,
            filter_lmax=self.filter_lmax,
            taper_points=self.taper_points,
            projection=self.projection,
            stereo_grid_spacing_km=self.stereo_grid_spacing_km,
            extent=self.extent,
        )
        t_pre = timeit.default_timer()
        print(f"    [Serial] Preprocessing time: {t_pre - t0:.4f}s")

        t0_detect = timeit.default_timer()
        detector = SimpleDetector.from_xarray(data_xr, variable_name=variable)
        raw_steps = _detect_and_link(
            detector,
            search_window_size=self.search_window_size,
            intensity_threshold=intensity_threshold,
            detection_mode=detection_mode,
            feature_point_method=self.feature_point_method,
        )
        raw_steps = _convert_stereo_steps(raw_steps, self.projection)

        t1 = timeit.default_timer()
        print(f"    [Serial] Detection time: {t1 - t0_detect:.4f}s")

        t2 = timeit.default_timer()
        tracks = _link_centers(
            raw_steps,
            primary_var=variable,
            mode=detection_mode,
            bounds=bounds,
            unit=variable_unit,
            processing=processing,
        )
        t3 = timeit.default_timer()
        print(f"    [Serial] Linking time: {t3 - t2:.4f}s")
        return tracks

    def track(
        self,
        data: TrackingInput,
        variable: str,
        *,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
        detection_mode: DetectionMode = "auto",
        intensity_threshold: float | None = None,
        engine: str | None = None,
    ) -> Tracks:
        import timeit

        t0 = timeit.default_timer()
        resolved_mode = resolve_mode(variable, detection_mode)

        time_range = None
        if start_time is not None or end_time is not None:
            st = coerce_time_input(start_time)
            et = coerce_time_input(end_time)
            time_range = TimeRange(start=st, end=et)

        if self.backend in ("mpi", "dask") and isinstance(
            data, (xr.DataArray, xr.Dataset)
        ):
            msg = (
                "Dask and MPI backends for SimpleTracker require a file path, "
                "not an xarray object."
            )
            raise NotImplementedError(msg)
        if self.backend == "mpi":
            from .concurrent import run_simple_mpi

            tracks = run_simple_mpi(
                data=str(data),
                variable=variable,
                time_range=time_range,
                detection_mode=resolved_mode,
                intensity_threshold=intensity_threshold,
                engine=engine,
                filter_lmin=self.filter_lmin,
                filter_lmax=self.filter_lmax,
                taper_points=self.taper_points,
                projection=self.projection,
                stereo_grid_spacing_km=self.stereo_grid_spacing_km,
                extent=self.extent,
                search_window_size=self.search_window_size,
                feature_point_method=self.feature_point_method,
            )
        elif self.backend == "dask":
            from .concurrent import run_simple_dask

            tracks = run_simple_dask(
                data=str(data),
                variable=variable,
                time_range=time_range,
                detection_mode=resolved_mode,
                workers=self.workers,
                chunk_size=self.chunk_size,
                intensity_threshold=intensity_threshold,
                engine=engine,
                filter_lmin=self.filter_lmin,
                filter_lmax=self.filter_lmax,
                taper_points=self.taper_points,
                projection=self.projection,
                stereo_grid_spacing_km=self.stereo_grid_spacing_km,
                extent=self.extent,
                search_window_size=self.search_window_size,
                feature_point_method=self.feature_point_method,
            )
        else:
            tracks = self._detect_serial(
                data,
                variable,
                time_range,
                resolved_mode,
                intensity_threshold=intensity_threshold,
                engine=engine,
            )

        t_end = timeit.default_timer()
        rank = 0
        if self.backend == "mpi":
            from mpi4py import MPI

            rank = MPI.COMM_WORLD.Get_rank()

        if rank == 0:
            print(f"Tracking time: {t_end - t0:.4f}s")

        return tracks
