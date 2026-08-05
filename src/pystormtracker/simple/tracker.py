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
from ..models.tracker import Backend, RawDetectionStep, Tracker, TrackingInput
from ..models.tracks import ProcessingStep, Tracks, TracksMetadata, _TracksBuilder
from ..models.units import (
    Mode,
    ModeOption,
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
    mode: Mode,
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
    size: int,
    threshold: float | None,
    mode: Literal["min", "max"],
    subgrid_refine: bool = False,
) -> list[RawDetectionStep]:
    """Worker task: Detects centers and returns raw results for central linking."""
    return detector.detect(
        size=size,
        threshold=threshold,
        minmaxmode=mode,
        subgrid_refine=subgrid_refine,
    )


def _convert_stereo_steps(
    raw_steps: list[RawDetectionStep],
    map_proj: Literal["global", "nh_stereo", "sh_stereo", "healpix"],
) -> list[RawDetectionStep]:
    """Convert projected detection coordinates back to latitude and longitude."""
    if map_proj not in ("nh_stereo", "sh_stereo"):
        return raw_steps

    from ..models.geo import stereo_to_latlon

    hemisphere = 1 if map_proj == "nh_stereo" else -1
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
        map_proj: Literal["global", "nh_stereo", "sh_stereo", "healpix"] = "global",
        resolution: float = 100.0,
        extent: MapExtent | None = None,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        size: int = 5,
        subgrid_refine: bool = False,
        backend: Backend = "serial",
        n_workers: int | None = None,
        max_chunk_size: int | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if map_proj not in ("global", "nh_stereo", "sh_stereo", "healpix"):
            raise ValueError(f"unsupported map_proj {map_proj!r}")
        if backend not in ("serial", "mpi", "dask"):
            raise ValueError(f"unsupported backend {backend!r}")
        if n_workers is not None and n_workers <= 0:
            raise ValueError("n_workers must be positive")
        if max_chunk_size is not None and max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        resolve_filter_bounds(lmin, lmax)
        if taper_points < 0:
            raise ValueError("taper_points must be nonnegative")

        self.map_proj = map_proj
        self.resolution = resolution
        self.extent = extent
        self.lmin = lmin
        self.lmax = lmax
        self.taper_points = taper_points
        self.size = size
        self.subgrid_refine = subgrid_refine
        self.backend = backend
        self.n_workers = n_workers
        self.max_chunk_size = max_chunk_size

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

    def _detect_serial(
        self,
        infile: TrackingInput,
        variable_name: str,
        time_range: TimeRange | None,
        mode: Mode,
        threshold: float | None = None,
        engine: str | None = None,
    ) -> Tracks:
        import timeit

        t0 = timeit.default_timer()
        data_xr = normalize_tracking_data(
            infile,
            variable_name,
            start_time=time_range.start if time_range else None,
            end_time=time_range.end if time_range else None,
            engine=engine,
        )
        data_xr, threshold, variable_unit = normalize_variable_units(
            data_xr,
            variable_name=variable_name,
            threshold=threshold,
        )
        bounds = spatial_bounds_from_xarray(data_xr)

        data_xr, processing = self.preprocess_standard_track(
            data_xr,
            lmin=self.lmin,
            lmax=self.lmax,
            taper_points=self.taper_points,
            map_proj=self.map_proj,
            resolution=self.resolution,
            extent=self.extent,
        )
        t_pre = timeit.default_timer()
        print(f"    [Serial] Preprocessing time: {t_pre - t0:.4f}s")

        t0_detect = timeit.default_timer()
        detector = SimpleDetector.from_xarray(data_xr, variable_name=variable_name)
        raw_steps = _detect_and_link(
            detector,
            size=self.size,
            threshold=threshold,
            mode=mode,
            subgrid_refine=self.subgrid_refine,
        )
        raw_steps = _convert_stereo_steps(raw_steps, self.map_proj)

        t1 = timeit.default_timer()
        print(f"    [Serial] Detection time: {t1 - t0_detect:.4f}s")

        t2 = timeit.default_timer()
        tracks = _link_centers(
            raw_steps,
            primary_var=variable_name,
            mode=mode,
            bounds=bounds,
            unit=variable_unit,
            processing=processing,
        )
        t3 = timeit.default_timer()
        print(f"    [Serial] Linking time: {t3 - t2:.4f}s")
        return tracks

    def track(
        self,
        infile: TrackingInput,
        variable_name: str,
        *,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
        mode: ModeOption = "auto",
        threshold: float | None = None,
        engine: str | None = None,
    ) -> Tracks:
        import timeit

        t0 = timeit.default_timer()
        resolved_mode = resolve_mode(variable_name, mode)

        time_range = None
        if start_time is not None or end_time is not None:
            st = coerce_time_input(start_time)
            et = coerce_time_input(end_time)
            time_range = TimeRange(start=st, end=et)

        if self.backend in ("mpi", "dask") and isinstance(
            infile, (xr.DataArray, xr.Dataset)
        ):
            msg = (
                "Dask and MPI backends for SimpleTracker require a file path, "
                "not an xarray object."
            )
            raise NotImplementedError(msg)
        if self.backend == "mpi":
            from .concurrent import run_simple_mpi

            tracks = run_simple_mpi(
                str(infile),
                variable_name,
                time_range,
                resolved_mode,
                threshold=threshold,
                engine=engine,
                lmin=self.lmin,
                lmax=self.lmax,
                taper_points=self.taper_points,
                map_proj=self.map_proj,
                resolution=self.resolution,
                extent=self.extent,
                size=self.size,
                subgrid_refine=self.subgrid_refine,
            )
        elif self.backend == "dask":
            from .concurrent import run_simple_dask

            tracks = run_simple_dask(
                str(infile),
                variable_name,
                time_range,
                resolved_mode,
                self.n_workers,
                max_chunk_size=self.max_chunk_size,
                threshold=threshold,
                engine=engine,
                lmin=self.lmin,
                lmax=self.lmax,
                taper_points=self.taper_points,
                map_proj=self.map_proj,
                resolution=self.resolution,
                extent=self.extent,
                size=self.size,
                subgrid_refine=self.subgrid_refine,
            )
        else:
            tracks = self._detect_serial(
                infile,
                variable_name,
                time_range,
                resolved_mode,
                threshold=threshold,
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
