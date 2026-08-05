from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from ..io.data_loader import normalize_tracking_data
from ..models import TimeRange, Tracks
from ..models.geo import SpatialBounds, spatial_bounds_from_xarray
from ..models.tracker import RawDetectionStep, get_int_option
from ..models.tracks import ProcessingStep, TracksBuilder, TracksMetadata
from ..models.units import (
    Mode,
    ModeOption,
    canonical_unit_for,
    normalize_variable_units,
    resolve_mode,
)
from ..preprocessing.tracking import Projection, preprocess_tracking_data
from ..time import (
    TimeInput,
    coerce_time_input,
    encode_time_values,
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
        (
            int(encode_time_values([step[0]])[0]),
            step[1],
            step[2],
            step[3],
        )
        for step in raw_steps
    ]
    builder = TracksBuilder(
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
        converted.append((time, lats, lons, values))
    return converted


class SimpleTracker:
    """
    A tracker implementing the PyStormTracker simple parallel algorithm.
    """

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
        infile: str | Path | xr.DataArray | xr.Dataset,
        variable_name: str,
        time_range: TimeRange | None,
        mode: Mode,
        threshold: float | None = None,
        engine: str | None = None,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        map_proj: Literal["global", "nh_stereo", "sh_stereo", "healpix"] = "global",
        nside: int | None = None,
        resolution: float = 100.0,
        extent: MapExtent | None = None,
        subgrid_refine: bool = False,
        **kwargs: float | str | None,
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
            lmin=lmin,
            lmax=lmax,
            taper_points=taper_points,
            map_proj=map_proj,
            nside=nside,
            resolution=resolution,
            extent=extent,
        )
        t_pre = timeit.default_timer()
        print(f"    [Serial] Preprocessing time: {t_pre - t0:.4f}s")

        t0_detect = timeit.default_timer()
        detector = SimpleDetector.from_xarray(data_xr, variable_name=variable_name)
        size = get_int_option(kwargs, "size", 5)
        raw_steps = _detect_and_link(
            detector,
            size=size,
            threshold=threshold,
            mode=mode,
            subgrid_refine=subgrid_refine,
        )
        raw_steps = _convert_stereo_steps(raw_steps, map_proj)

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
        overlap: int = 3,
        min_points: int = 1,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        nside: int | None = None,
        subgrid_refine: bool = False,
        **kwargs: float | str | None,
    ) -> Tracks:
        import timeit

        t0 = timeit.default_timer()
        resolved_mode = resolve_mode(variable_name, mode)

        time_range = None
        if start_time is not None or end_time is not None:
            st = coerce_time_input(start_time)
            et = coerce_time_input(end_time)
            time_range = TimeRange(start=st, end=et)

        if backend in ("mpi", "dask") and isinstance(
            infile, (xr.DataArray, xr.Dataset)
        ):
            msg = (
                "Dask and MPI backends for SimpleTracker require a file path, "
                "not an xarray object."
            )
            raise NotImplementedError(msg)
        if backend == "mpi":
            from .concurrent import run_simple_mpi

            tracks = run_simple_mpi(
                str(infile),
                variable_name,
                time_range,
                resolved_mode,
                threshold=threshold,
                engine=engine,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                nside=nside,
                map_proj=map_proj,
                resolution=resolution,
                extent=extent,
                subgrid_refine=subgrid_refine,
                **kwargs,
            )
        elif backend == "dask":
            from .concurrent import run_simple_dask

            tracks = run_simple_dask(
                str(infile),
                variable_name,
                time_range,
                resolved_mode,
                n_workers,
                max_chunk_size=max_chunk_size,
                threshold=threshold,
                engine=engine,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                nside=nside,
                map_proj=map_proj,
                resolution=resolution,
                extent=extent,
                subgrid_refine=subgrid_refine,
                **kwargs,
            )
        else:
            tracks = self._detect_serial(
                infile,
                variable_name,
                time_range,
                resolved_mode,
                threshold=threshold,
                engine=engine,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                nside=nside,
                map_proj=map_proj,
                resolution=resolution,
                extent=extent,
                subgrid_refine=subgrid_refine,
                **kwargs,
            )

        t_end = timeit.default_timer()
        rank = 0
        if backend == "mpi":
            from mpi4py import MPI

            rank = MPI.COMM_WORLD.Get_rank()

        if rank == 0:
            print(f"Tracking time: {t_end - t0:.4f}s")

        return tracks
