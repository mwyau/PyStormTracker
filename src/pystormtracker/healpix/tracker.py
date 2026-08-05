from __future__ import annotations

import timeit
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..hodges import constants
from ..io.data_loader import normalize_tracking_data
from ..models import Tracks
from ..models.geo import SpatialBounds, spatial_bounds_from_xarray
from ..models.tracker import RawDetectionStep, Tracker
from ..models.tracks import (
    ProcessingStep,
)
from ..models.units import Mode, ModeOption, normalize_variable_units, resolve_mode
from ..preprocessing.tracking import preprocess_tracking_data
from ..time import TimeInput
from .detector import HealpixDetector

if TYPE_CHECKING:
    from ..models.geo import MapExtent


def _detect_and_gather(
    detector: HealpixDetector,
    threshold: float | None,
    mode: Mode,
    min_points: int,
    subgrid_refine: bool,
) -> list[RawDetectionStep]:
    """Worker task: Detects centers on HEALPix and returns raw results."""
    return detector.detect(
        threshold=threshold,
        minmaxmode=mode,
        min_points=min_points,
        subgrid_refine=subgrid_refine,
    )


class HealpixTracker(Tracker):
    """
    A tracker specifically designed for 1D HEALPix grids.
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
        nside: int | None = None,
    ) -> tuple[xr.DataArray, tuple[ProcessingStep, ...]]:
        return preprocess_tracking_data(
            data,
            lmin=lmin,
            lmax=lmax,
            taper_points=taper_points,
            projection="healpix",
            nside=nside,
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
        overlap: int = 3,
        min_points: int = 1,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        nside: int | None = None,
        subgrid_refine: bool = True,
        **kwargs: float | str | None,
    ) -> Tracks:

        t0 = timeit.default_timer()
        resolved_mode = resolve_mode(variable_name, mode)

        if backend == "serial":
            # Normalize every supported public input to one selected DataArray.
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

            bounds: SpatialBounds | None = spatial_bounds_from_xarray(data_xr)
            processing: tuple[ProcessingStep, ...] = ()

            data_xr, processing = self.preprocess_standard_track(
                data_xr,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                nside=nside,
            )

            detector = HealpixDetector.from_xarray(data_xr, variable_name=variable_name)
            raw_steps = _detect_and_gather(
                detector,
                threshold=threshold,
                mode=resolved_mode,
                min_points=min_points,
                subgrid_refine=subgrid_refine,
            )
            from ..hodges.linker import HodgesLinker

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
                raw_steps,
                primary_var=variable_name,
                mode=resolved_mode,
                bounds=bounds,
                unit=stored_unit,
                processing=processing,
            )
        else:
            msg = f"Backend '{backend}' not yet implemented for HealpixTracker."
            raise NotImplementedError(msg)

        t_end = timeit.default_timer()
        print(f"Total HEALPix tracking time: {t_end - t0:.4f}s")

        return tracks
