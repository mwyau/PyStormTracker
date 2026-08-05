from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.data_loader import normalize_tracking_data
from ..models.geo import SpatialBounds, spatial_bounds_from_xarray
from ..models.time import TimeInput
from ..models.tracker import RawDetectionStep, Tracker, TrackingInput
from ..models.tracks import (
    ProcessingStep,
    Tracks,
)
from ..models.units import Mode, ModeOption, normalize_variable_units, resolve_mode
from ..preprocessing.tracking import (
    Projection,
    preprocess_tracking_data,
    resolve_filter_bounds,
)
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
        *,
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
        map_proj: Projection = "global",
        resolution: float = 100.0,
        extent: MapExtent | None = None,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        size: int = 5,
        min_points: int = constants.MIN_POINTS_DEFAULT,
        subgrid_refine: bool = True,
        max_chunk_size: int | None = None,
    ) -> None:
        """
        Initialize the Hodges Tracker.
        """
        if w1 < 0.0 or w2 < 0.0:
            raise ValueError("w1 and w2 must be nonnegative")
        if dmax <= 0.0:
            raise ValueError("dmax must be positive")
        if phimax < 0.0:
            raise ValueError("phimax must be nonnegative")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if min_lifetime <= 0:
            raise ValueError("min_lifetime must be positive")
        if max_missing < 0:
            raise ValueError("max_missing must be nonnegative")
        if size <= 0:
            raise ValueError("size must be positive")
        if min_points <= 0:
            raise ValueError("min_points must be positive")
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if taper_points < 0:
            raise ValueError("taper_points must be nonnegative")
        if map_proj not in ("global", "nh_stereo", "sh_stereo", "healpix"):
            raise ValueError(f"unsupported map_proj {map_proj!r}")
        if max_chunk_size is not None and max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        resolve_filter_bounds(lmin, lmax)

        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.n_iterations = n_iterations
        self.min_lifetime = min_lifetime
        self.max_missing = max_missing
        self.map_proj = map_proj
        self.resolution = resolution
        self.extent = extent
        self.lmin = lmin
        self.lmax = lmax
        self.taper_points = taper_points
        self.size = size
        self.min_points = min_points
        self.subgrid_refine = subgrid_refine
        self.max_chunk_size = max_chunk_size

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

        resolved_mode = resolve_mode(variable_name, mode)
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

        data_xr, processing = self.preprocess_standard_track(
            data_xr,
            lmin=self.lmin,
            lmax=self.lmax,
            taper_points=self.taper_points,
            map_proj=self.map_proj,
            resolution=self.resolution,
            extent=self.extent,
        )
        t1 = timeit.default_timer()
        print(f"    [Serial] Preprocessing time: {t1 - t0:.4f}s")

        if self.max_chunk_size is None:
            tracks = self._track_single_chunk_from_data(
                data_xr,
                primary_var=variable_name,
                mode=resolved_mode,
                bounds=bounds,
                threshold=threshold,
                unit=stored_unit,
                processing=processing,
            )
        else:
            # Detection can be partitioned, but linking must see the full series.
            from ..io.data_loader import DataLoader

            time_dim = DataLoader(data_xr).get_coords()[0]
            n_steps = data_xr.sizes[time_dim]
            detections: list[RawDetectionStep] = []

            for start_idx in range(0, n_steps, self.max_chunk_size):
                end_idx = min(start_idx + self.max_chunk_size, n_steps)
                chunk_data = data_xr.isel({time_dim: slice(start_idx, end_idx)})
                detections.extend(
                    self._detect_single_chunk_from_data(
                        chunk_data,
                        primary_var=variable_name,
                        mode=resolved_mode,
                        threshold=threshold,
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
    ) -> Tracks:
        detections = self._detect_single_chunk_from_data(
            data,
            primary_var=primary_var,
            mode=mode,
            threshold=threshold,
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
    ) -> list[RawDetectionStep]:
        import timeit

        t_detect_start = timeit.default_timer()
        detector = HodgesDetector.from_xarray(data, variable_name=primary_var)

        detections = detector.detect(
            size=self.size,
            threshold=threshold,
            minmaxmode=mode,
            min_points=self.min_points,
            subgrid_refine=self.subgrid_refine,
        )

        map_proj = data.attrs.get("map_proj", "global")
        if map_proj in ("nh_stereo", "sh_stereo"):
            from ..models.geo import stereo_to_latlon

            hemi = 1 if map_proj == "nh_stereo" else -1
            converted_detections: list[RawDetectionStep] = []
            for dt, lats, lons, values in detections:
                new_lats = np.zeros_like(lats)
                new_lons = np.zeros_like(lons)
                for i in range(len(lats)):
                    # Note: lons[i] is x, lats[i] is y
                    lat, lon = stereo_to_latlon(lons[i], lats[i], hemi)
                    new_lats[i] = lat
                    new_lons[i] = lon
                converted_detections.append(
                    RawDetectionStep(dt, new_lats, new_lons, values)
                )
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

        valid_indices = np.asarray(
            [index for index, tr in enumerate(tracks) if len(tr) >= self.min_lifetime],
            dtype=np.int64,
        )
        return tracks.subset(valid_indices)
