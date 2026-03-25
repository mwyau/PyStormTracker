from __future__ import annotations

from typing import Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..models import constants as model_constants
from ..models.tracker import Tracker
from ..models.tracks import Tracks
from ..preprocessing.spectral import SpectralFilter
from ..preprocessing.taper import TaperFilter
from . import constants
from .detector import HodgesDetector
from .linker import HodgesLinker


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
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
        sht_engine: Literal["auto", "shtns", "ducc0"] = "auto",
    ) -> xr.DataArray:
        """
        Applies standard TRACK preprocessing: Tapering -> Spherical Harmonic Filter.
        """
        # Ensure data is loaded into memory for spectral filtering
        if data.chunks:
            data = data.compute()

        # 1. Tapering
        if taper_points > 0:
            taper = TaperFilter(n_points=taper_points)
            data = cast(xr.DataArray, taper.filter(data))

        # 2. Spectral Filtering
        spectral_filter = SpectralFilter(lmin=lmin, lmax=lmax, sht_engine=sht_engine)
        data = spectral_filter.filter(data)

        return data

    def _splice_tracks(self, tracks_all: list[Tracks], _overlap: int) -> Tracks:
        """
        Splices tracks from multiple overlapping time chunks.
        Matching logic: if tracks in chunk N end with same points as chunk N+1 head.
        """
        if not tracks_all:
            return Tracks()

        final_tracks = tracks_all[0]

        for i in range(1, len(tracks_all)):
            next_chunk = tracks_all[i]
            matched_next_indices = set()
            current_tails = list(final_tracks)

            for tr_tail in current_tails:
                last_pt = tr_tail[-1]

                for idx_next, tr_next in enumerate(next_chunk):
                    if idx_next in matched_next_indices:
                        continue

                    first_pt = tr_next[0]

                    # Match if time, lat, lon are identical
                    if (
                        last_pt.time == first_pt.time
                        and abs(last_pt.lat - first_pt.lat) < 1e-5
                        and abs(last_pt.lon - first_pt.lon) < 1e-5
                    ):
                        # Splice: extend skipping the first overlapping point
                        for j in range(1, len(tr_next)):
                            tr_tail.append(tr_next[j])

                        matched_next_indices.add(idx_next)
                        break

            # Add unmatched tracks from next_chunk as new tracks
            for idx_next, tr_next in enumerate(next_chunk):
                if idx_next not in matched_next_indices:
                    final_tracks.append(tr_next)

        return final_tracks

    def track(
        self,
        infile: str,
        varname: str,
        start_time: str | np.datetime64 | None = None,
        end_time: str | np.datetime64 | None = None,
        mode: Literal["min", "max"] = "min",
        backend: Literal["serial", "mpi", "dask"] = "serial",
        n_workers: int | None = None,
        max_chunk_size: int | None = None,
        threshold: float | None = None,
        engine: str | None = None,
        overlap: int = model_constants.OVERLAP_DEFAULT,
        min_points: int = constants.MIN_POINTS_DEFAULT,
        filter: bool = True,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
        sht_engine: Literal["auto", "shtns", "ducc0"] = "auto",
        **kwargs: float | int | str | None,
    ) -> Tracks:
        """
        Runs the Hodges tracking algorithm.
        Supports time-chunking (RSPLICE) if max_chunk_size is provided.

        Args:
            infile: Path to the input data file.
            varname: Variable name to track.
            start_time, end_time: Time range for tracking.
            mode: Search for 'min' or 'max' extrema.
            backend: Processing backend (serial, mpi, dask).
            n_workers: Number of parallel workers.
            max_chunk_size: Number of steps per time chunk.
            threshold: Intensity threshold for detection.
            engine: Data loading engine (netcdf4, h5netcdf, etc).
            overlap: Overlap between chunks for splicing.
            min_points: Minimum grid points per object.
            filter: If True, apply spectral filtering.
            lmin, lmax: Spectral truncation range (default T5-42).
            taper_points: Boundary tapering points.
            sht_engine: SHT backend engine.
        """
        import timeit

        t_total_start = timeit.default_timer()

        # 1. Load and optionally filter data
        t0 = timeit.default_timer()
        detector_peek = HodgesDetector(infile, varname, engine=engine)
        if start_time is None or end_time is None:
            full_times = detector_peek.get_time()
            if start_time is None:
                start_time = full_times[0]
            if end_time is None:
                end_time = full_times[-1]

        data_xr = detector_peek.get_xarray(start_time, end_time)

        if filter:
            data_xr = self.preprocess_standard_track(
                data_xr,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                sht_engine=sht_engine,
            )
        t1 = timeit.default_timer()
        print(f"    [Serial] Preprocessing time: {t1 - t0:.4f}s")

        if max_chunk_size is None:
            tracks = self._track_single_chunk_from_data(
                data_xr,
                mode,
                threshold,
                min_points=min_points,
                **kwargs,
            )
        else:
            # 2. Time-chunking logic (RSPLICE-style)
            time_dim = detector_peek._loader.get_coords()[0]
            n_steps = data_xr.sizes[time_dim]
            tracks_all = []

            start_idx = 0
            while start_idx < n_steps:
                end_idx = min(start_idx + max_chunk_size, n_steps)
                chunk_data = data_xr.isel({time_dim: slice(start_idx, end_idx)})

                chunk_res = self._track_single_chunk_from_data(
                    chunk_data,
                    mode,
                    threshold,
                    min_points=min_points,
                    **kwargs,
                )
                tracks_all.append(chunk_res)

                if end_idx == n_steps:
                    break
                start_idx = end_idx - overlap

            tracks = self._splice_tracks(tracks_all, overlap)

        t_total_end = timeit.default_timer()
        print(f"Tracking time: {t_total_end - t_total_start:.4f}s")
        return tracks

    def _track_single_chunk_from_data(
        self,
        data: xr.DataArray,
        mode: Literal["min", "max"] = "min",
        threshold: float | None = None,
        min_points: int = constants.MIN_POINTS_DEFAULT,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        import timeit

        # 1. Detection
        t_detect_start = timeit.default_timer()
        detector = HodgesDetector.from_xarray(data)

        size = int(kwargs.get("size", 5))  # type: ignore[arg-type]

        detections = detector.detect(
            size=size, threshold=threshold, minmaxmode=mode, min_points=min_points
        )
        t_detect_end = timeit.default_timer()
        print(f"    [Serial] Detection time: {t_detect_end - t_detect_start:.4f}s")

        # 2. Linking (MGE with adaptive constraints)
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

        tracks = linker.link(detections)
        t_link_end = timeit.default_timer()
        print(f"    [Serial] Linking time: {t_link_end - t_link_start:.4f}s")

        # 3. Pruning
        valid_tracks = []
        for tr in tracks:
            if len(tr) >= self.min_lifetime:
                valid_tracks.append(tr)

        out = Tracks()
        for tr in valid_tracks:
            out.append(tr)

        return out
