from __future__ import annotations

from typing import Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..models.tracker import Tracker
from ..models.tracks import TimeRange, Tracks
from ..preprocessing.sh_filter import SphericalHarmonicFilter
from ..preprocessing.taper import TaperFilter
from .detector import HodgesDetector
from .linker import HodgesLinker


class HodgesTracker(Tracker):
    """
    A tracker implementing the Hodges (TRACK) algorithm with adaptive constraints.
    """

    # Standard TRACK legacy defaults (Hodges 1999)
    STANDARD_ZONES = np.array(
        [
            [0.0, 360.0, -90.0, -20.0, 6.5],
            [0.0, 360.0, -20.0, 20.0, 3.0],
            [0.0, 360.0, 20.0, 90.0, 6.5],
        ],
        dtype=np.float64,
    )
    STANDARD_ADAPT_THRESHOLDS = np.array([1.0, 2.0, 5.0, 8.0], dtype=np.float64)
    STANDARD_ADAPT_VALUES = np.array([1.0, 0.3, 0.1, 0.0], dtype=np.float64)

    def __init__(
        self,
        w1: float = 0.2,
        w2: float = 0.8,
        dmax: float = 6.5,
        phimax: float = 0.5,
        n_iterations: int = 3,
        min_lifetime: int = 3,
        max_missing: int = 0,
        zones: NDArray[np.float64] | None = None,
        adapt_thresholds: NDArray[np.float64] | None = None,
        adapt_values: NDArray[np.float64] | None = None,
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
            adapt_thresholds: Adaptive smoothness distance thresholds (4 points).
            adapt_values: Adaptive smoothness phi values (4 points).
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

        if zones is None and use_standard_constraints:
            self.zones = self.STANDARD_ZONES
        else:
            self.zones = zones

        if adapt_thresholds is None and use_standard_constraints:
            self.adapt_thresholds = self.STANDARD_ADAPT_THRESHOLDS
        else:
            self.adapt_thresholds = adapt_thresholds

        if adapt_values is None and use_standard_constraints:
            self.adapt_values = self.STANDARD_ADAPT_VALUES
        else:
            self.adapt_values = adapt_values

    def preprocess_standard_track(
        self, data: xr.DataArray, truncation: int = 42, taper_points: int = 10
    ) -> xr.DataArray:
        """
        Applies standard TRACK preprocessing: Tapering -> Spherical Harmonic Filter.
        """
        # 1. Tapering
        if taper_points > 0:
            taper = TaperFilter(n_points=taper_points)
            data = cast(xr.DataArray, taper.filter(data))

        # 2. Spectral Filtering
        sh_filter = SphericalHarmonicFilter(lmin=5, lmax=truncation)
        data = sh_filter.filter(data)

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
        overlap: int = 3,
        min_points: int = 1,
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
        """
        # Set default times if not provided
        if start_time is None or end_time is None:
            detector_peek = HodgesDetector(infile, varname, engine=engine)
            full_times = detector_peek.get_time()
            if start_time is None:
                start_time = full_times[0]
            if end_time is None:
                end_time = full_times[-1]

        if max_chunk_size is None:
            return self._track_single_chunk(
                infile,
                varname,
                start_time,
                end_time,
                mode,
                threshold,
                engine,
                min_points=min_points,
                **kwargs,
            )

        # 1. Determine time chunks
        detector = HodgesDetector(
            infile,
            varname,
            time_range=TimeRange(
                start=np.datetime64(start_time), end=np.datetime64(end_time)
            ),
            engine=engine,
        )
        times = detector.get_time()

        n_steps = len(times)
        tracks_all = []

        start_idx = 0
        while start_idx < n_steps:
            end_idx = min(start_idx + max_chunk_size, n_steps)

            t_start = times[start_idx]
            t_end = times[end_idx - 1]

            chunk_res = self._track_single_chunk(
                infile,
                varname,
                t_start,
                t_end,
                mode,
                threshold,
                engine,
                min_points=min_points,
                **kwargs,
            )
            tracks_all.append(chunk_res)

            if end_idx == n_steps:
                break
            start_idx = end_idx - overlap

        return self._splice_tracks(tracks_all, overlap)

    def _track_single_chunk(
        self,
        infile: str,
        varname: str,
        start_time: str | np.datetime64 | None = None,
        end_time: str | np.datetime64 | None = None,
        mode: Literal["min", "max"] = "min",
        threshold: float | None = None,
        engine: str | None = None,
        min_points: int = 1,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        # 1. Detection
        detector = HodgesDetector(
            pathname=infile,
            varname=varname,
            time_range=TimeRange(
                start=np.datetime64(start_time), end=np.datetime64(end_time)
            )
            if start_time
            else None,
            engine=engine,
        )

        size = int(kwargs.get("size", 5))  # type: ignore[arg-type]

        detections = detector.detect(
            size=size, threshold=threshold, minmaxmode=mode, min_points=min_points
        )

        # 2. Linking (MGE with adaptive constraints)
        linker = HodgesLinker(
            w1=self.w1,
            w2=self.w2,
            dmax=self.dmax,
            phimax=self.phimax,
            n_iterations=self.n_iterations,
            max_missing=self.max_missing,
            zones=self.zones,
            adapt_thresholds=self.adapt_thresholds,
            adapt_values=self.adapt_values,
        )

        tracks = linker.link(detections)

        # 3. Pruning
        valid_tracks = []
        for tr in tracks:
            if len(tr) >= self.min_lifetime:
                valid_tracks.append(tr)

        out = Tracks()
        for tr in valid_tracks:
            out.append(tr)

        return out
