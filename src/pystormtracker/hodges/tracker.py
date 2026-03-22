from __future__ import annotations

from typing import Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..models.tracker import Tracker
from ..models.tracks import Tracks
from ..preprocessing.sh_filter import SphericalHarmonicFilter
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

        if zones is None:
            if use_standard_constraints:
                self.zones = constants.TRACK_ZONES
            else:
                self.zones = np.zeros((0, 5), dtype=np.float64)
        else:
            self.zones = zones

        if adapt_thresholds is None:
            if use_standard_constraints:
                self.adapt_thresholds = constants.ADAPT_THRESHOLDS
            else:
                self.adapt_thresholds = np.zeros(0, dtype=np.float64)
        else:
            self.adapt_thresholds = adapt_thresholds

        if adapt_values is None:
            if use_standard_constraints:
                self.adapt_values = constants.ADAPT_VALUES
            else:
                self.adapt_values = np.zeros(0, dtype=np.float64)
        else:
            self.adapt_values = adapt_values

    def preprocess_standard_track(
        self,
        data: xr.DataArray,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
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
        sh_filter = SphericalHarmonicFilter(lmin=lmin, lmax=lmax)
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
        min_points: int = constants.MIN_POINTS_DEFAULT,
        filter: bool = True,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
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
        """
        # 1. Load and optionally filter data
        detector_peek = HodgesDetector(infile, varname, engine=engine)
        if start_time is None or end_time is None:
            full_times = detector_peek.get_time()
            if start_time is None:
                start_time = full_times[0]
            if end_time is None:
                end_time = full_times[-1]

        data_xr = detector_peek.get_xarray(start_time, end_time)

        if filter:
            data_xr = self.preprocess_standard_track(data_xr, lmin=lmin, lmax=lmax)

        if max_chunk_size is None:
            return self._track_single_chunk_from_data(
                data_xr,
                mode,
                threshold,
                min_points=min_points,
                **kwargs,
            )

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

        return self._splice_tracks(tracks_all, overlap)

    def _track_single_chunk_from_data(
        self,
        data: xr.DataArray,
        mode: Literal["min", "max"] = "min",
        threshold: float | None = None,
        min_points: int = constants.MIN_POINTS_DEFAULT,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        # 1. Detection
        print(f"DEBUG: Entering Detection with data shape {data.shape}", flush=True)
        detector = HodgesDetector.from_xarray(data)

        size = int(kwargs.get("size", 5))  # type: ignore[arg-type]

        detections = detector.detect(
            size=size, threshold=threshold, minmaxmode=mode, min_points=min_points
        )
        print(f"DEBUG: Detection complete. Found detections for {len(detections)} time steps.", flush=True)

        # 2. Linking (MGE with adaptive constraints)
        print("DEBUG: Entering Linking (MGE)...", flush=True)
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
        print(f"DEBUG: Linking complete. Found {len(tracks)} raw tracks.", flush=True)

        # 3. Pruning
        valid_tracks = []
        for tr in tracks:
            if len(tr) >= self.min_lifetime:
                valid_tracks.append(tr)

        out = Tracks()
        for tr in valid_tracks:
            out.append(tr)

        print(f"DEBUG: Pruning complete. Returning {len(out)} valid tracks.", flush=True)
        return out
