from __future__ import annotations

from typing import Literal

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

    def __init__(
        self,
        w1: float = 0.2,
        w2: float = 0.8,
        dmax: float = 5.0,
        phimax: float = 0.5,
        n_iterations: int = 3,
        min_lifetime: int = 3,
        max_missing: int = 0,
        zones: NDArray[np.float64] | None = None,
        adapt_thresholds: NDArray[np.float64] | None = None,
        adapt_values: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Initialize the Hodges Tracker.

        Args:
            w1 (float): Weight for direction in cost function.
            w2 (float): Weight for speed in cost function.
            dmax (float): Default maximum displacement in degrees.
            phimax (float): Penalty for phantom points (static cost).
            n_iterations (int): Number of MGE iterations (forward + backward).
            min_lifetime (int): Minimum number of steps for a valid track.
            max_missing (int): Maximum consecutive missing frames allowed.
            zones (np.ndarray): Regional dmax zones [lon_min, lon_max, lat_min, lat_max, dmax].
            adapt_thresholds (np.ndarray): Adaptive smoothness distance thresholds (4 points).
            adapt_values (np.ndarray): Adaptive smoothness phi values (4 points).
        """
        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.n_iterations = n_iterations
        self.min_lifetime = min_lifetime
        self.max_missing = max_missing
        
        self.zones = zones
        self.adapt_thresholds = adapt_thresholds
        self.adapt_values = adapt_values

    @classmethod
    def from_config(
        cls,
        zone_file: str | None = None,
        adapt_file: str | None = None,
        **kwargs
    ) -> HodgesTracker:
        """
        Creates a HodgesTracker instance loading regional/adaptive constraints from files.
        """
        tracker = cls(**kwargs)
        if zone_file:
            tracker.load_zones(zone_file)
        if adapt_file:
            tracker.load_adaptive_smoothness(adapt_file)
        return tracker

    def load_zones(self, filename: str) -> None:
        """Loads regional dmax zones from a TRACK-style zone.dat file."""
        with open(filename, "r") as f:
            lines = f.readlines()
            if not lines:
                return
            n_zones = int(lines[0].strip())
            zones = []
            for i in range(1, n_zones + 1):
                # Format: lon_min lon_max lat_min lat_max dmax
                zones.append([float(x) for x in lines[i].split()])
            self.zones = np.array(zones, dtype=np.float64)

    def load_adaptive_smoothness(self, filename: str) -> None:
        """Loads adaptive smoothness parameters from a TRACK-style adapt.dat file."""
        with open(filename, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                return
            # Line 1: distance thresholds
            self.adapt_thresholds = np.array([float(x) for x in lines[0].split()], dtype=np.float64)
            # Line 2: phi values
            self.adapt_values = np.array([float(x) for x in lines[1].split()], dtype=np.float64)

    def preprocess_standard_track(
        self,
        data: xr.DataArray,
        truncation: int = 42,
        taper_points: int = 10
    ) -> xr.DataArray:
        """
        Applies standard TRACK preprocessing: Tapering -> Spherical Harmonic Filter.
        """
        # 1. Tapering
        if taper_points > 0:
            taper = TaperFilter(n_points=taper_points)
            data = taper.filter(data)
            
        # 2. Spectral Filtering
        sh_filter = SphericalHarmonicFilter(truncation=truncation)
        data = sh_filter.filter(data)
        
        return data

    def _splice_tracks(self, tracks_all: list[Tracks], overlap: int) -> Tracks:
        """
        Splices tracks from multiple overlapping time chunks.
        Matching logic: if tracks in chunk N end at same time/lat/lon as tracks in chunk N+1.
        """
        if not tracks_all:
            return Tracks()
        
        final_tracks = Tracks()
        # Simply combining for now, a more sophisticated splicing based on overlap 
        # could be implemented here if bit-wise parity with TRACK's splicer is needed.
        # TRACK's splicer typically looks for track pieces that share at least 
        # 'overlap' points.
        
        for chunk_tracks in tracks_all:
            for track in chunk_tracks:
                final_tracks.append(track)
                
        # TODO: Implement point-matching splicing for overlapping windows
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
        engine: str | None = None,
        chunk_size: int | None = None,
        overlap: int = 3,
        **kwargs,
    ) -> Tracks:
        """
        Runs the Hodges tracking algorithm.
        Supports time-chunking (RSPLICE) if chunk_size is provided.
        """
        # Set default times if not provided
        if start_time is None or end_time is None:
            # We need to peek at the file to get full range
            detector_peek = HodgesDetector(infile, varname, engine=engine)
            full_times = detector_peek.get_time()
            if start_time is None:
                start_time = full_times[0]
            if end_time is None:
                end_time = full_times[-1]

        if chunk_size is None:
            return self._track_single_chunk(infile, varname, start_time, end_time, mode, engine, **kwargs)
        
        # 1. Determine time chunks
        detector = HodgesDetector(infile, varname, time_range=TimeRange(start_time, end_time), engine=engine)
        times = detector.get_time()
        
        n_steps = len(times)
        tracks_all = []
        
        start_idx = 0
        while start_idx < n_steps:
            end_idx = min(start_idx + chunk_size, n_steps)
            
            # Extract actual times for this chunk
            t_start = times[start_idx]
            t_end = times[end_idx - 1]
            
            chunk_res = self._track_single_chunk(infile, varname, t_start, t_end, mode, engine, **kwargs)
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
        engine: str | None = None,
        **kwargs,
    ) -> Tracks:
        # 1. Detection
        detector = HodgesDetector(
            pathname=infile,
            varname=varname,
            time_range=TimeRange(start_time, end_time) if start_time else None,
            engine=engine,
        )
        
        size = kwargs.get("size", 5)
        threshold = kwargs.get("threshold", None)
        
        detections = detector.detect(size=size, threshold=threshold, minmaxmode=mode)
        
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
        for track in tracks:
            if len(track) >= self.min_lifetime:
                valid_tracks.append(track)
                
        return Tracks(tracks=valid_tracks)
