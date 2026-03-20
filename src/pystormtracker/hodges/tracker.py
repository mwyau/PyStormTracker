from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..models.tracker import Tracker
from ..models.tracks import Tracks
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
        **kwargs,
    ) -> Tracks:
        """
        Runs the Hodges tracking algorithm.
        """
        # 1. Detection
        detector = HodgesDetector(
            pathname=infile,
            varname=varname,
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
