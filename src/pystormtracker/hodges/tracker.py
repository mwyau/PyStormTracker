from __future__ import annotations

from typing import Literal

import numpy as np

from ..models.tracks import Tracks
from ..models.tracker import Tracker
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
    ) -> None:
        """
        Initialize the Hodges Tracker.

        Args:
            w1 (float): Weight for direction in cost function.
            w2 (float): Weight for speed in cost function.
            dmax (float): Maximum displacement in degrees.
            phimax (float): Penalty for phantom points.
            n_iterations (int): Number of MGE iterations.
            min_lifetime (int): Minimum number of steps for a valid track.
        """
        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.n_iterations = n_iterations
        self.min_lifetime = min_lifetime

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
        
        # Determine detection parameters from kwargs or defaults
        size = kwargs.get("size", 5)
        threshold = kwargs.get("threshold", None)
        
        detections = detector.detect(size=size, threshold=threshold, minmaxmode=mode)
        
        # 2. Linking (MGE)
        linker = HodgesLinker(
            w1=self.w1,
            w2=self.w2,
            dmax=self.dmax,
            phimax=self.phimax,
            n_iterations=self.n_iterations,
        )
        
        tracks = linker.link(detections)
        
        # 3. Pruning
        valid_tracks = []
        for track in tracks:
            if len(track) >= self.min_lifetime:
                valid_tracks.append(track)
                
        return Tracks(tracks=valid_tracks)
