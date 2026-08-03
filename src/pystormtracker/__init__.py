from importlib.metadata import PackageNotFoundError, version

from .healpix.tracker import HealpixTracker
from .hodges.tracker import HodgesTracker
from .models import (
    Center,
    TimeRange,
    Track,
    Tracker,
    Tracks,
    TracksBuilder,
    TracksMetadata,
    TrackSummaryColumns,
    compute_track_summaries,
)
from .preprocessing.regrid import SpectralRegridder
from .simple import SimpleDetector, SimpleLinker, SimpleTracker

try:
    __version__ = version("pystormtracker")
except PackageNotFoundError:
    __version__ = "0.6.0.dev0"


__all__ = [
    "Center",
    "HealpixTracker",
    "HodgesTracker",
    "SimpleDetector",
    "SimpleLinker",
    "SimpleTracker",
    "SpectralRegridder",
    "TimeRange",
    "Track",
    "TrackSummaryColumns",
    "Tracker",
    "Tracks",
    "TracksBuilder",
    "TracksMetadata",
    "compute_track_summaries",
]
