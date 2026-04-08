from importlib.metadata import PackageNotFoundError, version

from .healpix.tracker import HealpixTracker
from .hodges.tracker import HodgesTracker
from .models import Center, TimeRange, Tracker, Tracks
from .preprocessing.regrid import SpectralRegridder
from .simple import SimpleDetector, SimpleLinker, SimpleTracker

try:
    __version__ = version("pystormtracker")
except PackageNotFoundError:
    __version__ = "0.5.0"

__all__ = [
    "Center",
    "HealpixTracker",
    "HodgesTracker",
    "SimpleDetector",
    "SimpleLinker",
    "SimpleTracker",
    "SpectralRegridder",
    "TimeRange",
    "Tracker",
    "Tracks",
]
