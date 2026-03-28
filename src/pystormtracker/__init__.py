from importlib.metadata import PackageNotFoundError, version

from .hodges.tracker import HodgesTracker
from .models import Center, TimeRange, Tracks
from .simple import SimpleDetector, SimpleLinker, SimpleTracker

try:
    __version__ = version("pystormtracker")
except PackageNotFoundError:
    __version__ = "0.5.0.dev0"

__all__ = [
    "Center",
    "HodgesTracker",
    "SimpleDetector",
    "SimpleLinker",
    "SimpleTracker",
    "TimeRange",
    "Tracks",
]
