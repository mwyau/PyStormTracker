from importlib.metadata import PackageNotFoundError, version

from .hodges.tracker import HodgesTracker
from .models import Center, TimeRange, Tracks
from .preprocessing import apply_sh_filter
from .simple import SimpleDetector, SimpleLinker, SimpleTracker

try:
    __version__ = version("pystormtracker")
except PackageNotFoundError:
    __version__ = "0.5.0.dev"

__all__ = [
    "Center",
    "HodgesTracker",
    "SimpleDetector",
    "SimpleLinker",
    "SimpleTracker",
    "TimeRange",
    "Tracks",
    "apply_sh_filter",
]
