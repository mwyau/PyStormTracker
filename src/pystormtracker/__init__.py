from importlib.metadata import PackageNotFoundError, version

from .healpix.tracker import HealpixTracker
from .hodges.tracker import HodgesTracker
from .models.tracker import Tracker
from .models.tracks import Track, Tracks
from .simple.tracker import SimpleTracker

try:
    __version__ = version("pystormtracker")
except PackageNotFoundError:
    __version__ = "0.6.1.dev1"


__all__ = [
    "HealpixTracker",
    "HodgesTracker",
    "SimpleTracker",
    "Track",
    "Tracker",
    "Tracks",
]
