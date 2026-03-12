from importlib.metadata import PackageNotFoundError, version

from .models import Center, Tracks
from .simple import SimpleDetector, SimpleLinker, SimpleTracker

try:
    __version__ = version("pystormtracker")
except PackageNotFoundError:
    __version__ = "0.4.0.dev"

__all__ = ["Center", "SimpleDetector", "SimpleLinker", "SimpleTracker", "Tracks"]
