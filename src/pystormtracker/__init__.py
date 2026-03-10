from importlib.metadata import PackageNotFoundError, version

from .models import Center, Grid, Tracks
from .simple import SimpleDetector, SimpleLinker

try:
    __version__ = version("pystormtracker")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.3.3-dev"

__all__ = ["Center", "Grid", "SimpleDetector", "SimpleLinker", "Tracks"]
