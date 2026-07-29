from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..models.constants import DEGTORAD, R_EARTH_KM
from ..models.geo import geod_dist_km


@dataclass(slots=True)
class Center:
    """Represents a detected storm center at a specific time and location."""

    time: np.datetime64
    lat: float
    lon: float
    vars: dict[str, float]

    def __repr__(self) -> str:
        return str(self.vars)

    def __str__(self) -> str:
        return f"[time={self.time}, lat={self.lat}, lon={self.lon}, vars={self.vars}]"

    def abs_dist(self, center: Center) -> float:
        """Calculate clamped great-circle distance in kilometers."""
        return float(geod_dist_km(self.lat, self.lon, center.lat, center.lon))

    def lat_dist(self, center: Center) -> float:
        """Calculates the latitudinal distance in km."""
        dlat = center.lat - self.lat
        return R_EARTH_KM * dlat * DEGTORAD

    def lon_dist(self, center: Center) -> float:
        """Calculates the longitudinal distance in km, adjusted for latitude."""
        avglat = (self.lat + center.lat) / 2
        dlon = center.lon - self.lon
        return R_EARTH_KM * dlon * DEGTORAD * math.cos(avglat * DEGTORAD)
