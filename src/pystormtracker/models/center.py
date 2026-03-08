from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Center:
    """Represents a detected storm center at a specific time and location."""

    time: float
    lat: float
    lon: float
    var: float

    # Earth radius in kilometers
    R: float = 6367.0
    # Conversion factor from degrees to radians
    DEGTORAD: float = math.pi / 180.0

    def __repr__(self) -> str:
        return str(self.var)

    def __str__(self) -> str:
        return f"[time={self.time}, lat={self.lat}, lon={self.lon}, var={self.var}]"

    def abs_dist(self, center: Center) -> float:
        """Haversine formula for calculating the great circle distance in km."""
        dlat = center.lat - self.lat
        dlon = center.lon - self.lon

        return (
            self.R
            * 2
            * math.asin(
                math.sqrt(
                    math.sin(dlat / 2 * self.DEGTORAD) ** 2
                    + math.cos(self.lat * self.DEGTORAD)
                    * math.cos(center.lat * self.DEGTORAD)
                    * math.sin(dlon / 2 * self.DEGTORAD) ** 2
                )
            )
        )

    def lat_dist(self, center: Center) -> float:
        """Calculates the latitudinal distance in km."""
        dlat = center.lat - self.lat
        return self.R * dlat * self.DEGTORAD

    def lon_dist(self, center: Center) -> float:
        """Calculates the longitudinal distance in km, adjusted for latitude."""
        avglat = (self.lat + center.lat) / 2
        dlon = center.lon - self.lon
        return self.R * dlon * self.DEGTORAD * math.cos(avglat * self.DEGTORAD)
