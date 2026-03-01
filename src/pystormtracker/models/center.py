import math
from typing import Any


class Center:
    R: float = 6367.0
    DEGTORAD: float = math.pi / 180.0

    def __init__(self, time: Any, lat: float, lon: float, var: Any) -> None:
        self.time = time
        self.lat = lat
        self.lon = lon
        self.var = var

    def __repr__(self) -> str:
        return str(self.var)

    def __str__(self) -> str:
        return f"[time={self.time}, lat={self.lat}, lon={self.lon}, var={self.var}]"

    def abs_dist(self, center: "Center") -> float:
        """Haversine formula for calculating the great circle distance"""

        if not isinstance(center, Center):
            raise TypeError("must be compared with a Center object")

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

    def lat_dist(self, center: "Center") -> float:
        if not isinstance(center, Center):
            raise TypeError("must be compared with a Center object")

        dlat = center.lat - self.lat

        return self.R * dlat * self.DEGTORAD

    def lon_dist(self, center: "Center") -> float:
        if not isinstance(center, Center):
            raise TypeError("must be compared with a Center object")

        avglat = (self.lat + center.lat) / 2
        dlon = center.lon - self.lon

        return self.R * dlon * self.DEGTORAD * math.cos(avglat * self.DEGTORAD)
