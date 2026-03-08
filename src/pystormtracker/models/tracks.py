from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import netCDF4

from .center import Center
from .time import TimeRange


@dataclass
class Track:
    """Represents a single storm track as a sequence of centers."""

    centers: list[Center] = field(default_factory=list)

    def __iter__(self) -> Iterator[Center]:
        return iter(self.centers)

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, index: int) -> Center:
        return self.centers[index]

    def append(self, center: Center) -> None:
        self.centers.append(center)

    def extend(self, other: Track) -> None:
        self.centers.extend(other.centers)

    def abs_dist(self, other: Track) -> float:
        """
        Distance between the last point of this track and
        the first point of another.
        """
        c1 = self.centers[-1]
        c2 = other[0]
        return c1.abs_dist(c2)


class Tracks:
    def __init__(self) -> None:
        self._tracks: list[Track] = []
        self.head: list[Track] = []
        self.tail: list[Track] = []
        self.time_range: TimeRange | None = None

    def __getitem__(self, index: int) -> Track:
        return self._tracks[index]

    def __setitem__(self, index: int, value: Track) -> None:
        self._tracks[index] = value

    def __iter__(self) -> Iterator[Track]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    def append(self, obj: Track) -> None:
        self._tracks.append(obj)

    @classmethod
    def from_imilast(cls, filename: Path | str) -> Tracks:
        """Loads tracks from an IMILAST format text file."""
        tracks_obj = cls()
        with open(filename) as f:
            lines = f.readlines()
            current_track_centers: list[Center] = []
            for line in lines[1:]:  # skip header
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "90":
                    if current_track_centers:
                        tracks_obj.append(Track(current_track_centers))
                    current_track_centers = []
                elif parts[0] == "00":
                    # Format: 00 CycloneNo StepNo DateI10 Year Month Day Hour Lon Lat
                    # Intensity1. Center takes (time, lat, lon, var).
                    lon, lat, var = float(parts[8]), float(parts[9]), float(parts[10])
                    time_val = float(parts[3])  # DateI10
                    current_track_centers.append(Center(time_val, lat, lon, var))
            if current_track_centers:
                tracks_obj.append(Track(current_track_centers))

        # Sort tracks by their first point's time, lat, lon for consistency
        tracks_obj._tracks.sort(
            key=lambda t: (t[0].time, t[0].lat, t[0].lon) if t else (0.0, 0.0, 0.0)
        )
        return tracks_obj

    def to_imilast(
        self,
        outfile: str,
        time_units: str = "",
        calendar: str = "standard",
        decimal_places: int = 4,
    ) -> None:
        """Exports tracks to an IMILAST format text file."""
        if not outfile.endswith(".txt"):
            outfile += ".txt"

        with open(outfile, "w", newline="") as f:
            header = (
                "99 00,CycloneNo,StepNo,DateI10,Year,Month,Day,Time,"
                "LongE,LatN,Intensity1\n"
            )
            f.write(header)

            for i, track in enumerate(self._tracks, start=1):
                f.write(f"90 {i} {len(track)}\n")
                for step, center in enumerate(track, start=1):
                    try:
                        dt_any: object = netCDF4.num2date(
                            [center.time], units=time_units, calendar=calendar
                        )
                        dt = dt_any[0]  # type: ignore[index]
                        yyyymmddhh = dt.strftime("%Y%m%d%H")
                        yyyy, mm, dd, hh = dt.year, dt.month, dt.day, dt.hour
                    except Exception:
                        # Fallback for DateI10 or unknown format
                        s_time = str(int(center.time))
                        if len(s_time) == 10:  # YYYYMMDDHH
                            yyyymmddhh = s_time
                            yyyy = int(s_time[:4])
                            mm = int(s_time[4:6])
                            dd = int(s_time[6:8])
                            hh = int(s_time[8:10])
                        else:
                            yyyymmddhh = "0000000000"
                            yyyy, mm, dd, hh = 0, 0, 0, 0

                    var_val = f"{float(center.var):.{decimal_places}f}"
                    lon = center.lon
                    if lon > 180:
                        lon -= 360

                    f.write(
                        f"00 {i} {step} {yyyymmddhh} {yyyy} {mm:02d} "
                        f"{dd:02d} {hh:02d} {lon:.2f} {center.lat:.2f} "
                        f"{var_val}\n"
                    )

    def compare(
        self,
        other: Tracks,
        length_diff_tol: int = 0,
        coord_tol: float = 1e-4,
        intensity_tol: float = 1e-4,
    ) -> None:
        """Compares this Tracks object with another for equality, ignoring order."""
        assert len(self) == len(other), (
            f"Track count mismatch: {len(self)} vs {len(other)}"
        )

        # Sort criteria for tracks: first center's time, then lat, then lon
        def track_key(t: Track) -> tuple[float, float, float]:
            if not t.centers:
                return (0.0, 0.0, 0.0)
            c = t.centers[0]
            return (c.time, c.lat, c.lon)

        sorted_self = sorted(self._tracks, key=track_key)
        sorted_other = sorted(other._tracks, key=track_key)

        for tr1, tr2 in zip(sorted_self, sorted_other, strict=False):
            assert abs(len(tr1) - len(tr2)) <= length_diff_tol, (
                f"Track length mismatch: {len(tr1)} vs {len(tr2)}"
            )

            # Robust matching using time as key for points within the track
            d1 = {c.time: c for c in tr1.centers}
            d2 = {c.time: c for c in tr2.centers}

            common_times = set(d1.keys()) & set(d2.keys())
            # Ensure we have a significant overlap
            assert len(common_times) >= min(len(tr1), len(tr2)) - length_diff_tol

            for t_val in common_times:
                c1, c2 = d1[t_val], d2[t_val]
                assert abs(c1.lat - c2.lat) <= coord_tol
                assert abs(c1.lon - c2.lon) <= coord_tol
                assert abs(c1.var - c2.var) <= intensity_tol
