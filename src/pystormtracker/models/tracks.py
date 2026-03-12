from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .center import Center


@dataclass
class TimeRange:
    """Metadata for the time range covered by a set of tracks."""

    start: np.datetime64
    end: np.datetime64
    step: np.timedelta64 | None = None


class Track:
    """Represents a single storm track. In the array-backed architecture,
    it acts as a view into the parent Tracks object."""

    def __init__(
        self,
        track_id: int,
        tracks: Tracks
    ) -> None:
        self.track_id = track_id
        self._tracks = tracks

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Track):
            return False
        if len(self) != len(other):
            return False
        for c1, c2 in zip(self, other, strict=False):
            if (
                c1.time != c2.time
                or c1.lat != c2.lat
                or c1.lon != c2.lon
                or c1.vars != c2.vars
            ):
                return False
        return True

    @property
    def indices(self) -> NDArray[np.int64]:
        return np.where(self._tracks.track_ids == self.track_id)[0]

    def __iter__(self) -> Iterator[Center]:
        idx = self.indices
        for i in idx:
            yield Center(
                self._tracks.times[i],
                float(self._tracks.lats[i]),
                float(self._tracks.lons[i]),
                {k: float(v[i]) for k, v in self._tracks.vars.items()},
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Center:
        idx = self.indices[index]
        return Center(
            self._tracks.times[idx],
            float(self._tracks.lats[idx]),
            float(self._tracks.lons[idx]),
            {k: float(v[idx]) for k, v in self._tracks.vars.items()},
        )

    def append(self, center: Center) -> None:
        self._tracks.track_ids = np.append(self._tracks.track_ids, self.track_id)
        self._tracks.times = np.append(self._tracks.times, center.time)
        self._tracks.lats = np.append(self._tracks.lats, center.lat)
        self._tracks.lons = np.append(self._tracks.lons, center.lon)

        for k, v in center.vars.items():
            if k not in self._tracks.vars:
                # If a new var appears, fill previous points with NaN
                self._tracks.vars[k] = np.full(len(self._tracks.track_ids) - 1, np.nan)
            self._tracks.vars[k] = np.append(self._tracks.vars[k], v)

    def extend(self, other: Track) -> None:
        idx = other.indices
        if self._tracks is not other._tracks:
            self._tracks.track_ids = np.concatenate(
                [self._tracks.track_ids, np.full(len(idx), self.track_id)]
            )
            self._tracks.times = np.concatenate(
                [self._tracks.times, other._tracks.times[idx]]
            )
            self._tracks.lats = np.concatenate(
                [self._tracks.lats, other._tracks.lats[idx]]
            )
            self._tracks.lons = np.concatenate(
                [self._tracks.lons, other._tracks.lons[idx]]
            )

            for k in other._tracks.vars:
                if k not in self._tracks.vars:
                    self._tracks.vars[k] = np.full(
                        len(self._tracks.track_ids) - len(idx), np.nan
                    )
                self._tracks.vars[k] = np.concatenate(
                    [self._tracks.vars[k], other._tracks.vars[k][idx]]
                )
        else:
            other._tracks.track_ids[idx] = self.track_id

    def abs_dist(self, other: Track | Center) -> float:
        c1 = self[-1]
        c2 = other[0] if hasattr(other, "__getitem__") else other
        return c1.abs_dist(c2)


class Tracks:
    def __init__(
        self,
        track_ids: NDArray[np.int64] | None = None,
        times: NDArray[np.datetime64] | None = None,
        lats: NDArray[np.float64] | None = None,
        lons: NDArray[np.float64] | None = None,
        vars_dict: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        if track_ids is not None:
            self.track_ids = np.asarray(track_ids, dtype=np.int64)
            self.times = np.asarray(times, dtype="datetime64[s]")
            self.lats = np.asarray(lats, dtype=np.float64)
            self.lons = np.asarray(lons, dtype=np.float64)
            if vars_dict:
                self.vars = {
                    k: np.asarray(v, dtype=np.float64) for k, v in vars_dict.items()
                }
            else:
                self.vars = {}
        else:
            self.track_ids = np.empty(0, dtype=np.int64)
            self.times = np.empty(0, dtype="datetime64[s]")
            self.lats = np.empty(0, dtype=np.float64)
            self.lons = np.empty(0, dtype=np.float64)
            self.vars = {}

        self.time_range: TimeRange | None = None
        self._next_id = 0

        # Keep track of tails and heads using array of track_ids
        self._head_ids: set[int] = set()
        self._tail_ids: set[int] = set()

    def add_track(self, centers: list[Center]) -> Track:
        """Helper to append a new track from a list of Centers."""
        tid = self._get_new_id()
        if not centers:
            return Track(tid, self)

        times = np.array([c.time for c in centers], dtype="datetime64[s]")
        lats = np.array([c.lat for c in centers], dtype=np.float64)
        lons = np.array([c.lon for c in centers], dtype=np.float64)

        # Consolidate vars from centers
        var_keys: set[str] = set()
        for c in centers:
            var_keys.update(c.vars.keys())

        self.track_ids = np.concatenate([self.track_ids, np.full(len(centers), tid)])
        self.times = np.concatenate([self.times, times])
        self.lats = np.concatenate([self.lats, lats])
        self.lons = np.concatenate([self.lons, lons])

        for k in var_keys:
            vals = np.array([c.vars.get(k, np.nan) for c in centers], dtype=np.float64)
            if k not in self.vars:
                self.vars[k] = np.full(len(self.track_ids) - len(centers), np.nan)
            self.vars[k] = np.concatenate([self.vars[k], vals])

        return Track(tid, self)

    @property
    def head(self) -> list[Track]:
        return [Track(tid, self) for tid in self._head_ids]

    @head.setter
    def head(self, val: list[Track]) -> None:
        self._head_ids = {t.track_id for t in val if t.track_id is not None}

    @property
    def tail(self) -> list[Track]:
        return [Track(tid, self) for tid in self._tail_ids]

    @tail.setter
    def tail(self, val: list[Track]) -> None:
        self._tail_ids = {t.track_id for t in val if t.track_id is not None}

    @property
    def unique_track_ids(self) -> list[int]:
        # Return unique track IDs in order of first appearance
        if len(self.track_ids) == 0:
            return []
        _, idx = np.unique(self.track_ids, return_index=True)
        return list(self.track_ids[np.sort(idx)])

    def __getitem__(self, index: int) -> Track:
        tid = self.unique_track_ids[index]
        return Track(tid, self)

    def __setitem__(self, index: int, value: Track) -> None:
        # Replaces track at index with value track
        tid = self.unique_track_ids[index]
        if value._tracks is self:
            idx = np.where(self.track_ids == tid)[0]
            self.track_ids[idx] = value.track_id
        else:
            # Replace physical data
            idx = np.where(self.track_ids != tid)[0]
            self.track_ids = self.track_ids[idx]
            self.times = self.times[idx]
            self.lats = self.lats[idx]
            self.lons = self.lons[idx]
            for k in list(self.vars.keys()):
                self.vars[k] = self.vars[k][idx]
            self.append(value)

    def __iter__(self) -> Iterator[Track]:
        for tid in self.unique_track_ids:
            yield Track(tid, self)

    def __len__(self) -> int:
        return len(self.unique_track_ids)

    def _get_new_id(self) -> int:
        self._next_id += 1
        while self._next_id in self.unique_track_ids:
            self._next_id += 1
        return self._next_id

    def append(self, obj: Track) -> None:
        if obj._tracks is self:
            return  # Already in here

        tid = self._get_new_id()

        assert obj._tracks is not None
        idx = obj.indices
        self.track_ids = np.concatenate([self.track_ids, np.full(len(idx), tid)])
        self.times = np.concatenate([self.times, obj._tracks.times[idx]])
        self.lats = np.concatenate([self.lats, obj._tracks.lats[idx]])
        self.lons = np.concatenate([self.lons, obj._tracks.lons[idx]])

        for k in obj._tracks.vars:
            if k not in self.vars:
                self.vars[k] = np.full(len(self.track_ids) - len(idx), np.nan)
            self.vars[k] = np.concatenate([self.vars[k], obj._tracks.vars[k][idx]])

        obj.track_id = tid
        obj._tracks = self

    def sort(self) -> None:
        """Sorts tracks by their first point's time, lat, then lon."""
        if len(self.track_ids) == 0:
            return

        first_indices_list = []
        u_ids = self.unique_track_ids
        for tid in u_ids:
            idx = np.where(self.track_ids == tid)[0][0]
            first_indices_list.append(idx)

        first_indices = np.array(first_indices_list)
        sort_keys = np.lexsort(
            (
                self.lons[first_indices],
                self.lats[first_indices],
                self.times[first_indices],
            )
        )

        sorted_u_ids = np.array(u_ids)[sort_keys]

        new_indices: list[int] = []
        for tid in sorted_u_ids:
            new_indices.extend(np.where(self.track_ids == tid)[0])

        new_indices_arr = np.array(new_indices)
        self.track_ids = self.track_ids[new_indices_arr]
        self.times = self.times[new_indices_arr]
        self.lats = self.lats[new_indices_arr]
        self.lons = self.lons[new_indices_arr]
        for k in list(self.vars.keys()):
            self.vars[k] = self.vars[k][new_indices_arr]

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

        self.sort()
        other.sort()

        for tr1, tr2 in zip(self, other, strict=False):
            assert abs(len(tr1) - len(tr2)) <= length_diff_tol, (
                f"Track length mismatch: {len(tr1)} vs {len(tr2)}"
            )

            d1 = {c.time: c for c in tr1}
            d2 = {c.time: c for c in tr2}

            common_times = set(d1.keys()) & set(d2.keys())
            assert len(common_times) >= min(len(tr1), len(tr2)) - length_diff_tol

            for t_val in common_times:
                c1, c2 = d1[t_val], d2[t_val]
                assert abs(c1.lat - c2.lat) <= coord_tol
                assert abs(c1.lon - c2.lon) <= coord_tol

                for k in c1.vars:
                    assert k in c2.vars, f"Variable {k} missing in right track"
                    assert abs(c1.vars[k] - c2.vars[k]) <= intensity_tol
