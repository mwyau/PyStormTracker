from collections.abc import Iterator
from typing import Any

from .detector import Center, RectGrid


class Tracks:
    def __init__(self, threshold: float = 500.0) -> None:

        self._tracks: list[list[Center]] = []
        self.head: list[int] = []
        self.tail: list[int] = []
        self.threshold: float = threshold
        self.tstart: Any | None = None
        self.tend: Any | None = None
        self.dt: Any | None = None

    def __getitem__(self, index: int) -> list[Center]:
        return self._tracks[index]

    def __setitem__(self, index: int, value: list[Center]) -> None:
        self._tracks[index] = value

    def __iter__(self) -> Iterator[list[Center]]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    def append(self, obj: list[Center]) -> None:
        self._tracks.append(obj)

    def match_center(self, centers: list[Center]) -> list[int | None]:

        ends = [self[i][-1] for i in self.tail]

        dforward: list[dict[int, float]] = [{} for _ in range(len(ends))]
        dbackward: list[dict[int, float]] = [{} for _ in range(len(centers))]

        for ic1, c1 in enumerate(ends):
            for ic2, c2 in enumerate(centers):
                dist = c1.abs_dist(c2)
                if dist < self.threshold:
                    dforward[ic1][ic2] = dist
                    dbackward[ic2][ic1] = dist

        matched: list[int | None] = [None for _ in range(len(centers))]

        while True:
            has_match = False

            for i, db in enumerate(dbackward):
                if matched[i] is None and len(db) > 0:
                    iforward = min(db, key=db.get)  # type: ignore
                    di = dforward[iforward]

                    if min(di, key=di.get) == i:  # type: ignore
                        matched[i] = iforward

                        db.clear()
                        for j in dbackward:
                            if iforward in j:
                                del j[iforward]
                        di.clear()

                        for j in dforward:
                            if i in j:
                                del j[i]

                        has_match = True

            if has_match is False:
                break

        return [self.tail[i] if i is not None else None for i in matched]

    def match_track(self, tracks: "Tracks") -> list[int | None]:

        centers = [tracks[i][0] for i in tracks.head]
        return self.match_center(centers)

    def append_center(self, centers: list[Center]) -> None:

        new_tail: list[int] = []

        matched_index = self.match_center(centers)

        for i, d in enumerate(matched_index):
            if self.tstart is None:
                self.append([centers[i]])
                self.head.append(len(self) - 1)
                new_tail.append(len(self) - 1)
            elif d is None or (
                self.tend is not None
                and self.dt is not None
                and centers[0].time - self.dt > self.tend
            ):
                self.append([centers[i]])
                new_tail.append(len(self) - 1)
            else:
                self[d].append(centers[i])
                new_tail.append(d)

        self.tail = new_tail

        self.tend = centers[0].time
        if self.tstart is None:
            self.tstart = centers[0].time
        elif self.dt is None:
            self.dt = self.tend - self.tstart

    def extend_track(self, tracks: "Tracks") -> None:

        new_tail: list[int] = []

        matched_index = self.match_track(tracks)
        matched_dict = {d: matched_index[i] for i, d in enumerate(tracks.head)}
        tail_dict = dict.fromkeys(tracks.tail)

        for i, d in enumerate(tracks):
            match_idx = matched_dict.get(i)
            if match_idx is not None:
                self[match_idx].extend(d)
                if i in tail_dict:
                    new_tail.append(match_idx)
            else:
                self.append(d)
                if i in tail_dict:
                    new_tail.append(len(self) - 1)

        self.tail = new_tail

        self.tend = tracks.tend


if __name__ == "__main__":
    import pickle
    import timeit

    print("Starting detector...")

    timer = timeit.default_timer()
    grid = RectGrid(pathname="../slp.2012.nc", varname="slp", trange=(0, 120))
    centers = grid.detect()

    print("Detection time: " + str(timeit.default_timer() - timer))

    print("Starting linker...")

    timer = timeit.default_timer()

    tracks = Tracks()

    for c in centers:
        tracks.append_center(c)

    print("Linking time: " + str(timeit.default_timer() - timer))

    num_tracks = len(
        [t for t in tracks if len(t) >= 8 and t[0].abs_dist(t[-1]) >= 1000.0]
    )

    print("Number of long tracks: " + str(num_tracks))

    with open("tracks.pickle", "wb") as f:
        pickle.dump(tracks, f)

    print("Starting multiple detector...")

    timer = timeit.default_timer()

    grid2 = RectGrid(pathname="../slp.2012.nc", varname="slp", trange=(0, 120))
    grids2 = grid2.split(4)

    centers2 = [g.detect() for g in grids2]

    print("Detection time: " + str(timeit.default_timer() - timer))

    print("Starting multiple linker...")

    timer = timeit.default_timer()

    tracks2: list[Tracks] = [Tracks() for _ in range(4)]
    for i, cs in enumerate(centers2):
        for c in cs:
            tracks2[i].append_center(c)

    tracksout = tracks2[0]

    for i in range(1, 4):
        tracksout.extend_track(tracks2[i])

    print("Detection time: " + str(timeit.default_timer() - timer))

    num_tracks = len(
        [t for t in tracksout if len(t) >= 8 and t[0].abs_dist(t[-1]) >= 1000.0]
    )

    print("Number of long tracks: " + str(num_tracks))
