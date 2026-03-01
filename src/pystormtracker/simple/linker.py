from ..models.center import Center
from ..models.tracks import Tracks


class SimpleLinker:
    def __init__(self, threshold: float = 500.0) -> None:
        self.threshold = threshold

    def match_center(self, tracks: Tracks, centers: list[Center]) -> list[int | None]:
        ends = [tracks[i][-1] for i in tracks.tail]

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

        return [tracks.tail[i] if i is not None else None for i in matched]

    def match_track(self, tracks1: Tracks, tracks2: Tracks) -> list[int | None]:
        centers = [tracks2[i][0] for i in tracks2.head]
        return self.match_center(tracks1, centers)

    def append_center(self, tracks: Tracks, centers: list[Center]) -> None:
        new_tail: list[int] = []

        matched_index = self.match_center(tracks, centers)

        for i, d in enumerate(matched_index):
            if tracks.tstart is None:
                tracks.append([centers[i]])
                tracks.head.append(len(tracks) - 1)
                new_tail.append(len(tracks) - 1)
            elif d is None or (
                tracks.tend is not None
                and tracks.dt is not None
                and centers[0].time - tracks.dt > tracks.tend
            ):
                tracks.append([centers[i]])
                new_tail.append(len(tracks) - 1)
            else:
                tracks[d].append(centers[i])
                new_tail.append(d)

        tracks.tail = new_tail

        tracks.tend = centers[0].time
        if tracks.tstart is None:
            tracks.tstart = centers[0].time
        elif tracks.dt is None:
            tracks.dt = tracks.tend - tracks.tstart

    def extend_track(self, tracks1: Tracks, tracks2: Tracks) -> None:
        new_tail: list[int] = []

        matched_index = self.match_track(tracks1, tracks2)
        matched_dict = {d: matched_index[i] for i, d in enumerate(tracks2.head)}
        tail_dict = dict.fromkeys(tracks2.tail)

        for i, d in enumerate(tracks2):
            match_idx = matched_dict.get(i)
            if match_idx is not None:
                tracks1[match_idx].extend(d)
                if i in tail_dict:
                    new_tail.append(match_idx)
            else:
                tracks1.append(d)
                if i in tail_dict:
                    new_tail.append(len(tracks1) - 1)

        tracks1.tail = new_tail
        tracks1.tend = tracks2.tend
