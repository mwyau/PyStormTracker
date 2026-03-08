from ..models.center import Center
from ..models.time import TimeRange
from ..models.tracks import Tracks


class SimpleLinker:
    def __init__(self, threshold: float = 500.0) -> None:
        self.threshold = threshold

    def match_center(
        self, tracks: Tracks, centers: list[Center]
    ) -> list[list[Center] | None]:
        """Matches a list of centers to the tails of existing tracks."""
        if not tracks.tail:
            return [None for _ in range(len(centers))]

        dforward: list[dict[int, float]] = [{} for _ in range(len(tracks.tail))]
        dbackward: list[dict[int, float]] = [{} for _ in range(len(centers))]

        for it, track in enumerate(tracks.tail):
            last_center = track[-1]
            for ic, center in enumerate(centers):
                dist = last_center.abs_dist(center)
                if dist < self.threshold:
                    dforward[it][ic] = dist
                    dbackward[ic][it] = dist

        matched: list[list[Center] | None] = [None for _ in range(len(centers))]

        while True:
            has_match = False
            for ic, db in enumerate(dbackward):
                if matched[ic] is None and len(db) > 0:
                    # Get the index of the closest track tail for this center
                    it_match = min(db, key=lambda k: db[k])
                    df = dforward[it_match]

                    # Check if this center is also the closest for that track tail
                    if min(df, key=lambda k: df[k]) == ic:
                        matched[ic] = tracks.tail[it_match]

                        # Clear resources to avoid double matching
                        db.clear()
                        for j in dbackward:
                            if it_match in j:
                                del j[it_match]
                        df.clear()
                        for j in dforward:
                            if ic in j:
                                del j[ic]

                        has_match = True
            if not has_match:
                break

        return matched

    def match_track(
        self, tracks1: Tracks, tracks2: Tracks
    ) -> list[list[Center] | None]:
        """Matches the heads of tracks2 to the tails of tracks1."""
        centers = [track[0] for track in tracks2.head]
        return self.match_center(tracks1, centers)

    def append_center(self, tracks: Tracks, centers: list[Center]) -> None:
        if not centers:
            tracks.tail = []
            return

        new_tail: list[list[Center]] = []
        matched_tracks = self.match_center(tracks, centers)

        for i, matched_track in enumerate(matched_tracks):
            if tracks.time_range is None:
                # First ever centers
                new_track = [centers[i]]
                tracks.append(new_track)
                tracks.head.append(new_track)
                new_tail.append(new_track)
            elif matched_track is None or (
                tracks.time_range.end is not None
                and tracks.time_range.step is not None
                and centers[0].time - tracks.time_range.step > tracks.time_range.end
            ):
                # No match or gap in time detected
                new_track = [centers[i]]
                tracks.append(new_track)
                new_tail.append(new_track)
            else:
                # Append to existing matched track
                matched_track.append(centers[i])
                new_tail.append(matched_track)

        tracks.tail = new_tail

        current_time = centers[0].time
        if tracks.time_range is None:
            tracks.time_range = TimeRange(start=current_time, end=current_time)
        else:
            if tracks.time_range.step is None:
                tracks.time_range.step = current_time - tracks.time_range.start
            tracks.time_range.end = current_time

    def extend_track(self, tracks1: Tracks, tracks2: Tracks) -> None:
        if not tracks2:
            return

        if not tracks1:
            tracks1._tracks = tracks2._tracks
            tracks1.head = tracks2.head
            tracks1.tail = tracks2.tail
            tracks1.time_range = tracks2.time_range
            return

        new_tail: list[list[Center]] = []
        matched_tracks = self.match_track(tracks1, tracks2)

        # Map of tracks2.head to the matched track in tracks1
        matched_map = {
            id(tracks2.head[i]): matched_tracks[i] for i in range(len(tracks2.head))
        }

        # Track which tracks in tracks2 are currently tails
        tail_ids = {id(t) for t in tracks2.tail}

        for track2 in tracks2:
            matched_track1 = matched_map.get(id(track2))
            if matched_track1 is not None:
                matched_track1.extend(track2)
                if id(track2) in tail_ids:
                    new_tail.append(matched_track1)
            else:
                tracks1.append(track2)
                if id(track2) in tail_ids:
                    new_tail.append(track2)

        tracks1.tail = new_tail
        if tracks1.time_range and tracks2.time_range:
            tracks1.time_range.end = tracks2.time_range.end
