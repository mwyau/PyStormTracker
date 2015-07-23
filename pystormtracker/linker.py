from detector import RectGrid, Center
try:
    from Queue import PriorityQueue
except ImportError:
    from queue import PriorityQueue

class Tracks(object):

    def __init__(self, threshold=500.):

        self.threshold = threshold
        self._tracks = []
        self._head = []
        self._tail = []
        self.tstart = None
        self.tend = None
        self.dt = None

    def __getitem__(self, index):
        return self._tracks[index]

    def __setitem__(self, index, value):
        self._tracks[index] = value

    def __iter__(self):
        return iter(self._tracks)

    def __len__(self):
        return len(self._tracks)

    def match_center(self, centers):

        ends = [self._tracks[i][-1] for i in self._tail]

        dforward = [{} for i in range(len(ends))]
        dbackward = [{} for i in range(len(centers))]

        for ic1, c1 in enumerate(ends):
            for ic2, c2 in enumerate(centers):
                dist = ends[ic1].abs_dist(centers[ic2])
                if dist < self.threshold:
                    dforward[ic1][ic2] = dist
                    dbackward[ic2][ic1] = dist

        matched = [None for i in range(len(centers))]

        while True:
            has_match = False
            for i, db in enumerate(dbackward):
                if matched[i] is None and len(db)>0:
                    iforward = min(db, key=db.get)
                    if len(dforward[iforward])>0 and \
                            min(dforward[iforward], key=dforward[iforward].get) == i:
                        matched[i] = iforward
                        dforward[iforward] = {}
                        for df in dforward:
                            if i in df:
                                del df[i]
                        has_match = True

            if has_match is False:
                break

        return [self._tail[i] if i is not None else None for i in matched]

    def match_track(self, tracks):

        centers = [tracks._tracks[i][0] for i in tracks._head]
        return self.match_center(centers)

    def append_center(self, centers):

        new_tail = []

        matched_index = self.match_center(centers)
        for i, d in enumerate(matched_index):
            if self.tstart is None:
                self._head.append(d)
            elif d is None or \
                    self.tend is not None and self.dt is not None and \
                    centers[0].time-self.dt > self.tend:
                self._tracks.append([centers[i]])
                new_tail.append(len(self._tracks)-1)
            else:
                self._tracks[d].append(centers[i])
                new_tail.append(d)

        self._tail = new_tail

        self.tend = centers[0].time
        if self.tstart is None:
            self.tstart = centers[0].time
        elif self.dt is None:
            self.dt = self.tend - self.tstart

    # def match_center(self, center, threshold=500.):

    #     q = PriorityQueue()
    #     for t in self.tail:
    #         dist = Center.abs_dist(center, self.tracks[t].centers[-1])
    #         if dist < threshold:
    #             q.put((dist, t))
    #     while not q.empty():
    #         return q.get()[1]
    #     return None

    # def append_center(self, centers):
    #     # Need to be rewritten
    #     if not self.tracks:
    #         for center in centers:
    #             track = Track()
    #             track.append(center)
    #             self.tracks.append(track)
    #             self.tail.append(len(self.tracks)-1)
    #     else:
    #         new_tail = []
    #         for center in centers:
    #             track_id = self.match_center(center)
    #             if track_id:
    #                 self.tracks[track_id].append(center)
    #                 new_tail.append(track_id)
    #             else:
    #                 track = Track()
    #                 track.append(center)
    #                 self.tracks.append(track)
    #                 new_tail.append(len(self.tracks)-1)
    #         self.tail = new_tail

if __name__ == "__main__":

    import timeit
    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    print("Starting linker...")

    timer = timeit.default_timer()
    grid = RectGrid(pathname="../slp.2012.nc", varname="slp", trange=(0,124))
    centers = grid.detect()

    print("Detection time: " + str(timeit.default_timer()-timer))
    timer = timeit.default_timer()

    tracks = Tracks()

    for c in centers:
        tracks.append_center(c)

    print("Linking time: " + str(timeit.default_timer()-timer))

    num_tracks = len([t for t in tracks if len(t)>=8 and t[0].abs_dist(t[-1])>=1000.])

    print("Number of long tracks: "+str(num_tracks))

    pickle.dump(tracks, open("tracks.pickle", "wb"))
