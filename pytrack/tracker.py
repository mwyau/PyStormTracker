import numpy as np
import numpy.ma as ma
import Nio

from detector import Center

try:
    from Queue import PriorityQueue
except ImportError:
    from queue import PriorityQueue

class Track(object):

    def __init__(self):
        self.endtime = None
        self.centers = []

    def append(self, center):
        if center.time > self.endtime or self.endtime is None:
            self.endtime = center.time
            self.centers.append(center)
        else:
            # print("Time mistatch")
            pass

class Tracks(object):

    def __init__(self):
        self.tracks = []
        self.tail = []

    def match_center(self, center, threshold=500.):
        q = PriorityQueue()
        for t in self.tail:
            dist = Center.abs_dist(center, self.tracks[t].centers[-1])
            if dist < threshold:
                q.put((dist, t))
        while not q.empty():
            return q.get()[1]
        return None

    def insert_centers(self, centers):
        if not self.tracks:
            for center in centers:
                track = Track()
                track.append(center)
                self.tracks.append(track)
                self.tail.append(len(self.tracks)-1)
        else:
            new_tail = []
            for center in centers:
                track_id = self.match_center(center)
                if track_id:
                    self.tracks[track_id].append(center)
                    new_tail.append(track_id)
                else:
                    track = Track()
                    track.append(center)
                    self.tracks.append(track)
                    new_tail.append(len(self.tracks)-1)
            self.tail = new_tail

if __name__ == "__main__":

    import detector
    import timeit
    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    timer = timeit.default_timer()

    var, time, lat, lon = detector.get_var_handle(filename="../slp.2012.nc")
    time = time[:]
    lat = lat[:]
    lon = lon[:]

    tracks = Tracks()

    for it, t in enumerate(time):
        psl = var[it,:,:]
        latlon = detector.detect_center_latlon(psl,threshold=10.)
        latlon_indices = latlon.nonzero()
        centers = []
        for i,j in np.transpose(latlon_indices):
            # print(t, lat[i], lon[j], psl[i,j])
            c = Center(t, lat[i], lon[j], psl[i,j])
            centers.append(c)

        tracks.insert_centers(centers)

    print(timeit.default_timer() - timer)

    pickle.dump(tracks, open("tracks.pickle", "wb"))
