import numpy as np
import numpy.ma as ma
import Nio

import detector
from detector import Center
from tracker import Tracks

if __name__ == "__main__":

    import timeit
    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    timer = timeit.default_timer()

    var, time, lat, lon = detector.get_var_handle(filename = \
            "/Users/mwyau/Documents/CESM/b.e11.B20TRC5CNBDRD.f09_g16.031.cam.h2.PSL.1990010100Z-2005123118Z.nc", \
            varname = "PSL")
    time = time[0:1459] # First year only: 365*4
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

    pickle.dump(tracks, open("tracks_cesm.pickle", "wb"))
