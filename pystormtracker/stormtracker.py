import sys, getopt
import timeit
from glob import glob
try:
    import cPickle as pickle
except ImportError:
    import pickle

from detector import RectGrid, Center
from linker import Tracks

if __name__ == "__main__":

    try:
        from mpi4py import MPI
        USE_MPI = True
    except ImportError:
        USE_MPI = False

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"hi:v:",["input=","var="])
    except getopt.GetoptError:
        print("stormtracker.py -i <input file> -v <variable name>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("stormtracker.py -i <input file> -v <variable name>")
            sys.exit()
        elif opt in ("-i", "--input"):
            pathname = arg
        elif opt in ("-v", "--var"):
            varname = arg

    trange = None

    if USE_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = size-1

    if USE_MPI is False or rank == root:
        print("Starting detector...")
        timer = timeit.default_timer()

    if USE_MPI:

        if rank == root:
            grid = RectGrid(pathname=pathname, varname=varname, trange=trange)
            grid = grid.split(size)
        else:
            grid = None

        grid = comm.scatter(grid, root=root)

    else:
        grid = RectGrid(pathname=pathname, varname=varname, trange=trange)

    centers = grid.detect()

    if USE_MPI:
        comm.Barrier()
    if USE_MPI is False or rank == root:
        print("Detection time: " + str(timeit.default_timer()-timer))
        timer = timeit.default_timer()
        print("Starting linker...")

    tracks = Tracks()

    for c in centers:
        tracks.append_center(c)

    if USE_MPI:

        tracks = comm.gather(tracks, root=root)
        if rank == root:
            for i in range(1,size):
                tracks[0].extend_track(tracks[i])
            tracks = tracks[0]

    if USE_MPI is False or rank == root:
        print("Linking time: " + str(timeit.default_timer()-timer))
        num_tracks = len([t for t in tracks if len(t)>=8 and t[0].abs_dist(t[-1])>=1000.])
        print("Number of long tracks: "+str(num_tracks))

        pickle.dump(tracks, open("tracks.pickle", "wb"))
