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
        opts, args = getopt.getopt(argv,"hi:v:o:n:",["input=","var=","output=","num="])
    except getopt.GetoptError:
        print("stormtracker.py -i <input file> -v <variable name> -o <output file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("stormtracker.py -i <input file> -v <variable name> -o <output file> -n <number of time steps>")
            sys.exit()
        elif opt in ("-i", "--input"):
            infile = arg
        elif opt in ("-v", "--var"):
            var = arg
        elif opt in ("-o", "--output"):
            outfile = arg
        elif opt in ("-n", "--num"):
            trange = (0, arg)
        else:
            trange = None

    timer = {}

    if USE_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = size-1

    if USE_MPI is False or rank == root:
        timer['detector'] = timeit.default_timer()

    if USE_MPI:

        if rank == root:
            grid = RectGrid(pathname=infile, varname=var, trange=trange)
            grid = grid.split(size)
        else:
            grid = None

        grid = comm.scatter(grid, root=root)

    else:
        grid = RectGrid(pathname=infile, varname=var, trange=trange)

    centers = grid.detect()

    if USE_MPI:
        comm.Barrier()
    if USE_MPI is False or rank == root:
        timer['detector'] = timeit.default_timer()-timer['detector']
        timer['linker'] = timeit.default_timer()

    tracks = Tracks()

    for c in centers:
        tracks.append_center(c)

    if USE_MPI:

        tracks = comm.gather(tracks, root=root)
        if rank == root:
            timer['combiner'] = timeit.default_timer()
            for i in range(1,size):
                tracks[0].extend_track(tracks[i])
            tracks = tracks[0]
            timer['combiner'] = timeit.default_timer()-timer['combiner']

    if USE_MPI is False or rank == root:
        timer['linker'] = timeit.default_timer()-timer['linker']
        print("Detector time: "+str(timer['detector']))
        print("Linker time: "+str(timer['linker']))
        print("Combiner time: "+str(timer['combiner']))
        num_tracks = len([t for t in tracks if len(t)>=8 and t[0].abs_dist(t[-1])>=1000.])
        print("Number of long tracks: "+str(num_tracks))

        pickle.dump(tracks, open(outfile, "wb"))
