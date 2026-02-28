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

    trange=None

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"hi:v:o:n:m:",["input=","var=","output=","num=","mode="])
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
            trange = (0, int(arg))
        elif opt in ("-m", "--mode"):
            mode = arg

    timer = {}

    if USE_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0

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

    centers = grid.detect(minmaxmode=mode)

    if USE_MPI:
        comm.Barrier()
    if USE_MPI is False or rank == root:
        timer['detector'] = timeit.default_timer()-timer['detector']
        timer['linker'] = timeit.default_timer()

    tracks = Tracks()

    for c in centers:
        tracks.append_center(c)

    if USE_MPI:

        timer['combiner'] = timeit.default_timer()

        nstripe = 2
        while nstripe <= size:
            if rank%nstripe == nstripe/2:
                comm.send(tracks, dest = rank-nstripe/2, tag=nstripe)
            elif rank%nstripe == 0:
                if rank+nstripe/2 < size:
                    tracks_recv = comm.recv(source=rank+nstripe/2, tag=nstripe)
                    tracks.extend_track(tracks_recv)
            nstripe = nstripe*2

        timer['combiner'] = timeit.default_timer()-timer['combiner']

    if USE_MPI is False or rank == root:
        timer['linker'] = timeit.default_timer()-timer['linker']
        print("Detector time: "+str(timer['detector']))
        print("Linker time: "+str(timer['linker']))
        print("Combiner time: "+str(timer['combiner']))
        num_tracks = len([t for t in tracks if len(t)>=8 and t[0].abs_dist(t[-1])>=1000.])
        print("Number of long tracks: "+str(num_tracks))

        pickle.dump(tracks, open(outfile, "wb"))
