import getopt
import pickle
import sys
import timeit
from typing import Literal

from .models import Tracks
from .simple import SimpleDetector, SimpleLinker


def main() -> None:
    try:
        from mpi4py import MPI

        USE_MPI: bool = True
    except ImportError:
        USE_MPI = False

    trange: tuple[int, int] | None = None

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(
            argv, "hi:v:o:n:m:", ["input=", "var=", "output=", "num=", "mode="]
        )
    except getopt.GetoptError:
        print("stormtracker.py -i <input file> -v <variable name> -o <output file>")
        sys.exit(2)

    infile: str = ""
    var: str = ""
    outfile: str = ""
    mode: Literal["min", "max"] = "min"

    for opt, arg in opts:
        if opt == "-h":
            print(
                "stormtracker.py -i <input file> -v <variable name> -o <output file> -n <number of time steps>"
            )
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
            if arg in ("min", "max"):
                mode = arg  # type: ignore

    timer: dict[str, float] = {}

    if USE_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0
    else:
        rank = 0
        root = 0

    if not USE_MPI or rank == root:
        timer["detector"] = timeit.default_timer()

    if USE_MPI:
        if rank == root:
            grid_obj = SimpleDetector(pathname=infile, varname=var, trange=trange)
            grids = grid_obj.split(size)
        else:
            grids = None

        grid = comm.scatter(grids, root=root)

    else:
        grid = SimpleDetector(pathname=infile, varname=var, trange=trange)

    centers = grid.detect(minmaxmode=mode)

    if USE_MPI:
        comm.Barrier()
    if not USE_MPI or rank == root:
        timer["detector"] = timeit.default_timer() - timer["detector"]
        timer["linker"] = timeit.default_timer()

    tracks = Tracks()
    linker = SimpleLinker()

    for c in centers:
        linker.append_center(tracks, c)

    if USE_MPI:
        timer["combiner"] = timeit.default_timer()

        nstripe = 2
        while nstripe <= size:
            if rank % nstripe == nstripe // 2:
                comm.send(tracks, dest=rank - nstripe // 2, tag=nstripe)
            elif rank % nstripe == 0:
                if rank + nstripe // 2 < size:
                    tracks_recv = comm.recv(source=rank + nstripe // 2, tag=nstripe)
                    linker.extend_track(tracks, tracks_recv)
            nstripe = nstripe * 2

        timer["combiner"] = timeit.default_timer() - timer["combiner"]

    if not USE_MPI or rank == root:
        timer["linker"] = timeit.default_timer() - timer["linker"]
        print("Detector time: " + str(timer["detector"]))
        print("Linker time: " + str(timer["linker"]))
        if "combiner" in timer:
            print("Combiner time: " + str(timer["combiner"]))
        num_tracks = len(
            [t for t in tracks if len(t) >= 8 and t[0].abs_dist(t[-1]) >= 1000.0]
        )
        print("Number of long tracks: " + str(num_tracks))

        with open(outfile, "wb") as f:
            pickle.dump(tracks, f)


if __name__ == "__main__":
    main()
