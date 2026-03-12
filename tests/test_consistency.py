import numpy as np
from pystormtracker.simple.tracker import SimpleTracker
from pystormtracker.utils.data_utils import fetch_era5_msl
import time


def test_consistency():
    infile = fetch_era5_msl(resolution="2.5x2.5")
    varname = "msl"

    tracker = SimpleTracker()

    print("Running baseline (Serial)...")
    t0 = time.time()
    tracks_serial = tracker.track(infile, varname, backend="serial")
    print(f"Serial finished in {time.time() - t0:.2f}s. Tracks: {len(tracks_serial)}")

    workers_to_test = [2, 4, 8, 16]

    for w in workers_to_test:
        print(f"\nRunning with {w} workers (Dask)...")
        t0 = time.time()
        tracks_parallel = tracker.track(infile, varname, backend="dask", n_workers=w)
        print(
            f"{w} workers finished in {time.time() - t0:.2f}s. Tracks: {len(tracks_parallel)}"
        )

        if len(tracks_serial) != len(tracks_parallel):
            print(
                f"FAILED: Mismatch with {w} workers! {len(tracks_serial)} vs {len(tracks_parallel)}"
            )
            # Deep check
            for i in range(min(len(tracks_serial), len(tracks_parallel))):
                if len(tracks_serial[i]) != len(tracks_parallel[i]):
                    print(
                        f"First track length mismatch at index {i}: {len(tracks_serial[i])} vs {len(tracks_parallel[i])}"
                    )
                    break
            raise ValueError("Consistency check failed")
        else:
            print(f"SUCCESS: {w} workers match baseline exactly.")


if __name__ == "__main__":
    test_consistency()
