"""Measure current package tracking time on a user-selected input.

The benchmark writes only temporary track output and prints compact JSON. It
does not download data or compare against an older package implementation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _run_case(
    input_path: Path,
    variable: str,
    detection_mode: str,
    frames: int | None,
    backend: str,
    workers: int | None,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="pystormtracker-benchmark-") as temp_dir:
        output = Path(temp_dir) / "tracks.trackjson"
        command = [
            sys.executable,
            "-m",
            "pystormtracker.cli",
            "track",
            "--input",
            str(input_path),
            "--variable",
            variable,
            "--detection-mode",
            detection_mode,
            "--output",
            str(output),
            "--backend",
            backend,
            "--no-progress",
        ]
        if frames is not None:
            command.extend(["--n-frames", str(frames)])
        if workers is not None:
            command.extend(["--workers", str(workers)])

        started = time.perf_counter()
        subprocess.run(command, check=True)
        elapsed = time.perf_counter() - started

    return {
        "backend": backend,
        "workers": workers or 1,
        "seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/data/era5/era5_msl_2025-12_2.5x2.5.nc"),
    )
    parser.add_argument("--variable", default="msl")
    parser.add_argument("--detection-mode", choices=("min", "max"), default="min")
    parser.add_argument("--frames", type=int)
    parser.add_argument(
        "--backends", nargs="+", choices=("serial", "dask"), default=["serial", "dask"]
    )
    parser.add_argument("--workers", nargs="+", type=int, default=[4])
    args = parser.parse_args()

    cases = []
    for backend in args.backends:
        worker_counts = [None] if backend == "serial" else args.workers
        cases.extend(
            _run_case(
                args.input,
                args.variable,
                args.detection_mode,
                args.frames,
                backend,
                workers,
            )
            for workers in worker_counts
        )
    print(json.dumps({"input": str(args.input), "results": cases}, indent=2))


if __name__ == "__main__":
    main()
