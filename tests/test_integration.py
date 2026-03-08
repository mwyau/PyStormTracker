import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pystormtracker.data import fetch_era5_msl, fetch_era5_vo850
from pystormtracker.models.tracks import Tracks
from pystormtracker.stormtracker import parse_args, run_tracker


def run_command_direct(cmd_args: list[str], use_mpi: bool = False) -> None:
    """Utility to run the tracker directly via function calls or MPI subprocess."""
    if use_mpi:
        base_cmd = f"{sys.executable} -m pystormtracker.stormtracker"
        full_cmd = f"mpiexec -n 2 {base_cmd} {' '.join(cmd_args)}"
        subprocess.run(full_cmd, shell=True, check=True, capture_output=True)
        return

    # Direct function call for Serial/Dask backends
    with patch.object(sys, "argv", ["stormtracker", *cmd_args]):
        args = parse_args()
        trange = (0, args.num) if args.num is not None else None
        run_tracker(
            infile=args.input,
            varname=args.var,
            outfile=args.output,
            trange=trange,
            mode=args.mode,
            backend=args.backend,
            n_workers=args.workers,
        )


def print_head(filename: Path | str, n: int = 15) -> None:
    """Prints the first n lines of a file."""
    print(f"\n--- First {n} lines of {os.path.basename(filename)} ---")
    with open(filename) as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            print(line.rstrip())
    print("-------------------------------------------------------\n")


def compare_tracks(
    file1: Path | str,
    file2: Path | str,
    length_diff_tol: int = 0,
    coord_tol: float = 1e-4,
    intensity_tol: float = 1e-4,
) -> None:
    """Compares two tracking files for equality using the Tracks class."""
    t1 = Tracks.from_imilast(file1)
    t2 = Tracks.from_imilast(file2)
    t1.compare(
        t2,
        length_diff_tol=length_diff_tol,
        coord_tol=coord_tol,
        intensity_tol=intensity_tol,
    )


@pytest.fixture(scope="module")
def test_data_msl() -> str:
    """Download MSL test data once per module."""
    return fetch_era5_msl()


@pytest.fixture(scope="module")
def test_data_vo() -> str:
    """Download VO test data once per module."""
    return fetch_era5_vo850()


@pytest.fixture(
    scope="module",
    params=[
        ("msl", "min", "-n 60"),
        ("vo", "max", "-n 60"),
    ],
    ids=["msl_min_fast", "vo_max_fast"],
)
def config(
    request: Any,  # noqa: ANN401
    test_data_msl: str,
    test_data_vo: str,
) -> tuple[str, str, str, str]:
    varname, mode, n_arg = request.param
    data_path = test_data_msl if varname == "msl" else test_data_vo
    return data_path, varname, mode, n_arg


@pytest.fixture(scope="module")
def shared_serial_output(
    tmp_path_factory: Any,  # noqa: ANN401
    config: tuple[str, str, str, str],
) -> Path:
    """Run serial once and share it across tests to save time."""
    data_path, varname, mode, n_arg = config
    temp_dir: Path = tmp_path_factory.mktemp("data")
    out_file = temp_dir / "integration_serial.txt"

    args = [
        "-i",
        data_path,
        "-v",
        varname,
        "-m",
        mode,
        "-o",
        str(out_file),
        "--backend",
        "serial",
    ]
    if n_arg:
        args.extend(n_arg.split())

    run_command_direct(args)

    # Verbose print the IMILAST format output
    print(f"\nConfiguration: Variable={varname}, Mode={mode}, Args={n_arg}")
    print_head(out_file, n=15)

    return Path(out_file)


def test_dask_vs_serial(
    shared_serial_output: Path, tmp_path: Path, config: tuple[str, str, str, str]
) -> None:
    """Integration test comparing Serial and Dask backends."""
    data_path, varname, mode, n_arg = config
    out_file = tmp_path / "integration_dask.txt"

    args = [
        "-i",
        data_path,
        "-v",
        varname,
        "-m",
        mode,
        "-o",
        str(out_file),
        "--backend",
        "dask",
        "--workers",
        "2",
    ]
    if n_arg:
        args.extend(n_arg.split())

    run_command_direct(args)
    compare_tracks(shared_serial_output, out_file)


def test_mpi_vs_serial(
    shared_serial_output: Path, tmp_path: Path, config: tuple[str, str, str, str]
) -> None:
    """Integration test comparing Serial and MPI backends."""
    try:
        subprocess.run("mpiexec -help", shell=True, capture_output=True)
    except FileNotFoundError:
        pytest.skip("mpiexec not found in path")

    data_path, varname, mode, n_arg = config
    mpi_out = tmp_path / "integration_mpi.txt"

    args = [
        "-i",
        data_path,
        "-v",
        varname,
        "-m",
        mode,
        "-o",
        str(mpi_out),
        "--backend",
        "mpi",
    ]
    if n_arg:
        args.extend(n_arg.split())

    run_command_direct(args, use_mpi=True)
    compare_tracks(shared_serial_output, mpi_out)


@pytest.mark.slow
def test_legacy_regression(test_data_msl: str, tmp_path: Path) -> None:
    """Regression test against v0.0.2 legacy output using Dask."""
    ref_file = "data/test/tracks/era5_msl_2.5x2.5_v0.0.2_imilast.txt"
    if not os.path.exists(ref_file):
        pytest.skip(f"Reference file {ref_file} not found")

    out_file = tmp_path / "legacy_regression.txt"
    args = [
        "-i",
        test_data_msl,
        "-v",
        "msl",
        "-m",
        "min",
        "-o",
        str(out_file),
        "--backend",
        "dask",
    ]
    run_command_direct(args)

    compare_tracks(
        ref_file, out_file, length_diff_tol=1, coord_tol=15.0, intensity_tol=500.0
    )
