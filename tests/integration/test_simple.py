from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

import pytest
import xarray as xr

from pystormtracker.io.imilast import read_imilast
from pystormtracker.models.tracks import Tracks
from pystormtracker.simple.tracker import SimpleTracker
from tests.utils import (
    DECEMBER_2025_END,
    DECEMBER_2025_START,
    fetch_era5_msl,
    get_integration_msl_path,
)

N_WORKERS = 4
DetectionMode = Literal["min", "max"]


def run_command_direct(cmd_args: list[str], use_mpi: bool = False) -> Tracks | None:
    """Utility to run the tracker directly via function calls or MPI subprocess."""
    # Prepend 'track' command if not present
    if cmd_args and cmd_args[0] not in ["track", "sample", "convert", "compare"]:
        cmd_args = ["track", *cmd_args]

    if use_mpi:
        # Keep executable and user arguments separate so paths and values are
        # passed without shell parsing.
        command = [
            "mpiexec",
            "-n",
            str(N_WORKERS),
            sys.executable,
            "-m",
            "pystormtracker.cli",
            *cmd_args,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"MPI Command failed: {e.cmd}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise
        out_idx = cmd_args.index("-o") + 1 if "-o" in cmd_args else -1
        if out_idx > 0 and out_idx < len(cmd_args):
            from pystormtracker.io.format import load_tracks

            return load_tracks(cmd_args[out_idx])
        return None

    # Direct function call for Serial/Dask backends
    import argparse

    from pystormtracker import compare, convert, sample, track

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    track.setup_parser(subparsers)
    sample.setup_parser(subparsers)
    convert.setup_parser(subparsers)
    compare.setup_parser(subparsers)

    args = parser.parse_args(cmd_args)
    if hasattr(args, "func"):
        args.func(args)
        from pystormtracker.io.format import load_tracks

        return load_tracks(args.output)
    return None


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


def compare_track_files(
    file1: Path | str,
    file2: Path | str,
) -> None:
    """Compares two tracking files for strict equality using the Tracks class."""
    t1 = read_imilast(file1)
    t2 = read_imilast(file2)

    # Backend comparisons must be identical
    assert t1 == t2


@pytest.fixture(scope="module")
def test_data_msl() -> str:
    """Return the committed December 2025 MSL integration input."""
    return str(get_integration_msl_path())


TRACK_CONFIGS = [
    pytest.param(("msl", "min"), id="msl_min"),
]


@pytest.fixture(scope="module", params=TRACK_CONFIGS)
def config_params(
    request: pytest.FixtureRequest,
) -> tuple[str, DetectionMode]:
    """Select the ordinary December 2025 tracking configurations."""
    return cast(tuple[str, DetectionMode], request.param)


@pytest.fixture(scope="module")
def config(
    config_params: tuple[str, DetectionMode],
    test_data_msl: str,
) -> tuple[str, str, DetectionMode]:
    variable_name, mode = config_params
    assert variable_name == "msl"
    return test_data_msl, variable_name, mode


@pytest.fixture(scope="module")
def serial_reference(
    tmp_path_factory: pytest.TempPathFactory,
    config: tuple[str, str, DetectionMode],
) -> tuple[Path, Tracks]:
    """Run serial once and share it across tests to save time."""
    data_path, variable_name, mode = config
    temp_dir: Path = tmp_path_factory.mktemp("data")
    out_file = temp_dir / "integration_serial.txt"

    args = [
        "-i",
        data_path,
        "--variable",
        variable_name,
        "-m",
        mode,
        "-o",
        str(out_file),
        "--backend",
        "serial",
    ]

    tracks = run_command_direct(args)
    assert tracks is not None

    # Verbose print the IMILAST format output
    print(f"\nConfiguration: Variable={variable_name}, Mode={mode}")
    print_head(out_file, n=15)

    return Path(out_file), tracks


@pytest.mark.integration
def test_dask_vs_serial(
    serial_reference: tuple[Path, Tracks],
    tmp_path: Path,
    config: tuple[str, str, DetectionMode],
) -> None:
    """Integration test comparing Serial and Dask backends."""
    serial_path, _ = serial_reference
    data_path, variable_name, mode = config
    out_file = tmp_path / "integration_dask.txt"

    args = [
        "-i",
        data_path,
        "--variable",
        variable_name,
        "-m",
        mode,
        "-o",
        str(out_file),
        "--backend",
        "dask",
        "--workers",
        str(N_WORKERS),
    ]

    run_command_direct(args)
    compare_track_files(serial_path, out_file)


@pytest.mark.integration
def test_mpi_vs_serial(
    serial_reference: tuple[Path, Tracks],
    tmp_path: Path,
    config: tuple[str, str, DetectionMode],
) -> None:
    """Integration test comparing Serial and MPI backends."""
    pytest.importorskip("mpi4py")

    # Check for mpiexec in PATH
    if shutil.which("mpiexec") is None:
        pytest.skip("mpiexec not found in PATH")

    data_path, variable_name, mode = config
    serial_path, _ = serial_reference
    mpi_out = tmp_path / "integration_mpi.txt"

    args = [
        "-i",
        data_path,
        "--variable",
        variable_name,
        "-m",
        mode,
        "-o",
        str(mpi_out),
        "--backend",
        "mpi",
    ]

    run_command_direct(args, use_mpi=True)
    compare_track_files(serial_path, mpi_out)


@pytest.mark.integration
@pytest.mark.data
def test_grib_vs_netcdf(
    serial_reference: tuple[Path, Tracks],
    tmp_path: Path,
    config: tuple[str, str, DetectionMode],
) -> None:
    """Test that tracking matches between NetCDF and GRIB inputs."""
    pytest.importorskip("cfgrib")

    _, variable_name, mode = config
    serial_path, _ = serial_reference

    grib_path = fetch_era5_msl(resolution="2.5x2.5", format="grib")

    out_file = tmp_path / "integration_grib.txt"

    with xr.open_dataset(grib_path, engine="cfgrib") as dataset:
        time_name = "valid_time" if "valid_time" in dataset.coords else "time"
        december = (
            dataset[variable_name]
            .sel({time_name: slice(DECEMBER_2025_START, DECEMBER_2025_END)})
            .load()
        )
    grib_tracks = SimpleTracker(backend="serial").track(
        data=december,
        variable=variable_name,
        detection_mode=mode,
    )
    grib_tracks.write(out_file)
    compare_track_files(serial_path, out_file)
