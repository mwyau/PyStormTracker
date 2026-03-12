from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pystormtracker.cli import main
from pystormtracker.io.imilast import read_imilast
from pystormtracker.utils.data_utils import fetch_era5_msl, fetch_era5_vo850

N_WORKERS = 4


def run_command_direct(cmd_args: list[str], use_mpi: bool = False) -> None:
    """Utility to run the tracker directly via function calls or MPI subprocess."""
    if use_mpi:
        base_cmd = f"{sys.executable} -m pystormtracker.cli"
        full_cmd = (
            f"mpiexec --oversubscribe -n {N_WORKERS} {base_cmd} {' '.join(cmd_args)}"
        )
        subprocess.run(full_cmd, shell=True, check=True, capture_output=True)
        return

    # Direct function call for Serial/Dask backends
    with patch.object(sys, "argv", ["stormtracker", *cmd_args]):
        main()


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
    t1 = read_imilast(file1)
    t2 = read_imilast(file2)

    # Requirement: PERFECT MATCH
    assert len(t1) == len(t2), f"Track count mismatch: {len(t1)} vs {len(t2)}"

    t1.compare(
        t2,
        length_diff_tol=length_diff_tol,
        coord_tol=coord_tol,
        intensity_tol=intensity_tol,
    )


@pytest.fixture(scope="module")
def test_data_msl() -> str:
    """Download MSL test data once per module."""
    return fetch_era5_msl(resolution="2.5x2.5")


@pytest.fixture(scope="module")
def test_data_vo() -> str:
    """Download VO test data once per module."""
    return fetch_era5_vo850(resolution="2.5x2.5")


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(("msl", "min"), id="msl_min_full"),
        pytest.param(("vo", "max"), id="vo_max_full"),
    ],
)
def config(
    request: pytest.FixtureRequest,
    test_data_msl: str,
    test_data_vo: str,
) -> tuple[str, str, str]:
    param: tuple[str, str] = request.param
    varname, mode = param
    data_path = test_data_msl if varname == "msl" else test_data_vo
    return data_path, varname, mode


@pytest.fixture(scope="module")
def shared_serial_output(
    tmp_path_factory: pytest.TempPathFactory,
    config: tuple[str, str, str],
) -> Path:
    """Run serial once and share it across tests to save time."""
    data_path, varname, mode = config
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

    run_command_direct(args)

    # Verbose print the IMILAST format output
    print(f"\nConfiguration: Variable={varname}, Mode={mode}")
    print_head(out_file, n=15)

    return Path(out_file)


@pytest.mark.integration
def test_dask_vs_serial(
    shared_serial_output: Path, tmp_path: Path, config: tuple[str, str, str]
) -> None:
    """Integration test comparing Serial and Dask backends."""
    data_path, varname, mode = config
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
        str(N_WORKERS),
    ]

    run_command_direct(args)
    compare_tracks(shared_serial_output, out_file)


@pytest.mark.integration
def test_mpi_vs_serial(
    shared_serial_output: Path, tmp_path: Path, config: tuple[str, str, str]
) -> None:
    """Integration test comparing Serial and MPI backends."""
    try:
        subprocess.run("mpiexec -help", shell=True, capture_output=True)
    except FileNotFoundError:
        pytest.skip("mpiexec not found in path")

    data_path, varname, mode = config
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

    run_command_direct(args, use_mpi=True)
    compare_tracks(shared_serial_output, mpi_out)


@pytest.mark.integration
def test_grib_serial(tmp_path: Path) -> None:
    """Test that tracking works correctly with GRIB input."""
    grib_path = fetch_era5_msl(resolution="2.5x2.5", format="grib")
    out_file = tmp_path / "integration_grib.txt"

    args = [
        "-i",
        grib_path,
        "-v",
        "msl",
        "-m",
        "min",
        "-o",
        str(out_file),
        "--backend",
        "serial",
    ]

    run_command_direct(args)
    assert out_file.exists()
    tracks = read_imilast(out_file)
    assert len(tracks) > 0


@pytest.mark.integration
def test_grib_vo_serial(tmp_path: Path) -> None:
    """Test that tracking works correctly with VO850 GRIB input."""
    grib_path = fetch_era5_vo850(resolution="2.5x2.5", format="grib")
    out_file = tmp_path / "integration_grib_vo.txt"

    args = [
        "-i",
        grib_path,
        "-v",
        "vo",
        "-m",
        "max",
        "-o",
        str(out_file),
        "--backend",
        "serial",
    ]

    run_command_direct(args)
    assert out_file.exists()
    tracks = read_imilast(out_file)
    assert len(tracks) > 0


@pytest.mark.integration
def test_legacy_regression(
    shared_serial_output: Path, config: tuple[str, str, str]
) -> None:
    """Regression test against v0.0.2 legacy output."""
    _, varname, _ = config
    # Only compare if we are running the full msl dataset
    if varname != "msl":
        pytest.skip("Legacy regression only applies to full msl dataset")

    ref_file = "data/test/tracks/era5_msl_2.5x2.5_v0.0.2_imilast.txt"
    if not os.path.exists(ref_file):
        pytest.skip(f"Reference file {ref_file} not found")

    compare_tracks(
        ref_file,
        shared_serial_output,
        length_diff_tol=1,
        coord_tol=15.0,
        intensity_tol=500.0,
    )
