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
    count_tol: int = 0,
    dist_tol: float | None = None,
) -> None:
    """Compares two tracking files for equality using the Tracks class."""
    t1 = read_imilast(file1)
    t2 = read_imilast(file2)

    # Requirement: Small tolerance for legacy regression if needed
    assert abs(len(t1) - len(t2)) <= count_tol, (
        f"Track count mismatch: {len(t1)} vs {len(t2)}"
    )

    t1.compare(
        t2,
        length_diff_tol=length_diff_tol,
        coord_tol=coord_tol,
        intensity_tol=intensity_tol,
        count_tol=count_tol,
        dist_tol=dist_tol,
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

    if varname == "vo":
        run_vo = request.config.getoption("--run-vo")
        run_all = request.config.getoption("--run-all")
        if not (run_vo or run_all):
            pytest.skip("VO tests skipped (use --run-vo or --run-all to run)")

    data_path = test_data_msl if varname == "msl" else test_data_vo
    return data_path, varname, mode


@pytest.fixture(scope="module")
def serial_reference(
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

    # Use 1e-5 threshold for VO to match legacy reference
    run_command_direct(args)

    # Verbose print the IMILAST format output
    print(f"\nConfiguration: Variable={varname}, Mode={mode}")
    print_head(out_file, n=15)

    return Path(out_file)


@pytest.mark.integration
def test_dask_vs_serial(
    serial_reference: Path, tmp_path: Path, config: tuple[str, str, str]
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
    compare_tracks(serial_reference, out_file)


@pytest.mark.integration
def test_mpi_vs_serial(
    serial_reference: Path, tmp_path: Path, config: tuple[str, str, str]
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
    compare_tracks(serial_reference, mpi_out)


@pytest.mark.integration
def test_grib_vs_netcdf(
    serial_reference: Path, tmp_path: Path, config: tuple[str, str, str]
) -> None:
    """Test that tracking matches between NetCDF and GRIB inputs."""
    _, varname, _ = config

    if varname == "msl":
        grib_path = fetch_era5_msl(resolution="2.5x2.5", format="grib")
    elif varname == "vo":
        grib_path = fetch_era5_vo850(resolution="2.5x2.5", format="grib")
    else:
        pytest.skip(f"No GRIB test for {varname}")

    out_file = tmp_path / "integration_grib.txt"

    args = [
        "-i",
        grib_path,
        "-v",
        varname,
        "-m",
        "min" if varname == "msl" else "max",
        "-o",
        str(out_file),
        "--backend",
        "serial",
    ]
    run_command_direct(args)
    compare_tracks(serial_reference, out_file)


@pytest.mark.integration
def test_legacy_regression(
    serial_reference: Path, config: tuple[str, str, str]
) -> None:
    """Regression test against v0.0.2 legacy output."""
    _, varname, _ = config

    if varname == "msl":
        ref_file = "data/test/tracks/era5_msl_2.5x2.5_v0.0.2_imilast.txt"
        l_tol, c_tol, i_tol, count_tol = 1, 15.0, 500.0, 1
    elif varname == "vo":
        ref_file = "data/test/tracks/era5_vo_2.5x2.5_1e-5_v0.0.2_imilast.txt"
        # algorithmic improvements lead to slight differences, but should still be close
        l_tol, c_tol, i_tol, count_tol = 5, 15.0, 1.0, 100
    else:
        pytest.skip(f"No legacy regression for {varname}")

    if not os.path.exists(ref_file):
        pytest.skip(f"Reference file {ref_file} not found")

    compare_tracks(
        ref_file,
        serial_reference,
        length_diff_tol=l_tol,
        coord_tol=c_tol,
        intensity_tol=i_tol,
        count_tol=count_tol,
        dist_tol=None,
    )
