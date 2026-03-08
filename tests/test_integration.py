import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from pystormtracker.data import fetch_era5_msl, fetch_era5_vo850
from pystormtracker.models.tracks import Tracks

# MS-MPI default path on Windows
MSMPI_BIN = r"C:\Program Files\Microsoft MPI\Bin"


def run_command(cmd: str, use_mpi: bool = False) -> str:
    """Utility to run shell commands and check success."""
    venv_bin = os.path.join(".venv", "Scripts", "stormtracker")
    # Fallback to just stormtracker if not in venv
    if not os.path.exists(venv_bin):
        venv_bin = "stormtracker"

    # Ensure MS-MPI bin is in path for Windows
    env = os.environ.copy()
    if os.path.exists(MSMPI_BIN):
        env["PATH"] = MSMPI_BIN + os.pathsep + env["PATH"]

    full_cmd = f"mpiexec -n 2 {venv_bin} {cmd}" if use_mpi else f"{venv_bin} {cmd}"

    result = subprocess.run(
        full_cmd, shell=True, capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, (
        f"Command failed: {full_cmd}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    return result.stdout


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
        ("msl", "min", "-n 120"),
        ("vo", "max", "-n 120"),
        ("msl", "min", ""),
        ("vo", "max", ""),
    ],
    ids=["msl_min_n120", "vo_max_n120", "msl_min_full", "vo_max_full"],
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
    temp_dir = tmp_path_factory.mktemp("data")
    out_file = temp_dir / "integration_serial.txt"
    run_command(
        f"-i {data_path} -v {varname} -m {mode} -o {out_file} {n_arg} --backend serial"
    )

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
    run_command(
        f"-i {data_path} -v {varname} -m {mode} -o {out_file} {n_arg} "
        "--backend dask --workers 2"
    )
    compare_tracks(shared_serial_output, out_file)


def test_mpi_vs_serial(
    shared_serial_output: Path, tmp_path: Path, config: tuple[str, str, str, str]
) -> None:
    """Integration test comparing Serial and MPI backends."""
    # Check if mpiexec is actually available
    env = os.environ.copy()
    if os.path.exists(MSMPI_BIN):
        env["PATH"] = MSMPI_BIN + os.pathsep + env["PATH"]

    try:
        subprocess.run("mpiexec -help", shell=True, capture_output=True, env=env)
    except FileNotFoundError:
        pytest.skip("mpiexec not found in path")

    data_path, varname, mode, n_arg = config

    mpi_out = tmp_path / "integration_mpi.txt"
    run_command(
        f"-i {data_path} -v {varname} -m {mode} -o {mpi_out} {n_arg} --backend mpi",
        use_mpi=True,
    )

    # Compare directly to shared_serial_output
    compare_tracks(shared_serial_output, mpi_out)


def test_legacy_regression(test_data_msl: str, tmp_path: Path) -> None:
    """Regression test against v0.0.2 legacy output using Dask."""
    ref_file = "data/test/tracks/era5_msl_2.5x2.5_v0.0.2_imilast.txt"
    if not os.path.exists(ref_file):
        pytest.skip(f"Reference file {ref_file} not found")

    out_file = tmp_path / "legacy_regression.txt"
    # Use Dask backend for speed with default workers
    run_command(f"-i {test_data_msl} -v msl -m min -o {out_file} --backend dask")
    compare_tracks(
        ref_file, out_file, length_diff_tol=1, coord_tol=15.0, intensity_tol=500.0
    )
