from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from utils import (
    fetch_era5_msl,
    fetch_era5_vo850,
    get_legacy_track_path,
)

from pystormtracker.cli import main
from pystormtracker.io.imilast import read_imilast

N_WORKERS = 2


def run_command_direct(cmd_args: list[str], use_mpi: bool = False) -> None:
    """Utility to run the tracker directly via function calls or MPI subprocess."""
    if use_mpi:
        base_cmd = f"{sys.executable} -m pystormtracker.cli"
        # We assume mpiexec is in the PATH (e.g., provided by openmpi or winget)
        full_cmd = f"mpiexec -n {N_WORKERS} {base_cmd} {' '.join(cmd_args)}"
        try:
            subprocess.run(
                full_cmd, shell=True, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"MPI Command failed: {e.cmd}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise
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
    coord_tol: float = 1e-5,
    intensity_tol: float = 1e-5,
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


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Custom parameterization to filter tests dynamically."""
    if "config_params" in metafunc.fixturenames:
        raw_params = [
            ("msl", "min", 60, "msl_min_short"),
            ("vo", "max", 60, "vo_max_short"),
            ("msl", "min", None, "msl_min_full"),
            ("vo", "max", None, "vo_max_full"),
        ]

        # Filter out 'short' variants for legacy regression as they have
        # no reference data
        if metafunc.function.__name__ == "test_legacy_regression":
            raw_params = [p for p in raw_params if p[2] is None]

        params = [pytest.param((p[0], p[1], p[2]), id=p[3]) for p in raw_params]

        metafunc.parametrize("config_params", params, scope="module")


@pytest.fixture(scope="module")
def config(
    request: pytest.FixtureRequest,
    config_params: tuple[str, str, int | None],
    test_data_msl: str,
    test_data_vo: str,
) -> tuple[str, str, str, int | None]:
    varname, mode, steps = config_params
    data_path = test_data_msl if varname == "msl" else test_data_vo

    # Full tests only run in CI or when --run-all is explicitly passed
    is_ci = os.environ.get("GITHUB_ACTIONS")
    run_all = request.config.getoption("--run-all")

    if steps is None and not (is_ci or run_all):
        pytest.skip("Full integration tests only run in CI or with --run-all")

    if varname == "vo" and steps is None:
        pytest.skip("vo_max_full integration tests are temporarily disabled.")

    return data_path, varname, mode, steps


@pytest.fixture(scope="module")
def serial_reference(
    tmp_path_factory: pytest.TempPathFactory,
    config: tuple[str, str, str, int | None],
) -> Path:
    """Run serial once and share it across tests to save time."""
    data_path, varname, mode, steps = config
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

    if steps:
        args.extend(["-n", str(steps)])

    run_command_direct(args)

    # Verbose print the IMILAST format output
    print(f"\nConfiguration: Variable={varname}, Mode={mode}, Steps={steps or 'Full'}")
    print_head(out_file, n=15)

    return Path(out_file)


@pytest.mark.integration
def test_dask_vs_serial(
    serial_reference: Path, tmp_path: Path, config: tuple[str, str, str, int | None]
) -> None:
    """Integration test comparing Serial and Dask backends."""
    data_path, varname, mode, steps = config
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

    if steps:
        args.extend(["-n", str(steps)])

    run_command_direct(args)
    compare_tracks(serial_reference, out_file)


@pytest.mark.integration
def test_mpi_vs_serial(
    serial_reference: Path, tmp_path: Path, config: tuple[str, str, str, int | None]
) -> None:
    """Integration test comparing Serial and MPI backends."""
    # Check for mpiexec in PATH
    if not shutil.which("mpiexec"):
        # Double check with a run in case it's a shell alias or something similar
        try:
            res = subprocess.run(
                "mpiexec -help", shell=True, capture_output=True, text=True
            )
            if res.returncode != 0:
                pytest.skip("mpiexec not found or failed to run")
        except Exception:
            pytest.skip("mpiexec not found in path")

    data_path, varname, mode, steps = config
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

    if steps:
        args.extend(["-n", str(steps)])

    run_command_direct(args, use_mpi=True)
    compare_tracks(serial_reference, mpi_out)


@pytest.mark.integration
def test_grib_vs_netcdf(
    serial_reference: Path, tmp_path: Path, config: tuple[str, str, str, int | None]
) -> None:
    """Test that tracking matches between NetCDF and GRIB inputs."""
    import xarray as xr

    # Check if cfgrib engine is available
    if "cfgrib" not in xr.backends.list_engines():
        pytest.skip("cfgrib engine not available (ecCodes likely missing)")

    _, varname, mode, steps = config

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
        mode,
        "-o",
        str(out_file),
        "--backend",
        "serial",
    ]

    if steps:
        args.extend(["-n", str(steps)])

    run_command_direct(args)
    compare_tracks(serial_reference, out_file)


@pytest.mark.integration
def test_legacy_regression(
    serial_reference: Path, config: tuple[str, str, str, int | None]
) -> None:
    """Regression test against v0.0.2 legacy output."""
    _, varname, _, _ = config

    if varname == "msl":
        ref_file = get_legacy_track_path("msl")
        l_tol, c_tol, i_tol, count_tol = 1, 15.0, 500.0, 1
    elif varname == "vo":
        pytest.skip("Legacy VO regression tests are temporarily disabled.")
        ref_file = get_legacy_track_path("vo")
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
