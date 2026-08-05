from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from utils import (
    fetch_era5_msl,
    fetch_era5_vo850,
    get_legacy_track_path,
)

from pystormtracker.io.imilast import read_imilast
from pystormtracker.metrics.compare import TrackComparisonConfig, compare_tracks
from pystormtracker.models.tracks import Tracks

N_WORKERS = 2


def run_command_direct(cmd_args: list[str], use_mpi: bool = False) -> Tracks | None:
    """Utility to run the tracker directly via function calls or MPI subprocess."""
    # Prepend 'track' command if not present
    if cmd_args and cmd_args[0] not in ["track", "sample", "convert", "compare"]:
        cmd_args = ["track", *cmd_args]

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

        params = [
            pytest.param(
                (variable_name, mode, steps),
                id=test_id,
                marks=pytest.mark.slow if steps is None else (),
            )
            for variable_name, mode, steps, test_id in raw_params
        ]

        metafunc.parametrize("config_params", params, scope="module")


@pytest.fixture(scope="module")
def config(
    request: pytest.FixtureRequest,
    config_params: tuple[str, str, int | None],
    test_data_msl: str,
    test_data_vo: str,
) -> tuple[str, str, str, int | None]:
    variable_name, mode, steps = config_params
    data_path = test_data_msl if variable_name == "msl" else test_data_vo

    # Full tests run in CI or when --run-slow/--run-all is explicitly passed.
    is_ci = os.environ.get("GITHUB_ACTIONS")
    run_all = request.config.getoption("--run-all")
    run_slow = request.config.getoption("--run-slow")

    if steps is None and not (is_ci or run_all or run_slow):
        pytest.skip(
            "Full integration tests only run in CI or with --run-slow/--run-all"
        )

    return data_path, variable_name, mode, steps


@pytest.fixture(scope="module")
def serial_reference(
    tmp_path_factory: pytest.TempPathFactory,
    config: tuple[str, str, str, int | None],
) -> tuple[Path, Tracks]:
    """Run serial once and share it across tests to save time."""
    data_path, variable_name, mode, steps = config
    temp_dir: Path = tmp_path_factory.mktemp("data")
    out_file = temp_dir / "integration_serial.txt"

    args = [
        "-i",
        data_path,
        "-v",
        variable_name,
        "-m",
        mode,
        "-o",
        str(out_file),
        "--backend",
        "serial",
    ]

    if steps:
        args.extend(["-n", str(steps)])

    tracks = run_command_direct(args)
    assert tracks is not None

    # Verbose print the IMILAST format output
    print(
        f"\nConfiguration: Variable={variable_name}, Mode={mode}, "
        f"Steps={steps or 'Full'}"
    )
    print_head(out_file, n=15)

    return Path(out_file), tracks


@pytest.mark.integration
def test_dask_vs_serial(
    serial_reference: tuple[Path, Tracks],
    tmp_path: Path,
    config: tuple[str, str, str, int | None],
) -> None:
    """Integration test comparing Serial and Dask backends."""
    serial_path, _ = serial_reference
    data_path, variable_name, mode, steps = config
    out_file = tmp_path / "integration_dask.txt"

    args = [
        "-i",
        data_path,
        "-v",
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

    if steps:
        args.extend(["-n", str(steps)])

    run_command_direct(args)
    compare_track_files(serial_path, out_file)


@pytest.mark.integration
def test_mpi_vs_serial(
    serial_reference: tuple[Path, Tracks],
    tmp_path: Path,
    config: tuple[str, str, str, int | None],
) -> None:
    """Integration test comparing Serial and MPI backends."""
    pytest.importorskip("mpi4py")

    # Check for mpiexec in PATH
    if shutil.which("mpiexec") is None:
        pytest.skip("mpiexec not found in PATH")

    data_path, variable_name, mode, steps = config
    serial_path, _ = serial_reference
    mpi_out = tmp_path / "integration_mpi.txt"

    args = [
        "-i",
        data_path,
        "-v",
        variable_name,
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
    compare_track_files(serial_path, mpi_out)


@pytest.mark.integration
def test_grib_vs_netcdf(
    serial_reference: tuple[Path, Tracks],
    tmp_path: Path,
    config: tuple[str, str, str, int | None],
) -> None:
    """Test that tracking matches between NetCDF and GRIB inputs."""
    pytest.importorskip("cfgrib")

    _, variable_name, mode, steps = config
    serial_path, _ = serial_reference

    if variable_name == "msl":
        grib_path = fetch_era5_msl(resolution="2.5x2.5", format="grib")
    elif variable_name == "vo":
        grib_path = fetch_era5_vo850(resolution="2.5x2.5", format="grib")
    else:
        pytest.skip(f"No GRIB test for {variable_name}")

    out_file = tmp_path / "integration_grib.txt"

    args = [
        "-i",
        grib_path,
        "-v",
        variable_name,
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
    compare_track_files(serial_path, out_file)


@pytest.mark.integration
@pytest.mark.slow
def test_legacy_regression(
    tmp_path: Path, config: tuple[str, str, str, int | None]
) -> None:
    """Regression test against v0.0.2 legacy output."""
    data_path, variable_name, mode, _ = config

    if variable_name == "msl":
        ref_file = get_legacy_track_path("msl")
        max_dist, min_overlap, min_match_rate = 220.0, 0.8, 0.95
    elif variable_name == "vo":
        ref_file = get_legacy_track_path("vo")
        max_dist, min_overlap, min_match_rate = 220.0, 0.8, 0.90
    else:
        pytest.skip(f"No legacy regression for {variable_name}")

    if not os.path.exists(ref_file):
        pytest.skip(f"Reference file {ref_file} not found")

    output_file = tmp_path / f"legacy_{variable_name}.txt"
    args = [
        "-i",
        data_path,
        "-v",
        variable_name,
        "-m",
        mode,
        "-o",
        str(output_file),
        "--backend",
        "serial",
    ]
    if variable_name == "vo":
        args.extend(["--threshold", "1e-4"])

    tracks_comp = run_command_direct(args)
    assert tracks_comp is not None
    tracks_ref = read_imilast(ref_file)

    comparison = compare_tracks(
        tracks_ref,
        tracks_comp,
        config=TrackComparisonConfig(
            max_mean_separation_deg=max_dist / 111.195,
            min_overlap_fraction=min_overlap,
        ),
    )

    match_rate = comparison.reference_coverage
    assert match_rate >= min_match_rate
