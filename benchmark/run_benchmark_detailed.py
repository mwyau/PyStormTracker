import json
import os
import re
import subprocess
import sys
import time

try:
    from pystormtracker.utils.data_utils import fetch_era5_msl
except ImportError:
    sys.path.insert(0, os.path.abspath("src"))
    from pystormtracker.utils.data_utils import fetch_era5_msl


def parse_output(output: str, version: str) -> dict[str, float]:
    # Parse detailed timings from stdout based on version
    # Initialize defaults
    results = {
        "detection": 0.0,
        "linking": 0.0,
        "export": 0.0,
        "total": 0.0,
        "wall": 0.0,
    }

    if version == "v0.3.3":
        # v0.3.3 format:
        # Detector time: X.XXXXs
        # Linker time: X.XXXXs
        # Combiner time: X.XXXXs (only MPI)
        # Total time: X.XXXXs (if tracked, but let's parse what's available)
        det_match = re.search(r"Detector time:\s*([0-9.]+)s", output)
        if det_match:
            results["detection"] = float(det_match.group(1))

        link_match = re.search(r"Linker time:\s*([0-9.]+)s", output)
        if link_match:
            results["linking"] = float(link_match.group(1))

        comb_match = re.search(r"Combiner time:\s*([0-9.]+)s", output)
        if comb_match:
            # Merge combiner into linking
            results["linking"] += float(comb_match.group(1))

        tot_match = re.search(r"Total time:\s*([0-9.]+)s", output)
        if tot_match:
            results["total"] = float(tot_match.group(1))

        # v0.3.3 might not explicitly log export time in all contexts, fallback to 0

    else:
        # v0.4.0 format:
        # [Serial] Detection time: X.XXXXs
        # [Serial] Linking time: X.XXXXs
        # [Dask] Cluster setup time: X.XXXXs
        # [Dask] Task execution & gather time: X.XXXXs
        # [Dask] Linking time: X.XXXXs
        # [MPI] Prep & Scatter time: X.XXXXs
        # [MPI] Detection & Gather time: X.XXXXs
        # [MPI] Linking time: X.XXXXs
        # Tracking time: X.XXXXs
        # Export time: X.XXXXs
        # Total time: X.XXXXs

        # Detection (summing setup/gather if present)
        det_match_serial = re.search(r"\[Serial\] Detection time:\s*([0-9.]+)s", output)
        if det_match_serial:
            results["detection"] = float(det_match_serial.group(1))

        det_match_dask = re.search(
            r"Task execution & gather time:\s*([0-9.]+)s", output
        )
        if det_match_dask:
            results["detection"] = float(det_match_dask.group(1))
            # Optional: Add setup time to detection overhead
            setup_match = re.search(r"Cluster setup time:\s*([0-9.]+)s", output)
            if setup_match:
                results["detection"] += float(setup_match.group(1))

        det_match_mpi = re.search(r"Detection & Gather time:\s*([0-9.]+)s", output)
        if det_match_mpi:
            results["detection"] = float(det_match_mpi.group(1))
            setup_match = re.search(r"Prep & Scatter time:\s*([0-9.]+)s", output)
            if setup_match:
                results["detection"] += float(setup_match.group(1))

        # Linking
        link_match = re.search(r"Linking time:\s*([0-9.]+)s", output)
        if link_match:
            results["linking"] = float(link_match.group(1))

        # Export
        exp_match = re.search(r"Export time:\s*([0-9.]+)s", output)
        if exp_match:
            results["export"] = float(exp_match.group(1))

        # Total explicitly logged by cli
        tot_match = re.search(r"Total time:\s*([0-9.]+)s", output)
        if tot_match:
            results["total"] = float(tot_match.group(1))

    return results


def run_cmd(cmd: str, version: str) -> dict[str, float] | None:
    print(f"Running: {cmd}")
    start = time.time()
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end = time.time()
    wall_time = end - start

    if res.returncode != 0:
        print(f"Command failed: {cmd}\n{res.stderr}")
        return None

    parsed = parse_output(res.stdout, version)
    parsed["wall"] = wall_time

    # If the CLI didn't log total time, fall back to Wall time
    if parsed["total"] == 0.0:
        parsed["total"] = wall_time

    # IO/Overhead is calculated as Total - (Detection + Linking + Export)
    # This accounts for process startup, xarray dataset loading, and Python imports.
    parsed["io_overhead"] = max(
        0.0,
        parsed["total"] - parsed["detection"] - parsed["linking"] - parsed["export"],
    )

    print(
        f"  -> Wall: {wall_time:.2f}s | Det: {parsed['detection']:.2f}s | "
        f"Link: {parsed['linking']:.2f}s | Exp: {parsed['export']:.2f}s | "
        f"IO/Startup: {parsed['io_overhead']:.2f}s"
    )
    return parsed


def main() -> None:
    f_25 = fetch_era5_msl(resolution="2.5x2.5")
    f_025 = fetch_era5_msl(resolution="0.25x0.25")

    datasets = {
        "2.5x2.5": {"path": f_25, "n": 360},
        "0.25x0.25": {"path": f_025, "n": 360},
    }

    version = sys.argv[1]

    if version == "v0.3.3":
        base_cmd = "uv run --directory worktrees/bench-v0.3.3 stormtracker"
        mpi_cmd = "uv run --directory worktrees/bench-v0.3.3 mpiexec --oversubscribe"
        mpi_target = "python -m pystormtracker.stormtracker"
    else:
        base_cmd = "uv run stormtracker"
        mpi_cmd = "uv run mpiexec --oversubscribe"
        mpi_target = "python -m pystormtracker.cli"

    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for res_name, info in datasets.items():
        results[res_name] = {}
        # -n limit back to 60 for 0.25 to prevent memory hang in sub-processes
        n_steps = 360 if res_name == "2.5x2.5" else 60
        args = f"-i {info['path']} -v msl -m min -o /tmp/bmk_out.txt -n {n_steps}"

        # Serial
        res_serial = run_cmd(f"{base_cmd} {args} -b serial", version)
        if res_serial:
            results[res_name]["serial"] = {"1": res_serial}

        # Dask
        results[res_name]["dask"] = {}
        for w in [2, 4, 8]:
            res_dask = run_cmd(f"{base_cmd} {args} -b dask -w {w}", version)
            if res_dask:
                results[res_name]["dask"][str(w)] = res_dask

        # MPI
        results[res_name]["mpi"] = {}
        for w in [2, 4, 8]:
            res_mpi = run_cmd(f"{mpi_cmd} -n {w} {mpi_target} {args} -b mpi", version)
            if res_mpi:
                results[res_name]["mpi"][str(w)] = res_mpi

    with open(f"benchmark_detailed_{version}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
