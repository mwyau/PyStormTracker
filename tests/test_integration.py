import os
import subprocess
import pytest
from pystormtracker.data import fetch_era5_msl, fetch_era5_vo850

# MS-MPI default path on Windows
MSMPI_BIN = r"C:\Program Files\Microsoft MPI\Bin"

def run_command(cmd, use_mpi=False):
    """Utility to run shell commands and check success."""
    venv_bin = os.path.join(".venv", "Scripts", "stormtracker")
    # Fallback to just stormtracker if not in venv
    if not os.path.exists(venv_bin):
        venv_bin = "stormtracker"
    
    # Ensure MS-MPI bin is in path for Windows
    env = os.environ.copy()
    if os.path.exists(MSMPI_BIN):
        env["PATH"] = MSMPI_BIN + os.pathsep + env["PATH"]
    
    if use_mpi:
        full_cmd = f"mpiexec -n 2 {venv_bin} {cmd}"
    else:
        full_cmd = f"{venv_bin} {cmd}"
        
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, env=env)
    assert result.returncode == 0, f"Command failed: {full_cmd}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    return result.stdout

def print_head(filename, n=15):
    """Prints the first n lines of a file."""
    print(f"\n--- First {n} lines of {os.path.basename(filename)} ---")
    with open(filename, 'r') as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            print(line.rstrip())
    print("-------------------------------------------------------\n")

def compare_tracks(file1, file2, length_diff_tol=0, coord_tol=1e-4, intensity_tol=1e-4):
    """Compares two tracking files for equality."""
    def parse_imilast(filename):
        tracks = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            current_track = []
            for line in lines[1:]: # skip header
                parts = line.split()
                if not parts: continue
                if parts[0] == "90":
                    if current_track:
                        tracks.append(current_track)
                    current_track = []
                elif parts[0] == "00":
                    current_track.append(parts[1:])
            if current_track:
                tracks.append(current_track)
        
        # Sort tracks by their first point's time, lat, lon to ensure order independence
        return sorted(tracks, key=lambda t: (t[0][2], float(t[0][7]), float(t[0][8])) if t else ("", 0.0, 0.0))

    t1 = parse_imilast(file1)
    t2 = parse_imilast(file2)
    
    assert len(t1) == len(t2), f"Track count mismatch: {len(t1)} vs {len(t2)}"
    
    for tr1, tr2 in zip(t1, t2):
        assert abs(len(tr1) - len(tr2)) <= length_diff_tol, f"Track length mismatch too large: {len(tr1)} vs {len(tr2)}"
        
        # Convert tracks to dicts keyed by DateI10 (index 2) for robust matching
        d1 = {p[2]: p for p in tr1}
        d2 = {p[2]: p for p in tr2}
        
        common_dates = set(d1.keys()) & set(d2.keys())
        # We expect most points to be common if they are the same track
        assert len(common_dates) >= min(len(tr1), len(tr2)) - length_diff_tol, "Too few common points in track matching"

        for date in common_dates:
            p1, p2 = d1[date], d2[date]
            # Check float fields: lon, lat, Intensity1
            # In new format, these are indices 7, 8, 9
            for i in range(7, 9): # lon, lat
                val1, val2 = float(p1[i]), float(p2[i])
                assert abs(val1 - val2) <= coord_tol, f"Float mismatch at {date} index {i}: {val1} vs {val2}"
            
            # Intensity should be more stable, but can differ if linking picks a different center
            val1, val2 = float(p1[9]), float(p2[9])
            assert abs(val1 - val2) <= intensity_tol, f"Intensity mismatch at {date}: {val1} vs {val2}"

@pytest.fixture(scope="module")
def test_data_msl():
    """Download MSL test data once per module."""
    return fetch_era5_msl()

@pytest.fixture(scope="module")
def test_data_vo():
    """Download VO test data once per module."""
    return fetch_era5_vo850()

@pytest.fixture(scope="module", params=[
    ("msl", "min", "-n 120"),
    ("vo", "max", "-n 120"),
    ("msl", "min", ""),
    ("vo", "max", "")
], ids=["msl_min_n120", "vo_max_n120", "msl_min_full", "vo_max_full"])
def config(request, test_data_msl, test_data_vo):
    varname, mode, n_arg = request.param
    data_path = test_data_msl if varname == "msl" else test_data_vo
    return data_path, varname, mode, n_arg

@pytest.fixture(scope="module")
def shared_serial_output(tmp_path_factory, config):
    """Run serial once and share it across tests to save time."""
    data_path, varname, mode, n_arg = config
    temp_dir = tmp_path_factory.mktemp("data")
    out_file = temp_dir / "integration_serial.txt"
    run_command(f"-i {data_path} -v {varname} -m {mode} -o {out_file} {n_arg} --backend serial")
    
    # Verbose print the IMILAST format output
    print(f"\nConfiguration: Variable={varname}, Mode={mode}, Args={n_arg}")
    print_head(out_file, n=15)
    
    return out_file

def test_dask_vs_serial(shared_serial_output, tmp_path, config):
    """Integration test comparing Serial and Dask backends."""
    data_path, varname, mode, n_arg = config
    out_file = tmp_path / "integration_dask.txt"
    run_command(f"-i {data_path} -v {varname} -m {mode} -o {out_file} {n_arg} --backend dask --workers 2")
    compare_tracks(shared_serial_output, out_file)

def test_mpi_vs_serial(shared_serial_output, tmp_path, config):
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
    run_command(f"-i {data_path} -v {varname} -m {mode} -o {mpi_out} {n_arg} --backend mpi", use_mpi=True)
    
    # Compare directly to shared_serial_output
    compare_tracks(shared_serial_output, mpi_out)

def test_legacy_regression(test_data_msl, tmp_path):
    """Regression test against v0.0.2 legacy output using Dask."""
    ref_file = "data/test/tracks/era5_msl_2.5x2.5_v0.0.2_imilast.txt"
    if not os.path.exists(ref_file):
        pytest.skip(f"Reference file {ref_file} not found")
        
    out_file = tmp_path / "legacy_regression.txt"
    # Use Dask backend for speed with default workers
    run_command(f"-i {test_data_msl} -v msl -m min -o {out_file} --backend dask")
    compare_tracks(ref_file, out_file, length_diff_tol=1, coord_tol=15.0, intensity_tol=500.0)
