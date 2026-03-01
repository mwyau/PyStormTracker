import os
import subprocess
import pandas as pd
import pytest

# Paths to test data and outputs
TEST_NC = "data/slp.2012.nc"
SERIAL_OUT = "integration_serial.csv"
DASK_OUT = "integration_dask.csv"
MPI_OUT = "integration_mpi.csv"

# MS-MPI default path on Windows
MSMPI_BIN = r"C:\Program Files\Microsoft MPI\Bin"

def run_command(cmd, use_mpi=False):
    """Utility to run shell commands and check success."""
    venv_bin = os.path.join(".venv", "Scripts", "stormtracker")
    
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

def compare_csvs(file1, file2):
    """Compares two tracking CSV files for equality."""
    df1 = pd.read_csv(file1).sort_values(["track_id", "time"]).reset_index(drop=True)
    df2 = pd.read_csv(file2).sort_values(["track_id", "time"]).reset_index(drop=True)
    
    # Check shape
    assert df1.shape == df2.shape, f"Shape mismatch: {df1.shape} vs {df2.shape}"
    
    # Check content (allowing small float differences)
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False, atol=1e-4)

@pytest.fixture(scope="module", autouse=True)
def shared_serial_output():
    """Run serial once and share it across tests to save time."""
    if not os.path.exists(TEST_NC):
        pytest.skip("Test data not found")
    
    run_command(f"-i {TEST_NC} -v slp -o {SERIAL_OUT} -n 5 --backend serial")
    yield SERIAL_OUT
    if os.path.exists(SERIAL_OUT):
        os.remove(SERIAL_OUT)

def test_dask_vs_serial(shared_serial_output):
    """Integration test comparing Serial and Dask backends."""
    run_command(f"-i {TEST_NC} -v slp -o {DASK_OUT} -n 5 --backend dask --workers 2")
    compare_csvs(shared_serial_output, DASK_OUT)
    if os.path.exists(DASK_OUT):
        os.remove(DASK_OUT)

def test_mpi_vs_serial(shared_serial_output):
    """Integration test comparing Serial and MPI backends."""
    # Check if mpiexec is actually available
    env = os.environ.copy()
    if os.path.exists(MSMPI_BIN):
        env["PATH"] = MSMPI_BIN + os.pathsep + env["PATH"]
    
    try:
        subprocess.run("mpiexec -help", shell=True, capture_output=True, env=env)
    except FileNotFoundError:
        pytest.skip("mpiexec not found in path")

    run_command(f"-i {TEST_NC} -v slp -o {MPI_OUT} -n 10 --backend mpi", use_mpi=True)
    
    # Note: We run 10 steps for MPI to test split logic, 
    # so we need a fresh serial run for comparison if n is different.
    SERIAL_10 = "integration_serial_10.csv"
    run_command(f"-i {TEST_NC} -v slp -o {SERIAL_10} -n 10 --backend serial")
    
    compare_csvs(SERIAL_10, MPI_OUT)
    
    if os.path.exists(MPI_OUT): os.remove(MPI_OUT)
    if os.path.exists(SERIAL_10): os.remove(SERIAL_10)
