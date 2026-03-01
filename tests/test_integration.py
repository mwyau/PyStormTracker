import os
import subprocess
import pandas as pd
import pytest

# Paths to test data and outputs
TEST_NC = "data/slp.2012.nc"
SERIAL_OUT = "integration_serial.csv"
DASK_OUT = "integration_dask.csv"
MPI_OUT = "integration_mpi.csv"

def run_command(cmd):
    """Utility to run shell commands and check success."""
    # Use the absolute path to the stormtracker script in the venv
    venv_bin = os.path.join(".venv", "Scripts", "stormtracker")
    full_cmd = f"{venv_bin} {cmd}"
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
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

@pytest.mark.skipif(not os.path.exists(TEST_NC), reason="Test data not found")
def test_backends_integration():
    """Integration test comparing Serial and Dask backends."""
    
    # 1. Run Serial
    run_command(f"-i {TEST_NC} -v slp -o {SERIAL_OUT} -n 5 --backend serial")
    
    # 2. Run Dask
    run_command(f"-i {TEST_NC} -v slp -o {DASK_OUT} -n 5 --backend dask --workers 2")
    
    # 3. Compare Serial vs Dask
    compare_csvs(SERIAL_OUT, DASK_OUT)
    
    # Clean up
    if os.path.exists(SERIAL_OUT): os.remove(SERIAL_OUT)
    if os.path.exists(DASK_OUT): os.remove(DASK_OUT)

@pytest.mark.skip(reason="MPI requires MS-MPI and mpiexec to be installed")
def test_mpi_integration():
    """Integration test for MPI backend (skipped by default)."""
    # This would be run as:
    # mpiexec -n 2 .venv/Scripts/stormtracker -i data/slp.2012.nc -v slp -o integration_mpi.csv -n 5 --backend mpi
    run_command(f"mpiexec -n 2 stormtracker -i {TEST_NC} -v slp -o {MPI_OUT} -n 5 --backend mpi")
    
    if os.path.exists(MPI_OUT): os.remove(MPI_OUT)
