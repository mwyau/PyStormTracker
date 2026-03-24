import os
import time
import numpy as np
import xarray as xr
from pystormtracker.preprocessing.derivatives import apply_wind_derivatives

def run_benchmark():
    # Attempt to locate the high-resolution dataset
    data_dir = os.path.expanduser("~/PyStormTracker-Data")
    uv_file = os.path.join(data_dir, "era5_uv850_2025-2026_djf_0.25x0.25.nc")
    
    if not os.path.exists(uv_file):
        print(f"Data file not found: {uv_file}")
        return

    print(f"Loading data from {uv_file}...")
    ds_uv = xr.open_dataset(uv_file)
    
    # Extract 360 time steps (approx 90 days if 6-hourly)
    num_steps = 360
    u = ds_uv.u.isel(valid_time=slice(0, num_steps), pressure_level=0).load()
    v = ds_uv.v.isel(valid_time=slice(0, num_steps), pressure_level=0).load()
    
    print(f"Data shape: {u.shape}")
    print("Running benchmarks...")
    print("-" * 50)
    print(f"{'Engine':<10} | {'Total Time (s)':<15} | {'Time/Frame (s)':<15}")
    print("-" * 50)
    
    engines = ["shtns", "ducc0", "shtools"]
    
    for engine in engines:
        try:
            start_time = time.perf_counter()
            # nthreads=0 tells ducc0/shtools to use all available cores.
            # shtns uses OMP_NUM_THREADS (which is usually all cores by default).
            _div, _vort = apply_wind_derivatives(u, v, engine=engine, nthreads=0)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            time_per_frame = total_time / num_steps
            
            print(f"{engine:<10} | {total_time:<15.4f} | {time_per_frame:<15.4f}")
        except Exception as e:
            print(f"{engine:<10} | {'FAILED':<15} | {str(e)}")

if __name__ == "__main__":
    run_benchmark()
