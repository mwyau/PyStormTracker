
import os
import numpy as np
import xarray as xr
from pystormtracker.preprocessing import SpectralFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MSL_FILE = os.path.join(BASE_DIR, "data/test/era5/era5_msl_2025120100_2.5x2.5.nc")

TEST_CASES = [
    {
        "name": "T5-42",
        "lmin": 5,
        "lmax": 42,
        "ref": os.path.join(BASE_DIR, "data/test/era5/era5_msl_2025120100_2.5x2.5_t5-42_ncl.nc"),
    },
    {
        "name": "T0-42",
        "lmin": 0,
        "lmax": 42,
        "ref": os.path.join(BASE_DIR, "data/test/era5/era5_msl_2025120100_2.5x2.5_t0-42_ncl.nc"),
    },
]

def calculate_stats(filtered, reference):
    f = filtered.values.flatten().astype(np.float64)
    r = reference.values.flatten().astype(np.float64)
    
    diff = f - r
    rmse = np.sqrt(np.mean(diff**2))
    abs_err = np.mean(np.abs(diff))
    
    # Relative error (using mean of absolute values of reference)
    rel_err = rmse / np.mean(np.abs(r))
    
    corr = np.corrcoef(f, r)[0, 1]
    
    return {
        "rmse": rmse,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "corr": corr
    }

def main():
    if not os.path.exists(MSL_FILE):
        print(f"Base MSL file not found: {MSL_FILE}")
        return

    ds_msl = xr.open_dataset(MSL_FILE)
    msl = ds_msl.msl

    engines = ["shtns", "ducc0"]
    
    header = f"{'Case':<10} {'Engine':<10} {'RMSE':<18} {'Abs Err':<18} {'Rel Err':<18} {'Corr':<18}"
    print(header)
    print("-" * len(header))

    for case in TEST_CASES:
        if not os.path.exists(case["ref"]):
            print(f"Reference file not found for {case['name']}: {case['ref']}")
            continue
            
        ds_ref = xr.open_dataset(case["ref"])
        ref = ds_ref.msl
        
        for engine in engines:
            try:
                filt = SpectralFilter(lmin=case["lmin"], lmax=case["lmax"], sht_engine=engine)
                filtered = filt.filter(msl)
                
                stats = calculate_stats(filtered, ref)
                print(f"{case['name']:<10} {engine:<10} {stats['rmse']:<18.8f} {stats['abs_err']:<18.8f} {stats['rel_err']:<18.8e} {stats['corr']:<18.12e}")
            except Exception as e:
                print(f"{case['name']:<10} {engine:<10} Error: {e}")

if __name__ == "__main__":
    main()
