import os
import numpy as np
import xarray as xr
from pystormtracker.preprocessing.kinematics import apply_vort_div

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIND_FILE = os.path.join(BASE_DIR, "data/test/era5/era5_uv850_2025120100_2.5x2.5.nc")
VODIV_FILE = os.path.join(BASE_DIR, "data/test/era5/era5_vodv850_2025120100_2.5x2.5_ncl.nc")

def calculate_stats(calc, ref):
    c = calc.values.flatten().astype(np.float64)
    r = ref.values.flatten().astype(np.float64)
    
    diff = c - r
    rmse = np.sqrt(np.mean(diff**2))
    abs_err = np.mean(np.abs(diff))
    
    # Relative error (using mean of absolute values of reference)
    rel_err = rmse / np.mean(np.abs(r))
    
    corr = np.corrcoef(c, r)[0, 1]
    
    return {
        "rmse": rmse,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "corr": corr
    }

def main():
    if not os.path.exists(WIND_FILE) or not os.path.exists(VODIV_FILE):
        print(f"Test data not found.")
        return

    ds_uv = xr.open_dataset(WIND_FILE)
    ds_ref = xr.open_dataset(VODIV_FILE)

    u, v = ds_uv.u, ds_uv.v
    vo_ref, dv_ref = ds_ref.vo, ds_ref.dv

    engines = ["ducc0"]
    
    header = f"{'Variable':<10} {'Engine':<10} {'RMSE':<18} {'Abs Err':<18} {'Rel Err':<18} {'Corr':<18}"
    print(header)
    print("-" * len(header))

    for engine in engines:
        try:
            div, vort = apply_vort_div(u, v, sht_engine=engine)
            
            # Vorticity
            vo_stats = calculate_stats(vort, vo_ref)
            if engine == engines[0]:
                print(f"Ref Mean Abs (Vorticity): {np.mean(np.abs(vo_ref.values)):.8e}")
            print(f"{'Vorticity':<10} {engine:<10} {vo_stats['rmse']:<18.8f} {vo_stats['abs_err']:<18.8f} {vo_stats['rel_err']:<18.12f} {vo_stats['corr']:<18.12f}")
            
            # Divergence
            dv_stats = calculate_stats(div, dv_ref)
            if engine == engines[0]:
                print(f"Ref Mean Abs (Divergence): {np.mean(np.abs(dv_ref.values)):.8e}")
            print(f"{'Diverg.':<10} {engine:<10} {dv_stats['rmse']:<18.8f} {dv_stats['abs_err']:<18.8f} {dv_stats['rel_err']:<18.12f} {dv_stats['corr']:<18.12f}")
        except Exception as e:
            print(f"Engine {engine} Error: {e}")

if __name__ == "__main__":
    main()
