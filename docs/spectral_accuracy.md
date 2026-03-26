# Spectral Filtering Accuracy

> [!NOTE]
> As of version 0.5.0, `PyStormTracker` has transitioned to using **ducc0** as the exclusive spectral backend. Support for **SHTns** and the `sht_engine` parameter has been removed to simplify the codebase and leverage the superior multi-frame performance and portability of `ducc0`. This document remains as a technical reference for the accuracy and performance trade-offs that informed this decision.

This document details the accuracy and performance of the spherical harmonic transform (SHT) backends used for spectral filtering and kinematic derivative calculations in `PyStormTracker`, comparing them against NCL (NCAR Command Language) as the ground truth reference.

## Methodology

Accuracy is evaluated using **Root Mean Square Error (RMSE)**, **Relative Error**, and **Pearson Correlation Coefficient** for ERA5 Mean Sea Level Pressure (MSL) and Wind (U/V) data. 

Two spatial resolutions are tested:
- **2.5°x2.5°**: A low-resolution grid (73x144) where T42 truncation (requiring ~85 latitudes) results in aliasing.
- **0.25°x0.25°**: A high-resolution grid (721x1440) where T42 truncation is alias-free.

All `SHTns` results are obtained using `polar_opt=0.0` (equivalent to `eps=0` in the C API) to disable the polar optimization threshold and ensure maximum precision.

## Accuracy & Performance Metrics

### Spectral Filtering (MSL)

#### Resolution: 2.5°x2.5° (ERA5)
*Note: Higher error in this resolution is primarily due to aliasing on the coarse grid. Times are for a single frame.*

| Engine | Truncation | RMSE (Pa) | Rel. Error | Correlation | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SHTns** | T5-42 | 0.45711949 | 6.40e-04 | 0.999999897272 | 0.0350 |
| **ducc0** | T5-42 | 0.05266872 | 7.37e-05 | 0.999999998663 | 0.0030 |
| **SHTns** | T0-42 | 0.45729944 | 4.53e-06 | 0.999999918845 | 0.0009 |
| **ducc0** | T0-42 | 0.05369308 | 5.31e-07 | 0.999999998880 | 0.0019 |

#### Resolution: 0.25°x0.25° (ERA5)
*Note: This resolution satisfies the sampling theorem for T42, resulting in near-perfect parity. Times are for a single frame.*

| Engine | Truncation | RMSE (Pa) | Rel. Error | Correlation | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SHTns** | T5-42 | 0.00003643 | 5.16e-08 | 1.000000000000 | 0.0665 |
| **ducc0** | T5-42 | 0.01276583 | 1.81e-05 | 0.999999999928 | 0.0041 |
| **SHTns** | T0-42 | 0.00486497 | 4.81e-08 | 0.999999999994 | 0.0141 |
| **ducc0** | T0-42 | 0.02114745 | 2.09e-07 | 0.999999999831 | 0.0038 |

### Kinematic Derivatives (Vorticity & Divergence)
*Evaluated on 0.25°x0.25° ERA5 grid. Comparison against NCL `uv2vrdvF` reference.*

| Engine | Variable | RMSE (s⁻¹) | Correlation | Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **SHTns** | Vorticity | 9.05e-06 | 0.9899 | 0.1017 |
| **ducc0** | Vorticity | **1.74e-14** | **1.0000** | **0.0576** |
| **SHTns** | Divergence | 8.27e-06 | 0.9917 | 0.1017 |
| **ducc0** | Divergence | **1.68e-14** | **1.0000** | **0.0576** |

### Full Dataset Benchmark (0.25°x0.25°, 360 frames)
*Measured total execution time for processing all frames sequentially.*

| Engine | Total Time (s) | Time per Frame (ms) | Speed Comparison |
| :--- | :--- | :--- | :--- |
| **SHTns** | 5.9887 | 16.64 | 1.0x |
| **ducc0** | 0.9428 | 2.62 | **6.3x faster** |

## Impact of Polar Optimization (SHTns)

For high-resolution grids (0.25°), disabling the polar optimization threshold (`polar_opt=0.0`) ensures maximum precision. A comparison with the default setting (`1e-10`) shows:
- **Accuracy**: RMSE difference is negligible (~10⁻¹⁵ Pa).
- **Performance**: The default setting is approximately 1.25% faster.

Disabling the optimization is recommended for cases where strict bit-wise parity with double-precision ground truth is required.

## Performance vs. Accuracy

The choice of engine involves a significant trade-off between performance and bit-wise parity with legacy tools:

1.  **SHTns**:
    *   **Pros**: Exceptional accuracy on high-resolution alias-free grids (RMSE ~10⁻⁵ Pa). Provides the closest match to NCL/Spherepack ground truth for scalar filtering.
    *   **Cons**: Slower than `ducc0` for multi-frame workloads. Higher RMSE on coarse grids due to aliasing handling differences.

2.  **ducc0**:
    *   **Pros**: **Extremely fast** (6.3x faster than SHTns for 0.25° data). Achieves near bit-wise parity with NCL for kinematics (RMSE ~10⁻¹⁴ s⁻¹). **No external C dependencies** (self-contained pocketFFT implementation).
    *   **Cons**: Slightly less accurate than SHTns on high-resolution grids compared to NCL ground truth for scalar filtering.

## Summary

For standard storm tracking applications, both engines are scientifically equivalent. **ducc0** is the recommended default for its superior performance, robustness, and ease of installation. **SHTns** (with `polar_opt=0.0`) is preferred when maximum bit-wise parity with NCL is required for high-resolution validation of scalar fields.
