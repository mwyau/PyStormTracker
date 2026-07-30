# Spectral Filtering Accuracy

PyStormTracker supports multiple backends for spherical harmonic transforms (SHT) used in spectral filtering and kinematic derivative calculations. The choice of engine can be controlled via the `sht_engine` parameter in the Python API.

## Methodology

Accuracy is evaluated using **Root Mean Square Error (RMSE)**, **Relative Error**, and **Pearson Correlation Coefficient** for ERA5 Mean Sea Level Pressure (MSL) and Wind (U/V) data.

Two spatial resolutions are tested:
- **2.5°x2.5°**: A low-resolution grid (73x144) where T42 truncation (requiring ~85 latitudes) results in aliasing.
- **0.25°x0.25°**: A high-resolution grid (721x1440) where T42 truncation is alias-free.

## SHT Engines

### 1. ducc0 (Default)
**ducc0** is the standard backend for PyStormTracker. It is a high-performance C++ library that provides excellent multi-frame performance and bit-wise parity with modern SHT implementations.

*   **Pros**: Extremely fast, robust, and handles aliased coarse grids gracefully. No external C dependencies (self-contained).
*   **Cons**: Slightly less accurate than specialized legacy libraries like SHTns on specific high-resolution scalar fields.

### 2. SHTns (Legacy Reference)
**SHTns** was the primary backend in earlier versions. While no longer the default, its performance metrics serve as a high-precision benchmark for scalar filtering.

*   **Pros**: Exceptional accuracy on high-resolution alias-free grids for scalar fields.
*   **Cons**: Slower than `ducc0` for multi-frame workloads. Requires complex external C compilation. Significant numerical discrepancy (~10⁻⁶ RMSE) in kinematic derivative calculations compared to `Spherepack` (NCL).

## Accuracy & Performance Metrics

### Spectral Filtering (MSL)

#### Cross-Engine Accuracy (vs NCL Reference)

**Resolution: 2.5°x2.5° (ERA5)**
*Note: Higher error in this resolution is primarily due to aliasing on the coarse grid.*

| Engine | Truncation | RMSE (Pa) | Rel. Error | Correlation | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SHTns** | T5-42 | 0.45711949 | 6.40e-04 | 0.999999897272 | 0.0350 |
| **ducc0** | T5-42 | 0.05266872 | 7.37e-05 | 0.999999998663 | 0.0030 |
| **SHTns** | T0-42 | 0.45729944 | 4.53e-06 | 0.999999918845 | 0.0009 |
| **ducc0** | T0-42 | 0.05369308 | 5.31e-07 | 0.999999998880 | 0.0019 |

**Resolution: 0.25°x0.25° (ERA5)**
*Note: This resolution satisfies the sampling theorem for T42, resulting in near-perfect parity.*

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

## Impact of Polar Optimization (SHTns)

For high-resolution grids (0.25°), disabling the polar optimization threshold (`polar_opt=0.0`) ensures maximum precision. A comparison with the default setting (`1e-10`) shows:
- **Accuracy**: RMSE difference is negligible (~10⁻¹⁵ Pa).
- **Performance**: The default setting is approximately 1.25% faster.

Disabling the optimization is recommended for cases where strict bit-wise parity with double-precision ground truth is required.

## Summary

For standard storm tracking applications, all engines are scientifically equivalent. **ducc0** is the recommended default for its balance of speed, robustness, and ease of installation. It also provides the highest parity with legacy `Spherepack` references for kinematics. **SHTns** (with `polar_opt=0.0`) remains a high-precision benchmark for scalar fields, though it was passed over as the primary engine due to its numerical discrepancies in derivative calculations compared to standard meteorological tools.
