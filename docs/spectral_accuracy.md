# Spectral Filtering Accuracy

This document details the accuracy of the spherical harmonic transform (SHT) backends used for spectral filtering in `PyStormTracker`, comparing them against NCL (NCAR Command Language) as the ground truth reference.

## Methodology

Accuracy is evaluated using **Root Mean Square Error (RMSE)** and **Pearson Correlation Coefficient** for ERA5 Mean Sea Level Pressure (MSL) data. Two spatial resolutions are tested:
- **2.5°x2.5°**: A low-resolution grid (73x144) where T42 truncation (requiring ~85 latitudes) results in aliasing.
- **0.25°x0.25°**: A high-resolution grid (721x1440) where T42 truncation is alias-free.

All `SHTns` results are obtained using `polar_opt=0.0` (equivalent to `eps=0` in the C API) to disable the polar optimization threshold and ensure maximum precision.

## Accuracy Metrics

### Resolution: 2.5°x2.5° (ERA5)
*Note: High RMSE in this resolution is primarily due to aliasing on the coarse grid.*

| Engine | Truncation | RMSE (Pa) | Correlation |
| :--- | :--- | :--- | :--- |
| **SHTns** | T5-42 | 0.45711949 | 0.999999897272 |
| **ducc0** | T5-42 | 0.05266872 | 0.999999998663 |
| **SHTns** | T0-42 | 0.45729944 | 0.999999918845 |
| **ducc0** | T0-42 | 0.05369308 | 0.999999998880 |

### Resolution: 0.25°x0.25° (ERA5)
*Note: This resolution satisfies the sampling theorem for T42, resulting in near-perfect parity.*

| Engine | Truncation | RMSE (Pa) | Correlation |
| :--- | :--- | :--- | :--- |
| **SHTns** | T5-42 | 0.00003643 | 1.000000000000 |
| **ducc0** | T5-42 | 0.01276583 | 0.999999999928 |
| **SHTns** | T0-42 | 0.00486497 | 0.999999999994 |
| **ducc0** | T0-42 | 0.02114745 | 0.999999999831 |

## Performance vs. Accuracy

While both engines provide extreme structural correlation (> 0.9999), the choice of engine involves a trade-off between performance and bit-wise parity with legacy tools:

1.  **SHTns**:
    *   **Pros**: Significant performance advantage (18-35% faster) when `polar_opt=0.0` is used. Matches NCL/Spherepack extremely closely on high-resolution grids.
    *   **Cons**: Higher RMSE on coarse grids where aliasing occurs, as its internal handling of the sampling theorem differs slightly from NCL's `shaec` implementation.

2.  **ducc0**:
    *   **Pros**: Consistent low RMSE across all resolutions. Robust and easier to install (no external C libraries).
    *   **Cons**: Slightly slower than SHTns for large-scale iterative transforms.

## Summary

For standard storm tracking applications, both engines are scientifically equivalent. `SHTns` is recommended for high-throughput processing of high-resolution data, while `ducc0` provides the most stable accuracy on coarse or unconventional grids.
