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

### 2. JAX (Experimental)
The **JAX** backend provides a JAX-native implementation of the SHT algorithms. It is designed for researchers looking to leverage GPU acceleration or automatic differentiation in their preprocessing pipelines.

*   **Pros**: GPU/TPU support via JAX. Machine-precision parity with `ducc0` for resolved harmonics.
*   **Cons**: Requires the `jax` extra (`pip install pystormtracker[jax]`). Subject to significant aliasing errors on coarse grids (e.g., 2.5°) where `lmax` is close to the grid's Nyquist frequency.

## Accuracy & Performance Metrics

### Spectral Filtering (MSL) Parity: JAX vs ducc0

| Resolution | Target lmax | Mean Relative Error | Max Relative Error | RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **0.25°x0.25°** | 42 | **2.77e-15** | 5.49e-09 | 4.03e-16 |
| **1.0°x1.0°** | 42 | **2.67e-14** | 1.15e-10 | 5.93e-16 |
| **2.5°x2.5°** | 42 | 5.81e-03 | 2.71e+00 | 7.24e-04 |

*Note: The high relative error at 2.5° is primarily due to differences in integration quadrature and aliasing treatment between the JAX-native matrix-vector approach and ducc0's optimized kernels.*

### Kinematic Derivatives (Vorticity & Divergence)
*Evaluated on 0.25°x0.25° ERA5 grid. Comparison against NCL `uv2vrdvF` reference.*

| Engine | Variable | RMSE (s⁻¹) | Correlation | Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **ducc0** | Vorticity | **1.74e-14** | **1.0000** | **0.0576** |
| **jax** | Vorticity | **1.12e-10** | **1.0000** | **3.49** |

*JAX timings are measured on CPU; significantly faster performance is expected on CUDA-enabled devices.*

## Usage & Validation

To use the JAX engine in the Python API:

```python
from pystormtracker.preprocessing import SpectralFilter

filt = SpectralFilter(lmin=5, lmax=42)
filtered = filt.filter(data, sht_engine="jax")
```

### Safety Limits
The JAX backend enforces strict resolution checks to prevent invalid transforms:
- **ValueError**: Raised if `lmax > ny - 2` (matching `ducc0` Clenshaw-Curtis limits).
- **UserWarning**: Issued if `lmax > ny / 2`, warning of potential aliasing artifacts.

## Summary

For most users, **ducc0** remains the recommended backend for its balance of speed and robustness. The **JAX** backend is an excellent choice for high-resolution (0.25° or 0.5°) datasets where alias-free transforms ensure scientific parity with the C++ backends while enabling JAX-native acceleration.
