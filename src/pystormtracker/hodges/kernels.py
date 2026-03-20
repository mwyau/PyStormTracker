from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ..utils.geo import DEGTORAD, geod_dist

R_EARTH = 6367.0  # Radius of Earth in km


@nb.njit(parallel=True, cache=True)
def _numba_hodges_extrema(
    frame: NDArray[np.float64],
    size: int,
    threshold: float,
    is_min: bool,
) -> NDArray[np.float64]:
    """
    Finds local extrema in a 2D frame.
    Handles longitude periodicity (assumed last dimension).
    """
    ny, nx = frame.shape
    extrema = np.zeros_like(frame)
    half = size // 2

    for i in nb.prange(ny):
        for j in range(nx):
            val = frame[i, j]
            if is_min:
                if val > threshold:
                    continue
            else:
                if val < threshold:
                    continue

            is_extrema = True
            for di in range(-half, half + 1):
                ni = i + di
                if ni < 0 or ni >= ny:
                    continue
                for dj in range(-half, half + 1):
                    if di == 0 and dj == 0:
                        continue
                    nj = (j + dj) % nx
                    nval = frame[ni, nj]
                    if is_min:
                        if nval < val:
                            is_extrema = False
                            break
                    else:
                        if nval > val:
                            is_extrema = False
                            break
                if not is_extrema:
                    break
            
            if is_extrema:
                extrema[i, j] = 1.0
    
    return extrema


@nb.njit(cache=True)
def _numba_get_centers(
    extrema: NDArray[np.float64],
    frame: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Extracts coordinates and values of detected extrema."""
    idx = np.where(extrema > 0)
    r_idx = idx[0]
    c_idx = idx[1]
    vals = np.zeros(len(r_idx), dtype=np.float64)
    for i in range(len(r_idx)):
        vals[i] = frame[r_idx[i], c_idx[i]]
    return r_idx, c_idx, vals


@nb.njit(cache=True)
def subgrid_refine(
    frame: NDArray[np.float64],
    r: int,
    c: int,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Refines the extremum position using local quadratic interpolation.
    Fits f(y, x) = a*y^2 + b*x^2 + c*y*x + d*y + e*x + f
    Returns (refined_lat, refined_lon, refined_val).
    """
    ny, nx = frame.shape
    
    # Boundary check for 3x3 neighborhood
    if r < 1 or r >= ny - 1:
        return lat[r], lon[c], frame[r, c]
    
    # Extract 3x3 neighborhood
    # Longitude is periodic
    cm = (c - 1) % nx
    cp = (c + 1) % nx
    
    z = np.zeros((3, 3))
    z[0, 0] = frame[r-1, cm]
    z[0, 1] = frame[r-1, c]
    z[0, 2] = frame[r-1, cp]
    z[1, 0] = frame[r, cm]
    z[1, 1] = frame[r, c]
    z[1, 2] = frame[r, cp]
    z[2, 0] = frame[r+1, cm]
    z[2, 1] = frame[r+1, c]
    z[2, 2] = frame[r+1, cp]
    
    # Coefficients of the quadratic surface (simple finite differences)
    # d2f/dy2
    f_yy = z[0, 1] - 2*z[1, 1] + z[2, 1]
    # d2f/dx2
    f_xx = z[1, 0] - 2*z[1, 1] + z[1, 2]
    # d2f/dy dx
    f_yx = 0.25 * (z[2, 2] - z[2, 0] - z[0, 2] + z[0, 0])
    # df/dy
    f_y = 0.5 * (z[2, 1] - z[0, 1])
    # df/dx
    f_x = 0.5 * (z[1, 2] - z[1, 0])
    
    det = f_yy * f_xx - f_yx**2
    if abs(det) < 1e-10:
        return lat[r], lon[c], frame[r, c]
    
    dy = (f_yx * f_x - f_xx * f_y) / det
    dx = (f_yx * f_y - f_yy * f_x) / det
    
    # If refinement is too far (> 1 grid point), fallback to grid center
    if abs(dy) > 1.0 or abs(dx) > 1.0:
        return lat[r], lon[c], frame[r, c]
        
    # Interpolate lat/lon
    # Handle lat spacing
    dlat = lat[r+1] - lat[r] if r < ny - 1 else lat[r] - lat[r-1]
    # Handle lon spacing (assuming regular)
    dlon = lon[1] - lon[0] if nx > 1 else 0.0
    if cp < cm: # Wrapped
         dlon = (lon[cp] + 360.0 - lon[cm]) / 2.0
    
    ref_lat = lat[r] + dy * dlat
    ref_lon = lon[c] + dx * dlon
    if ref_lon >= 360.0: ref_lon -= 360.0
    if ref_lon < 0.0: ref_lon += 360.0
    
    ref_val = z[1, 1] + 0.5 * (f_y * dy + f_x * dx)
    
    return ref_lat, ref_lon, ref_val


@nb.njit(cache=True)
def geod_dev(
    p0_lat: float, p0_lon: float,
    p1_lat: float, p1_lon: float,
    p2_lat: float, p2_lon: float,
    w1: float, w2: float
) -> float:
    """
    Spherical cost function (Hodges 1995, 1999).
    Calculates the deviation over three consecutive points.
    """
    # 1. Distances (angular)
    alpha1 = geod_dist(p0_lat, p0_lon, p1_lat, p1_lon)
    alpha2 = geod_dist(p1_lat, p1_lon, p2_lat, p2_lon)
    
    if alpha1 <= 0.0 and alpha2 <= 0.0:
        return 0.0
    if alpha1 <= 0.0 or alpha2 <= 0.0:
        return w2 # Penalty for zero speed
        
    # 2. Tangent vector deviation
    # Convert to unit cartesian
    x0 = np.cos(p0_lat * DEGTORAD) * np.cos(p0_lon * DEGTORAD)
    y0 = np.cos(p0_lat * DEGTORAD) * np.sin(p0_lon * DEGTORAD)
    z0 = np.sin(p0_lat * DEGTORAD)
    
    x1 = np.cos(p1_lat * DEGTORAD) * np.cos(p1_lon * DEGTORAD)
    y1 = np.cos(p1_lat * DEGTORAD) * np.sin(p1_lon * DEGTORAD)
    z1 = np.sin(p1_lat * DEGTORAD)
    
    x2 = np.cos(p2_lat * DEGTORAD) * np.cos(p2_lon * DEGTORAD)
    y2 = np.cos(p2_lat * DEGTORAD) * np.sin(p2_lon * DEGTORAD)
    z2 = np.sin(p2_lat * DEGTORAD)
    
    # Dot product p0.p1 and p2.p1
    dot01 = x0*x1 + y0*y1 + z0*z1
    dot21 = x2*x1 + y2*y1 + z2*z1
    
    s1 = np.sin(alpha1)
    s2 = np.sin(alpha2)
    
    # Tangent vectors at p1
    # T1 = (p0 - (p0.p1)p1) / sin(alpha1)
    t1x = (x0 - dot01*x1) / s1
    t1y = (y0 - dot01*y1) / s1
    t1z = (z0 - dot01*z1) / s1
    
    # T2 = (p2.p1)p1 - p2 / sin(alpha2) (direction reversed to align)
    t2x = (dot21*x1 - x2) / s2
    t2y = (dot21*y1 - y2) / s2
    t2z = (dot21*z1 - z2) / s2
    
    # Dot product of tangent vectors
    dot_t = t1x*t2x + t1y*t2y + t1z*t2z
    
    # 3. Combined cost
    # Smoothness (direction) + Speed variation
    phi = w1 * (1.0 - dot_t) + w2 * (1.0 - 2.0 * np.sqrt(alpha1 * alpha2) / (alpha1 + alpha2))
    
    return max(0.0, phi)


@nb.njit(cache=True)
def get_regional_dmax(lat: float, lon: float, zones: NDArray[np.float64], default_dmax: float) -> float:
    """
    Returns dmax for a given lat/lon based on regional zones.
    zones array shape (n_zones, 5): [lon_min, lon_max, lat_min, lat_max, dmax]
    """
    if zones.shape[0] == 0:
        return default_dmax
        
    for i in range(zones.shape[0]):
        lon_min, lon_max, lat_min, lat_max, dmax = zones[i]
        
        # Latitude check
        if lat < lat_min or lat > lat_max:
            continue
            
        # Longitude check (periodic)
        in_lon = False
        if lon_min > lon_max: # Wraps around
            if lon >= lon_min or lon <= lon_max:
                in_lon = True
        else:
            if lon >= lon_min and lon <= lon_max:
                in_lon = True
                
        if in_lon:
            return dmax
            
    return default_dmax


@nb.njit(cache=True)
def get_adaptive_phimax(
    mean_dist: float, 
    adapt_thresholds: NDArray[np.float64], 
    adapt_values: NDArray[np.float64],
    default_phimax: float
) -> float:
    """
    Returns phimax based on adaptive smoothness piecewise linear function.
    adapt_thresholds: shape (4,) distances in degrees.
    adapt_values: shape (4,) phi values.
    """
    if adapt_thresholds.shape[0] < 4:
        return default_phimax
        
    # thresholds are in degrees, convert mean_dist from radians to degrees if needed
    # (assuming mean_dist passed in degrees here for simplicity, or we convert it)
    
    d = mean_dist
    
    if d < adapt_thresholds[0]:
        return adapt_values[0]
    
    if d >= adapt_thresholds[3]:
        return adapt_values[3]
        
    if d >= adapt_thresholds[0] and d < adapt_thresholds[1]:
        # Interpolate 0 and 1
        slope = (adapt_values[1] - adapt_values[0]) / (adapt_thresholds[1] - adapt_thresholds[0])
        return adapt_values[0] + slope * (d - adapt_thresholds[0])
        
    if d >= adapt_thresholds[1] and d < adapt_thresholds[2]:
        # Interpolate 1 and 2
        slope = (adapt_values[2] - adapt_values[1]) / (adapt_thresholds[2] - adapt_thresholds[1])
        return adapt_values[1] + slope * (d - adapt_thresholds[1])
        
    if d >= adapt_thresholds[2] and d < adapt_thresholds[3]:
        # Interpolate 2 and 3
        slope = (adapt_values[3] - adapt_values[2]) / (adapt_thresholds[3] - adapt_thresholds[2])
        return adapt_values[2] + slope * (d - adapt_thresholds[2])
        
    return default_phimax
