from __future__ import annotations

import functools
from typing import Literal

import ducc0
import numpy as np
import jax
import jax.numpy as jnp
from numpy.typing import NDArray

@functools.lru_cache(maxsize=16)
def _get_sht_matrices(
    ny: int, 
    nx: int, 
    lmax: int, 
    mmax: int | None = None,
    geometry: str = "CC",
    spin: int = 0
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """
    Precomputes Legendre matrices and quadrature weights for JAX SHT.
    Using ducc0 as a reference for bit-wise parity.
    """
    if mmax is None:
        mmax = min(lmax, (nx - 1) // 2)

    weights = ducc0.sht.get_gridweights(geometry=geometry, ntheta=ny)
    theta = np.linspace(0, np.pi, ny)
    
    plm_list = []
    for m in range(mmax + 1):
        l_start = max(m, spin)
        if l_start > lmax:
            continue
            
        n_l_active = lmax - l_start + 1
        m_mat = np.zeros((n_l_active, ny), dtype=np.float64)
        
        for i_l_rel in range(n_l_active):
            l_val = l_start + i_l_rel
            n_comp = 1 if spin == 0 else 2
            alm_m = np.zeros((n_comp, lmax + 1), dtype=np.complex128)
            alm_m[0, l_val] = 1.0
            leg = ducc0.sht.alm2leg(
                alm=alm_m, spin=spin, lmax=lmax, theta=theta, mval=np.array([m]), mstart=np.array([0])
            )
            # leg shape is (n_spin, ntheta, n_m)
            m_mat[i_l_rel, :] = leg[0, :, 0].real
        plm_list.append(m_mat)
    
    plm = np.concatenate(plm_list, axis=0)
    
    if spin == 1:
        plm_b_list = []
        for m in range(mmax + 1):
            l_start = max(m, spin)
            if l_start > lmax: continue
            n_l_active = lmax - l_start + 1
            m_mat_b = np.zeros((n_l_active, ny), dtype=np.float64)
            for i_l_rel in range(n_l_active):
                l_val = l_start + i_l_rel
                alm_m = np.zeros((2, lmax + 1), dtype=np.complex128)
                alm_m[0, l_val] = 1.0
                leg = ducc0.sht.alm2leg(
                    alm=alm_m, spin=spin, lmax=lmax, theta=theta, mval=np.array([m]), mstart=np.array([0])
                )
                m_mat_b[i_l_rel, :] = leg[1, :, 0].real
            plm_b_list.append(m_mat_b)
        plm_b = np.concatenate(plm_b_list, axis=0)
        return jnp.asarray(plm), jnp.asarray(weights), jnp.asarray(mmax), jnp.asarray(plm_b)

    return jnp.asarray(plm), jnp.asarray(weights), jnp.asarray(mmax), None

def jax_analysis_2d(
    frame: jnp.ndarray, 
    lmax: int, 
    mmax: int | None = None, 
    geometry: str = "CC",
    spin: int = 0
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-native SHT analysis (Forward) matching ducc0."""
    ny, nx = frame.shape[-2], frame.shape[-1]
    if mmax is None:
        mmax = min(lmax, (nx - 1) // 2)

    if spin == 0:
        plm, weights, _, _ = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=0)
        f_m = jnp.fft.rfft(frame, axis=-1)[:, :mmax + 1]
        weighted_f_m = f_m * (weights[:, jnp.newaxis] / nx)
        
        alm_list = []
        curr = 0
        for m in range(mmax + 1):
            n_l = lmax - m + 1
            alm_m = jnp.matmul(plm[curr : curr + n_l, :], weighted_f_m[:, m])
            alm_list.append(alm_m)
            curr += n_l
        return jnp.concatenate(alm_list)
    
    elif spin == 1:
        plm_e, weights, _, plm_b = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=1)
        v_theta_m = jnp.fft.rfft(frame[0], axis=-1)[:, :mmax + 1]
        v_phi_m = jnp.fft.rfft(frame[1], axis=-1)[:, :mmax + 1]
        
        w_v_theta_m = v_theta_m * (weights[:, jnp.newaxis] / nx)
        w_v_phi_m = v_phi_m * (weights[:, jnp.newaxis] / nx)
        
        alm_e_list = []
        alm_b_list = []
        curr = 0
        for m in range(mmax + 1):
            l_start = max(m, 1)
            if l_start > lmax: continue
            n_l = lmax - l_start + 1
            e_m = jnp.matmul(plm_e[curr:curr+n_l], w_v_theta_m[:, m]) + \
                  1j * jnp.matmul(plm_b[curr:curr+n_l], w_v_phi_m[:, m])
            b_m = jnp.matmul(plm_b[curr:curr+n_l], w_v_theta_m[:, m]) - \
                  1j * jnp.matmul(plm_e[curr:curr+n_l], w_v_phi_m[:, m])
            alm_e_list.append(e_m)
            alm_b_list.append(b_m)
            curr += n_l
        return jnp.concatenate(alm_e_list), jnp.concatenate(alm_b_list)

    raise ValueError(f"Spin {spin} not supported in jax_sht")

def jax_synthesis_2d(
    alm: jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray], 
    ny: int, 
    nx: int, 
    lmax: int, 
    mmax: int | None = None, 
    geometry: str = "CC",
    spin: int = 0
) -> jnp.ndarray:
    """JAX-native SHT synthesis (Inverse) matching ducc0."""
    if mmax is None:
        mmax = min(lmax, (nx - 1) // 2)

    if spin == 0:
        plm, _, _, _ = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=0)
        f_m_cols = []
        curr = 0
        for m in range(mmax + 1):
            n_l = lmax - m + 1
            alm_m = alm[curr : curr + n_l]
            f_m_cols.append(jnp.matmul(alm_m, plm[curr : curr + n_l]))
            curr += n_l
        
        f_m_active = jnp.stack(f_m_cols, axis=1)
        f_m = jnp.pad(f_m_active, ((0, 0), (0, nx // 2 + 1 - (mmax + 1))))
        return jnp.fft.irfft(f_m, n=nx, axis=-1) * nx
    
    elif spin == 1:
        alm_e, alm_b = alm
        plm_e, _, _, plm_b = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=1)
        v_theta_m_cols = []
        v_phi_m_cols = []
        curr = 0
        for m in range(mmax + 1):
            l_start = max(m, 1)
            if l_start > lmax: continue
            n_l = lmax - l_start + 1
            e_m = alm_e[curr:curr+n_l]
            b_m = alm_b[curr:curr+n_l]
            
            vt_m = jnp.matmul(e_m, plm_e[curr:curr+n_l]) + jnp.matmul(b_m, plm_b[curr:curr+n_l])
            vp_m = 1j * (jnp.matmul(e_m, plm_b[curr:curr+n_l]) - jnp.matmul(b_m, plm_e[curr:curr+n_l]))
            
            v_theta_m_cols.append(vt_m)
            v_phi_m_cols.append(vp_m)
            curr += n_l
            
        vt_m = jnp.pad(jnp.stack(v_theta_m_cols, axis=1), ((0, 0), (0, nx // 2 + 1 - (mmax + 1))))
        vp_m = jnp.pad(jnp.stack(v_phi_m_cols, axis=1), ((0, 0), (0, nx // 2 + 1 - (mmax + 1))))
        
        return jnp.stack([
            jnp.fft.irfft(vt_m, n=nx, axis=-1) * nx,
            jnp.fft.irfft(vp_m, n=nx, axis=-1) * nx
        ])

    raise ValueError(f"Spin {spin} not supported in jax_sht")
