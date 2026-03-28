from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, cast

import ducc0
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    pass


@functools.lru_cache(maxsize=16)
def _get_sht_matrices(
    ny: int,
    nx: int,
    lmax: int,
    mmax: int | None = None,
    geometry: str = "CC",
    spin: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """
    Precomputes Legendre matrices and quadrature weights for JAX SHT.

    Using ducc0 as a reference for bit-wise parity.

    Layout: For each m, l goes from m to lmax.
    Matches ducc0 storage convention.
    """
    if mmax is None:
        mmax = min(lmax, (nx - 1) // 2)

    weights = ducc0.sht.get_gridweights(geometry=geometry, ntheta=ny)
    theta = np.linspace(0, np.pi, ny)

    plm_list = []
    for m in range(mmax + 1):
        # Even for spin > 0, we store modes starting from m to lmax
        # to match ducc0's packed storage layout.
        n_l_m = lmax - m + 1
        m_mat = np.zeros((n_l_m, ny), dtype=np.float64)

        for i_l_rel in range(n_l_m):
            l_val = m + i_l_rel
            if l_val < spin:
                continue

            n_comp = 1 if spin == 0 else 2
            alm_m = np.zeros((n_comp, lmax + 1), dtype=np.complex128)
            alm_m[0, l_val] = 1.0
            leg = ducc0.sht.alm2leg(
                alm=alm_m,
                spin=spin,
                lmax=lmax,
                theta=theta,
                mval=np.array([m]),
                mstart=np.array([0]),
            )
            # leg shape is (n_spin, ntheta, n_m)
            m_mat[i_l_rel, :] = leg[0, :, 0].real
        plm_list.append(m_mat)

    plm = np.concatenate(plm_list, axis=0)

    if spin == 1:
        plm_b_list = []
        for m in range(mmax + 1):
            n_l_m = lmax - m + 1
            m_mat_b = np.zeros((n_l_m, ny), dtype=np.float64)
            for i_l_rel in range(n_l_m):
                l_val = m + i_l_rel
                if l_val < spin:
                    continue
                alm_m = np.zeros((2, lmax + 1), dtype=np.complex128)
                alm_m[0, l_val] = 1.0
                leg = ducc0.sht.alm2leg(
                    alm=alm_m,
                    spin=spin,
                    lmax=lmax,
                    theta=theta,
                    mval=np.array([m]),
                    mstart=np.array([0]),
                )
                m_mat_b[i_l_rel, :] = leg[1, :, 0].real
            plm_b_list.append(m_mat_b)
        plm_b = np.concatenate(plm_b_list, axis=0)
        return (
            jnp.asarray(plm),
            jnp.asarray(weights),
            jnp.asarray(mmax),
            jnp.asarray(plm_b),
        )

    return jnp.asarray(plm), jnp.asarray(weights), jnp.asarray(mmax), None


def _validate_resolution(ny: int, lmax: int, geometry: str) -> None:
    """Validate lmax against grid resolution to prevent aliasing."""
    if geometry == "CC" and lmax > ny - 2:
        raise ValueError(
            f"Too few latitude rings ({ny}) for analysis up to requested lmax ({lmax}) "
            "with CC geometry. Max allowed lmax is ny - 2."
        )

    if lmax > ny // 2:
        warnings.warn(
            f"lmax ({lmax}) is more than half the latitude resolution (ny={ny}). "
            "Results may be subject to significant aliasing at lower resolutions "
            "with the JAX backend.",
            UserWarning,
            stacklevel=3,
        )


def jax_analysis_2d(
    frame: jnp.ndarray,
    lmax: int,
    mmax: int | None = None,
    geometry: str = "CC",
    spin: int = 0,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-native SHT analysis (Forward) matching ducc0."""
    ny, nx = frame.shape[-2], frame.shape[-1]
    _validate_resolution(ny, lmax, geometry)

    if mmax is None:
        mmax = min(lmax, (nx - 1) // 2)

    if spin == 0:
        res = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=0)
        plm, weights, _mmax_cached, _ = res
        f_m = jnp.fft.rfft(frame, axis=-1)[:, : mmax + 1]
        weighted_f_m = f_m * (weights[:, jnp.newaxis] / nx)

        alm_list = []
        curr = 0
        for m in range(mmax + 1):
            n_l = lmax - m + 1
            alm_m = jnp.matmul(plm[curr : curr + n_l, :], weighted_f_m[:, m])
            alm_list.append(alm_m)
            curr += n_l
        return jnp.concatenate(alm_list)

    if spin == 1:
        res_spin1 = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=1)
        plm_e, weights, _mmax_cached, plm_b = res_spin1
        # plm_b is guaranteed to be non-None for spin=1
        plm_b = cast(jnp.ndarray, plm_b)

        v_theta_m = jnp.fft.rfft(frame[0], axis=-1)[:, : mmax + 1]
        v_phi_m = jnp.fft.rfft(frame[1], axis=-1)[:, : mmax + 1]

        w_v_theta_m = v_theta_m * (weights[:, jnp.newaxis] / nx)
        w_v_phi_m = v_phi_m * (weights[:, jnp.newaxis] / nx)

        alm_e_list = []
        alm_b_list = []
        curr = 0
        for m in range(mmax + 1):
            n_l = lmax - m + 1
            e_m = jnp.matmul(
                plm_e[curr : curr + n_l], w_v_theta_m[:, m]
            ) + jnp.matmul(plm_b[curr : curr + n_l], w_v_phi_m[:, m])
            b_m = jnp.matmul(
                plm_e[curr : curr + n_l], w_v_phi_m[:, m]
            ) - jnp.matmul(plm_b[curr : curr + n_l], w_v_theta_m[:, m])
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
    spin: int = 0,
) -> jnp.ndarray:
    """JAX-native SHT synthesis (Inverse) matching ducc0."""
    _validate_resolution(ny, lmax, geometry)

    if mmax is None:
        mmax = min(lmax, (nx - 1) // 2)

    if spin == 0:
        # alm must be jnp.ndarray for spin=0
        alm_arr = cast(jnp.ndarray, alm)
        res = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=0)
        plm, _, _, _ = res
        f_m_cols = []
        curr = 0
        for m in range(mmax + 1):
            n_l = lmax - m + 1
            alm_m = alm_arr[curr : curr + n_l]
            f_m_cols.append(jnp.matmul(alm_m, plm[curr : curr + n_l]))
            curr += n_l

        f_m_active = jnp.stack(f_m_cols, axis=1)
        f_m = jnp.pad(f_m_active, ((0, 0), (0, nx // 2 + 1 - (mmax + 1))))
        return jnp.fft.irfft(f_m, n=nx, axis=-1) * nx

    if spin == 1:
        # alm must be tuple[jnp.ndarray, jnp.ndarray] for spin=1
        alm_e, alm_b = cast(tuple[jnp.ndarray, jnp.ndarray], alm)
        res_spin1 = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=1)
        plm_e, _, _, plm_b = res_spin1
        plm_b = cast(jnp.ndarray, plm_b)

        v_theta_m_cols = []
        v_phi_m_cols = []
        curr = 0
        for m in range(mmax + 1):
            n_l = lmax - m + 1
            e_m = alm_e[curr : curr + n_l]
            b_m = alm_b[curr : curr + n_l]

            vt_m = jnp.matmul(e_m, plm_e[curr : curr + n_l]) - jnp.matmul(
                b_m, plm_b[curr : curr + n_l]
            )
            vp_m = jnp.matmul(e_m, plm_b[curr : curr + n_l]) + jnp.matmul(
                b_m, plm_e[curr : curr + n_l]
            )

            v_theta_m_cols.append(vt_m)
            v_phi_m_cols.append(vp_m)
            curr += n_l

        vt_m = jnp.pad(
            jnp.stack(v_theta_m_cols, axis=1), ((0, 0), (0, nx // 2 + 1 - (mmax + 1)))
        )
        vp_m = jnp.pad(
            jnp.stack(v_phi_m_cols, axis=1), ((0, 0), (0, nx // 2 + 1 - (mmax + 1)))
        )

        return jnp.stack(
            [
                jnp.fft.irfft(vt_m, n=nx, axis=-1) * nx,
                jnp.fft.irfft(vp_m, n=nx, axis=-1) * nx,
            ]
        )

    raise ValueError(f"Spin {spin} not supported in jax_sht")
