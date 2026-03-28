from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, cast

import ducc0
import jax
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

    Layout: For each m, l goes from 0 to lmax (padded for vmap support).
    Matches ducc0 storage convention up to padding.
    """
    if mmax is None:
        mmax = min(lmax, (nx - 1) // 2)

    weights = ducc0.sht.get_gridweights(geometry=geometry, ntheta=ny)
    theta = np.linspace(0, np.pi, ny)

    plm_list = []
    for m in range(mmax + 1):
        # We store modes from l=0 to lmax for uniform shape (vmap support).
        # Modes l < m or l < spin will remain zero.
        m_mat = np.zeros((lmax + 1, ny), dtype=np.float64)

        for l_val in range(m, lmax + 1):
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
            m_mat[l_val, :] = leg[0, :, 0].real
        plm_list.append(m_mat)

    plm = np.stack(plm_list, axis=0)

    if spin == 1:
        plm_b_list = []
        for m in range(mmax + 1):
            m_mat_b = np.zeros((lmax + 1, ny), dtype=np.float64)
            for l_val in range(m, lmax + 1):
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
                m_mat_b[l_val, :] = leg[1, :, 0].real
            plm_b_list.append(m_mat_b)
        plm_b = np.stack(plm_b_list, axis=0)
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
    """JAX-native SHT analysis (Forward) matching ducc0. Handles (..., ny, nx)."""
    ny, nx = frame.shape[-2], frame.shape[-1]
    _validate_resolution(ny, lmax, geometry)

    if mmax is None:
        mmax = min(lmax, (nx - 1) // 2)

    if spin == 0:
        res = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=0)
        plm, weights, _mmax_cached, _ = res
        f_m = jnp.fft.rfft(frame, axis=-1)[..., : mmax + 1]

        # weights shape (ny,), f_m shape (..., ny, m)
        weighted_f_m = f_m * (weights.reshape((ny, 1)) / nx)

        # plm: (m, l, n), weighted_f_m: (... n, m)
        # Result: (m, ..., l)
        alm = jnp.einsum("mln,...nm->m...l", plm, weighted_f_m)
        return alm

    if spin == 1:
        res_spin1 = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=1)
        plm_e, weights, _mmax_cached, plm_b = res_spin1
        plm_b = cast(jnp.ndarray, plm_b)

        # frame is (2, ..., ny, nx)
        v_theta_m = jnp.fft.rfft(frame[0], axis=-1)[..., : mmax + 1]
        v_phi_m = jnp.fft.rfft(frame[1], axis=-1)[..., : mmax + 1]

        w_v_theta_m = v_theta_m * (weights.reshape((ny, 1)) / nx)
        w_v_phi_m = v_phi_m * (weights.reshape((ny, 1)) / nx)

        alm_e = jnp.einsum("mln,...nm->m...l", plm_e, w_v_theta_m) + jnp.einsum(
            "mln,...nm->m...l", plm_b, w_v_phi_m
        )
        alm_b = jnp.einsum("mln,...nm->m...l", plm_e, w_v_phi_m) - jnp.einsum(
            "mln,...nm->m...l", plm_b, w_v_theta_m
        )

        return alm_e, alm_b

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
    """JAX-native SHT synthesis (Inverse) matching ducc0. Handles (m, ..., l)."""
    _validate_resolution(ny, lmax, geometry)

    if mmax is None:
        mmax = min(lmax, (nx - 1) // 2)

    if spin == 0:
        alm_arr = cast(jnp.ndarray, alm)  # (m, ..., l)
        res = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=0)
        plm, _, _, _ = res  # (m, l, n)

        # plm: (m, l, n), alm_arr: (m, ..., l)
        # Result: (... n, m)
        f_m_active = jnp.einsum("mln,m...l->...nm", plm, alm_arr)

        f_m = jnp.pad(
            f_m_active,
            ((0, 0),) * (f_m_active.ndim - 1) + ((0, nx // 2 + 1 - (mmax + 1)),),
        )
        return jnp.fft.irfft(f_m, n=nx, axis=-1) * nx

    if spin == 1:
        alm_e, alm_b = cast(tuple[jnp.ndarray, jnp.ndarray], alm)
        res_spin1 = _get_sht_matrices(ny, nx, lmax, mmax, geometry, spin=1)
        plm_e, _, _, plm_b = res_spin1
        plm_b = cast(jnp.ndarray, plm_b)

        vt_m_active = jnp.einsum("mln,m...l->...nm", plm_e, alm_e) - jnp.einsum(
            "mln,m...l->...nm", plm_b, alm_b
        )
        vp_m_active = jnp.einsum("mln,m...l->...nm", plm_e, alm_b) + jnp.einsum(
            "mln,m...l->...nm", plm_b, alm_e
        )

        vt_m = jnp.pad(
            vt_m_active,
            ((0, 0),) * (vt_m_active.ndim - 1) + ((0, nx // 2 + 1 - (mmax + 1)),),
        )
        vp_m = jnp.pad(
            vp_m_active,
            ((0, 0),) * (vp_m_active.ndim - 1) + ((0, nx // 2 + 1 - (mmax + 1)),),
        )

        return jnp.stack(
            [
                jnp.fft.irfft(vt_m, n=nx, axis=-1) * nx,
                jnp.fft.irfft(vp_m, n=nx, axis=-1) * nx,
            ],
            axis=0,
        )

    raise ValueError(f"Spin {spin} not supported in jax_sht")


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _jax_filter_core(
    data: jnp.ndarray,
    plm: jnp.ndarray,
    weights: jnp.ndarray,
    lmin: int,
    lmax: int,
    mmax: int,
    nx: int,
) -> jnp.ndarray:
    """JIT-compiled core filtering logic."""
    ny = data.shape[-2]

    # Forward transform
    f_m = jnp.fft.rfft(data, axis=-1)[..., : mmax + 1]
    weighted_f_m = f_m * (weights.reshape((ny, 1)) / nx)
    alm = jnp.einsum("mln,...nm->m...l", plm, weighted_f_m)

    # Apply Bandpass Mask
    l_arr = jnp.arange(lmax + 1)
    mask = l_arr >= lmin
    alm_f = alm * mask

    # Inverse transform
    f_m_active = jnp.einsum("mln,m...l->...nm", plm, alm_f)
    f_m_pad = jnp.pad(
        f_m_active, ((0, 0),) * (f_m_active.ndim - 1) + ((0, nx // 2 + 1 - (mmax + 1)),)
    )
    return jnp.fft.irfft(f_m_pad, n=nx, axis=-1) * nx


def jax_filter(
    data: np.ndarray,
    lmin: int,
    lmax: int,
) -> np.ndarray:
    """Filters data using JAX-native SHT. Handles (..., ny, nx)."""
    import jax

    jax.config.update("jax_enable_x64", True)

    ny, nx = data.shape[-2], data.shape[-1]
    mmax = min(lmax, nx // 2 - 1)

    plm, weights, _, _ = _get_sht_matrices(ny, nx, lmax, mmax, geometry="CC", spin=0)
    data_jax = jax.device_put(data)

    out = _jax_filter_core(data_jax, plm, weights, lmin, lmax, mmax, nx)
    return np.asarray(out)
