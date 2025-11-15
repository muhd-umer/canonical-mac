"""Ellipsoid method initialization for MIMO MAC."""

import numpy as np
from scipy.linalg import sqrtm, svd

from fmwaterfill_gn import fmwaterfill_gn


def start_ellipse_mimo(H, bu, w, cb, Lx, idx_start, idx_end):
    """Initialize ellipsoid method for MIMO MAC.

    Returns the initial ellipsoid shape matrix and center for the minPMAC solver
    by approximating user energies via successive water-filling sweeps.

    Args:
        H: Channel matrices, shape (Ly, Ltot, N); complex or real
        bu: Target rates per user, shape (U,)
        w: Energy weights per user, shape (U,)
        cb: Baseband type (1=complex, 2=real)
        Lx: Number of antennas per user, shape (U,)
        idx_start: Starting antenna index per user, shape (U,)
        idx_end: Ending antenna index per user, shape (U,)

    Returns:
        A: Ellipsoid shape matrix, shape (U, U)
        g: Ellipsoid center (dual variables), shape (U,)
        w: Energy weights (returned as-is), shape (U,)
    """
    Ly, Ltot, N = H.shape
    U = len(Lx)

    tmax = np.zeros(U)

    # for each user, compute water-filling solution with perturbed rate
    for u in range(U):
        en = np.zeros((U, N))
        b = bu.copy()
        b[u] = b[u] + 1  # perturb by 1 nat (as in reference)

        # iterate through users in reverse order
        for u1 in range(U - 1, -1, -1):
            # initialize noise covariance for each tone
            Rnoise = np.zeros((Ly, Ly, N), dtype=H.dtype)

            for index in range(N):
                Rnoise[:, :, index] = np.eye(Ly, dtype=H.dtype)

            # add interference from other users
            for u2 in range(U - 1, -1, -1):
                if u2 != u1:
                    for index in range(N):
                        ant_idx = slice(idx_start[u2], idx_end[u2] + 1)
                        H_u2 = H[:, ant_idx, index]

                        # for siso, en(u2, index) is scalar energy
                        if Lx[u2] == 1:
                            Rnoise[:, :, index] += en[u2, index] * H_u2 @ H_u2.conj().T
                        else:
                            # for mimo, use scalar approximation (energy distributed equally)
                            energy_per_ant = en[u2, index] / Lx[u2]
                            Rnoise[:, :, index] += energy_per_ant * (
                                H_u2 @ H_u2.conj().T
                            )

            # compute effective channel gains per tone
            ant_idx = slice(idx_start[u1], idx_end[u1] + 1)
            g_tones = np.zeros(N)

            for index in range(N):
                H_u1 = H[:, ant_idx, index]
                # compute whitened channel: sqrtm(Rnoise)^{-1} * H_u1
                h_temp = np.linalg.solve(sqrtm(Rnoise[:, :, index]), H_u1)

                if Lx[u1] == 1:
                    # siso case: use squared norm
                    g_tones[index] = np.real(h_temp.conj().T @ h_temp)
                else:
                    # mimo case: use maximum singular value squared
                    sv = svd(h_temp, compute_uv=False)
                    g_tones[index] = sv[0] ** 2

            # water-filling allocation
            b_temp, E_temp = fmwaterfill_gn(g_tones, b[u1] / N, 0, cb)
            en[u1, :] = E_temp

        # compute weighted energy sum
        tmax[u] = np.sum(w * np.sum(en, axis=1))

    # initialize ellipsoid center and shape
    g = tmax / 2
    A = np.eye(U) * np.sum(g**2)

    return A, g, w
