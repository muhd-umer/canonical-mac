"""Lagrange dual function evaluation for MIMO MAC."""

import numpy as np

from minptone_mimo import minptone_mimo


def lagrange_dual_mimo(
    H, theta, w, bu_min, Lx, idx_start, idx_end, R_init_all=None, state_init_all=None
):
    """Compute Lagrange dual function for MIMO MAC.

    Evaluates the dual objective by optimizing each tone independently via
    minptone_mimo, returning both the per-user rates in nats and the stacked
    covariance matrices. Supports warm-starting with initial covariances and
    L-BFGS states.

    Args:
        H: Channel matrices, shape (Ly, Ltot, N); complex or real
        theta: Dual variables, shape (U,)
        w: Energy weights, shape (U,)
        bu_min: Target rates per user, shape (U,)
        Lx: Number of antennas per user, shape (U,)
        idx_start: Starting antenna indices, shape (U,)
        idx_end: Ending antenna indices, shape (U,)
        R_init_all: Optional warm-start covariances, shape (Ltot, Ltot, N)
        state_init_all: Optional list of N L-BFGS states for warm-starting

    Returns:
        f: Dual objective value (scalar)
        bun: Per-tone rates in nats, shape (U, N)
        Rxx_all: Transmit covariances per tone, shape (Ltot, Ltot, N)
        state_all: List of N L-BFGS states for warm-starting next iteration
    """
    Ly, Ltot, N = H.shape
    U = len(Lx)

    f = 0
    bun = np.zeros((U, N))
    rxx_dtype = np.result_type(H.dtype, np.complex128)
    Rxx_all = np.zeros((Ltot, Ltot, N), dtype=rxx_dtype)

    if state_init_all is None:
        state_init_all = []

    states_cell = []

    # optimize each tone
    for n in range(N):
        # extract warm-start covariance for this tone
        if R_init_all is None:
            R_init_n = None
        else:
            R_init_n = R_init_all[:, :, n]

        # extract warm-start state for this tone
        if len(state_init_all) <= n:
            state_init_n = None
        else:
            state_init_n = state_init_all[n]

        # optimize this tone
        temp_f, b_n, Rxx_n, state_n = minptone_mimo(
            H[:, :, n], theta, w, Lx, idx_start, idx_end, R_init_n, state_init_n
        )

        f += temp_f
        bun[:, n] = b_n
        Rxx_all[:, :, n] = Rxx_n
        states_cell.append(state_n)

    # subtract target rate term
    f = f - np.dot(theta, bu_min)

    return f, bun, Rxx_all, states_cell
