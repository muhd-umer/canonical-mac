"""Per-tone dual maximization for MIMO MAC using L-BFGS."""

import numpy as np
from scipy.linalg import cholesky, eigh


def minptone_mimo(H, theta, w, Lx, idx_start, idx_end, R_init=None, state_in=None):
    """Dual maximization on a single tone for MIMO MAC.

    Solves the per-tone Lagrangian maximization that appears inside the ellipsoid
    method for the MIMO minPMAC problem. This implementation avoids CVX and works
    directly with the covariance matrices for each user by applying a limited
    memory quasi-Newton ascent on the dual objective.

    Args:
        H: Channel matrix for this tone, shape (Ly, Ltot); complex or real
        theta: Dual variables (not yet sorted), shape (U,)
        w: User energy weights, shape (U,)
        Lx: Number of antennas per user, shape (U,)
        idx_start: Starting indices for each user's antennas, shape (U,)
        idx_end: Ending indices for each user's antennas, shape (U,)
        R_init: Optional warm-start covariance matrix, shape (Ltot, Ltot)
        state_in: Optional dict with L-BFGS history for warm-start

    Returns:
        f: Scalar value theta' * b - sum_u w_u * trace(R_u)
        b: Rate vector in nats/channel use, shape (U,)
        Rxx: Transmit covariance matrix for this tone, shape (Ltot, Ltot)
        state_out: Dict retaining L-BFGS warm-start data
    """
    Ly, _ = H.shape
    U = len(Lx)

    # sort users according to theta (descending) to match polymatroid vertex
    order = np.argsort(-theta)
    stheta = theta[order]
    sw = w[order]

    # build per-user channel blocks for the sorted order
    H_blocks = []
    block_sizes = Lx[order]

    for u in range(U):
        orig = order[u]
        cols = slice(idx_start[orig], idx_end[orig] + 1)
        H_blocks.append(H[:, cols])

    # coefficients alpha_k = 0.5 * (theta_k - theta_{k+1}), theta_{U+1}=0
    alpha = 0.5 * (stheta - np.append(stheta[1:], 0))

    # validate and prepare state
    state_valid = True
    if state_in is None:
        state_valid = False
    elif "order" not in state_in or not np.array_equal(state_in["order"], order):
        state_in = None
        state_valid = False
    elif "block_sizes" not in state_in or not np.array_equal(
        state_in["block_sizes"], block_sizes
    ):
        state_in = None
        state_valid = False

    # prepare initial covariance blocks
    if not state_valid:
        if R_init is None:
            R_init_blocks = None
        else:
            R_init_blocks = []
            for u in range(U):
                orig = order[u]
                cols = slice(idx_start[orig], idx_end[orig] + 1)
                Ru = R_init[cols, cols]
                R_init_blocks.append(0.5 * (Ru + Ru.conj().T))
    else:
        R_init_blocks = None

    # run l-bfgs optimization
    R_blocks, F_opt, state_internal = maximize_dual_lbfgs(
        H_blocks, alpha, sw, block_sizes, R_init_blocks, state_in
    )

    # compute achieved rates (nats) for sorted users
    b_sorted = np.zeros(U)
    S_prev = np.eye(Ly, dtype=H.dtype)

    for u in range(U):
        S_curr = S_prev + H_blocks[u] @ R_blocks[u] @ H_blocks[u].conj().T
        S_curr = 0.5 * (S_curr + S_curr.conj().T)
        b_sorted[u] = 0.5 * np.real(logdet_spd(S_curr) - logdet_spd(S_prev))
        S_prev = S_curr

    # map rates back to original ordering
    b = np.zeros(U)
    b[order] = b_sorted

    # assemble full covariance matrix in original antenna order
    Ltot = np.sum(Lx)
    Rxx_dtype = np.result_type(H.dtype, np.complex128)
    Rxx = np.zeros((Ltot, Ltot), dtype=Rxx_dtype)

    for u in range(U):
        orig = order[u]
        cols = slice(idx_start[orig], idx_end[orig] + 1)
        Ru = R_blocks[u]
        Rxx[cols, cols] = 0.5 * (Ru + Ru.conj().T)

    # objective value
    energy = 0
    for u in range(U):
        orig = order[u]
        energy += w[orig] * np.real(np.trace(R_blocks[u]))

    f = np.real(np.dot(theta, b) - energy)

    # finalize state for warm-starting
    state_out = finalize_state(state_internal, order, block_sizes, R_blocks)

    return f, b, Rxx, state_out


def compute_objective_and_gradient(H_blocks, B_blocks, alpha, w):
    """Evaluate dual objective and gradient for given B blocks.

    Args:
        H_blocks: List of U channel matrices
        B_blocks: List of U Cholesky factors (R = B*B')
        alpha: Coefficients from theta sorting, shape (U,)
        w: Energy weights, shape (U,)

    Returns:
        F: Dual objective value
        grad_phi_blocks: List of U gradient matrices
        R_blocks: List of U covariance matrices (R = B*B')
    """
    U = len(H_blocks)
    Ly = H_blocks[0].shape[0]
    dtype = H_blocks[0].dtype
    eye_Ly = np.eye(Ly, dtype=dtype)
    S = eye_Ly.copy()
    logdets = np.zeros(U)
    invS_cache = []
    energy_term = 0

    for u in range(U):
        Hu = H_blocks[u]
        Bu = B_blocks[u]
        HB = Hu @ Bu
        S = S + HB @ HB.conj().T
        S = 0.5 * (S + S.conj().T)

        try:
            L = cholesky(S, lower=True)
        except np.linalg.LinAlgError:
            # add regularization if not positive definite
            S = S + 1e-9 * eye_Ly
            L = cholesky(S, lower=True)

        logdets[u] = np.real(2 * np.sum(np.log(np.diag(L))))
        Linv = np.linalg.solve(L, eye_Ly)
        invS = Linv.conj().T @ Linv
        invS_cache.append(0.5 * (invS + invS.conj().T))
        energy_term += w[u] * np.real(np.sum(np.conj(Bu.flatten()) * Bu.flatten()))

    F = np.sum(alpha * logdets) - energy_term

    # compute gradients via backward pass
    grad_phi_blocks = []
    R_blocks = []
    tail = np.zeros((Ly, Ly), dtype=dtype)

    for u in range(U - 1, -1, -1):
        tail = tail + alpha[u] * invS_cache[u]
        Hu = H_blocks[u]
        Bu = B_blocks[u]
        Lu = Bu.shape[0]
        grad_R = Hu.conj().T @ (tail @ Hu) - w[u] * np.eye(Lu, dtype=dtype)
        grad_R = 0.5 * (grad_R + grad_R.conj().T)
        grad_phi_blocks.insert(0, -2 * grad_R @ Bu)
        Ru = Bu @ Bu.conj().T
        R_blocks.insert(0, 0.5 * (Ru + Ru.conj().T))

    return F, grad_phi_blocks, R_blocks


def maximize_dual_lbfgs(
    H_blocks, alpha, w, block_sizes, R_init_blocks=None, state_in=None
):
    """L-BFGS maximization of dual objective.

    Args:
        H_blocks: List of U channel matrices
        alpha: Coefficients from theta sorting, shape (U,)
        w: Energy weights, shape (U,)
        block_sizes: Antenna counts for sorted users, shape (U,)
        R_init_blocks: Optional list of U initial covariance matrices
        state_in: Optional dict with L-BFGS history

    Returns:
        R_blocks: List of U optimal covariance matrices
        F_opt: Optimal dual objective value
        state_out: Dict with L-BFGS state for warm-starting
    """
    # l-bfgs parameters
    max_it = 150
    tol_grad = 1e-5
    m_hist = 8
    step_init = 1.0
    step_min = 1e-10
    beta = 0.3
    armijo_c = 3.33e-04
    max_linesearch = 18
    U = len(H_blocks)
    dtype = H_blocks[0].dtype

    # prepare state
    state_valid, prepared_state = prepare_lbfgs_state(state_in, block_sizes, U)

    if state_valid:
        B_blocks = prepared_state["B_blocks"]
        S_hist = prepared_state["S_hist"]
        Y_hist = prepared_state["Y_hist"]
    else:
        # initialize B_blocks
        B_blocks = []
        for u in range(U):
            if R_init_blocks is not None:
                Ru = R_init_blocks[u]
                Ru = 0.5 * (Ru + Ru.conj().T)
                B_blocks.append(initialize_factor(Ru))
            else:
                B_blocks.append(np.sqrt(1e-6) * np.eye(block_sizes[u], dtype=dtype))

        S_hist = []
        Y_hist = []

    # pack into vector
    x = pack_complex_blocks(B_blocks, block_sizes)

    if state_valid:
        S_hist, Y_hist = sanitize_history(S_hist, Y_hist, len(x), m_hist)

    # initial evaluation
    phi, F_opt, grad, B_blocks, R_blocks = evaluate_lbfgs_state(
        x, H_blocks, alpha, w, block_sizes
    )
    grad_norm = np.linalg.norm(grad)

    # main l-bfgs loop
    for iter in range(max_it):
        if grad_norm < tol_grad:
            break

        # compute search direction
        direction = lbfgs_direction(grad, S_hist, Y_hist)
        dir_deriv = dot_real(grad, direction)

        # reset if not descent direction
        if dir_deriv >= 0:
            direction = -grad
            dir_deriv = dot_real(grad, direction)
            S_hist = []
            Y_hist = []

        # backtracking line search
        step = step_init
        accepted = False
        x_candidate = None
        phi_candidate = None
        grad_candidate = None
        B_candidate = None
        R_candidate = None

        for ls in range(max_linesearch):
            x_trial = x + step * direction
            phi_trial, F_trial, grad_trial, B_trial, R_trial = evaluate_lbfgs_state(
                x_trial, H_blocks, alpha, w, block_sizes
            )

            # armijo condition
            if phi_trial <= phi + armijo_c * step * dir_deriv:
                accepted = True
                x_candidate = x_trial
                phi_candidate = phi_trial
                grad_candidate = grad_trial
                B_candidate = B_trial
                R_candidate = R_trial
                F_opt = F_trial
                break

            step = step * beta
            if step < step_min:
                break

        # if line search failed, try gradient descent
        if not accepted:
            direction = -grad
            dir_deriv = dot_real(grad, direction)
            step = step_init

            for ls in range(max_linesearch):
                x_trial = x + step * direction
                phi_trial, F_trial, grad_trial, B_trial, R_trial = evaluate_lbfgs_state(
                    x_trial, H_blocks, alpha, w, block_sizes
                )

                if phi_trial <= phi + armijo_c * step * dir_deriv:
                    accepted = True
                    x_candidate = x_trial
                    phi_candidate = phi_trial
                    grad_candidate = grad_trial
                    B_candidate = B_trial
                    R_candidate = R_trial
                    F_opt = F_trial
                    break

                step = step * beta
                if step < step_min:
                    break

            if not accepted:
                break

            # reset history after gradient descent
            S_hist = []
            Y_hist = []

        # update l-bfgs history
        s = x_candidate - x
        y = grad_candidate - grad

        if dot_real(s, y) > 1e-12:
            if len(S_hist) == m_hist:
                S_hist = S_hist[1:]
                Y_hist = Y_hist[1:]

            S_hist.append(s)
            Y_hist.append(y)
        else:
            S_hist = []
            Y_hist = []

        # update state
        x = x_candidate
        phi = phi_candidate
        grad = grad_candidate
        B_blocks = B_candidate
        R_blocks = R_candidate
        grad_norm = np.linalg.norm(grad)

    # prepare output state
    state_out = {
        "B_blocks": B_blocks,
        "S_hist": S_hist,
        "Y_hist": Y_hist,
        "x": x,
        "grad": grad,
        "phi": phi,
        "F": F_opt,
    }

    return R_blocks, F_opt, state_out


def evaluate_lbfgs_state(x, H_blocks, alpha, w, block_sizes):
    """Evaluate objective and gradient at point x.

    Args:
        x: Vectorized B_blocks (real vector)
        H_blocks: List of U channel matrices
        alpha: Coefficients, shape (U,)
        w: Weights, shape (U,)
        block_sizes: Antenna counts, shape (U,)

    Returns:
        phi: -F (minimization objective)
        F_val: F (maximization objective)
        grad_vec: Gradient vector
        B_blocks: List of U Cholesky factors
        R_blocks: List of U covariance matrices
    """
    B_blocks = unpack_complex_blocks(x, block_sizes)
    F_val, grad_blocks, R_blocks = compute_objective_and_gradient(
        H_blocks, B_blocks, alpha, w
    )
    grad_vec = pack_complex_blocks(grad_blocks, block_sizes)
    phi = -F_val
    return phi, F_val, grad_vec, B_blocks, R_blocks


def lbfgs_direction(grad, S_hist, Y_hist):
    """Compute L-BFGS search direction via two-loop recursion.

    Args:
        grad: Gradient vector
        S_hist: List of previous step vectors
        Y_hist: List of previous gradient differences

    Returns:
        direction: Search direction
    """
    if len(S_hist) == 0:
        return -grad

    k = len(S_hist)
    alpha_vals = np.zeros(k)
    rho = np.zeros(k)
    q = grad.copy()

    # first loop (backward)
    for i in range(k - 1, -1, -1):
        s = S_hist[i]
        y = Y_hist[i]
        denom = dot_real(y, s)

        if denom <= 1e-12:
            rho[i] = 0
            alpha_vals[i] = 0
            continue

        rho[i] = 1 / denom
        alpha_vals[i] = rho[i] * dot_real(s, q)
        q = q - alpha_vals[i] * y

    # compute initial Hessian approximation
    sk = S_hist[-1]
    yk = Y_hist[-1]
    denom = dot_real(yk, yk)

    if denom <= 1e-12:
        H0 = 1
    else:
        H0 = dot_real(sk, yk) / denom

    r = H0 * q

    # second loop (forward)
    for i in range(k):
        y = Y_hist[i]
        s = S_hist[i]

        if rho[i] == 0:
            beta = 0
        else:
            beta = rho[i] * dot_real(y, r)

        r = r + s * (alpha_vals[i] - beta)

    return -r


def pack_complex_blocks(blocks, block_sizes):
    """Pack list of complex matrices into real vector.

    Args:
        blocks: List of U complex matrices
        block_sizes: Antenna counts, shape (U,)

    Returns:
        vec: Real vector of size 2*sum(block_sizes^2)
    """
    U = len(blocks)
    total_len = 2 * np.sum(block_sizes**2)
    vec = np.zeros(total_len)
    offset = 0

    for u in range(U):
        block = blocks[u]
        flat = block.flatten()
        len_flat = len(flat)
        vec[offset : offset + len_flat] = np.real(flat)
        vec[offset + len_flat : offset + 2 * len_flat] = np.imag(flat)
        offset += 2 * len_flat

    return vec


def unpack_complex_blocks(vec, block_sizes):
    """Unpack real vector into list of complex matrices.

    Args:
        vec: Real vector
        block_sizes: Antenna counts, shape (U,)

    Returns:
        blocks: List of U complex matrices
    """
    U = len(block_sizes)
    blocks = []
    offset = 0

    for u in range(U):
        len_flat = block_sizes[u] ** 2
        real_part = vec[offset : offset + len_flat]
        imag_part = vec[offset + len_flat : offset + 2 * len_flat]
        block = (real_part + 1j * imag_part).reshape(block_sizes[u], block_sizes[u])
        blocks.append(block)
        offset += 2 * len_flat

    return blocks


def dot_real(a, b):
    """Real part of dot product."""
    return np.real(np.sum(a * b))


def prepare_lbfgs_state(state_in, block_sizes, U):
    """Validate and prepare L-BFGS state for warm-starting.

    Args:
        state_in: Input state dict (or None)
        block_sizes: Antenna counts, shape (U,)
        U: Number of users

    Returns:
        is_valid: True if state is valid
        state_out: Prepared state dict
    """
    is_valid = False
    state_out = {}

    if state_in is None or not isinstance(state_in, dict):
        return is_valid, state_out

    if "B_blocks" not in state_in:
        return is_valid, state_out

    if "block_sizes" in state_in:
        if len(state_in["block_sizes"]) != len(block_sizes):
            return is_valid, state_out
        if not np.array_equal(state_in["block_sizes"], block_sizes):
            return is_valid, state_out

    if len(state_in["B_blocks"]) != U:
        return is_valid, state_out

    for u in range(U):
        Bu = state_in["B_blocks"][u]
        if Bu is None or Bu.shape != (block_sizes[u], block_sizes[u]):
            return is_valid, state_out

    state_out = state_in.copy()

    if "S_hist" not in state_out or not isinstance(state_out["S_hist"], list):
        state_out["S_hist"] = []

    if "Y_hist" not in state_out or not isinstance(state_out["Y_hist"], list):
        state_out["Y_hist"] = []

    is_valid = True
    return is_valid, state_out


def sanitize_history(S_hist_in, Y_hist_in, dim, m_hist):
    """Clean up L-BFGS history to ensure correct dimensions.

    Args:
        S_hist_in: Input list of step vectors
        Y_hist_in: Input list of gradient differences
        dim: Expected dimension
        m_hist: Maximum history length

    Returns:
        S_hist_out: Sanitized step vectors
        Y_hist_out: Sanitized gradient differences
    """
    if len(S_hist_in) == 0 or len(Y_hist_in) == 0:
        return [], []

    k = min(len(S_hist_in), len(Y_hist_in), m_hist)
    start_idx = max(0, len(S_hist_in) - k)
    S_hist_out = []
    Y_hist_out = []

    for idx in range(start_idx, len(S_hist_in)):
        s = S_hist_in[idx]
        y = Y_hist_in[idx]

        if len(s) == dim and len(y) == dim:
            S_hist_out.append(s)
            Y_hist_out.append(y)

    return S_hist_out, Y_hist_out


def finalize_state(state_internal, order, block_sizes, R_blocks):
    """Package L-BFGS state for output.

    Args:
        state_internal: Internal state dict
        order: User ordering, shape (U,)
        block_sizes: Antenna counts, shape (U,)
        R_blocks: List of U covariance matrices

    Returns:
        state_out: Finalized state dict
    """
    if state_internal is None or not isinstance(state_internal, dict):
        state_internal = {}

    state_out = state_internal.copy()
    state_out["order"] = order
    state_out["block_sizes"] = block_sizes
    state_out["R_blocks"] = R_blocks

    return state_out


def initialize_factor(R):
    """Initialize Cholesky factor from covariance matrix.

    Args:
        R: Covariance matrix (Hermitian positive semi-definite)

    Returns:
        B: Cholesky factor such that R = B*B'
    """
    if R is None or R.size == 0:
        return np.zeros_like(R)

    R = 0.5 * (R + R.conj().T)

    try:
        L = cholesky(R, lower=True)
        return L
    except np.linalg.LinAlgError:
        # fallback using eigendecomposition for semi-definite cases
        d, V = eigh(R)
        d = np.real(d)
        d[d < 0] = 0
        return V @ np.diag(np.sqrt(d))


def logdet_spd(M):
    """Log-determinant of symmetric/Hermitian positive definite matrix.

    Args:
        M: Symmetric or Hermitian positive definite matrix

    Returns:
        val: log(det(M))
    """
    M = 0.5 * (M + M.conj().T)

    try:
        L = cholesky(M, lower=True)
    except np.linalg.LinAlgError:
        # add regularization if not positive definite
        M = M + 1e-9 * np.eye(M.shape[0], dtype=M.dtype)
        L = cholesky(M, lower=True)

    return 2 * np.sum(np.log(np.diag(L)))
