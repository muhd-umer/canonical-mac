"""Minimum power multi-user MIMO MAC solver (non-CVX implementation)."""

import time
import itertools
import numpy as np
from scipy.linalg import cholesky
import cvxpy as cp

try:
    from cvxpy.reductions.solvers.conic_solvers import mosek_conif as mosek_iface
except Exception:  # pragma: no cover - mosek optional during import
    mosek_iface = None

from start_ellipse_mimo import start_ellipse_mimo
from lagrange_dual_mimo import lagrange_dual_mimo


def _patch_mosek_sparse_indices():
    """Force cvxpy's MOSEK interface to pass int32 sparse indices."""
    if mosek_iface is None:
        return

    sparse_module = mosek_iface.sp.sparse
    current_find = sparse_module.find

    if getattr(current_find, "_minpmac_force_int32", False):
        return

    def find_int32(*args, **kwargs):
        rows, cols, vals = current_find(*args, **kwargs)
        if rows.dtype != np.int32:
            rows = rows.astype(np.int32, copy=False)
        if cols.dtype != np.int32:
            cols = cols.astype(np.int32, copy=False)
        return rows, cols, vals

    find_int32._minpmac_force_int32 = True
    sparse_module.find = find_int32


_patch_mosek_sparse_indices()


def minpmac_mimo(H, Lx, bu_min, w, cb):
    """Minimum power multi-user MIMO MAC solver.

    Computes the optimal covariance matrices that satisfy user rate targets with
    minimum weighted energy for both SISO and MIMO MAC configurations.

    Args:
        H: Channel matrix, shape (Ly, sum(Lx), N); complex or real
        Lx: Antennas per user, shape (U,)
        bu_min: Target rates per user in bits/channel use, shape (U,)
        w: Positive energy weights, shape (U,)
        cb: Baseband type (1=complex, 2=real)

    Returns:
        FEAS_FLAG: Feasibility flag (0=infeasible, 1=single order, 2=time-sharing)
        bu_a: Achieved user rates in bits/channel use, shape (U,)
        info: Dict with detailed solution metrics including:
              - frac: Time-sharing fractions for active orders (empty if single order)
              - bun: Per-tone bit allocations for the returned solution
              - Eu_avg: Average energy per user
              - elapsed_time: Runtime in seconds
    """
    start_time = time.time()

    Ly, Ltot, N = H.shape
    U = len(Lx)

    # validate inputs
    if np.sum(Lx) != Ltot:
        raise ValueError(
            f"sum(Lx) = {np.sum(Lx)} must equal total TX antennas = {Ltot}"
        )

    if len(bu_min) != U or len(w) != U:
        raise ValueError(f"bu_min and w must have length U={U}")

    if not np.all(w > 0):
        raise ValueError("Energy weights w must be positive")

    if not np.all(bu_min >= 0):
        raise ValueError("Target rates bu_min must be non-negative")

    if not np.all(Lx >= 1):
        raise ValueError("Each user must have at least 1 antenna")

    # compute antenna index mapping (convert to 0-based indexing)
    idx_end = np.cumsum(Lx) - 1
    idx_start = np.concatenate([[0], idx_end[:-1] + 1])

    # convert to column vectors and scale rates
    bu_min_bits = np.asarray(bu_min, dtype=float).flatten()
    w = np.asarray(w, dtype=float).flatten()
    H = np.asarray(H)
    bu_internal = (1 / (3 - cb)) * bu_min_bits

    # scale rates for optimization (nats)
    bu_scaled = bu_internal * np.log(2)

    print(f"minPMACMIMO: Solving for {U} users, {N} tones, {Ly} RX antennas")
    print(f"Antenna configuration: {Lx}, total={Ltot}")

    # ellipsoid method to find optimal theta
    err = 1e-9
    count = 0

    # initialize ellipsoid
    A, theta, _ = start_ellipse_mimo(H, bu_internal, w, cb, Lx, idx_start, idx_end)
    Rxx_warm = None
    state_warm = []

    while True:
        # solve dual problem for current theta
        _, bun_internal, Rxx_opt, state_warm = lagrange_dual_mimo(
            H, theta, w, bu_scaled, Lx, idx_start, idx_end, Rxx_warm, state_warm
        )
        Rxx_warm = Rxx_opt

        # compute subgradient
        g = np.sum(bun_internal, axis=1) - bu_scaled

        # check stopping criteria
        if np.sqrt(g @ A @ g) <= err:
            break

        # update ellipsoid
        tmp = A @ g / np.sqrt(g @ A @ g)
        theta = theta - (1 / (U + 1)) * tmp
        A = (U**2) / (U**2 - 1) * (A - (2 / (U + 1)) * np.outer(tmp, tmp))

        # ensure theta is feasible (non-negative)
        ind = np.where(theta < 0)[0]

        while len(ind) > 0:
            g = np.zeros(U)
            g[ind[0]] = -1
            tmp = A @ g / np.sqrt(g @ A @ g)
            theta = theta - (1 / (U + 1)) * tmp
            A = (U**2) / (U**2 - 1) * (A - (2 / (U + 1)) * np.outer(tmp, tmp))
            ind = np.where(theta < 0)[0]

        count += 1

        if count > 2000:  # prevent infinite loops
            print("Warning: Ellipsoid method did not converge")
            break

    bu_min = bu_min_bits

    # cluster users by lagrange multipliers
    clusters, theta_unique = cluster_theta_values(theta)

    # generate all possible decoding orders
    all_orders = generate_decoding_orders(clusters)

    print(f"Found {len(clusters)} clusters, {len(all_orders)} possible decoding orders")

    # compute solution based on number of orders
    if len(all_orders) == 1:
        # single decoding order
        order = all_orders[0]
        bu_achieved, b_achieved = compute_achieved_rates_mimo(
            H, Rxx_opt, order, Ly, U, N, cb, Lx, idx_start, idx_end
        )

        FEAS_FLAG = 1
        bu_a = bu_achieved

        # store single-order results
        info = create_info_struct_mimo(
            H,
            Lx,
            bu_min,
            w,
            cb,
            theta,
            theta_unique,
            clusters,
            [order],
            np.array([1]),
            [Rxx_opt],
            bu_achieved,
            b_achieved,
            "Solved",
            idx_start,
            idx_end,
        )
        info["frac"] = None
        info["bun"] = b_achieved

    else:
        # multiple orders; solve for time-sharing weights
        weights, bu_vertices, bun_orders, orderings = solve_time_sharing_mimo(
            H, Rxx_opt, all_orders, bu_min, Ly, U, N, cb, Lx, idx_start, idx_end
        )

        if weights is None:
            FEAS_FLAG = 0
            bu_a = np.zeros(U)
            info = {
                "sol_status": "Failed",
                "feasible": False,
                "time_sharing_failed": True,
            }
            elapsed_time = time.time() - start_time
            info["elapsed_time"] = elapsed_time
            print(
                f"minPMACMIMO completed in {elapsed_time:.3f} seconds, FEAS_FLAG={FEAS_FLAG}"
            )
            return FEAS_FLAG, bu_a, info

        # compute weighted average of achieved rates
        weights = weights.flatten()
        bu_a = bu_vertices @ weights

        # aggregate per-tone bit allocations with time-sharing weights
        bun_avg = np.zeros((U, N))
        for k in range(len(weights)):
            bun_avg += bun_orders[k] * weights[k]

        FEAS_FLAG = 2

        # store time-sharing results
        info = create_info_struct_mimo(
            H,
            Lx,
            bu_min,
            w,
            cb,
            theta,
            theta_unique,
            clusters,
            orderings,
            weights,
            [Rxx_opt],
            bu_vertices,
            bun_orders,
            "Solved",
            idx_start,
            idx_end,
        )
        info["frac"] = weights
        info["bun"] = bun_avg

    # compute average energies
    info["Eu_avg"] = compute_average_energies_mimo(
        Rxx_opt, U, N, Lx, idx_start, idx_end
    )
    elapsed_time = time.time() - start_time
    info["elapsed_time"] = elapsed_time

    print(f"minPMACMIMO completed in {elapsed_time:.3f} seconds, FEAS_FLAG={FEAS_FLAG}")

    return FEAS_FLAG, bu_a, info


def cluster_theta_values(theta):
    """Cluster users by dual variable values.

    Args:
        theta: Dual variables, shape (U,)

    Returns:
        clusters: List of lists, each containing user indices in a cluster
        theta_unique: Unique theta values, shape (num_clusters,)
    """
    tol = 1e-8
    theta_vec = theta.flatten()

    # sort theta values
    sorted_indices = np.argsort(theta_vec)
    sorted_theta = theta_vec[sorted_indices]

    # find unique values within tolerance
    theta_unique = []
    clusters = []
    current_cluster = [sorted_indices[0]]
    current_val = sorted_theta[0]

    for i in range(1, len(sorted_theta)):
        if abs(sorted_theta[i] - current_val) <= tol:
            # add to current cluster
            current_cluster.append(sorted_indices[i])
        else:
            # start new cluster
            theta_unique.append(current_val)
            clusters.append(current_cluster)
            current_cluster = [sorted_indices[i]]
            current_val = sorted_theta[i]

    # add last cluster
    theta_unique.append(current_val)
    clusters.append(current_cluster)

    return clusters, np.array(theta_unique)


def generate_decoding_orders(clusters):
    """Generate all possible decoding orders from clusters.

    Args:
        clusters: List of lists, each containing user indices

    Returns:
        orders: List of tuples, each representing a decoding order
    """
    if len(clusters) == 1:
        # single cluster - return all permutations
        return list(itertools.permutations(clusters[0]))

    # multiple clusters - generate cartesian product of cluster permutations
    cluster_perms = [list(itertools.permutations(cluster)) for cluster in clusters]

    # compute cartesian product
    all_orders = []
    for combo in itertools.product(*cluster_perms):
        # concatenate tuples
        order = sum(combo, ())
        all_orders.append(order)

    return all_orders


def compute_achieved_rates_mimo(H, Rxx, order, Ly, U, N, cb, Lx, idx_start, idx_end):
    """Compute achieved rates for a given decoding order.

    Args:
        H: Channel matrices, shape (Ly, Ltot, N)
        Rxx: Transmit covariances, shape (Ltot, Ltot, N)
        order: Decoding order (tuple of user indices)
        Ly: Number of RX antennas
        U: Number of users
        N: Number of tones
        cb: Baseband type
        Lx: Antennas per user
        idx_start: Starting antenna indices
        idx_end: Ending antenna indices

    Returns:
        bu_achieved: Total achieved rates per user, shape (U,)
        b_achieved: Per-tone achieved rates, shape (U, N)
    """
    b_achieved = np.zeros((U, N))
    cols_per_order = []

    for k in range(U):
        user_idx = order[k]
        cols_per_order.append(slice(idx_start[user_idx], idx_end[user_idx] + 1))

    eye_Ly = np.eye(Ly, dtype=H.dtype)
    inv_log2_cb = 1 / (np.log(2) * cb)

    for n in range(N):
        Rn = Rxx[:, :, n]

        # precompute H*R*H' for each user
        suffix_cache = []
        for k in range(U):
            cols = cols_per_order[k]
            Hv = H[:, cols, n]
            Rv = Rn[cols, cols]
            HRH = Hv @ Rv @ Hv.conj().T
            suffix_cache.append(0.5 * (HRH + HRH.conj().T))

        # compute rates via backward pass
        suffix_sum = np.zeros((Ly, Ly), dtype=H.dtype)
        suffix_next_logdet = 0

        for pos in range(U - 1, -1, -1):
            logdet_interf = suffix_next_logdet
            suffix_sum = suffix_sum + suffix_cache[pos]
            S = eye_Ly + suffix_sum
            S = 0.5 * (S + S.conj().T)

            try:
                L = cholesky(S, lower=True)
            except np.linalg.LinAlgError:
                S = S + 1e-9 * eye_Ly
                L = cholesky(S, lower=True)

            logdet_signal = 2 * np.sum(np.log(np.diag(L)))
            suffix_next_logdet = logdet_signal
            user_idx = order[pos]
            rate_bits = (logdet_signal - logdet_interf) * inv_log2_cb
            b_achieved[user_idx, n] = np.real(rate_bits)

    bu_achieved = np.sum(b_achieved, axis=1)
    return bu_achieved, b_achieved


def solve_time_sharing_mimo(
    H, Rxx, orders, bu_min, Ly, U, N, cb, Lx, idx_start, idx_end
):
    """Solve time-sharing optimization using CVXPY.

    Args:
        H: Channel matrices, shape (Ly, Ltot, N)
        Rxx: Transmit covariances, shape (Ltot, Ltot, N)
        orders: List of decoding orders
        bu_min: Target rates, shape (U,)
        Ly: Number of RX antennas
        U: Number of users
        N: Number of tones
        cb: Baseband type
        Lx: Antennas per user
        idx_start: Starting antenna indices
        idx_end: Ending antenna indices

    Returns:
        weights: Time-sharing weights for active orders
        bu_vertices: Achieved rates per order, shape (U, num_orders)
        bun_vertices: List of per-tone rates for each order
        orderings: List of active decoding orders
    """
    num_orders = len(orders)
    bu_vertices = np.zeros((U, num_orders))
    orderings = []
    bun_vertices = []

    # compute achieved rates for each order
    for k in range(num_orders):
        orderings.append(orders[k])
        bu_achieved, bun_matrix = compute_achieved_rates_mimo(
            H, Rxx, orders[k], Ly, U, N, cb, Lx, idx_start, idx_end
        )
        bu_vertices[:, k] = bu_achieved
        bun_vertices.append(bun_matrix)

    # solve time-sharing LP using cvxpy
    tol = 1e-3
    weights_var = cp.Variable(num_orders, nonneg=True)
    z = cp.Variable(num_orders, boolean=True)

    objective = cp.Minimize(cp.sum(z))
    constraints = [
        cp.sum(weights_var) == 1,
        bu_vertices @ weights_var >= bu_min - tol,
        weights_var <= z,
    ]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.MOSEK, verbose=False)
    except Exception as e:
        print(f"Time-sharing optimization failed: {e}")
        return None, None, None, None

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Time-sharing optimization status: {problem.status}")
        return None, None, None, None

    # extract active vertices
    weights = weights_var.value
    active_idx = weights > 1e-8
    weights = weights[active_idx]
    weights = weights / np.sum(weights)  # renormalize
    bu_vertices = bu_vertices[:, active_idx]
    bun_vertices = [bun_vertices[i] for i in range(num_orders) if active_idx[i]]
    orderings = [orderings[i] for i in range(num_orders) if active_idx[i]]

    return weights, bu_vertices, bun_vertices, orderings


def compute_average_energies_mimo(Rxx, U, N, Lx, idx_start, idx_end):
    """Compute average energy per user across tones.

    Args:
        Rxx: Transmit covariances, shape (Ltot, Ltot, N)
        U: Number of users
        N: Number of tones
        Lx: Antennas per user
        idx_start: Starting antenna indices
        idx_end: Ending antenna indices

    Returns:
        Eu_avg: Average energy per user, shape (U,)
    """
    Eu_avg = np.zeros(U)

    for u in range(U):
        ant_idx = slice(idx_start[u], idx_end[u] + 1)
        total_energy = 0

        for n in range(N):
            total_energy += np.real(np.trace(Rxx[ant_idx, ant_idx, n]))

        Eu_avg[u] = total_energy / N

    return Eu_avg


def create_info_struct_mimo(
    H,
    Lx,
    bu_min,
    w,
    cb,
    theta,
    theta_unique,
    clusters,
    orderings,
    weights,
    Rxx_cell,
    bu_vertices,
    b_achieved,
    cvx_status,
    idx_start,
    idx_end,
):
    """Create info dictionary with solution details.

    Args:
        (Various solution parameters)

    Returns:
        info: Dictionary with solution details
    """
    info = {
        "H": H,
        "Lx": Lx,
        "bu_min": bu_min,
        "w": w,
        "cb": cb,
        "theta": theta,
        "theta_unique": theta_unique,
        "clusters": clusters,
        "orderings": orderings,
        "weights": weights,
        "Rxx": Rxx_cell,
        "bu_vertices": bu_vertices,
        "b_achieved": b_achieved,
        "sol_status": cvx_status,
        "feasible": True,
        "idx_start": idx_start,
        "idx_end": idx_end,
    }

    return info
