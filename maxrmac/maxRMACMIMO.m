function [Rxxs, Eun, w, bun] = maxRMACMIMO(H, Lxu, Eu, theta, cb)
    %MAXRMACMIMO Maximize weighted rate sum with per-user energy constraints
    %   [Rxxs, Eun, w, bun] = MAXRMACMIMO(H, Lxu, Eu, theta, cb) maximizes the
    %   weighted sum rate for a (noise-whitened) multi-user MIMO MAC subject to
    %   per-user total energy constraints.
    %
    %   This is a non-CVX implementation that solves the Lagrangian dual in the
    %   energy multipliers and evaluates the per-tone inner maximization via a
    %   limited-memory quasi-Newton method over Cholesky-like factors.
    %
    %   Inputs:
    %       H       channel tensor [Ly, sum(Lxu), N] with user antennas stacked.
    %               For SISO, sum(Lxu)=U and H is Ly-by-U-by-N.
    %       Lxu     antennas per user (scalar or 1-by-U).
    %       Eu      per-user energy constraints (length-U vector).
    %       theta   per-user rate weights (length-U vector).
    %       cb      baseband type: 1 for complex, 2 for real.
    %
    %   Outputs:
    %       Rxxs    U-by-N cell array of covariance matrices, or
    %               Lxu_max-by-Lxu_max-by-U-by-N tensor if Lxu is scalar.
    %       Eun     U-by-N energy allocation, Eun(u,n)=trace(Rxx(u,n)).
    %       w       U-by-1 dual variables for energy constraints.
    %       bun     U-by-N per-user per-tone rates (bits).

    [Ly, Ltot, N_in] = size(H);
    theta = reshape(theta, [], 1);
    Eu = reshape(Eu, [], 1);
    U = length(theta);

    if length(Eu) ~= U
        error('Eu must have length U=%d', U);
    end

    if cb ~= 1 && cb ~= 2
        error('cb must be 1 (complex) or 2 (real)');
    end

    if isscalar(Lxu)
        Lxu_vec = ones(1, U) * Lxu;
        uniform_flag = true;
    else
        Lxu_vec = reshape(Lxu, 1, []);
        uniform_flag = false;
    end

    if length(Lxu_vec) ~= U
        error('Lxu must be a scalar or a length-U vector');
    end

    if sum(Lxu_vec) ~= Ltot
        error('sum(Lxu)=%d must equal size(H,2)=%d', sum(Lxu_vec), Ltot);
    end

    % dual update parameters
    max_dual_iter = 800;
    tol_energy = 1e-6;
    dual_step = 0.5;
    dual_step_min = 0.01;
    dual_step_max = 1.0;
    dual_step_shrink = 0.5;
    dual_step_grow = 1.1;
    dual_step_shrink_tol = 0.02;
    dual_step_grow_tol = 0.05;
    dual_log_clip = 4;
    w_min = 1e-14;
    w_max = 1e14;
    energy_zero_tol = 1e-14;
    energy_abs_tol = 1e-12;
    warn_tol_energy = 1e-3;

    N = N_in;
    single_tone_flag = (N_in == 1);
    scale_single = (3 - cb);

    if N_in > 1 && cb == 2

        if mod(N_in, 2) ~= 0
            error('for cb=2 and N>1, N must be even');
        end

        H = H(:, :, 1:(N_in / 2));
        N = N_in / 2;
    elseif single_tone_flag
        Eu = Eu / scale_single;
        H = reshape(H, Ly, Ltot, 1);
        N = 1;
    end

    idx_end = cumsum(Lxu_vec);
    idx_start = [1, idx_end(1:end - 1) + 1];

    [~, idx_sort] = sort(theta, 'descend');
    stheta = theta(idx_sort);
    delta = stheta - [stheta(2:end); 0];

    Eu_sorted = Eu(idx_sort);
    Lxu_sorted = Lxu_vec(idx_sort);
    idx_start_sorted = idx_start(idx_sort);
    idx_end_sorted = idx_end(idx_sort);

    w_sorted = max(w_min, ones(U, 1));
    w_sorted(Eu_sorted <= energy_zero_tol) = w_max;

    states = cell(N, 1);
    Rxx_sorted = cell(U, N);
    Eun_sorted = zeros(U, N);

    prev_err = Inf;
    step = dual_step;
    converged = false;

    last_Rxx = [];
    last_Eun = [];

    % special case; single tone has fixed per-user energies
    if N == 1 && all(Lxu_sorted == 1)
        [Rxx_sorted, Eun_sorted, w_sorted] = solve_single_tone(H(:, :, 1), Eu_sorted, delta, idx_start_sorted, idx_end_sorted, Ly, U);
        last_Rxx = Rxx_sorted;
        last_Eun = Eun_sorted;
        converged = true;
    end

    for iter = 1:max_dual_iter

        if converged
            break;
        end

        [Rxx_iter, Eun_iter, states] = eval_primal(H, Lxu_sorted, idx_start_sorted, idx_end_sorted, delta, w_sorted, states);

        last_Rxx = Rxx_iter;
        last_Eun = Eun_iter;

        E_used = sum(Eun_iter, 2);
        mismatch = E_used - Eu_sorted;
        rel_err = mismatch ./ max(Eu_sorted, energy_zero_tol);
        active = Eu_sorted > energy_zero_tol;

        if any(active)
            err = max(abs(rel_err(active)));
        else
            err = 0;
        end

        ok = true;

        for u = 1:U

            if Eu_sorted(u) <= energy_zero_tol
                ok = ok && (abs(E_used(u)) <= energy_abs_tol);
            else
                active_ok = abs(mismatch(u)) / max(Eu_sorted(u), 1e-12) <= tol_energy;
                inactive_ok = (w_sorted(u) <= w_min * 10) && (mismatch(u) < 0);
                ok = ok && (active_ok || inactive_ok);
            end

        end

        if ok
            converged = true;
            break;
        end

        if isfinite(prev_err)

            if err > prev_err * (1 + dual_step_shrink_tol)
                step = max(dual_step_min, step * dual_step_shrink);
            elseif err < prev_err * (1 - dual_step_grow_tol)
                step = min(dual_step_max, step * dual_step_grow);
            end

        end

        prev_err = err;

        ratio = max(E_used, energy_abs_tol) ./ max(Eu_sorted, energy_abs_tol);
        log_ratio = log(ratio);
        log_ratio = max(-dual_log_clip, min(dual_log_clip, log_ratio));
        w_new = w_sorted .* exp(step * log_ratio);
        w_new(Eu_sorted <= energy_zero_tol) = w_max;
        w_sorted = min(w_max, max(w_min, w_new));
    end

    % ensure returned primal corresponds to final w
    if converged
        Rxx_sorted = last_Rxx;
        Eun_sorted = last_Eun;
    else
        [Rxx_sorted, Eun_sorted, states] = eval_primal(H, Lxu_sorted, idx_start_sorted, idx_end_sorted, delta, w_sorted, states);
    end

    if ~converged && max_dual_iter > 0
        E_used_final = sum(Eun_sorted, 2);
        mismatch_final = E_used_final - Eu_sorted;
        rel_err_final = mismatch_final ./ max(Eu_sorted, energy_zero_tol);
        active = Eu_sorted > energy_zero_tol;

        if any(active)
            err_final = max(abs(rel_err_final(active)));
        else
            err_final = 0;
        end

        if err_final > warn_tol_energy
            warning('maxRMACMIMO:dualNotConverged', ...
                'dual update did not converge within %d iterations (max rel energy error %.2e)', ...
                max_dual_iter, err_final);
        end

    end

    % compute final rates in the sorted order
    bun_sorted = compute_rates(H, Rxx_sorted, Lxu_sorted, idx_start_sorted, idx_end_sorted, cb, U, N, Ly);

    % map outputs back to original user order
    w = zeros(U, 1);
    w(idx_sort) = w_sorted;

    Eun = zeros(U, N);
    Eun(idx_sort, :) = Eun_sorted;

    bun = zeros(U, N);
    bun(idx_sort, :) = bun_sorted;

    % assemble covariance outputs
    Lxu_max = max(Lxu_vec);

    if uniform_flag
        Rxxs = zeros(Lxu_max, Lxu_max, U, N);

        for us = 1:U
            u = idx_sort(us);
            Lu = Lxu_sorted(us);

            for n = 1:N
                Rxxs(1:Lu, 1:Lu, u, n) = Rxx_sorted{us, n};
            end

        end

    else
        Rxxs = cell(U, N);

        for us = 1:U
            u = idx_sort(us);

            for n = 1:N
                Rxxs{u, n} = Rxx_sorted{us, n};
            end

        end

    end

    % match CVX convention for N==1 energy scaling
    if single_tone_flag
        Eun = scale_single * Eun;

        if uniform_flag
            Rxxs(:, :, :, 1) = scale_single * Rxxs(:, :, :, 1);
        else

            for u = 1:U
                Rxxs{u, 1} = scale_single * Rxxs{u, 1};
            end

        end

    end

end

function [Rxx_sorted, Eun_sorted, w_sorted] = solve_single_tone(Hn, Eu_sorted, delta, idx_start, idx_end, Ly, U)
    % SOLVE_SINGLE_TONE Single tone SISO/SIMO case with fixed per-user energies

    Rxx_sorted = cell(U, 1);
    Eun_sorted = zeros(U, 1);

    for u = 1:U
        Rxx_sorted{u, 1} = Eu_sorted(u);
        Eun_sorted(u, 1) = Eu_sorted(u);
    end

    % compute KKT multipliers w from gradient of objective at this primal point
    invS = cell(U, 1);
    S = eye(Ly);

    for u = 1:U
        cols = idx_start(u):idx_end(u);
        Hu = Hn(:, cols);
        S = S + Hu * Eu_sorted(u) * Hu';
        S = 0.5 * (S + S');
        [L, flag] = chol(S, 'lower');

        if flag ~= 0
            [L, flag] = chol(S +1e-12 * eye(Ly), 'lower');
        end

        if flag ~= 0
            error('failed to factorize covariance matrix in single tone solver');
        end

        Linv = L \ eye(Ly);
        invS{u} = Linv' * Linv;
    end

    w_sorted = zeros(U, 1);
    tail = zeros(Ly, Ly);

    for u = U:-1:1
        tail = tail + delta(u) * invS{u};
        cols = idx_start(u):idx_end(u);
        Hu = Hn(:, cols);
        w_sorted(u) = real(Hu' * tail * Hu);
    end

end

function bun = compute_rates(H, Rxx_all, Lxu, idx_start, idx_end, cb, U, N, Ly)
    %COMPUTE_RATES Compute per-user per-tone rates from covariance matrices

    bun = zeros(U, N);
    eye_Ly = eye(Ly);
    inv_log2_cb = 1 / (log(2) * cb);

    for n = 1:N
        Hn = H(:, :, n);
        S_prev = eye_Ly;
        logdet_prev = 0;

        for u = 1:U
            cols = idx_start(u):idx_end(u);
            Hu = Hn(:, cols);
            HRH = Hu * Rxx_all{u, n} * Hu';
            HRH = 0.5 * (HRH + HRH');
            S_curr = S_prev + HRH;
            S_curr = 0.5 * (S_curr + S_curr');

            logdet_curr = logdet_stable(S_curr);
            bun(u, n) = (logdet_curr - logdet_prev) * inv_log2_cb;
            S_prev = S_curr;
            logdet_prev = logdet_curr;
        end

    end

end

function val = logdet_stable(M)
    %LOGDET_STABLE Numerically stable log-determinant computation

    M = 0.5 * (M + M');
    [L, flag] = chol(M, 'lower');

    if flag ~= 0
        M = M +1e-12 * eye(size(M, 1));
        [L, flag] = chol(M, 'lower');
    end

    if flag ~= 0
        eigvals = real(eig(M));
        eigvals = max(eigvals, 1e-15);
        val = sum(log(eigvals));
        return;
    end

    val = 2 * sum(log(diag(L)));
end
