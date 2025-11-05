function [f, b, Rxx] = minPtone_mimo(H, theta, w, Lx, idx_start, idx_end, R_init)
    %MINPTONE_MIMO Dual maximization on a single tone for MIMO MAC
    %   [f, b, Rxx] = MINPTONE_MIMO(H, theta, w, Lx, idx_start, idx_end) solves the
    %   per-tone Lagrangian maximization that appears inside the ellipsoid method
    %   for the MIMO minPMAC problem. This implementation avoids CVX and works
    %   directly with the covariance matrices for each user by applying a limited
    %   memory quasi-newton ascent on the dual objective.
    %
    %   Inputs:
    %       H           Ly-by-Ltot channel matrix for this tone (users stacked).
    %       theta       U-by-1 dual variables (not yet sorted).
    %       w           U-by-1 user energy weights.
    %       Lx          U-by-1 number of antennas per user.
    %       idx_start   U-by-1 starting indices for each user's antennas.
    %       idx_end     U-by-1 ending indices for each user's antennas.
    %       R_init      (optional) Ltot-by-Ltot warm-start covariance matrix.
    %
    %   Outputs:
    %       f     scalar value theta' * b - sum_u w_u * trace(R_u).
    %       b     U-by-1 rate vector in nats/channel use (consistent with SISO).
    %       Rxx   Ltot-by-Ltot transmit covariance matrix for this tone.

    [Ly, ~] = size(H);
    U = length(Lx);

    % sort users according to theta (descending) to match polymatroid vertex
    [stheta, order] = sort(-theta);
    stheta = -stheta;
    sw = w(order);

    % build per-user channel blocks for the sorted order
    H_blocks = cell(U, 1);
    block_sizes = Lx(order);

    for u = 1:U
        orig = order(u);
        cols = idx_start(orig):idx_end(orig);
        H_blocks{u} = H(:, cols);
    end

    % coefficients alpha_k = 0.5 * (theta_k - theta_{k+1}), theta_{U+1}=0
    alpha = 0.5 * (stheta - [stheta(2:U); 0]);

    if nargin < 7 || isempty(R_init)
        R_init_blocks = [];
    else
        R_init_blocks = cell(U, 1);

        for u = 1:U
            orig = order(u);
            cols = idx_start(orig):idx_end(orig);
            R_init_blocks{u} = symmetrize(R_init(cols, cols));
        end

    end

    [R_blocks, ~] = maximize_dual_lbfgs(H_blocks, alpha, sw, block_sizes, R_init_blocks);

    % compute achieved rates (nats) for sorted users
    b_sorted = zeros(U, 1);
    S_prev = eye(Ly);

    for u = 1:U
        S_curr = S_prev + H_blocks{u} * R_blocks{u} * H_blocks{u}';
        b_sorted(u) = 0.5 * real(logdet_spd(S_curr) - logdet_spd(S_prev));
        S_prev = symmetrize(S_curr);
    end

    % map rates back to original ordering
    b = zeros(U, 1);
    b(order) = b_sorted;

    % assemble full covariance matrix in original antenna order
    Ltot = sum(Lx);
    Rxx = zeros(Ltot, Ltot);

    for u = 1:U
        orig = order(u);
        cols = idx_start(orig):idx_end(orig);
        Rxx(cols, cols) = symmetrize(R_blocks{u});
    end

    % objective value
    energy = 0;

    for u = 1:U
        orig = order(u);
        energy = energy + w(orig) * trace(R_blocks{u});
    end

    f = real(theta' * b - energy);
end

function [F, grad_blocks] = compute_objective_and_gradient(H_blocks, R_blocks, alpha, w)
    % evaluate dual objective and its gradient for the current covariance set.
    U = numel(H_blocks);
    Ly = size(H_blocks{1}, 1);
    S = eye(Ly);
    logdets = zeros(U, 1);
    chol_factors = cell(U, 1);

    for u = 1:U
        S = S + H_blocks{u} * R_blocks{u} * H_blocks{u}';
        S = symmetrize(S);
        [L, flag] = chol(S, 'lower');

        if flag ~= 0
            S = S +1e-9 * eye(Ly);
            L = chol(S, 'lower');
        end

        chol_factors{u} = L;
        logdets(u) = 2 * sum(log(diag(L)));
    end

    energy_term = 0;

    for u = 1:U
        energy_term = energy_term + w(u) * trace(R_blocks{u});
    end

    F = sum(alpha .* logdets) - energy_term;

    if nargout >= 2
        grad_blocks = cell(U, 1);

        for u = 1:U
            Hu = H_blocks{u};
            grad_u = zeros(size(R_blocks{u}));

            for k = u:U
                Lk = chol_factors{k};
                Y = Lk \ Hu;
                Z = Lk' \ Y;
                grad_u = grad_u + alpha(k) * (Hu' * Z);
            end

            grad_u = grad_u - w(u) * eye(size(grad_u, 1));
            grad_blocks{u} = symmetrize(grad_u);
        end

    end

end

function [R_blocks, F_opt] = maximize_dual_lbfgs(H_blocks, alpha, w, block_sizes, R_init_blocks)
    % lbfgs parameters tuned for fast convergence on smooth dual
    max_it = 200;
    tol_grad = 1e-6;
    m_hist = 10;
    step_init = 1.0;
    step_min = 1e-10;
    beta = 0.5;
    armijo_c = 1e-4;
    max_linesearch = 15;
    U = numel(H_blocks);
    B_blocks = cell(U, 1);

    for u = 1:U

        if nargin >= 5 && ~isempty(R_init_blocks)
            Ru = symmetrize(R_init_blocks{u});
            B_blocks{u} = initialize_factor(Ru);
        else
            B_blocks{u} = sqrt(1e-6) * eye(block_sizes(u));
        end

    end

    x = pack_complex_blocks(B_blocks, block_sizes);
    [phi, F_opt, grad, B_blocks, R_blocks] = evaluate_lbfgs_state(x, H_blocks, alpha, w, block_sizes);
    grad_norm = norm(grad);
    S_hist = cell(0, 1);
    Y_hist = cell(0, 1);

    for iter = 1:max_it

        if grad_norm < tol_grad
            break;
        end

        direction = lbfgs_direction(grad, S_hist, Y_hist);
        dir_deriv = dot_real(grad, direction);

        if dir_deriv >= 0
            direction = -grad;
            dir_deriv = dot_real(grad, direction);
            S_hist = cell(0, 1);
            Y_hist = cell(0, 1);
        end

        step = step_init;
        accepted = false;
        x_candidate = [];
        phi_candidate = [];
        grad_candidate = [];
        B_candidate = [];
        R_candidate = [];

        for ls = 1:max_linesearch
            x_trial = x + step * direction;
            [phi_trial, F_trial, grad_trial, B_trial, R_trial] = evaluate_lbfgs_state(x_trial, H_blocks, alpha, w, block_sizes);

            if phi_trial <= phi + armijo_c * step * dir_deriv
                accepted = true;
                x_candidate = x_trial;
                phi_candidate = phi_trial;
                grad_candidate = grad_trial;
                B_candidate = B_trial;
                R_candidate = R_trial;
                F_opt = F_trial;
                break;
            end

            step = step * beta;

            if step < step_min
                break;
            end

        end

        if ~accepted
            direction = -grad;
            dir_deriv = dot_real(grad, direction);
            step = step_init;

            for ls = 1:max_linesearch
                x_trial = x + step * direction;
                [phi_trial, F_trial, grad_trial, B_trial, R_trial] = evaluate_lbfgs_state(x_trial, H_blocks, alpha, w, block_sizes);

                if phi_trial <= phi + armijo_c * step * dir_deriv
                    accepted = true;
                    x_candidate = x_trial;
                    phi_candidate = phi_trial;
                    grad_candidate = grad_trial;
                    B_candidate = B_trial;
                    R_candidate = R_trial;
                    F_opt = F_trial;
                    break;
                end

                step = step * beta;

                if step < step_min
                    break;
                end

            end

            if ~accepted
                break;
            end

            S_hist = cell(0, 1);
            Y_hist = cell(0, 1);
        end

        s = x_candidate - x;
        y = grad_candidate - grad;

        if dot_real(s, y) > 1e-12

            if numel(S_hist) == m_hist
                S_hist = S_hist(2:end);
                Y_hist = Y_hist(2:end);
            end

            S_hist{end + 1} = s;
            Y_hist{end + 1} = y;

        else
            S_hist = cell(0, 1);
            Y_hist = cell(0, 1);
        end

        x = x_candidate;
        phi = phi_candidate;
        grad = grad_candidate;
        B_blocks = B_candidate;
        R_blocks = R_candidate;
        grad_norm = norm(grad);
    end

end

function [phi, F_val, grad_vec, B_blocks, R_blocks] = evaluate_lbfgs_state(x, H_blocks, alpha, w, block_sizes)
    B_blocks = unpack_complex_blocks(x, block_sizes);
    U = numel(B_blocks);
    R_blocks = cell(U, 1);

    for u = 1:U
        R_blocks{u} = symmetrize(B_blocks{u} * B_blocks{u}');
    end

    [F_val, grad_R] = compute_objective_and_gradient(H_blocks, R_blocks, alpha, w);
    grad_blocks = cell(U, 1);

    for u = 1:U
        grad_blocks{u} = -2 * grad_R{u} * B_blocks{u};
    end

    grad_vec = pack_complex_blocks(grad_blocks, block_sizes);
    phi = -F_val;
end

function direction = lbfgs_direction(grad, S_hist, Y_hist)

    if isempty(S_hist)
        direction = -grad;
        return;
    end

    k = numel(S_hist);
    alpha = zeros(k, 1);
    rho = zeros(k, 1);
    q = grad;

    for i = k:-1:1
        s = S_hist{i};
        y = Y_hist{i};
        denom = dot_real(y, s);

        if denom <= 1e-12
            rho(i) = 0;
            alpha(i) = 0;
            continue;
        end

        rho(i) = 1 / denom;
        alpha(i) = rho(i) * dot_real(s, q);
        q = q - alpha(i) * y;
    end

    sk = S_hist{end};
    yk = Y_hist{end};
    denom = dot_real(yk, yk);

    if denom <= 1e-12
        H0 = 1;
    else
        H0 = dot_real(sk, yk) / denom;
    end

    r = H0 * q;

    for i = 1:k
        y = Y_hist{i};
        s = S_hist{i};

        if rho(i) == 0
            beta = 0;
        else
            beta = rho(i) * dot_real(y, r);
        end

        r = r + s * (alpha(i) - beta);
    end

    direction = -r;
end

function vec = pack_complex_blocks(blocks, block_sizes)
    U = numel(blocks);
    total_len = 0;

    for u = 1:U
        total_len = total_len + 2 * block_sizes(u) * block_sizes(u);
    end

    vec = zeros(total_len, 1);
    offset = 0;

    for u = 1:U
        block = blocks{u};
        flat = block(:);
        len = numel(flat);
        vec(offset + (1:len)) = real(flat);
        vec(offset + len + (1:len)) = imag(flat);
        offset = offset + 2 * len;
    end

end

function blocks = unpack_complex_blocks(vec, block_sizes)
    U = numel(block_sizes);
    blocks = cell(U, 1);
    offset = 0;

    for u = 1:U
        len = block_sizes(u) * block_sizes(u);
        real_part = vec(offset + (1:len));
        imag_part = vec(offset + len + (1:len));
        block = reshape(real_part + 1j * imag_part, block_sizes(u), block_sizes(u));
        blocks{u} = block;
        offset = offset + 2 * len;
    end

end

function val = dot_real(a, b)
    val = real(sum(a .* b));
end

function B = initialize_factor(R)

    if isempty(R)
        B = zeros(size(R));
        return;
    end

    R = symmetrize(R);
    [L, flag] = chol(R, 'lower');

    if flag == 0
        B = L;
        return;
    end

    % fallback using eigen decomposition for semi-definite cases
    [V, D] = eig(R, 'vector');
    d = real(D);
    d(d < 0) = 0;
    B = V * diag(sqrt(d));
end

function A = symmetrize(A)
    A = 0.5 * (A + A');
end

function R = project_to_psd(M)
    M = symmetrize(M);
    [V, D] = eig(M);
    d = real(diag(D));
    d(d < 0) = 0;
    R = V * diag(d) * V';
    R = symmetrize(R);
end

function val = logdet_spd(M)
    M = symmetrize(M);
    [L, flag] = chol(M, 'lower');

    if flag ~= 0
        M = M +1e-9 * eye(size(M, 1));
        L = chol(M, 'lower');
    end

    val = 2 * sum(log(diag(L)));
end
