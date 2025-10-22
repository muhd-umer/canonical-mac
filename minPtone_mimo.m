function [f, b, Rxx] = minPtone_mimo(H, theta, w, Lx, idx_start, idx_end)
    %MINPTONE_MIMO  Dual maximization on a single tone for MIMO MAC
    %   [f, b, Rxx] = MINPTONE_MIMO(H, theta, w, Lx, idx_start, idx_end)
    %   solves the per-tone Lagrangian maximization that appears inside the
    %   ellipsoid method for the MIMO minPMAC problem. This implementation
    %   avoids CVX and works directly with the covariance matrices for each
    %   user by applying a projected gradient ascent on the dual objective.
    %
    %   Inputs:
    %     H           Ly-by-Ltot channel matrix for this tone (users stacked)
    %     theta       U-by-1 dual variables (not yet sorted)
    %     w           U-by-1 user energy weights
    %     Lx          U-by-1 number of antennas per user
    %     idx_start   U-by-1 starting indices for each user's antennas
    %     idx_end     U-by-1 ending indices for each user's antennas
    %
    %   Outputs:
    %     f     scalar value theta' * b - sum_u w_u * trace(R_u)
    %     b     U-by-1 rate vector (in nats/channel use, consistent with SISO)
    %     Rxx   Ltot-by-Ltot transmit covariance matrix for this tone

    [Ly, ~] = size(H);
    U = length(Lx);

    % Sort users according to theta (descending) to match polymatroid vertex
    [stheta, order] = sort(-theta);
    stheta = -stheta;
    sw = w(order);

    % Build per-user channel blocks for the sorted order
    H_blocks = cell(U, 1);
    block_sizes = Lx(order);

    for u = 1:U
        orig = order(u);
        cols = idx_start(orig):idx_end(orig);
        H_blocks{u} = H(:, cols);
    end

    % Coefficients alpha_k = 0.5 * (theta_k - theta_{k+1}), theta_{U+1}=0
    alpha = 0.5 * (stheta - [stheta(2:U); 0]);

    % Initialize strictly feasible PSD covariances (small identity)
    R_blocks = cell(U, 1);

    for u = 1:U
        R_blocks{u} = 1e-6 * eye(block_sizes(u));
    end

    % Gradient-ascent parameters
    max_it = 500;
    tol_grad = 1e-6;
    step_init = 1.0;
    step_min = 1e-10;
    beta = 0.5;
    armijo_c = 1e-4;

    [F_prev, grad_blocks] = compute_objective_and_gradient(H_blocks, R_blocks, alpha, sw);

    for iter = 1:max_it
        grad_norm = 0;

        for u = 1:U
            grad_norm = grad_norm + norm(grad_blocks{u}, 'fro') ^ 2;
        end

        grad_norm = sqrt(grad_norm);

        if grad_norm < tol_grad
            break;
        end

        step = step_init;
        accepted = false;

        while step > step_min
            trial_blocks = cell(U, 1);

            for u = 1:U
                delta = step * symmetrize(grad_blocks{u});
                trial_blocks{u} = project_to_psd(R_blocks{u} + delta);
            end

            [F_trial, ~] = compute_objective_and_gradient(H_blocks, trial_blocks, alpha, sw);

            % Directional derivative surrogate for Armijo test
            dir_improve = 0;

            for u = 1:U
                dir_improve = dir_improve + real(trace(grad_blocks{u}' * (trial_blocks{u} - R_blocks{u})));
            end

            if dir_improve <= 0
                step = step * beta;
                continue;
            end

            if F_trial >= F_prev + armijo_c * step * dir_improve
                accepted = true;
                R_blocks = trial_blocks;
                F_prev = F_trial;
                break;
            end

            step = step * beta;
        end

        if ~accepted
            % No productive direction detected; terminate
            break;
        end

        [~, grad_blocks] = compute_objective_and_gradient(H_blocks, R_blocks, alpha, sw);
    end

    % Compute achieved rates (nats) for sorted users
    b_sorted = zeros(U, 1);
    S_prev = eye(Ly);

    for u = 1:U
        S_curr = S_prev + H_blocks{u} * R_blocks{u} * H_blocks{u}';
        b_sorted(u) = 0.5 * real(logdet_spd(S_curr) - logdet_spd(S_prev));
        S_prev = symmetrize(S_curr);
    end

    % Map rates back to original ordering
    b = zeros(U, 1);
    b(order) = b_sorted;

    % Assemble full covariance matrix in original antenna order
    Ltot = sum(Lx);
    Rxx = zeros(Ltot, Ltot);

    for u = 1:U
        orig = order(u);
        cols = idx_start(orig):idx_end(orig);
        Rxx(cols, cols) = symmetrize(R_blocks{u});
    end

    % Objective value
    energy = 0;

    for u = 1:U
        orig = order(u);
        energy = energy + w(orig) * trace(R_blocks{u});
    end

    f = real(theta' * b - energy);
end

function [F, grad_blocks] = compute_objective_and_gradient(H_blocks, R_blocks, alpha, w)
    % Evaluate dual objective and its gradient for the current covariance set.
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
