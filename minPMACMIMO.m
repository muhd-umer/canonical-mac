function [FEAS_FLAG, bu_a, info] = minPMACMIMO(H, Lx, bu_min, w, cb)
    %MINPMACMIMO Minimum power multi-user MIMO MAC (non-CVX implementation)
    %   [FEAS_FLAG, bu_a, info] = MINPMACMIMO(H, Lx, bu_min, w, cb) computes the
    %   optimal covariance matrices that satisfy user rate targets with minimum
    %   weighted energy for both SISO and MIMO MAC configurations.
    %
    %   Inputs:
    %       H       channel matrix [Ly, sum(Lx), N] concatenated across users.
    %       Lx      antennas per user [1, U] where sum(Lx) equals total TX antennas.
    %       bu_min  target rates per user [1, U] in bits per channel use.
    %       w       positive energy weights [1, U].
    %       cb      baseband type: 1 for complex, 2 for real (affects rate scaling).
    %
    %   Outputs:
    %       FEAS_FLAG   feasibility flag: 0 infeasible, 1 single order, 2 time-sharing.
    %       bu_a        achieved user rates in bits per channel use.
    %       info        structure with detailed solution metrics including:
    %                   frac   time-sharing fractions for active decoding orders (empty if single order).
    %                   bun    per-tone bit allocations for the returned solution.

    tic;
    [Ly, Ltot, N] = size(H);
    U = length(Lx);

    % validate inputs
    if sum(Lx) ~= Ltot
        error('sum(Lx) = %d must equal total TX antennas = %d', sum(Lx), Ltot);
    end

    if length(bu_min) ~= U || length(w) ~= U
        error('bu_min and w must have length U=%d', U);
    end

    if ~all(w > 0)
        error('Energy weights w must be positive');
    end

    if ~all(bu_min >= 0)
        error('Target rates bu_min must be non-negative');
    end

    if ~all(Lx >= 1)
        error('Each user must have at least 1 antenna');
    end

    % compute antenna index mapping
    idx_end = cumsum(Lx);
    idx_start = [1, idx_end(1:end - 1) + 1];

    % convert to column vectors
    bu_min_bits = double(bu_min(:));
    w = double(w(:));
    H = double(H);
    bu_internal = (1 / (3 - cb)) * bu_min_bits;

    % scale rates for optimization (nats)
    bu_scaled = bu_internal * log(2);

    fprintf('minPMACMIMO: Solving for %d users, %d tones, %d RX antennas\n', U, N, Ly);
    fprintf('Antenna configuration: [%s], total=%d\n', num2str(Lx), Ltot);

    %% ellipsoid method to find optimal theta
    err = 1e-9;
    count = 0;

    % initialize ellipsoid
    [A, theta] = startEllipse_mimo(H, bu_internal, w, cb, Lx, idx_start, idx_end);
    Rxx_warm = [];
    state_warm = cell(N, 1);

    while true
        % solve dual problem for current theta
        [~, bun_internal, Rxx_opt, state_warm] = Lag_dual_f_mimo(H, theta, w, bu_scaled, Lx, idx_start, idx_end, Rxx_warm, state_warm);
        Rxx_warm = Rxx_opt;

        % compute subgradient
        g = sum(bun_internal, 2) - bu_scaled;

        % check stopping criteria
        if sqrt(g' * A * g) <= err
            break;
        end

        % update ellipsoid
        tmp = A * g / sqrt(g' * A * g);
        theta = theta - 1 / (U + 1) * tmp;
        A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * (tmp * tmp'));

        % ensure theta is feasible (non-negative)
        ind = find(theta < 0);

        while ~isempty(ind)
            g = zeros(U, 1);
            g(ind(1)) = -1;
            tmp = A * g / sqrt(g' * A * g);
            theta = theta - 1 / (U + 1) * tmp;
            A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * (tmp * tmp'));
            ind = find(theta < 0);
        end

        count = count + 1;

        if count > 1000
            warning('Ellipsoid method did not converge');
            break;
        end

    end

    bu_min = bu_min_bits;

    %% cluster users by lagrange multipliers
    [clusters, theta_unique] = cluster_theta_values(theta);

    %% generate all possible decoding orders
    all_orders = generate_decoding_orders(clusters);

    fprintf('Found %d clusters, %d possible decoding orders\n', ...
        length(clusters), size(all_orders, 1));

    %% compute solution based on number of orders
    if size(all_orders, 1) == 1
        % single decoding order
        order = all_orders(1, :);
        [bu_achieved, b_achieved] = compute_achieved_rates_mimo(H, Rxx_opt, order, ...
            Ly, U, N, cb, Lx, idx_start, idx_end);

        FEAS_FLAG = 1;
        bu_a = bu_achieved';

        % store single-order results
        info = create_info_struct_mimo(H, Lx, bu_min, w, cb, theta, theta_unique, ...
            clusters, {order}, 1, {Rxx_opt}, bu_achieved, b_achieved, 'Solved', ...
            idx_start, idx_end);
        info.frac = [];
        info.bun = b_achieved;

    else
        % multiple orders; solve for time-sharing weights
        [weights, bu_vertices, bun_orders, orderings] = solve_time_sharing_mimo(H, ...
            Rxx_opt, all_orders, bu_min, Ly, U, N, cb, Lx, idx_start, idx_end);

        if isempty(weights)
            FEAS_FLAG = 0;
            bu_a = zeros(1, U);
            info = struct('sol_status', 'Failed', 'feasible', false, ...
                'time_sharing_failed', true);
            return;
        end

        % compute weighted average of achieved rates
        weights = weights(:);
        bu_a = (bu_vertices * weights)';
        bun_avg = zeros(U, N);

        for k = 1:length(weights)
            % aggregate per-tone bit allocations with time-sharing weights
            bun_avg = bun_avg + bun_orders{k} * weights(k);
        end

        FEAS_FLAG = 2;

        % store time-sharing results
        info = create_info_struct_mimo(H, Lx, bu_min, w, cb, theta, theta_unique, ...
            clusters, orderings, weights, {Rxx_opt}, bu_vertices, bun_orders, 'Solved', ...
            idx_start, idx_end);
        info.frac = weights;
        info.bun = bun_avg;

    end

    % compute average energies
    info.Eu_avg = compute_average_energies_mimo(Rxx_opt, U, N, Lx, idx_start, idx_end);
    info.elapsed_time = toc;

    fprintf('minPMACMIMO completed in %.3f seconds, FEAS_FLAG=%d\n', info.elapsed_time, FEAS_FLAG);
end

function [clusters, theta_unique] = cluster_theta_values(theta)
    tol = 1e-8;
    theta_vec = double(theta(:));
    [theta_unique, ~, groups] = uniquetol(theta_vec, tol);
    num_clusters = length(theta_unique);
    clusters = cell(num_clusters, 1);

    for k = 1:num_clusters
        clusters{k} = find(groups == k)';
    end

end

function orders = generate_decoding_orders(clusters)

    if length(clusters) == 1
        orders = perms(clusters{1});
    else
        cluster_perms = cell(length(clusters), 1);

        for k = 1:length(clusters)
            cluster_perms{k} = perms(clusters{k});
        end

        orders = cluster_perms{1};

        for k = 2:length(clusters)
            new_orders = [];

            for i = 1:size(orders, 1)

                for j = 1:size(cluster_perms{k}, 1)
                    new_orders = [new_orders; [orders(i, :), cluster_perms{k}(j, :)]];
                end

            end

            orders = new_orders;
        end

    end

end

function [bu_achieved, b_achieved] = compute_achieved_rates_mimo(H, Rxx, order, ...
        Ly, U, N, cb, Lx, idx_start, idx_end)
    b_achieved = zeros(U, N);
    cols_per_order = cell(U, 1);

    for k = 1:U
        user_idx = order(k);
        cols_per_order{k} = idx_start(user_idx):idx_end(user_idx);
    end

    eye_Ly = eye(Ly);
    inv_log2 = 1 / log(2);
    suffix_cache = cell(U, 1);

    for n = 1:N
        Rn = Rxx(:, :, n);

        for k = 1:U
            cols = cols_per_order{k};
            Hv = H(:, cols, n);
            Rv = Rn(cols, cols);
            suffix_cache{k} = symmetrize_local(Hv * (Rv * Hv'));
        end

        suffix_sum = zeros(Ly, Ly);
        suffix_next_logdet = 0;

        for pos = U:-1:1
            logdet_interf = suffix_next_logdet;
            suffix_sum = suffix_sum + suffix_cache{pos};
            logdet_signal = logdet_spd_local(eye_Ly + suffix_sum);
            suffix_next_logdet = logdet_signal;
            user_idx = order(pos);
            rate_bits = (logdet_signal - logdet_interf) * inv_log2 / cb;
            b_achieved(user_idx, n) = real(rate_bits);
        end

    end

    bu_achieved = sum(b_achieved, 2);
end

function M = symmetrize_local(M)
    M = 0.5 * (M + M');
end

function val = logdet_spd_local(M)
    M = symmetrize_local(M);
    [L, flag] = chol(M, 'lower');

    if flag ~= 0
        diag_jitter = 1e-9;
        M = M + diag_jitter * eye(size(M, 1));
        L = chol(M, 'lower');
    end

    val = 2 * sum(log(diag(L)));
end

function [weights, bu_vertices, bun_vertices, orderings] = solve_time_sharing_mimo(H, ...
        Rxx, orders, bu_min, Ly, U, N, cb, Lx, idx_start, idx_end)
    num_orders = size(orders, 1);
    bu_vertices = zeros(U, num_orders);
    orderings = cell(num_orders, 1);
    bun_vertices = cell(num_orders, 1);

    for k = 1:num_orders
        orderings{k} = orders(k, :);
        [bu_achieved, bun_matrix] = compute_achieved_rates_mimo(H, Rxx, orders(k, :), ...
            Ly, U, N, cb, Lx, idx_start, idx_end);
        bu_vertices(:, k) = bu_achieved;
        bun_vertices{k} = bun_matrix;
    end

    tol = 1e-3;
    cvx_begin quiet
    cvx_solver mosek
    variable weights(num_orders) nonnegative
    variable z(num_orders) binary
    minimize(sum(z))
    subject to
    sum(weights) == 1;
    bu_vertices * weights >= bu_min - tol;
    weights <= z;
    cvx_end

    if ~strcmp(cvx_status, 'Solved') && ~strcmp(cvx_status, 'Inaccurate/Solved')
        weights = [];
        return;
    end

    active_idx = weights > 1e-8;
    weights = weights(active_idx);
    weights = weights / sum(weights);
    bu_vertices = bu_vertices(:, active_idx);
    bun_vertices = bun_vertices(active_idx);
    orderings = orderings(active_idx);
end

function Eu_avg = compute_average_energies_mimo(Rxx, U, N, Lx, idx_start, idx_end)
    Eu_avg = zeros(1, U);

    for u = 1:U
        ant_idx = idx_start(u):idx_end(u);
        total_energy = 0;

        for n = 1:N
            total_energy = total_energy + trace(Rxx(ant_idx, ant_idx, n));
        end

        Eu_avg(u) = total_energy / N;
    end

end

function info = create_info_struct_mimo(H, Lx, bu_min, w, cb, theta, theta_unique, ...
        clusters, orderings, weights, Rxx_cell, bu_vertices, b_achieved, cvx_status, ...
        idx_start, idx_end)
    info = struct();
    info.H = H;
    info.Lx = Lx;
    info.bu_min = bu_min;
    info.w = w;
    info.cb = cb;
    info.theta = theta;
    info.theta_unique = theta_unique;
    info.clusters = clusters;
    info.orderings = orderings;
    info.weights = weights;
    info.Rxx = Rxx_cell;
    info.bu_vertices = bu_vertices;
    info.b_achieved = b_achieved;
    info.sol_status = cvx_status;
    info.feasible = true;
    info.idx_start = idx_start;
    info.idx_end = idx_end;
end
