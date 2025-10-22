function [FEAS_FLAG, bu_a, info] = minPMACMIMO(H, Lx, bu_min, w, cb)
    % MINPMACMIMO - Minimum power multi-user MIMO MAC (non-CVX implementation)
    %
    % SYNTAX:
    %   [FEAS_FLAG, bu_a, info] = minPMACMIMO(H, Lx, bu_min, w, cb)
    %
    % INPUTS:
    %   H       - Channel matrix [Ly, sum(Lx), N] concatenated across users
    %   Lx      - Antennas per user [1, U] where sum(Lx) = total TX antennas
    %   bu_min  - Target rates per user [1, U] (bits per channel use)
    %   w       - Energy weights [1, U] (positive weights for energy objective)
    %   cb      - Baseband type: 1=complex, 2=real (affects rate calculation)
    %
    % OUTPUTS:
    %   FEAS_FLAG - Feasibility: 0=infeasible, 1=single order, 2=time-sharing
    %   bu_a      - Achieved rates [1, U] (bits per channel use)
    %   info      - Structure with detailed results

    tic;
    [Ly, Ltot, N] = size(H);
    U = length(Lx);

    % Validate inputs
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

    % Compute antenna index mapping
    idx_end = cumsum(Lx);
    idx_start = [1, idx_end(1:end - 1) + 1];

    % Convert to column vectors
    bu_min = double(bu_min(:));
    w = double(w(:));
    H = double(H);

    bu_min = 1 / (3 - cb) * bu_min;

    % Scale rates for optimization
    bu_scaled = bu_min * log(2);

    fprintf('minPMACMIMO: Solving for %d users, %d tones, %d RX antennas\n', U, N, Ly);
    fprintf('Antenna configuration: [%s], total=%d\n', num2str(Lx), Ltot);

    % For SISO case, use reference minPMAC implementation
    if all(Lx == 1)
        addpath(fullfile(fileparts(mfilename('fullpath')), 'src'));
        % minPMAC expects original (unscaled) rate in bits
        % We've already scaled bu_min by 1/(3-cb), so reverse it
        bu_min_orig = (3 - cb) * bu_min;
        [Eun, theta, bun, ~, ~, ~] = minPMAC(H, bu_min_orig, w, cb);

        % Convert to Rxx format (diagonal matrices for SISO)
        Rxx_opt = zeros(Ltot, Ltot, N);
        for n = 1:N
            for u = 1:U
                Rxx_opt(u, u, n) = Eun(u, n);
            end
        end

        % minPMAC handles all scaling, so restore bu_min for later use
        bu_min = bu_min_orig;
    else
        %% Ellipsoid method to find optimal theta for true MIMO case
        err = 1e-9;
        count = 0;

        % Initialize ellipsoid
        [A, g, w] = startEllipse_mimo(H, bu_scaled, w, cb, Lx, idx_start, idx_end);
        theta = g;

        while true
            % Solve dual problem for current theta
            [~, bun, Rxx_opt] = Lag_dual_f_mimo(H, theta, w, bu_scaled, Lx, idx_start, idx_end);

            % Compute subgradient
            g = sum(bun, 2) - bu_scaled;

            % Check stopping criteria
            if sqrt(g' * A * g) <= err
                break;
            end

            % Update ellipsoid
            tmp = A * g / sqrt(g' * A * g);
            theta = theta - 1 / (U + 1) * tmp;
            A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * (tmp * tmp'));

            % Ensure theta is feasible (non-negative)
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

        % Convert rates back from nats to bits
        bun = (3 - cb) * bun / log(2);
        bu_min = (3 - cb) * bu_min;
    end

    bu_scaled = (3 - cb) * bu_scaled / log(2);

    %% Cluster users by Lagrange multipliers
    [clusters, theta_unique] = cluster_theta_values(theta);

    %% Generate all possible decoding orders
    all_orders = generate_decoding_orders(clusters);

    fprintf('Found %d clusters, %d possible decoding orders\n', ...
        length(clusters), size(all_orders, 1));

    %% Compute solution based on number of orders
    if size(all_orders, 1) == 1
        % Single decoding order
        order = all_orders(1, :);
        [bu_achieved, b_achieved] = compute_achieved_rates_mimo(H, Rxx_opt, order, ...
            Ly, U, N, cb, Lx, idx_start, idx_end);

        FEAS_FLAG = 1;
        bu_a = bu_achieved';

        % Store single-order results
        info = create_info_struct_mimo(H, Lx, bu_min, w, cb, theta, theta_unique, ...
            clusters, {order}, 1, {Rxx_opt}, bu_achieved, b_achieved, 'Solved', ...
            idx_start, idx_end);
    else
        % Multiple orders - solve for time-sharing weights
        [weights, bu_vertices, orderings] = solve_time_sharing_mimo(H, Rxx_opt, ...
            all_orders, bu_min, Ly, U, N, cb, Lx, idx_start, idx_end);

        if isempty(weights)
            FEAS_FLAG = 0;
            bu_a = zeros(1, U);
            info = struct('sol_status', 'Failed', 'feasible', false, ...
                'time_sharing_failed', true);
            return;
        end

        % Compute weighted average of achieved rates
        bu_a = (bu_vertices * weights)';
        FEAS_FLAG = 2;

        % Store time-sharing results
        info = create_info_struct_mimo(H, Lx, bu_min, w, cb, theta, theta_unique, ...
            clusters, orderings, weights, {Rxx_opt}, bu_vertices, [], 'Solved', ...
            idx_start, idx_end);
    end

    % Compute average energies
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

    for n = 1:N
        Rn = Rxx(:, :, n);

        for k = 1:U
            u = order(k);
            signal_plus_intf = zeros(Ly, Ly);

            for j = k:U
                v = order(j);
                ant_idx = idx_start(v):idx_end(v);
                H_v = H(:, ant_idx, n);
                R_v = Rn(ant_idx, ant_idx);
                signal_plus_intf = signal_plus_intf + H_v * R_v * H_v';
            end

            interference = zeros(Ly, Ly);

            for j = (k + 1):U
                v = order(j);
                ant_idx = idx_start(v):idx_end(v);
                H_v = H(:, ant_idx, n);
                R_v = Rn(ant_idx, ant_idx);
                interference = interference + H_v * R_v * H_v';
            end

            rate_with_signal = log2(det(eye(Ly) + signal_plus_intf));
            rate_interference_only = log2(det(eye(Ly) + interference));
            b_achieved(u, n) = real(rate_with_signal - rate_interference_only) / cb;
        end

    end

    bu_achieved = sum(b_achieved, 2);
end

function [weights, bu_vertices, orderings] = solve_time_sharing_mimo(H, Rxx, orders, ...
        bu_min, Ly, U, N, cb, Lx, idx_start, idx_end)
    num_orders = size(orders, 1);
    bu_vertices = zeros(U, num_orders);
    orderings = cell(num_orders, 1);

    for k = 1:num_orders
        orderings{k} = orders(k, :);
        [bu_achieved, ~] = compute_achieved_rates_mimo(H, Rxx, orders(k, :), ...
            Ly, U, N, cb, Lx, idx_start, idx_end);
        bu_vertices(:, k) = bu_achieved;
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
