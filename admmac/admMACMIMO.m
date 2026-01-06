function [FEAS_FLAG, bu_a, Rxxs, Eun, theta, w, info] = admMACMIMO(H, Lxu, bu, Eu, cb)
    %ADMMACMIMO Determine feasibility of target rates under energy constraints
    %   [FEAS_FLAG, bu_a, Rxxs, Eun, theta, w, info] = ADMMACMIMO(H, Lxu, bu, Eu, cb)
    %   determines whether the target rate vector bu is feasible for the
    %   (noise-whitened) multi-user MIMO MAC channel H given per-user energy
    %   constraints Eu.
    %
    %   Inputs:
    %       H       channel tensor [Ly, sum(Lxu), N] with user antennas stacked.
    %               For SISO (Lxu=1), H is Ly-by-U-by-N. H(:,:,n) is the channel
    %               for the n-th tone.
    %       Lxu     antennas per user (scalar or 1-by-U vector). If scalar, every
    %               user has Lxu transmit antennas; otherwise user u has Lxu(u)
    %               transmit antennas. Use Lxu=1 for SISO case.
    %       bu      target rate of each user, length-U vector (bits per channel use).
    %       Eu      energy/symbol constraint for each user, length-U vector.
    %       cb      baseband type: 1 for complex, 2 for real.
    %
    %   Outputs:
    %       FEAS_FLAG   feasibility indicator:
    %                   0 = target rates infeasible
    %                   1 = feasible via single decoding order
    %                   2 = feasible via time-sharing between multiple vertices
    %       bu_a        U-by-1 vector of achieved sum rate per user.
    %       Rxxs        transmit covariance matrices:
    %                   - if FEAS_FLAG=0: returns 0
    %                   - if Lxu is scalar: Lxu-by-Lxu-by-U-by-N tensor
    %                   - if Lxu is vector: U-by-N cell array of matrices
    %       Eun         U-by-N matrix of per-user per-tone energy allocations.
    %                   If FEAS_FLAG=0, returns 0.
    %       theta       U-by-1 Lagrangian multipliers for rate constraints.
    %       w           U-by-1 Lagrangian multipliers for energy constraints.
    %       info        detailed solution information (depends on FEAS_FLAG):
    %                   - if FEAS_FLAG=0: empty cell
    %                   - if FEAS_FLAG=1: 1-row table with {bu_v, bun, order}
    %                   - if FEAS_FLAG=2: v-row table with {bu_v, bun, order, frac, clusterID}

    % add paths for dependencies
    thisDir = fileparts(mfilename('fullpath'));
    addpath(fullfile(thisDir, '..', 'maxrmac'));
    addpath(fullfile(thisDir, '..', 'minpmac'));

    tstart = tic;
    [Ly, Ltot, N] = size(H);
    Eu = reshape(Eu, [], 1);
    bu = reshape(bu, [], 1);
    bu_min = bu;
    U = length(Eu);
    bu_a = zeros(U, 1);

    % initialize theta and ellipsoid matrix
    theta = ones(U, 1) + 0.05 * (1:U)';
    A = eye(U) * U;
    count = 0;

    % tolerances
    err = 1e-6;
    conv_tol = 1e-6;
    max_iter = 2000;

    FEAS_FLAG = 0;

    % check if Lxu was given as scalar
    if isscalar(Lxu)
        SCALAR_FLAG = true;
        Lxu_vec = ones(1, U) * Lxu;
    else
        SCALAR_FLAG = false;
        Lxu_vec = reshape(Lxu, 1, []);
    end

    % validate input dimensions
    if sum(Lxu_vec) ~= Ltot
        error('sum(Lxu)=%d must equal size(H,2)=%d', sum(Lxu_vec), Ltot);
    end

    % store last computed values for returning when infeasible
    last_Rxxs = 0;
    last_Eun = 0;
    last_bun = [];

    % ellipsoid method main loop
    while count < max_iter
        count = count + 1;

        % get the next vertex via maxRMACMIMO
        [Rxxstemp, Eun_iter, w, bun] = maxRMACMIMO(H, Lxu, Eu, theta, cb);

        % store last values
        last_Rxxs = Rxxstemp;
        last_Eun = Eun_iter;
        last_bun = bun;

        bu_v = sum(bun, 2);
        g = bu_v - bu;
        bu_a = bu_v;

        % infeasibility check: theta'*g < -err
        if theta' * g < -err
            FEAS_FLAG = 0;
            Rxxs = 0;
            Eun = 0;
            info = {};
            return
        end

        % achievable by current vertex: all users meet target
        if min(g) >= 0
            FEAS_FLAG = 1;
            Rxxs = Rxxstemp;
            Eun = Eun_iter;
            [~, Itheta] = sort(theta);
            order = Itheta';
            info = table(bu_v', {bun}, order, ...
                'VariableNames', {'bu_v', 'bun', 'order'});
            return
        end

        [stheta, Itheta] = sort(theta, 'descend');
        delta = -diff([stheta; 0]);
        rel_delta = delta ./ max(stheta, err);
        spots = rel_delta < 0.10;
        spots(end) = 0;

        if ~any(spots)
            % no vertex-share, update ellipsoid and continue
            [theta, A] = update_ellipsoid(theta, A, g, U);
            continue;
        end

        % two or more equal-theta users detected
        % form clusters and check per-cluster feasibility
        [clusters, sizeclus, set_thetaeq, numclus] = form_clusters(spots, Itheta, U);

        CLUSTER_FEAS_FLAG = check_cluster_feasibility(clusters, bu_v, bu, err, Itheta, spots, U);

        if ~CLUSTER_FEAS_FLAG
            [theta, A] = update_ellipsoid(theta, A, g, U);
            continue;
        end

        % vertex sharing needed, reorder users by theta
        [H_reord, Lxu_reord, Rxxs_reord, bu_a_reord, bu_v_reord, bun_reord, ...
             bu_min_reord, index_start, index_end] = reorder_by_theta(H, Lxu_vec, Rxxstemp, ...
            bu_a, bu_v, bun, bu_min, Itheta, SCALAR_FLAG, N, U);

        % process each cluster for vertex sharing
        [FEAS_FLAG, Big_info, bu_a_reord] = process_clusters(H_reord, Lxu_reord, Rxxs_reord, ...
            bu_v_reord, bun_reord, bu_min_reord, index_start, index_end, ...
            clusters, sizeclus, set_thetaeq, numclus, err, conv_tol, cb, Ly, N, U);

        if FEAS_FLAG == 0
            [theta, A] = update_ellipsoid(theta, A, g, U);
            continue;
        end

        break;
    end

    if FEAS_FLAG == 0
        Rxxs = 0;
        Eun = 0;
        info = {};
        return
    end

    % restore original user ordering
    [info, bu_a, Rxxs] = restore_ordering(Big_info, bu_a_reord, Rxxs_reord, ...
        Itheta, N, U, SCALAR_FLAG);

    % set Eun from last iteration
    Eun = last_Eun;

    elapsed = toc(tstart);
    fprintf('admMACMIMO: completed in %.3f s (FEAS_FLAG=%d, iters=%d)\n', elapsed, FEAS_FLAG, count);
end

%% helper functions

function [theta, A] = update_ellipsoid(theta, A, g, U)
    %UPDATE_ELLIPSOID Update theta and ellipsoid matrix A via ellipsoid method
    tmp = A * g / sqrt(g' * A * g);
    theta = theta - 1 / (U + 1) * tmp;
    A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * (tmp * tmp'));

    % project theta to non-negative
    ind = find(theta < 0);

    while ~isempty(ind)
        g_proj = zeros(U, 1);
        g_proj(ind(1)) = -1;
        tmp = A * g_proj / sqrt(g_proj' * A * g_proj);
        theta = theta - 1 / (U + 1) * tmp;
        A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * (tmp * tmp'));
        ind = find(theta < 0);
    end

    theta_norm = norm(theta);

    if theta_norm < 0.1
        theta = theta / theta_norm;
    end

end

function [clusters, sizeclus, set_thetaeq, numclus] = form_clusters(spots, Itheta, U)
    %FORM_CLUSTERS Group users into equal-theta clusters
    sizeclus = [];
    numclus = 0;
    set_thetaeq = [];
    clusters = {};
    flagclus = 0;

    for i = 1:U

        if spots(i) && ~flagclus
            % start new cluster
            numclus = numclus + 1;
            set_thetaeq = [set_thetaeq, i];
            sizeclus = [sizeclus, 1];
            flagclus = 1;
        elseif spots(i)
            % continue cluster
            sizeclus(end) = sizeclus(end) + 1;
        elseif flagclus
            % end cluster
            sizeclus(end) = sizeclus(end) + 1;
            flagclus = 0;
            clusters{end + 1} = Itheta(set_thetaeq(end):set_thetaeq(end) + sizeclus(end) - 1)';
        end

    end

    if flagclus && numclus > 0
        sizeclus(end) = sizeclus(end) + 1;
        clusters{end + 1} = Itheta(set_thetaeq(end):set_thetaeq(end) + sizeclus(end) - 1)';
    end

end

function CLUSTER_FEAS_FLAG = check_cluster_feasibility(clusters, bu_v, bu, err, Itheta, spots, U)
    %CHECK_CLUSTER_FEASIBILITY Check if each cluster meets its target rate sum
    CLUSTER_FEAS_FLAG = 1;
    flagclus = 0;
    sizeclus = [];
    set_thetaeq = [];
    numclus = 0;

    for i = 1:U

        if spots(i) && ~flagclus
            numclus = numclus + 1;
            set_thetaeq = [set_thetaeq, i];
            sizeclus = [sizeclus, 1];
            flagclus = 1;
        elseif spots(i)
            sizeclus(end) = sizeclus(end) + 1;
        elseif flagclus
            sizeclus(end) = sizeclus(end) + 1;
            flagclus = 0;
            us = Itheta(set_thetaeq(end):set_thetaeq(end) + sizeclus(end) - 1);

            if sum(bu_v(us) - bu(us)) < -err * length(us)
                CLUSTER_FEAS_FLAG = 0;
                return
            end

        else
            flagclus = 0;

            if bu_v(Itheta(i)) - bu(Itheta(i)) < -err
                CLUSTER_FEAS_FLAG = 0;
                return
            end

        end

    end

    if flagclus && numclus > 0
        sizeclus(end) = sizeclus(end) + 1;
        us = Itheta(set_thetaeq(end):set_thetaeq(end) + sizeclus(end) - 1);

        if sum(bu_v(us) - bu(us)) < -err * length(us)
            CLUSTER_FEAS_FLAG = 0;
            return
        end

    end

end

function [H_reord, Lxu_reord, Rxxs_reord, bu_a_reord, bu_v_reord, bun_reord, ...
              bu_min_reord, index_start, index_end] = reorder_by_theta(H, Lxu_vec, Rxxstemp, ...
    bu_a, bu_v, bun, bu_min, Itheta, SCALAR_FLAG, N, U)
%REORDER_BY_THETA Reorder all data structures by theta ordering
bu_a_reord = bu_a(Itheta);
bu_v_reord = bu_v(Itheta);
bun_reord = bun(Itheta, :);
bu_min_reord = bu_min(Itheta);

% reorder channel matrix
index_end_orig = cumsum(Lxu_vec);
index_start_orig = [1, index_end_orig(1:end - 1) + 1];
hidx = [];

for i = Itheta'
    hidx = [hidx, index_start_orig(i):index_end_orig(i)];
end

Lxu_reord = Lxu_vec(Itheta);
index_end = cumsum(Lxu_reord);
index_start = [1, index_end(1:end - 1) + 1];
H_reord = H(:, hidx, :);

% reorder covariance matrices
if SCALAR_FLAG
    Rxxs_reord = cell(U, N);
    nd = ndims(Rxxstemp);

    for n = 1:N

        for u = 1:U

            if nd == 4
                Rxxs_reord{u, n} = Rxxstemp(:, :, Itheta(u), n);
            elseif nd == 3
                Rxxs_reord{u, n} = Rxxstemp(:, :, Itheta(u));
            else
                Rxxs_reord{u, n} = Rxxstemp;
            end

        end

    end

else

    if iscell(Rxxstemp)
        Rxxs_reord = cell(U, N);

        for n = 1:N
            Rxxs_reord(:, n) = Rxxstemp(Itheta, n);
        end

    else
        Rxxs_reord = cell(U, N);
        nd = ndims(Rxxstemp);

        for n = 1:N

            for u = 1:U

                if nd == 4
                    Rxxs_reord{u, n} = Rxxstemp(:, :, Itheta(u), n);
                elseif nd == 3
                    Rxxs_reord{u, n} = Rxxstemp(:, :, Itheta(u));
                else
                    Rxxs_reord{u, n} = Rxxstemp;
                end

            end

        end

    end

end

end

function [FEAS_FLAG, Big_info, bu_a] = process_clusters(H, Lxu, Rxxs, bu_v, bun, bu_min, ...
        index_start, index_end, clusters, sizeclus, set_thetaeq, numclus, err, conv_tol, cb, Ly, N, U)
    %PROCESS_CLUSTERS Process each cluster for vertex sharing

    Big_info = [];

    % save initial values
    initialbu_v = bu_v';
    initialbun = bun;

    % create initial info table
    initialinfo = table(bu_v', {bun}, U:-1:1, ...
        'VariableNames', {'bu_v', 'bun', 'order'});

    % format all permuted orders for clusters
    order = repmat(1:U, sum(sizeclus - 1), 1);
    cumsize = cumsum([0, sizeclus - 1]);

    for jdx = 1:numclus
        u_range = set_thetaeq(jdx):set_thetaeq(jdx) + sizeclus(jdx) - 1;

        for i = 1:sizeclus(jdx) - 1
            order(cumsize(jdx) + i, u_range) = circshift(u_range, i);
        end

    end

    bu_a = bu_v;

    % process each cluster
    for jdx = 1:numclus
        FEAS_FLAG = 0;

        % initialize cumulative interference
        if jdx == 1
            Sinit = repmat(eye(Ly), 1, 1, N);
            cumrateinit = zeros(1, N);

            for n = 1:N

                for i = 1:set_thetaeq(1) - 1
                    Sinit(:, :, n) = Sinit(:, :, n) + ...
                        H(:, index_start(i):index_end(i), n) * Rxxs{i, n} * ...
                        H(:, index_start(i):index_end(i), n)';
                end

                cumrateinit(n) = (1 / cb) * real(log2(det(Sinit(:, :, n))));
            end

        else

            for n = 1:N

                for i = set_thetaeq(jdx - 1):set_thetaeq(jdx - 1) + sizeclus(jdx - 1) - 1
                    Sinit(:, :, n) = Sinit(:, :, n) + ...
                        H(:, index_start(i):index_end(i), n) * Rxxs{i, n} * ...
                        H(:, index_start(i):index_end(i), n)';
                end

                cumrateinit(n) = (1 / cb) * real(log2(det(Sinit(:, :, n))));
            end

        end

        u_range = set_thetaeq(jdx):set_thetaeq(jdx) + sizeclus(jdx) - 1;
        bu_v_local = initialbu_v';
        known_vertices = initialinfo;
        bd_vertices = bu_v_local(u_range)';
        vertices = de2bi_local(0:2 ^ sizeclus(jdx) - 1, sizeclus(jdx)) .* bd_vertices;

        % check if achievable without time sharing
        if min(bd_vertices - bu_min(u_range)') >= -conv_tol
            info_row = known_vertices(1, :);
            info_row.frac = 1;
            info_row.clusterID = jdx;

            if jdx == 1
                Big_info = info_row;
            else
                Big_info = [Big_info; info_row];
            end

            FEAS_FLAG = 2;
            continue;
        end

        bd_V = 1;

        % iterate through permutations within cluster
        for idx = cumsize(jdx) + 1:cumsize(jdx) + sizeclus(jdx) - 1
            cumrate = zeros(sizeclus(jdx) + 1, N);
            cumrate(1, :) = cumrateinit;

            for n = 1:N
                rel_idx = 2;
                S = Sinit(:, :, n);

                for u = u_range
                    u_or = order(idx, u);
                    S = S + H(:, index_start(u_or):index_end(u_or), n) * ...
                        Rxxs{u_or, n} * H(:, index_start(u_or):index_end(u_or), n)';
                    cumrate(rel_idx, n) = (1 / cb) * real(log2(det(S)));
                    rel_idx = rel_idx + 1;
                end

            end

            bun_new = initialbun;
            bun_new(u_range, :) = diff(cumrate);
            bun_new(order(idx, :), :) = bun_new;
            bu_v_new = sum(bun_new, 2);

            bd_V = bd_V + 1;
            bd_vertices_extend = [bd_vertices; bu_v_new(u_range)'];
            known_vertices = [known_vertices; {bu_v_new', {bun_new}, order(idx, end:-1:1)}];
            vertices = [vertices; de2bi_local(0:2 ^ sizeclus(jdx) - 1, sizeclus(jdx)) .* bu_v_new(u_range)'];

            try
                tess = convhulln(vertices);
                vertices = vertices(unique(tess), :);
                bd_vertices = intersect(bd_vertices_extend, vertices, 'rows', 'stable');
                tess = convhulln(vertices);
            catch
                continue;
            end

            if size(bd_vertices, 1) < bd_V
                to_remove = setdiff(bd_vertices_extend, bd_vertices, 'rows');

                for k = size(known_vertices, 1):-1:1
                    bu_v_k = known_vertices.bu_v(k, :);

                    if any(ismember(to_remove, bu_v_k(u_range), 'rows'))
                        known_vertices(k, :) = [];
                    end

                end

                bd_V = size(bd_vertices, 1);
            end

            % check if target is inside convex hull
            if inhull(bu_min(u_range)', vertices, tess, conv_tol)
                FEAS_FLAG = 2;
                frac = bd_vertices' \ bu_min(u_range);
                a_v = find(frac >= err);
                info_rows = known_vertices(a_v, :);
                info_rows.frac = frac(a_v);
                info_rows.frac = info_rows.frac / sum(info_rows.frac);
                info_rows.clusterID = ones(length(a_v), 1) * jdx;
                bu_a(u_range) = bd_vertices(a_v, :)' * info_rows.frac;

                if jdx == 1
                    Big_info = info_rows;
                else
                    Big_info = [Big_info; info_rows];
                end

                break;
            end

        end

        if FEAS_FLAG == 0
            return;
        end

    end

    % ensure FEAS_FLAG is set if we made it through all clusters
    if isempty(Big_info)
        FEAS_FLAG = 0;
    elseif FEAS_FLAG == 1
        FEAS_FLAG = 2;
    end

end

function [info, bu_a, Rxxs] = restore_ordering(Big_info, bu_a_reord, Rxxs_reord, Itheta, N, U, SCALAR_FLAG)
    %RESTORE_ORDERING Restore original user ordering to all outputs

    info = Big_info;
    bd_V = size(info.bu_v, 1);

    % restore bun ordering; handle both cell and non-cell formats
    if iscell(info.bun)
        tempbun_cell = cell(bd_V, 1);

        for v = 1:bd_V
            bun_v = info.bun{v};

            if iscell(bun_v)
                bun_v = bun_v{1};
            end

            bun_restored = zeros(U, N);
            bun_restored(Itheta, :) = bun_v;
            tempbun_cell{v} = bun_restored;
        end

        info.bun = tempbun_cell;
    else
        tempbun = info.bun;
        tempbun(Itheta, :) = tempbun;
        info.bun = {tempbun};
    end

    % restore bu_a ordering
    bu_a = zeros(U, 1);
    bu_a(Itheta) = bu_a_reord;

    % restore bu_v ordering
    bu_v_restored = zeros(bd_V, U);

    for v = 1:bd_V
        bu_v_restored(v, Itheta) = info.bu_v(v, :);
    end

    info.bu_v = bu_v_restored;

    % restore decoding order; handle both cell and matrix formats
    if iscell(info.order)
        new_order = cell(bd_V, 1);

        for v = 1:bd_V
            ord_v = info.order{v};

            if iscell(ord_v)
                ord_v = ord_v{1};
            end

            new_order{v} = Itheta(ord_v)';
        end

        info.order = new_order;
    else
        tmporder = zeros(bd_V, U);

        for v = 1:bd_V
            tmporder(v, :) = Itheta(info.order(v, :))';
        end

        info.order = mat2cell(tmporder, ones(1, bd_V), U);
    end

    % restore Rxxs ordering
    Rxxs = cell(U, N);

    for n = 1:N

        for u = 1:U
            Rxxs{Itheta(u), n} = Rxxs_reord{u, n};
        end

    end

end

function b = de2bi_local(d, n)
    %DE2BI_LOCAL Convert decimal to binary matrix
    % avoids dependency on communications toolbox
    m = length(d);
    b = zeros(m, n);

    for i = 1:m
        val = d(i);

        for j = 1:n
            b(i, j) = mod(val, 2);
            val = floor(val / 2);
        end

    end

end
