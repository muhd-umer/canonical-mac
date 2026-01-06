% function [FEAS_FLAG, bu_a, info] = admMAC_cvx(H, Lxu, bu, Eu, cb)
%
% admMAC_rate_region determines whether the target rate vector bu is
% feasible for (noise-whitened) channel H and energy/symbol Eu via rate region
%
%   Input arguments:
%       - H: Ly-by-Lx-by-N channel matrix. H(:,:,n) denotes the channel for
%           the n-th tone.
%       - Lxu: number of transmit antennas of each user. It can be either a
%           scalar or a length-U vector. If it is a scalar, every user has
%           Lxu transmit antennas; otherwise user u has Lxu(u) transmit
%           antennas.
%       - bu: target rate of each user, length-U vector.
%       - Eu: Energy/symbol on each user, length-U vector.
%   Outputs:
%       - FEAS_FLAG: indicator of achievability. FEAS_FLAG=0 if the target
%           is not achievable; FEAS_FLAG=1 if the target is achievable by a
%           single ordering; FEAS_FLAG=2 if the target is achievable by
%           time-sharing
%       - bu_a: U-by-1 vector showing achieved sum rate of each user.
%       - Rxxs: U-by-N cell array containing Rxx(u,n)'s. If the rate target
%               is infeasible, output 0.
%       - Eun: U-by-N matrix showing users' transmit energy on each tone.
%               If infeasible, output 0.
%       - theta: U-by-1 Lagrangian multiplier w.r.t. target rates
%       - w: U-by-1 Lagrangian multiplier w.r.t. energy constraints
%       - info: various length output depending on FEAS_FLAG
%           --if FEAS_FLAG=0: empty
%           --if FEAS_FLAG=1: 1-by-3 table containing
%               {bu_v, bun, order} corresponds to the single vertex
%           --if FEAS_FLAG=2: v-by-5 cell array, with each row representing
%               a time-shared vertex {bu_v, bun, order, frac, clusterID}
%
%      info's row entries in detail (one row for each vertex shared
%       - bu_v: 1-by-U vector showing achieved rate of each user
%       - bun: a cell containing U-by-N matrix showing users' rate on each
%               tone. If infeasible, output 0.
%       - order: 1-by-U vector showing decoding order
%       - frac: fraction of dimensions for each vertex in time share (FF =2
%           ONLY), per cluster
%       -- clusterID 1 has largest common-theta, cluster 2 has second largest
%          common-theta group, and so on
%
%      Subroutines called
%       - maxRMAC_cvx
%       - maxRMACMIMO_cvx
%************************************************************************
function [FEAS_FLAG, bu_a, Rxxs, Eun, theta, w, info] = admMAC_cvx(H, Lxu, bu, Eu, cb)

    tstart = tic;
    [Ly, ~, N] = size(H);
    Eu = reshape(Eu, [], 1);
    bu = reshape(bu, [], 1);
    bu_min = bu;
    %Rxxs = cell(1,N);
    U = length(Eu);
    bu_a = zeros(1, U);
    theta = ones(U, 1) + 0.05 * (1:U)';
    A = eye(U) * U;
    count = 0;
    % error tolerance
    err = 1e-6;
    % tolerance of convex hull boundary
    conv_tol = 1e-6;
    FEAS_FLAG = 0;

    SCALAR_FLAG = (max(Lxu) == 1);

    while 1
        count = count + 1;

        % ------------------------ next vertex ----------
        % get the next vertex
        if SCALAR_FLAG
            [Eun, w, bun] = maxRMAC_cvx(H, Eu, theta, cb);
            Rxxstemp = num2cell(Eun);
        else
            [Rxxs, Eun, w, bun] = maxRMACMIMO_cvx(H, Lxu, Eu, theta, cb);
            Rxxstemp = Rxxs;
        end

        bu_v = sum(bun, 2);
        g = bu_v - bu;
        bu_a = bu_v;

        if theta' * g < -err % infeasible criteria
            FEAS_FLAG = 0;
            Rxxs = 0;
            Eun = 0;
            info = {};
            return
        end

        if min(g) >= 0 % achievable by current vertex
            FEAS_FLAG = 1;
            Rxxs = Rxxstemp;
            [~, Itheta] = sort(theta);
            info = table(bu_v', {bun}, Itheta', 'VariableNames', {'bu_v', 'bun', 'order'});
            return
        end

        % check if there are any equal-theta clusters
        [stheta, Itheta] = sort(theta, 'descend');
        delta = -diff([stheta; 0]);
        spots = delta < max(err, 1e-2 * norm(theta)); % whether a user (sorted by theta) is in the same cluster as the next user
        spots(end) = 0;

        if ~any(spots) % no vertex-share, update theta and continue to next iteration
            updateEllipsoid
            continue;
        end

        % ---------------------TWO OR MORE EQUAL-THETA USERS ------------------
        % This point of while loop is only reached if there are equal-theta users.
        % This section forms the equal-theta clusters and check is sum(bu_v)
        % of each cluster meets sum(bu_min) target
        % ---------------------------------------------------------------

        sizeclus = []; % size of each cluster
        numclus = 0; % number of clusters (>1 equal-theta groups)
        set_thetaeq = []; % index of the first entry of each cluster
        CLUSTER_FEAS_FLAG = 1; % whether the current vertex / cluster of vertex meets bu_min target
        flagclus = 0;

        for i = 1:U

            if spots(i) && ~flagclus % create a new set
                numclus = numclus + 1;
                set_thetaeq = [set_thetaeq, i];
                sizeclus = [sizeclus, 1];
                flagclus = 1;
            elseif spots(i) % add user to an existing set
                sizeclus(end) = sizeclus(end) + 1;
            elseif flagclus % last user to add
                sizeclus(end) = sizeclus(end) + 1;
                flagclus = 0;
                us = Itheta(set_thetaeq(end):set_thetaeq(end) + sizeclus(end) - 1);

                if sum(bu_v(us) - bu(us)') < -err * length(us)
                    CLUSTER_FEAS_FLAG = 0;
                    break
                end

            else % user not in a cluster
                flagclus = 0;

                if bu_v(Itheta(i)) - bu(Itheta(i)) < -err
                    CLUSTER_FEAS_FLAG = 0;
                    break
                end

            end

        end

        if ~CLUSTER_FEAS_FLAG % update theta and continue to next iteration
            updateEllipsoid
            continue
        end

        %---------------------TWO OR MORE EQUAL-THETA USERS --------------
        % This section vertex shares for the equal-theta clusters.
        %
        % Only equal-theta users vertices' rate subsets are evaluated.
        % sizeset is the size of a cluster = # of equal-theta users
        % There is no need for U! orders to be searched (which is usually a much
        % larger number, just the sizeclus within each cluster, while sizeset+1 is
        % the total number of orders that are searched (over all clusters).
        %
        % order changes with respect to Itheta ONLY within the clusters so that
        % different vertices can be shared to get the correct data rate bu_min
        % (bu input) instead of just sum(bu_min)
        % ---------------------------------------------------------------

        % --------- REORDER USERS ACCORDING TO THETA --------------------
        % This ordering will need eventually to be reversed before returing
        % from this function.
        % ---------------------------------------------------------------
        bu_a = bu_a(Itheta);
        bu_v = bu_v(Itheta);
        bun = bun(Itheta, :);
        bu_min = bu_min(Itheta);
        % handle scalar Lxu
        if isscalar(Lxu)
            Lxu_vec = ones(1, U) * Lxu;
        else
            Lxu_vec = reshape(Lxu, 1, []);
        end

        index_end = cumsum(Lxu_vec);
        index_start = [1, index_end(1:end - 1) + 1];
        hidx = [];

        for i = Itheta'
            hidx = [hidx, index_start(i):index_end(i)];
        end

        Lxu_vec = Lxu_vec(Itheta);
        index_end = cumsum(Lxu_vec);
        index_start = [1, index_end(1:end - 1) + 1];
        H = H(:, hidx, :);

        for tone = 1:N
            Rxxstemp(:, tone) = Rxxstemp(Itheta, tone);
        end

        % ----------- CREATE INFO TABLE & ADD FIRST VERTEX ----------
        initialbu_v = bu_v'; % save this value as it is needed if more than 1 cluster
        initialbun = bun;
        firstvertices = de2bi(0:2 ^ U - 1) .* initialbu_v;

        initialinfo = table(bu_v', {bun}, U:-1:1, 'VariableNames', {'bu_v', 'bun', 'order'});

        % format the clusters and permuted orders
        order = repmat(1:U, sum(sizeclus - 1), 1);
        cumsize = cumsum([0, sizeclus - 1]);

        for jdx = 1:numclus
            u_range = set_thetaeq(jdx):set_thetaeq(jdx) + sizeclus(jdx) - 1; % users in cluster jdx

            for i = 1:sizeclus(jdx) - 1
                order(cumsize(jdx) + i, u_range) = circshift(u_range, i); % all permutated orders (excluding 1,2,...U in each cluster)
            end

        end

        % ----- For each cluster -----------
        for jdx = 1:numclus
            FEAS_FLAG = 0;

            if jdx == 1
                Sinit = repmat(eye(Ly), 1, 1, N);
                cumrateinit = zeros(1, N);

                for n = 1:N

                    for i = 1:set_thetaeq(1) - 1
                        Sinit(:, :, n) = Sinit(:, :, n) + H(:, index_start(i):index_end(i), n) * Rxxstemp{i, n} * H(:, index_start(i):index_end(i), n)';
                    end

                    cumrateinit(n) = (1 / cb) * real(log2(det(Sinit(:, :, n))));
                end

            else

                for n = 1:N

                    for i = set_thetaeq(jdx - 1):set_thetaeq(jdx - 1) + sizeclus(jdx - 1) - 1
                        Sinit(:, :, n) = Sinit(:, :, n) + H(:, index_start(i):index_end(i), n) * Rxxstemp{i, n} * H(:, index_start(i):index_end(i), n)';
                    end

                    cumrateinit(n) = (1 / cb) * real(log2(det(Sinit(:, :, n))));
                end

            end

            u_range = set_thetaeq(jdx):set_thetaeq(jdx) + sizeclus(jdx) - 1;
            bu_v = initialbu_v';
            known_vertices = initialinfo; % detailed info of boundary vertices
            bd_vertices = bu_v(u_range)'; % track critical boundary vertices
            vertices = de2bi(0:2 ^ sizeclus(jdx) - 1) .* bd_vertices;

            if min(bd_vertices - bu_min(u_range)') >= -conv_tol % achievable w/o time share
                info = known_vertices;
                info.frac = 1;
                info.clusterID = jdx;

                if jdx == 1
                    Big_info = info;
                else
                    Big_info = [Big_info; info];
                end

                continue;
            end

            bd_V = 1;

            for idx = cumsize(jdx) + 1:cumsize(jdx) + sizeclus(jdx) - 1 % add more vertices in cluster jdx
                cumrate = zeros(sizeclus(jdx) + 1, N);
                cumrate(1, :) = cumrateinit;

                for n = 1:N
                    rel_idx = 2;
                    S = Sinit(:, :, n);

                    for u = u_range
                        u_or = order(idx, u);
                        S = S + H(:, index_start(u_or):index_end(u_or), n) * Rxxstemp{u_or, n} * H(:, index_start(u_or):index_end(u_or), n)';
                        cumrate(rel_idx, n) = (1 / cb) * real(log2(det(S)));
                        rel_idx = rel_idx + 1;
                    end

                end

                bun = initialbun;
                bun(u_range, :) = diff(cumrate);
                bun(order(idx, :), :) = bun;
                bu_v = sum(bun, 2);

                bd_V = bd_V + 1;
                bd_vertices_extend = [bd_vertices; bu_v(u_range)'];
                known_vertices = [known_vertices; {bu_v', {bun}, order(idx, end:-1:1)}];
                vertices = [vertices; de2bi(0:2 ^ sizeclus(jdx) - 1) .* bu_v(u_range)'];
                tess = convhulln(vertices);
                vertices = vertices(unique(tess), :);
                bd_vertices = intersect(bd_vertices_extend, vertices, 'rows', 'stable');
                tess = convhulln(vertices);
                % delete inner vertices
                if size(bd_vertices, 1) < bd_V
                    to_remove = setdiff(bd_vertices_extend, bd_vertices, 'rows');
                    known_vertices(ismember(known_vertices.bu_v(u_range), to_remove, 'rows'), :) = [];
                    bd_V = size(bd_vertices, 1);
                end

                if inhull(bu_min(u_range)', vertices, tess, conv_tol) % bu_min achievable by time-share
                    FEAS_FLAG = 2;
                    frac = bd_vertices' \ bu_min(u_range);
                    a_v = find(frac >= err); % active vertices in time-share
                    info = known_vertices(a_v, :);
                    info.frac = frac(a_v);
                    info.frac = info.frac / sum(info.frac);
                    info.clusterID = ones(length(a_v), 1) * jdx;
                    bu_a(u_range) = bd_vertices(a_v, :)' * info.frac;
                    break;
                end

            end

            if FEAS_FLAG == 0
                break;
            end

            % write big info table
            if jdx == 1
                Big_info = info;
            else
                Big_info = [Big_info; info];
            end

        end

        if FEAS_FLAG ~= 0
            break;
        end

    end

    info = Big_info;
    bd_V = size(info.bu_v, 1);

    % ----------- RESTORE ORIGINAL ORDER TO ALL ----------------
    %
    tempbun = reshape(cell2mat(info.bun), U, bd_V, N);
    tempbun(Itheta, :, :) = tempbun;
    tempbun = permute(tempbun, [2 1 3]);
    info.bun = mat2cell(tempbun, [ones(1, bd_V)], U, N);
    %
    bu_a(Itheta) = bu_a;
    info.bu_v(:, Itheta) = info.bu_v;
    %
    tmporder = Itheta(info.order);
    info.order = mat2cell(tmporder, ones(1, bd_V), U);

    for tone = 1:N
        Rxxstemp(Itheta, :) = Rxxstemp;
    end

    Rxxs = Rxxstemp;
    toc(tstart)

    function updateEllipsoid
        tmp = A * g / sqrt(g' * A * g);
        theta = theta - 1 / (U + 1) * tmp;
        A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * (tmp * tmp'));
        ind = find(theta < zeros(U, 1));

        while ~isempty(ind) % helps numerically, as thetas > 0
            g = zeros(U, 1);
            g(ind(1)) = -1;
            tmp = A * g / sqrt(g' * A * g);
            theta = theta - 1 / (U + 1) * tmp;
            A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * (tmp * tmp'));
            ind = find(theta < zeros(U, 1));
        end

    end

end
