function f = eval_f_mimo(theta, H, Rxx, w, Lx, idx_start, idx_end, ind, Ly)
    % EVAL_F_MIMO - Evaluates the Lagrangian objective for MIMO

    U = length(Lx);
    theta_diff = 0.5 * (theta - [theta(2:U); 0]);

    RYY = zeros(Ly, Ly, U);
    RYY(:, :, 1) = eye(Ly);

    for u = 1:U
        ant_idx = idx_start(ind(u)):idx_end(ind(u));
        H_u = H(:, ant_idx);
        R_u = Rxx(ant_idx, ant_idx);

        if u == 1
            RYY(:, :, u) = eye(Ly) + H_u * R_u * H_u';
        else
            RYY(:, :, u) = RYY(:, :, u - 1) + H_u * R_u * H_u';
        end

    end

    f = 0;

    for u = 1:U
        ant_idx = idx_start(ind(u)):idx_end(ind(u));
        R_u = Rxx(ant_idx, ant_idx);

        f = f + theta_diff(u) * log(det(RYY(:, :, u)));
        f = f + log(det(R_u) +1e-10); % Barrier for positive definiteness
        f = f - w(u) * trace(R_u);
    end

end
