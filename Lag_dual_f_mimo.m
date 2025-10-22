function [f, bun, Rxx_all] = Lag_dual_f_mimo(H, theta, w, bu_min, Lx, idx_start, idx_end)
    % LAG_DUAL_F_MIMO - Computes Lagrange dual function for MIMO MAC
    %
    % Optimizes over each tone independently

    [Ly, Ltot, N] = size(H);
    U = length(Lx);

    f = 0;
    bun = zeros(U, N);
    Rxx_all = zeros(Ltot, Ltot, N);

    % Optimize each tone
    for n = 1:N
        [temp_f, b_n, Rxx_n] = minPtone_mimo(H(:, :, n), theta, w, Lx, idx_start, idx_end);
        f = f + temp_f;
        bun(:, n) = b_n;
        Rxx_all(:, :, n) = Rxx_n;
    end

    f = f - theta' * bu_min;
end
