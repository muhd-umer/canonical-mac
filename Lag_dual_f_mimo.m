function [f, bun, Rxx_all] = Lag_dual_f_mimo(H, theta, w, bu_min, Lx, idx_start, idx_end, R_init_all)
    %LAG_DUAL_F_MIMO Compute Lagrange dual function for MIMO MAC
    %   [f, bun, Rxx_all] = LAG_DUAL_F_MIMO(H, theta, w, bu_min, Lx, idx_start, idx_end)
    %   evaluates the dual objective by optimizing each tone independently via
    %   MINPTONE_MIMO, returning both the per-user rates in nats and the stacked
    %   covariance matrices.
    %   If R_INIT_ALL is provided, it warm-starts the per-tone optimizers.

    [Ly, Ltot, N] = size(H);
    U = length(Lx);

    f = 0;
    bun = zeros(U, N);
    Rxx_all = zeros(Ltot, Ltot, N);

    if nargin < 8
        R_init_all = [];
    end

    % optimize each tone
    for n = 1:N

        if isempty(R_init_all)
            R_init_n = [];
        else
            R_init_n = R_init_all(:, :, n);
        end

        [temp_f, b_n, Rxx_n] = minPtone_mimo(H(:, :, n), theta, w, Lx, idx_start, idx_end, R_init_n);
        f = f + temp_f;
        bun(:, n) = b_n;
        Rxx_all(:, :, n) = Rxx_n;
    end

    f = f - theta' * bu_min;
end
