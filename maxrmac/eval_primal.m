function [Rxx_all, Eun_all, states] = eval_primal(H, Lxu, idx_start, idx_end, delta, w, states)
    %EVAL_PRIMAL Evaluate primal solution by solving each tone
    %   [Rxx_all, Eun_all, states] = EVAL_PRIMAL(H, Lxu, idx_start, idx_end, delta, w, states)
    %   calls solve_tone for each frequency tone and aggregates results.
    %
    %   Inputs:
    %       H           Ly-by-Ltot-by-N channel tensor
    %       Lxu         1-by-U antennas per user (sorted order)
    %       idx_start   1-by-U starting antenna indices (sorted order)
    %       idx_end     1-by-U ending antenna indices (sorted order)
    %       delta       U-by-1 rate weight differences
    %       w           U-by-1 per-user Lagrange multipliers for energy constraints
    %       states      N-by-1 cell array of per-tone warm-start states
    %
    %   Outputs:
    %       Rxx_all     U-by-N cell array of covariance matrices
    %       Eun_all     U-by-N energy allocation matrix
    %       states      N-by-1 cell array of updated per-tone states

    [Ly, ~, N] = size(H);
    U = length(Lxu);
    Rxx_all = cell(U, N);
    Eun_all = zeros(U, N);

    for n = 1:N
        Hn = H(:, :, n);
        H_blocks = cell(U, 1);

        for u = 1:U
            cols = idx_start(u):idx_end(u);
            H_blocks{u} = Hn(:, cols);
        end

        state_in = [];

        if ~isempty(states) && numel(states) >= n
            state_in = states{n};
        end

        [Rxx_n, E_n, state_out] = solve_tone(H_blocks, Lxu, delta, w, Ly, state_in);
        states{n} = state_out;

        for u = 1:U
            Rxx_all{u, n} = Rxx_n{u};
            Eun_all(u, n) = E_n(u);
        end

    end

end
