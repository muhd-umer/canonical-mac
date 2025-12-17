function [Rxx_opt, E_opt, state_out] = solve_tone(H_blocks, Lxu, delta, w, Ly, state_in)
    %SOLVE_TONE Solve single-tone optimization using L-BFGS
    %   [Rxx_opt, E_opt, state_out] = SOLVE_TONE(H_blocks, Lxu, delta, w, Ly, state_in)
    %   solves the per-tone rate maximization problem using L-BFGS optimization
    %   over Cholesky factors for automatic PSD constraint satisfaction.
    %
    %   Maximizes: sum_u delta(u) * log|S_u| - sum_u w(u) * trace(Rxx_u)
    %   where S_u = I + sum_{j=1}^u H_j * Rxx_j * H_j'
    %
    %   Inputs:
    %       H_blocks    U-by-1 cell array of per-user channel matrices
    %       Lxu         1-by-U antennas per user (sorted order)
    %       delta       U-by-1 rate weight differences
    %       w           U-by-1 per-user Lagrange multipliers for energy constraints
    %       Ly          number of receiver antennas
    %       state_in    (optional) warm-start state from previous solve
    %
    %   Outputs:
    %       Rxx_opt     U-by-1 cell array of optimal covariance matrices
    %       E_opt       U-by-1 vector of per-user energies
    %       state_out   state structure for warm-starting next solve

    U = length(Lxu);

    % l-bfgs parameters
    max_iter = 300;
    tol_grad = 1e-5;
    m_hist = 8;
    step_init = 1.0;
    step_min = 1e-12;
    beta = 0.5;
    armijo_c = 1e-4;
    max_ls = 30;

    % initialize B factors
    B_blocks = cell(U, 1);

    if isstruct(state_in) && isfield(state_in, 'B_blocks') && numel(state_in.B_blocks) == U
        ok = true;

        for u = 1:U
            Bu = state_in.B_blocks{u};

            if isempty(Bu) || any(size(Bu) ~= [Lxu(u), Lxu(u)])
                ok = false;
                break;
            end

        end

        if ok
            B_blocks = state_in.B_blocks;
        end

    end

    if isempty(B_blocks{1})

        for u = 1:U
            B_blocks{u} = sqrt(1e-6) * eye(Lxu(u));
        end

    end

    % lbfgs state
    S_hist = {};
    Y_hist = {};

    [phi, grad_blocks, R_blocks] = eval_gradobj(B_blocks, H_blocks, delta, w, Ly);

    if isinf(phi)

        for u = 1:U
            B_blocks{u} = sqrt(1e-6) * eye(Lxu(u));
        end

        S_hist = {};
        Y_hist = {};
        [phi, grad_blocks, R_blocks] = eval_gradobj(B_blocks, H_blocks, delta, w, Ly);
    end

    grad = pack_blocks(grad_blocks, Lxu);
    x = pack_blocks(B_blocks, Lxu);

    for iter = 1:max_iter

        if norm(grad) < tol_grad
            break;
        end

        dir = lbfgs_direction(grad, S_hist, Y_hist);
        dir_deriv = real(grad' * dir);

        if dir_deriv >= 0
            dir = -grad;
            dir_deriv = real(grad' * dir);
            S_hist = {};
            Y_hist = {};
        end

        step = step_init;
        accepted = false;
        x_new = x;
        phi_new = phi;
        grad_new = grad;
        B_new = B_blocks;
        R_new = R_blocks;

        for ls = 1:max_ls
            x_trial = x + step * dir;
            B_trial = unpack_blocks(x_trial, Lxu);
            [phi_trial, grad_trial_blocks, R_trial] = eval_gradobj(B_trial, H_blocks, delta, w, Ly);

            if ~isinf(phi_trial) && phi_trial <= phi + armijo_c * step * dir_deriv
                accepted = true;
                x_new = x_trial;
                phi_new = phi_trial;
                grad_new = pack_blocks(grad_trial_blocks, Lxu);
                B_new = B_trial;
                R_new = R_trial;
                break;
            end

            step = step * beta;

            if step < step_min
                break;
            end

        end

        if ~accepted
            dir = -grad;
            dir_deriv = real(grad' * dir);
            step = step_init;

            for ls = 1:max_ls
                x_trial = x + step * dir;
                B_trial = unpack_blocks(x_trial, Lxu);
                [phi_trial, grad_trial_blocks, R_trial] = eval_gradobj(B_trial, H_blocks, delta, w, Ly);

                if ~isinf(phi_trial) && phi_trial <= phi + armijo_c * step * dir_deriv
                    accepted = true;
                    x_new = x_trial;
                    phi_new = phi_trial;
                    grad_new = pack_blocks(grad_trial_blocks, Lxu);
                    B_new = B_trial;
                    R_new = R_trial;
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

            S_hist = {};
            Y_hist = {};
        end

        s = x_new - x;
        y = grad_new - grad;
        sy = real(s' * y);

        if sy > 1e-12
            S_hist{end + 1} = s;
            Y_hist{end + 1} = y;

            if length(S_hist) > m_hist
                S_hist = S_hist(2:end);
                Y_hist = Y_hist(2:end);
            end

        else
            S_hist = {};
            Y_hist = {};
        end

        x = x_new;
        phi = phi_new;
        grad = grad_new;
        B_blocks = B_new;
        R_blocks = R_new;
    end

    Rxx_opt = cell(U, 1);
    E_opt = zeros(U, 1);

    for u = 1:U
        Ru = B_blocks{u} * B_blocks{u}';
        Ru = 0.5 * (Ru + Ru');
        Rxx_opt{u} = Ru;
        E_opt(u) = real(trace(Ru));
    end

    state_out = struct();
    state_out.B_blocks = B_blocks;
end

function dir = lbfgs_direction(grad, S_hist, Y_hist)
    %LBFGS_DIRECTION Compute L-BFGS search direction

    if isempty(S_hist)
        dir = -grad;
        return;
    end

    k = length(S_hist);
    alpha = zeros(k, 1);
    rho = zeros(k, 1);
    q = grad;

    for i = k:-1:1
        s = S_hist{i};
        y = Y_hist{i};
        denom = real(y' * s);

        if denom <= 1e-12
            rho(i) = 0;
            alpha(i) = 0;
            continue;
        end

        rho(i) = 1 / denom;
        alpha(i) = rho(i) * real(s' * q);
        q = q - alpha(i) * y;
    end

    sk = S_hist{end};
    yk = Y_hist{end};
    denom = real(yk' * yk);

    if denom <= 1e-12
        H0 = 1;
    else
        H0 = real(sk' * yk) / denom;
    end

    r = H0 * q;

    for i = 1:k

        if rho(i) == 0
            continue;
        end

        y = Y_hist{i};
        s = S_hist{i};
        beta = rho(i) * real(y' * r);
        r = r + s * (alpha(i) - beta);
    end

    dir = -r;
end

function vec = pack_blocks(blocks, Lxu)
    %PACK_BLOCKS Pack cell array of matrices into a vector

    U = length(Lxu);
    total = 0;

    for u = 1:U
        total = total + 2 * Lxu(u) ^ 2;
    end

    vec = zeros(total, 1);
    offset = 0;

    for u = 1:U
        B = blocks{u};
        len = numel(B);
        vec(offset + (1:len)) = real(B(:));
        vec(offset + len + (1:len)) = imag(B(:));
        offset = offset + 2 * len;
    end

end

function blocks = unpack_blocks(vec, Lxu)
    %UNPACK_BLOCKS Unpack vector into cell array of matrices

    U = length(Lxu);
    blocks = cell(U, 1);
    offset = 0;

    for u = 1:U
        Lu = Lxu(u);
        len = Lu ^ 2;
        re = vec(offset + (1:len));
        im = vec(offset + len + (1:len));
        blocks{u} = reshape(re + 1j * im, Lu, Lu);
        offset = offset + 2 * len;
    end

end
