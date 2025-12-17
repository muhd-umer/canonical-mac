function [phi, grad_blocks, R_blocks] = eval_gradobj(B_blocks, H_blocks, delta, w, Ly)
    %EVAL_GRADOBJ Evaluate objective and gradient for L-BFGS optimization
    %   [phi, grad_blocks, R_blocks] = EVAL_GRADOBJ(B_blocks, H_blocks, delta, w, Ly)
    %   evaluates the negative dual objective (for minimization) and computes
    %   gradients with respect to the Cholesky factors B.
    %
    %   Inputs:
    %       B_blocks    U-by-1 cell array of Cholesky factors (Rxx = B*B')
    %       H_blocks    U-by-1 cell array of per-user channel matrices
    %       delta       U-by-1 rate weight differences
    %       w           U-by-1 per-user Lagrange multipliers for energy constraints
    %       Ly          number of receiver antennas
    %
    %   Outputs:
    %       phi         scalar negative objective (for minimization)
    %       grad_blocks U-by-1 cell array of gradients w.r.t. B_u
    %       R_blocks    U-by-1 cell array of covariance matrices Rxx_u

    U = numel(H_blocks);
    eye_Ly = eye(Ly);
    S = eye_Ly;
    logdet_S = zeros(U, 1);
    invS = cell(U, 1);
    R_blocks = cell(U, 1);

    energy_term = 0;

    % forward pass; compute S_u and log|S_u|
    for u = 1:U
        Hu = H_blocks{u};
        Bu = B_blocks{u};
        Ru = Bu * Bu';
        R_blocks{u} = Ru;

        HRH = Hu * Ru * Hu';
        S = S + HRH;
        S = 0.5 * (S + S');

        [L, flag] = chol(S, 'lower');

        if flag ~= 0
            [L, flag] = chol(S +1e-9 * eye_Ly, 'lower');
        end

        if flag ~= 0
            phi = Inf;
            grad_blocks = [];
            R_blocks = [];
            return;
        end

        logdet_S(u) = 2 * sum(log(diag(L)));
        Linv = L \ eye_Ly;
        invS{u} = Linv' * Linv;

        energy_term = energy_term + w(u) * sum(real(conj(Bu(:)) .* Bu(:)));
    end

    F = sum(delta .* logdet_S) - real(energy_term);
    phi = -F; % minimize negative

    % backward pass; compute gradients
    grad_blocks = cell(U, 1);
    tail = zeros(Ly, Ly);

    for u = U:-1:1
        tail = tail + delta(u) * invS{u};
        Hu = H_blocks{u};
        Bu = B_blocks{u};
        Lu = size(Bu, 1);

        % gradient of F w.r.t. Rxx_u
        grad_R = Hu' * tail * Hu - w(u) * eye(Lu);
        grad_R = 0.5 * (grad_R + grad_R');

        % gradient of F w.r.t. Bu (from chain rule -> d/dB = 2 * grad_R * B)
        grad_B = 2 * grad_R * Bu;

        % negate for minimization
        grad_blocks{u} = -grad_B;
    end

end
