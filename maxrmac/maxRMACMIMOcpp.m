function [Rxxs, Eun, w, bun] = maxRMACMIMOcpp(H, Lxu, Eu, theta, cb)
    %MAXRMACMIMOCPP C++ accelerated weighted rate sum maximization
    %   [Rxxs, Eun, w, bun] = MAXRMACMIMOCPP(H, Lxu, Eu, theta, cb) provides
    %   identical functionality to maxRMACMIMO but uses optimized C++ code.
    %
    %   This function requires the compiled MEX file 'maxrmac_mex' to be on
    %   the MATLAB path.
    %
    %   Inputs:
    %       H       channel tensor [Ly, sum(Lxu), N] with user antennas stacked.
    %       Lxu     antennas per user (scalar or 1-by-U).
    %       Eu      per-user energy constraints (length-U vector).
    %       theta   per-user rate weights (length-U vector).
    %       cb      baseband type: 1 for complex, 2 for real.
    %
    %   Outputs:
    %       Rxxs    U-by-N cell array of covariance matrices, or
    %               Lxu_max-by-Lxu_max-by-U-by-N tensor if Lxu is scalar.
    %       Eun     U-by-N energy allocation.
    %       w       U-by-1 dual variables for energy constraints.
    %       bun     U-by-N per-user per-tone rates (bits).
    %
    %   See also: maxRMACMIMO

    [Ly, Ltot, N] = size(H);
    theta = reshape(theta, [], 1);
    Eu = reshape(Eu, [], 1);
    U = length(theta);

    if length(Eu) ~= U
        error('Eu must have length U=%d', U);
    end

    if cb ~= 1 && cb ~= 2
        error('cb must be 1 (complex) or 2 (real)');
    end

    if isscalar(Lxu)
        Lxu_vec = ones(1, U) * Lxu;
        uniform_flag = true;
    else
        Lxu_vec = reshape(Lxu, 1, []);
        uniform_flag = false;
    end

    if length(Lxu_vec) ~= U
        error('Lxu must be a scalar or a length-U vector');
    end

    if sum(Lxu_vec) ~= Ltot
        error('sum(Lxu)=%d must equal size(H,2)=%d', sum(Lxu_vec), Ltot);
    end

    if isreal(H)
        H = complex(H);
    end

    [Rxx_cell, Eun, w, bun] = maxrmac_mex(H, Lxu_vec, Eu, theta, cb);

    N = size(bun, 2);

    Lxu_max = max(Lxu_vec);

    if uniform_flag
        Rxxs = zeros(Lxu_max, Lxu_max, U, N);

        for u = 1:U

            for n = 1:N
                Rxxs(1:Lxu_vec(u), 1:Lxu_vec(u), u, n) = Rxx_cell{u, n};
            end

        end

    else
        Rxxs = Rxx_cell;
    end

end
