function Rxxm = bc2mac(Rxxb, Hmac)
    %BC2MAC Convert BC covariance matrices to the dual MAC covariances
    %   Rxxm = BC2MAC(Rxxb, Hmac) maps a set of broadcast-channel covariance
    %   matrices to the covariance matrices of the dual MAC that achieve the
    %   same rate point under the standard BC/MAC duality transform.
    %
    %   Inputs:
    %       Rxxb    Ly-by-Ly-by-U array where Rxxb(:, :, u) is the BC
    %               covariance matrix for user u.
    %       Hmac    Ly-by-Lx-by-U tensor of noise-whitened MAC channel
    %               matrices. The corresponding dual BC channel is
    %               conj(permute(Hmac(:, :, end:-1:1), [2, 1, 3])).
    %
    %   Output:
    %       Rxxm    Ly-by-Ly-by-U array of dual MAC covariance matrices.
    %               The user order is restored to match the input ordering.
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    [Ly, ~, U] = size(Hmac);
    H = conj(permute(Hmac(:, :, end:-1:1), [2, 1, 3]));
    [Lx, ~, ~] = size(H);

    % A stores BC-side noise-plus-crosstalk terms, B stores MAC-side terms
    A = zeros(Lx, Lx, U);
    B = zeros(Ly, Ly, U);
    Rxxm = zeros(Ly, Ly, U);
    Rxxb_sum = zeros(Lx, Lx);

    % construct cumulative BC covariance terms
    A(:, :, 1) = eye(Lx);

    for u = 1:(U - 1)
        Rxxb_sum = Rxxb_sum + Rxxb(:, :, u);
        A(:, :, u + 1) = eye(Lx) + H(:, :, u + 1) * Rxxb_sum * H(:, :, u + 1)';
    end

    B(:, :, U) = eye(Ly);
    Hmac = Hmac(:, :, U:-1:1);

    for u = U:-1:1
        temp_A = pinv(sqrtm(A(:, :, u)));
        temp_B = pinv(sqrtm(B(:, :, u)));

        [F, ~, M] = svd(temp_B' * Hmac(:, :, u) * temp_A', 'econ');

        Rxxm(:, :, u) = temp_A * M * F' * sqrtm(B(:, :, u))' * Rxxb(:, :, u) ...
            * sqrtm(B(:, :, u)) * F * M' * temp_A';

        if u ~= 1
            B(:, :, u - 1) = B(:, :, u) + Hmac(:, :, u) * Rxxm(:, :, u) * Hmac(:, :, u)';
        end

    end

    Rxxm(:, :, U:-1:1) = Rxxm;
end
