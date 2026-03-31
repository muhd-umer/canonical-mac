function Rxxb = mac2bc(Rxxm, Hmac)
    %MAC2BC Convert MAC covariance matrices to the dual BC covariances
    %   Rxxb = MAC2BC(Rxxm, Hmac) maps a set of MAC covariance matrices to
    %   the covariance matrices of the dual BC that achieve the same rate
    %   point under the standard BC/MAC duality transform.
    %
    %   Inputs:
    %       Rxxm    Lx-by-Lx-by-U array of MAC covariance matrices. Users are
    %               interpreted in the decoding order induced by the input.
    %       Hmac    Ly-by-Lx-by-U tensor of noise-whitened MAC channel
    %               matrices for the same user ordering.
    %
    %   Output:
    %       Rxxb    Ly-by-Ly-by-U array of dual BC covariance matrices. The
    %               output ordering matches the dual BC convention implied by
    %               the reversal used internally.

    [Ly, Lx, U] = size(Hmac);

    % reverse ordering to align the implementation with the textbook form.
    Hmac = Hmac(:, :, U:-1:1);
    Rxxm = Rxxm(:, :, U:-1:1);

    Rxxb = zeros(Ly, Ly, U);
    Rxxb_total = zeros(Ly, Ly);

    % A stores BC-side noise-plus-crosstalk terms, B stores MAC-side terms.
    A = zeros(Lx, Lx, U);
    B = zeros(Ly, Ly, U);

    B(:, :, U) = eye(Ly);

    for k = U:-1:2
        B(:, :, k - 1) = B(:, :, k) + Hmac(:, :, k) * Rxxm(:, :, k) * Hmac(:, :, k)';
    end

    A(:, :, 1) = eye(Lx);

    for k = 1:U
        temp_A = inv(sqrtm(A(:, :, k)));
        temp_B = inv(sqrtm(B(:, :, k)));

        [F, ~, M] = svd(temp_B' * Hmac(:, :, k) * temp_A', 'econ');
        Rxxb(:, :, k) = temp_B' * F * M' * sqrtm(A(:, :, k)) * Rxxm(:, :, k) ...
            * sqrtm(A(:, :, k))' * M * F' * temp_B;

        Rxxb_total = Rxxb_total + Rxxb(:, :, k);

        if k ~= U
            A(:, :, k + 1) = eye(Lx) + Hmac(:, :, k + 1)' * Rxxb_total * Hmac(:, :, k + 1);
        end

    end

end
