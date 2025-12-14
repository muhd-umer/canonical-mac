% function [Eun, w, bun] = maxRESMAC_cvx(H, Eu, theta, cb)
%
% maxRMAC_cvx Maximizes weighted rate sum subject to energy constraint, each
% user has ONLY ONE transmit antenna. It uses CVX and mosek. It only works
% for Lxu=1 on all users.
%
%   INPUTS:
% H(:,u,n): Ly x U x N channel matrix. Ly = number of receiver antennas,
%     U = number of users and N = number of tones.
%
%     If the channel is real-bbd (cb=2), maxPMAC_cvx realizes user data rates
%     over all the tones (or equivalently positive and negative
%     freqs). This means the input H must be conjugate symmetric H_n =
%     H_{N-n}^*.  The program will reduce N by 2 and focus energy on the
%     lower half of frequencies. Thus N>1 must be even for real channels.
%     For real N=1 channel, the program runs and will produce half the data
%     rates for the same complex N=1 channel.
%
%     By constast if cb=1 (cplx bbd), maxRMAC_cvx need not have a conjugate
%     symmetric H and N is not reduced.
%
% Eu: TOTAL SUM-ENERGY/symbol constraint for all user inputs.
%
% theta: weight on each user's rate, length-U vector.
%
% cb: =1 if H is complex baseband, and cb=2 if H is real baseband.
%
%   OUTPUTS:
% Eun:    U x N energy distribution. Eun(u,n) is user u's energy allocation
%         on tone n.
% w:   U x 1 Lagrangian multiplier w.r.t. energy constraints
% bun: U by N bit distributions for all users.
% ***********************************************************************
function [Eun, w, bun] = maxRESMAC_cvx(H, Eu, theta, cb)
    [Ly, U, N] = size(H);

    if N > 1

        if cb == 2
            H = H(:, :, 1:N / 2);
            N = N / 2;
        end

    end

    Eu = reshape(Eu, [], 1);
    theta = reshape(theta, [], 1);
    [stheta, idx] = sort(theta, 'descend');
    delta = -diff([stheta; 0]);
    sH = H(:, idx, :);
    Gs = zeros(Ly, Ly, U, N);

    parfor n = 1:N

        for u = 1:U
            Gs(:, :, u, n) = sH(:, u, n) * sH(:, u, n)';
        end

    end

    cvx_begin quiet
    cvx_solver mosek
    variable Eun(U, N) nonnegative
    dual variable w
    expression r(U, N)
    S = cumsum(Gs .* repelem(reshape(Eun, 1, 1, U, N), Ly, Ly, 1, 1), 3);

    for u = 1:U

        for n = 1:N
            r(u, n) = log_det(S(:, :, u, n) + eye(Ly));
        end

    end

    maximize sum(delta' * r)
    subject to
    w:sum(sum(Eun, 2)) <= Eu;
    cvx_end

    S = cumsum(Gs .* repelem(reshape(Eun, 1, 1, U, N), Ly, Ly, 1, 1), 3) + eye(Ly);
    cumrate = zeros(U, N);

    for u = 1:U

        for n = 1:N
            cumrate(u, n) = (1 / cb) * real(log2(det(S(:, :, u, n))));
        end

    end

    bun(idx, :) = diff([zeros(1, N); cumrate]);
    w(idx) = w;
end
