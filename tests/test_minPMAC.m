% TEST_MINPMAC Test script for minPMAC implementations
%
% This script tests both SISO and MIMO versions of minPMAC with
% random channel matrices to verify correctness of implementation.

clear;
clc;
cvx_clear;
rng(42);

fprintf('[test] SISO minPMAC\n');
Ly = 2;
U = 3;
N = 4;
H_siso = (randn(Ly, U, N) + 1j * randn(Ly, U, N)) / sqrt(2);
bu_min = [2, 1.5, 2.5];
w = [1, 1, 1];
cb = 1;

try
    [flag, bu_a, info] = minPMAC_reftd(H_siso, bu_min, w, cb);

    fprintf('  ::: Results :::\n');
    fprintf('  - FEAS_FLAG: %d\n', flag);
    fprintf('  - Target rates: [%.2f, %.2f, %.2f]\n', bu_min);
    fprintf('  - Achieved rates: [%.2f, %.2f, %.2f]\n', bu_a);
    fprintf('  - Rate error: [%.3f, %.3f, %.3f]\n', bu_a - bu_min);
    fprintf('  - Sol. status: %s\n', info.sol_status);
    fprintf('  - Clusters found: %d\n', length(info.clusters));
    fprintf('  - Orders: %d\n', length(info.orderings));

    if flag > 0
        fprintf('  [✓] SISO test PASSED\n');
    else
        fprintf('  [x] SISO test FAILED (infeasible)\n');
    end

catch ME
    fprintf('  [x] SISO test FAILED with error: %s\n', ME.message);
end

fprintf('\n');

fprintf('[test] MIMO minPMAC\n');
Ly = 3;
Lx = [2, 3, 2];
U = length(Lx);
Ltot = sum(Lx);
N = 6;
H_mimo = (randn(Ly, Ltot, N) + 1j * randn(Ly, Ltot, N)) / sqrt(2);
bu_min = [3, 4, 2.5];
w = [1, 2, 1];
cb = 1;

try
    [flag, bu_a, info] = minPMACMIMO_reftd(H_mimo, Lx, bu_min, w, cb);

    fprintf('  ::: Results :::\n');
    fprintf('  - Antenna config: [%s], total=%d\n', num2str(Lx), Ltot);
    fprintf('  - FEAS_FLAG: %d\n', flag);
    fprintf('  - Target rates: [%.2f, %.2f, %.2f]\n', bu_min);
    fprintf('  - Achieved rates: [%.2f, %.2f, %.2f]\n', bu_a);
    fprintf('  - Rate error: [%.3f, %.3f, %.3f]\n', bu_a - bu_min);
    fprintf('  - Sol. status: %s\n', info.sol_status);
    fprintf('  - Clusters found: %d\n', length(info.clusters));
    fprintf('  - Orders: %d\n', length(info.orderings));

    if flag > 0
        fprintf('  [✓] MIMO test PASSED\n');
    else
        fprintf('  [x] MIMO test FAILED (infeasible)\n');
    end

catch ME
    fprintf('  [x] MIMO test FAILED with error: %s\n', ME.message);
end

fprintf('\n');

fprintf('[test] Stress Test\n');
Ly = 4;
Lx = [3, 2, 4, 2, 1, 2];
U = length(Lx);
Ltot = sum(Lx);
N = 8;
H_large = zeros(Ly, Ltot, N);

for n = 1:N
    H_base = (randn(Ly, Ltot) + 1j * randn(Ly, Ltot)) / sqrt(2);
    H_large(:, :, n) = H_base + 0.1 * eye(Ly, Ltot);
end

bu_min = [2, 1.8, 2.2, 1.5, 2, 1.8];
w = ones(1, U);
cb = 1;

try
    tic;
    [flag, bu_a, info] = minPMACMIMO_reftd(H_large, Lx, bu_min, w, cb);
    elapsed = toc;

    fprintf('  ::: Results :::\n');
    fprintf('  - System size: %d RX, %d users, %d total TX, %d tones\n', ...
        Ly, U, Ltot, N);
    fprintf('  - FEAS_FLAG: %d\n', flag);
    fprintf('  - Target rates: [%s]\n', sprintf('%.2f ', bu_min));
    fprintf('  - Achieved rates: [%s]\n', sprintf('%.2f ', bu_a));
    fprintf('  - Max rate error: %.3f\n', max(abs(bu_a - bu_min)));
    fprintf('  - Computation time: %.3f seconds\n', elapsed);
    fprintf('  - Sol. status: %s\n', info.sol_status);

    if flag > 0 && max(abs(bu_a - bu_min)) < 0.1
        fprintf('  [✓] Stress test PASSED\n');
    else
        fprintf('  [x] Stress test FAILED\n');
    end

catch ME
    fprintf('  [x] Stress test FAILED with error: %s\n', ME.message);
end

fprintf('\n');

fprintf('[test] Real Baseband\n');
H_real = randn(2, 3, 4);
bu_min = [1, 1, 1];
w = [1, 1, 1];
cb = 2; % real baseband

try
    [flag, bu_a, info] = minPMAC_reftd(H_real, bu_min, w, cb);

    fprintf('  ::: Results :::\n');
    fprintf('  - FEAS_FLAG: %d\n', flag);
    fprintf('  - Target rates: [%.2f, %.2f, %.2f]\n', bu_min);
    fprintf('  - Achieved rates: [%.2f, %.2f, %.2f]\n', bu_a);
    fprintf('  - Sol. status: %s\n', info.sol_status);

    if flag > 0
        fprintf('  [✓] Real baseband test PASSED\n');
    else
        fprintf('  [x] Real baseband test FAILED\n');
    end

catch ME
    fprintf('  [x] Real baseband test FAILED with error: %s\n', ME.message);
end

fprintf('\n');

fprintf('[test] Infeasible Case\n');
% create a rank-deficient channel that limits capacity
Ly = 2;
U = 3;
N = 1;

% create rank-1 channel (very limited capacity)
h_base = [1; 0.5];
H_infeas = zeros(Ly, U, N);
H_infeas(:, 1, 1) = 0.01 * h_base;
H_infeas(:, 2, 1) = 0.01 * h_base;
H_infeas(:, 3, 1) = 0.01 * h_base;

% request rates that exceed single-tone capacity
bu_min_impossible = [8, 8, 8];
w = [1, 1, 1];
cb = 1;

try
    [flag, bu_a, info] = minPMAC_reftd(H_infeas, bu_min_impossible, w, cb);

    fprintf('  ::: Results :::\n');
    fprintf('  - FEAS_FLAG: %d\n', flag);
    fprintf('  - Sol. status: %s\n', info.sol_status);

    % check if CVX detected infeasibility
    if strcmp(info.sol_status, 'Infeasible') || flag == 0
        fprintf('  [✓] Infeasible test PASSED (correctly detected infeasibility)\n');
    else
        fprintf('  [x] Infeasible test FAILED (should be infeasible)\n');
        fprintf('  - Achieved rates: [%.2f, %.2f, %.2f]\n', bu_a);
    end

catch ME
    fprintf('  [x] Infeasible test FAILED with error: %s\n', ME.message);
end
