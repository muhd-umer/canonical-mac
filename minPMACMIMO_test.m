%% minPMACMIMO_test.m - Test non-CVX MIMO MAC implementation

clear all; close all; clc;

%% =======================================================================
%% TEST 1: SISO Case (Reference test from specification)
%% =======================================================================
fprintf('=============================================================\n');
fprintf('TEST 1: SISO Case (1 antenna per user)\n');
fprintf('=============================================================\n\n');

H_siso = zeros(1, 2, 4);
H_siso(:, :, 1) = [80 60];
H_siso(:, :, 2) = [40 30];
H_siso(:, :, 3) = [50 50];
H_siso(:, :, 4) = [30 40];

bu_min_siso = [24 24];
cb = 1; % complex baseband
w_siso = [1 1]; % equal energy weights
Lx_siso = [1 1]; % SISO case (1 antenna per user)

fprintf('Channel Configuration:\n');
fprintf('  Users: %d\n', length(Lx_siso));
fprintf('  Antennas per user: [%s]\n', num2str(Lx_siso));
fprintf('  Tones: %d\n', size(H_siso, 3));
fprintf('  RX antennas: %d\n', size(H_siso, 1));
fprintf('  Target rates: [%s] bits/channel use\n', num2str(bu_min_siso));
fprintf('\n');

%% Run optimization
[FEAS_FLAG_siso, bu_a_siso, info_siso] = minPMACMIMO(H_siso, Lx_siso, bu_min_siso, w_siso, cb);

%% Display results
fprintf('\n--- SISO Results ---\n');
fprintf('Feasibility Flag: %d ', FEAS_FLAG_siso);

if FEAS_FLAG_siso == 1
    fprintf('(single decoding order)\n');
elseif FEAS_FLAG_siso == 2
    fprintf('(time-sharing required)\n');
else
    fprintf('(infeasible)\n');
end

fprintf('\nAchieved Rates:\n');
for u = 1:length(Lx_siso)
    fprintf('  User %d: %.4f bits/channel use\n', u, bu_a_siso(u));
end

fprintf('\nTotal Energy (sum of all Rxx): %.4f\n', sum(info_siso.Rxx{:}, 'all'));
fprintf('Expected optimal value: 6.4746\n');
fprintf('Difference from expected: %.6f\n', abs(sum(info_siso.Rxx{:}, 'all') - 6.4746));

fprintf('\nAverage Energy per User:\n');
for u = 1:length(Lx_siso)
    fprintf('  User %d: %.4f\n', u, info_siso.Eu_avg(u));
end

if FEAS_FLAG_siso >= 1
    fprintf('\nDecoding Order:\n');
    fprintf('  Order: [%s]\n', num2str(info_siso.orderings{1}));
    fprintf('  (User %d decoded last/best position)\n', info_siso.orderings{1}(end));
end

fprintf('\nElapsed Time: %.3f seconds\n', info_siso.elapsed_time);

%% Verify rates meet constraints
fprintf('\n');
if all(bu_a_siso >= bu_min_siso - 0.01)
    fprintf('✓ SISO rate constraints satisfied\n');
else
    fprintf('✗ SISO rate constraints NOT satisfied\n');
    fprintf('  Shortfall: [%s]\n', num2str(bu_min_siso - bu_a_siso));
end

if abs(sum(info_siso.Rxx{:}, 'all') - 6.4746) < 0.01
    fprintf('✓ SISO optimal energy achieved\n');
else
    fprintf('✗ SISO optimal energy NOT achieved\n');
end

%% =======================================================================
%% TEST 2: True MIMO Case (Multiple antennas per user)
%% =======================================================================
fprintf('\n\n=============================================================\n');
fprintf('TEST 2: MIMO Case (2 antennas per user)\n');
fprintf('=============================================================\n\n');

% Create MIMO channel: 2 RX antennas, 2 users with 2 TX antennas each, 2 tones
Ly_mimo = 2;
Lx_mimo = [2 2];
N_mimo = 2;
Ltot_mimo = sum(Lx_mimo);

H_mimo = zeros(Ly_mimo, Ltot_mimo, N_mimo);

% Tone 1: Random channel matrices for each user
% User 1 (antennas 1-2)
H_mimo(:, 1:2, 1) = [3.2 2.1; 1.8 2.9];
% User 2 (antennas 3-4)
H_mimo(:, 3:4, 1) = [2.5 3.0; 2.2 1.9];

% Tone 2: Different channel realization
% User 1 (antennas 1-2)
H_mimo(:, 1:2, 2) = [2.9 2.5; 2.0 3.1];
% User 2 (antennas 3-4)
H_mimo(:, 3:4, 2) = [2.8 2.3; 1.7 2.6];

bu_min_mimo = [10 10]; % Target rates in bits/channel use
w_mimo = [1 1]; % Equal energy weights

fprintf('Channel Configuration:\n');
fprintf('  Users: %d\n', length(Lx_mimo));
fprintf('  Antennas per user: [%s]\n', num2str(Lx_mimo));
fprintf('  Total TX antennas: %d\n', Ltot_mimo);
fprintf('  Tones: %d\n', N_mimo);
fprintf('  RX antennas: %d\n', Ly_mimo);
fprintf('  Target rates: [%s] bits/channel use\n', num2str(bu_min_mimo));
fprintf('\n');

%% Run optimization
[FEAS_FLAG_mimo, bu_a_mimo, info_mimo] = minPMACMIMO(H_mimo, Lx_mimo, bu_min_mimo, w_mimo, cb);

%% Display results
fprintf('\n--- MIMO Results ---\n');
fprintf('Feasibility Flag: %d ', FEAS_FLAG_mimo);

if FEAS_FLAG_mimo == 1
    fprintf('(single decoding order)\n');
elseif FEAS_FLAG_mimo == 2
    fprintf('(time-sharing required)\n');
else
    fprintf('(infeasible)\n');
end

fprintf('\nAchieved Rates:\n');
for u = 1:length(Lx_mimo)
    fprintf('  User %d: %.4f bits/channel use\n', u, bu_a_mimo(u));
end

fprintf('\nTotal Energy (sum of all Rxx): %.4f\n', sum(info_mimo.Rxx{:}, 'all'));

fprintf('\nAverage Energy per User:\n');
for u = 1:length(Lx_mimo)
    fprintf('  User %d: %.4f\n', u, info_mimo.Eu_avg(u));
end

fprintf('\nCovariance Matrix Structure:\n');
for n = 1:N_mimo
    fprintf('  Tone %d:\n', n);
    Rxx_n = info_mimo.Rxx{1}(:, :, n);
    for u = 1:length(Lx_mimo)
        idx_start = [1, cumsum(Lx_mimo(1:end-1)) + 1];
        idx_end = cumsum(Lx_mimo);
        ant_idx = idx_start(u):idx_end(u);
        fprintf('    User %d Rxx:\n', u);
        disp(Rxx_n(ant_idx, ant_idx));
    end
end

if FEAS_FLAG_mimo >= 1
    fprintf('Decoding Order:\n');
    fprintf('  Order: [%s]\n', num2str(info_mimo.orderings{1}));
    fprintf('  (User %d decoded last/best position)\n', info_mimo.orderings{1}(end));
end

fprintf('\nElapsed Time: %.3f seconds\n', info_mimo.elapsed_time);

%% Verify rates meet constraints
fprintf('\n');
if all(bu_a_mimo >= bu_min_mimo - 0.01)
    fprintf('✓ MIMO rate constraints satisfied\n');
else
    fprintf('✗ MIMO rate constraints NOT satisfied\n');
    fprintf('  Shortfall: [%s]\n', num2str(bu_min_mimo - bu_a_mimo));
end

%% =======================================================================
%% Summary
%% =======================================================================
fprintf('\n\n=============================================================\n');
fprintf('TEST SUMMARY\n');
fprintf('=============================================================\n');
fprintf('SISO Test: %s\n', ternary(all(bu_a_siso >= bu_min_siso - 0.01) && ...
    abs(sum(info_siso.Rxx{:}, 'all') - 6.4746) < 0.01, 'PASS', 'FAIL'));
fprintf('MIMO Test: %s\n', ternary(all(bu_a_mimo >= bu_min_mimo - 0.01), 'PASS', 'FAIL'));
fprintf('=============================================================\n');

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
