%TEST_ADMMACMIMO Collection of tests for admMACMIMO

clear;
clc;
rng(42);

cpp_available = exist('admmac_mex', 'file') == 3;

if cpp_available
    fprintf('[info] MEX implementation available\n\n');
else
    fprintf('[info] MEX not found; testing MATLAB only\n');
    fprintf('[info] Build with: cd build && cmake .. && make admmac_mex\n\n');
end

%% test definitions
tests = struct();

tests(1).name = 'basic siso feas';
H = zeros(1, 2, 4);
H(:, :, 1) = [80 60]; H(:, :, 2) = [40 30];
H(:, :, 3) = [50 50]; H(:, :, 4) = [30 40];
tests(1).H = H; tests(1).Lxu = [1 1];
tests(1).bu = [5; 5]; tests(1).Eu = [1; 1];
tests(1).cb = 1; tests(1).expect_infeasible = false;

tests(2).name = 'infeas low energy';
tests(2).H = tests(1).H; tests(2).Lxu = [1 1];
tests(2).bu = [5; 5]; tests(2).Eu = [0.001; 0.001];
tests(2).cb = 1; tests(2).expect_infeasible = true;

tests(3).name = 'infeas high rate';
tests(3).H = tests(1).H; tests(3).Lxu = [1 1];
tests(3).bu = [100; 100]; tests(3).Eu = [1; 1];
tests(3).cb = 1; tests(3).expect_infeasible = true;

tests(4).name = 'vertex sharing';
H = zeros(1, 3, 2);
H(:, :, 1) = [80 80 80]; H(:, :, 2) = [60 60 60];
tests(4).H = H; tests(4).Lxu = [1 1 1];
tests(4).bu = [3; 2.5; 2]; tests(4).Eu = [1; 1; 1];
tests(4).cb = 1; tests(4).expect_infeasible = false;

tests(5).name = 'basic mimo';
H = zeros(2, 4, 2);
H(:, 1:2, 1) = [3.2 2.1; 1.8 2.9]; H(:, 3:4, 1) = [2.5 3.0; 2.2 1.9];
H(:, 1:2, 2) = [2.9 2.5; 2.0 3.1]; H(:, 3:4, 2) = [2.8 2.3; 1.7 2.6];
tests(5).H = H; tests(5).Lxu = [2 2];
tests(5).bu = [5; 5]; tests(5).Eu = [2; 2];
tests(5).cb = 1; tests(5).expect_infeasible = false;

tests(6).name = 'mimo multi-tone';
tests(6).H = (randn(2, 4, 8) + 1j * randn(2, 4, 8)) / sqrt(2);
tests(6).Lxu = [2 2];
tests(6).bu = [8; 8]; tests(6).Eu = [3; 3];
tests(6).cb = 1; tests(6).expect_infeasible = false;

tests(7).name = 'single tone';
H = reshape([80 60], 1, 2, 1);
tests(7).H = H; tests(7).Lxu = [1 1];
tests(7).bu = [2; 2]; tests(7).Eu = [0.5; 0.5];
tests(7).cb = 1; tests(7).expect_infeasible = false;

tests(8).name = 'three users';
H = zeros(2, 3, 4);
H(:, :, 1) = [5 4 3; 4 5 4]; H(:, :, 2) = [4 3 5; 3 4 5];
H(:, :, 3) = [3 5 4; 5 3 4]; H(:, :, 4) = [4 4 4; 4 4 4];
tests(8).H = H; tests(8).Lxu = [1 1 1];
tests(8).bu = [2; 2; 2]; tests(8).Eu = [1; 1; 1];
tests(8).cb = 1; tests(8).expect_infeasible = false;

tests(9).name = 'mimo unequal ant';
tests(9).H = (randn(3, 5, 2) + 1j * randn(3, 5, 2)) / sqrt(2);
tests(9).Lxu = [2 3];
tests(9).bu = [3; 3]; tests(9).Eu = [2; 3];
tests(9).cb = 1; tests(9).expect_infeasible = false;

tests(10).name = 'real baseband';
H = randn(2, 3, 4);
tests(10).H = H; tests(10).Lxu = [1 1 1];
tests(10).bu = [1; 1; 1]; tests(10).Eu = [1; 1; 1];
tests(10).cb = 2; tests(10).expect_infeasible = false;

tests(11).name = 'uniform Lxu';
tests(11).H = (randn(2, 6, 3) + 1j * randn(2, 6, 3)) / sqrt(2);
tests(11).Lxu = 2;
tests(11).bu = [4; 4; 4]; tests(11).Eu = [2; 2; 2];
tests(11).cb = 1; tests(11).expect_infeasible = false;

tests(12).name = 'stress';
tests(12).H = (randn(4, 9, 64) + 1j * randn(4, 9, 64)) / sqrt(2);
tests(12).Lxu = [2 1 2 1 2 1];
tests(12).bu = [4; 5; 4; 3; 4; 3]; tests(12).Eu = [2; 3; 2; 1.5; 2; 1.5];
tests(12).cb = 1; tests(12).expect_infeasible = false;

tests(13).name = 'boundary feas';
H = zeros(1, 2, 2);
H(:, :, 1) = [50 40]; H(:, :, 2) = [45 35];
tests(13).H = H; tests(13).Lxu = [1 1];
tests(13).bu = [4; 4]; tests(13).Eu = [0.5; 0.5];
tests(13).cb = 1; tests(13).expect_infeasible = false;

%% run MATLAB tests
fprintf('[matlab]\n\n');

matlab_results = run_all_tests(@(H, Lxu, bu, Eu, cb) admMACMIMO(H, Lxu, bu, Eu, cb, false), tests);

%% run C++ tests
if cpp_available
    fprintf('[cpp]\n\n');

    cpp_results = run_all_tests(@(H, Lxu, bu, Eu, cb) admMACMIMO(H, Lxu, bu, Eu, cb, true), tests);

    %% comparison table
    fprintf('[comparison]\n');

    fprintf('%-18s | %5s %5s | %10s %10s | %8s %8s | %7s\n', ...
        'test', 'f[.m]', 'f[.c]', 'margin[.m]', 'margin[.c]', 'ms[.m]', 'ms[.cpp]', 'speedup');
    fprintf('%s\n', repmat('-', 1, 90));

    for t = 1:length(tests)
        mr = matlab_results(t);
        cr = cpp_results(t);
        speedup = mr.elapsed_ms / max(cr.elapsed_ms, 0.01);
        fprintf('%-18s | %5d %5d | %10.4f %10.4f | %8.2f %8.2f | %6.1fx\n', ...
            tests(t).name, mr.feas_flag, cr.feas_flag, ...
            mr.min_margin, cr.min_margin, mr.elapsed_ms, cr.elapsed_ms, speedup);
    end

    fprintf('\n');
end

%% helper functions

function results = run_all_tests(solver, tests)
    results = [];

    for t = 1:length(tests)
        tc = tests(t);
        fprintf('[test] %s\n', tc.name);
        result = run_test(solver, tc.H, tc.Lxu, tc.bu, tc.Eu, tc.cb, tc.name, tc.expect_infeasible);
        results = [results; result];
        fprintf('\n');
    end

end

function result = run_test(solver, H, Lxu, bu, Eu, cb, test_name, expect_infeasible)
    result.passed = true;
    result.elapsed_ms = 0;
    result.feas_flag = 0;
    result.min_margin = -Inf;

    try
        tic;
        [FEAS_FLAG, bu_a, Rxxs, Eun, theta, w, info] = solver(H, Lxu, bu, Eu, cb);
        elapsed = toc;
        result.elapsed_ms = elapsed * 1000;
        result.feas_flag = FEAS_FLAG;

        bu = bu(:);
        Eu = Eu(:);
        U = length(bu);

        fprintf('  config: %d users, bu = [%s], Eu = [%s]\n', U, ...
            num2str(bu', '%.2g '), num2str(Eu', '%.2g '));
        fprintf('  FEAS_FLAG: %d\n', FEAS_FLAG);

        if FEAS_FLAG > 0
            fprintf('  target rates: [%s]\n', num2str(bu'));
            fprintf('  achieved rates: [%s]\n', num2str(bu_a'));
            fprintf('  rate margin: [%s]\n', num2str((bu_a - bu)'));
            fprintf('  theta: [%s]\n', num2str(theta'));
            fprintf('  w: [%s]\n', num2str(w'));
            fprintf('  elapsed: %.3f ms\n', result.elapsed_ms);

            rate_margin = min(bu_a - bu);
            result.min_margin = rate_margin;

            if rate_margin >= -1e-4
                fprintf('  [p] all target rates met (min margin: %.6f)\n', rate_margin);
            else
                fprintf('  [x] some target rates not met (min margin: %.6f)\n', rate_margin);
                result.passed = false;
            end

            if ~isequal(Eun, 0)
                energy_used = sum(Eun, 2);
                energy_margin = Eu - energy_used;

                if min(energy_margin) >= -1e-4
                    fprintf('  [p] all energy constraints satisfied\n');
                else
                    fprintf('  [x] some energy constraints violated\n');
                    result.passed = false;
                end

            end

            if ~isequal(Rxxs, 0)
                psd_ok = check_psd(Rxxs);

                if psd_ok
                    fprintf('  [p] covariance matrices are PSD\n');
                else
                    fprintf('  [x] some covariance matrices are not PSD\n');
                    result.passed = false;
                end

            end

            if ~isempty(info)

                if istable(info)
                    fprintf('  [p] info table has %d rows\n', size(info, 1));
                else
                    fprintf('  [p] info structure present\n');
                end

            end

        else
            fprintf('  achieved rates: [%s]\n', num2str(bu_a'));
            fprintf('  elapsed: %.3f ms\n', result.elapsed_ms);
            result.min_margin = -Inf;
        end

        if expect_infeasible

            if FEAS_FLAG == 0
                fprintf('  [p] infeasible case correctly detected\n');
            else
                fprintf('  [x] expected infeasible but got FEAS_FLAG=%d\n', FEAS_FLAG);
                result.passed = false;
            end

        else

            if FEAS_FLAG > 0
                fprintf('  [p] test passed: %s\n', test_name);
            else
                fprintf('  [x] test failed: expected feasible but got FEAS_FLAG=0\n');
                result.passed = false;
            end

        end

    catch ME
        result.passed = false;
        fprintf('  [x] test crashed: %s\n', ME.message);

        if ~isempty(ME.stack)
            fprintf('  at %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end

    end

end

function psd_ok = check_psd(Rxxs)
    psd_ok = true;

    if iscell(Rxxs)

        for u = 1:size(Rxxs, 1)

            for n = 1:size(Rxxs, 2)
                Rxx = Rxxs{u, n};

                if ~isempty(Rxx)
                    eigvals = eig(0.5 * (Rxx + Rxx'));

                    if any(real(eigvals) < -1e-9)
                        psd_ok = false;
                        return;
                    end

                end

            end

        end

    else

        for u = 1:size(Rxxs, 3)

            for n = 1:size(Rxxs, 4)
                Rxx = Rxxs(:, :, u, n);
                eigvals = eig(0.5 * (Rxx + Rxx'));

                if any(real(eigvals) < -1e-9)
                    psd_ok = false;
                    return;
                end

            end

        end

    end

end
