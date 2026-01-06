%TEST_ADMMACMIMO Collection of tests for admMACMIMO

clear;
clc;
rng(42);

fprintf('[test] basic siso feasible (2 users, 4 tones)\n');
H_siso = zeros(1, 2, 4);
H_siso(:, :, 1) = [80 60];
H_siso(:, :, 2) = [40 30];
H_siso(:, :, 3) = [50 50];
H_siso(:, :, 4) = [30 40];
Lxu = [1 1];
bu = [5; 5];
Eu = [1; 1];
cb = 1;

run_test(@() admMACMIMO(H_siso, Lxu, bu, Eu, cb), bu, Eu, 'basic siso feasible');

fprintf('[test] siso infeasible (low energy)\n');
bu_inf = [5; 5];
Eu_inf = [0.001; 0.001];

run_test(@() admMACMIMO(H_siso, Lxu, bu_inf, Eu_inf, cb), bu_inf, Eu_inf, ...
    'siso infeasible (low energy)', true);

fprintf('[test] siso infeasible (high rate)\n');
bu_high = [100; 100];
Eu_high = [1; 1];

run_test(@() admMACMIMO(H_siso, Lxu, bu_high, Eu_high, cb), bu_high, Eu_high, ...
    'siso infeasible (high rate)', true);

fprintf('[test] siso vertex sharing (equal channels)\n');
H_eq = zeros(1, 3, 2);
H_eq(:, :, 1) = [80 80 80];
H_eq(:, :, 2) = [60 60 60];
Lxu_eq = [1 1 1];
bu_eq = [3; 2.5; 2];
Eu_eq = [1; 1; 1];

run_test(@() admMACMIMO(H_eq, Lxu_eq, bu_eq, Eu_eq, cb), bu_eq, Eu_eq, ...
'siso vertex sharing');

fprintf('[test] basic mimo (2 users, 2x2, 2 tones)\n');
Ly = 2;
Lxu_mimo = [2 2];
N = 2;
H_mimo = zeros(Ly, sum(Lxu_mimo), N);
H_mimo(:, 1:2, 1) = [3.2 2.1; 1.8 2.9];
H_mimo(:, 3:4, 1) = [2.5 3.0; 2.2 1.9];
H_mimo(:, 1:2, 2) = [2.9 2.5; 2.0 3.1];
H_mimo(:, 3:4, 2) = [2.8 2.3; 1.7 2.6];
bu_mimo = [5; 5];
Eu_mimo = [2; 2];

run_test(@() admMACMIMO(H_mimo, Lxu_mimo, bu_mimo, Eu_mimo, cb), ...
    bu_mimo, Eu_mimo, 'basic mimo');

fprintf('[test] mimo multi-tone (8 tones)\n');
Ly = 2;
Lxu_mt = [2, 2];
N = 8;
H_multi = (randn(Ly, sum(Lxu_mt), N) + 1j * randn(Ly, sum(Lxu_mt), N)) / sqrt(2);
bu_mt = [8; 8];
Eu_mt = [3; 3];

run_test(@() admMACMIMO(H_multi, Lxu_mt, bu_mt, Eu_mt, cb), ...
    bu_mt, Eu_mt, 'mimo multi-tone');

fprintf('[test] single tone\n');
H_single = reshape([80 60], 1, 2, 1);
Lxu_single = [1 1];
bu_single = [2; 2];
Eu_single = [0.5; 0.5];

run_test(@() admMACMIMO(H_single, Lxu_single, bu_single, Eu_single, cb), ...
    bu_single, Eu_single, 'single tone');

fprintf('[test] three users siso\n');
H_3u = zeros(2, 3, 4);
H_3u(:, :, 1) = [5 4 3; 4 5 4];
H_3u(:, :, 2) = [4 3 5; 3 4 5];
H_3u(:, :, 3) = [3 5 4; 5 3 4];
H_3u(:, :, 4) = [4 4 4; 4 4 4];
Lxu_3u = [1 1 1];
bu_3u = [2; 2; 2];
Eu_3u = [1; 1; 1];

run_test(@() admMACMIMO(H_3u, Lxu_3u, bu_3u, Eu_3u, cb), ...
    bu_3u, Eu_3u, 'three users siso');

fprintf('[test] mimo with unequal antennas\n');
Ly = 3;
Lxu_mixed = [2 3];
N = 2;
H_mixed = (randn(Ly, sum(Lxu_mixed), N) + 1j * randn(Ly, sum(Lxu_mixed), N)) / sqrt(2);
bu_mixed = [3; 3];
Eu_mixed = [2; 3];

run_test(@() admMACMIMO(H_mixed, Lxu_mixed, bu_mixed, Eu_mixed, cb), ...
    bu_mixed, Eu_mixed, 'mimo unequal antennas');

fprintf('[test] real baseband\n');
H_real = randn(2, 3, 4);
Lxu_real = [1 1 1];
bu_real = [1; 1; 1];
Eu_real = [1; 1; 1];
cb_real = 2;

run_test(@() admMACMIMO(H_real, Lxu_real, bu_real, Eu_real, cb_real), ...
    bu_real, Eu_real, 'real baseband');

fprintf('[test] uniform antennas (scalar Lxu)\n');
Ly = 2;
U = 3;
Lxu_scalar = 2;
N = 3;
H_uniform = (randn(Ly, U * Lxu_scalar, N) + 1j * randn(Ly, U * Lxu_scalar, N)) / sqrt(2);
bu_uniform = [4; 4; 4];
Eu_uniform = [2; 2; 2];

run_test(@() admMACMIMO(H_uniform, Lxu_scalar, bu_uniform, Eu_uniform, cb), ...
    bu_uniform, Eu_uniform, 'uniform antennas');

fprintf('[test] stress test\n');
Ly = 4;
Lxu_stress = [2 1 2 1 2 1];
N = 64;
H_stress = (randn(Ly, sum(Lxu_stress), N) + 1j * randn(Ly, sum(Lxu_stress), N)) / sqrt(2);
bu_stress = [4; 5; 4; 3; 4; 3];
Eu_stress = [2; 3; 2; 1.5; 2; 1.5];

run_test(@() admMACMIMO(H_stress, Lxu_stress, bu_stress, Eu_stress, cb), ...
    bu_stress, Eu_stress, 'stress test');

fprintf('[test] boundary feasibility\n');
H_bnd = zeros(1, 2, 2);
H_bnd(:, :, 1) = [50 40];
H_bnd(:, :, 2) = [45 35];
bu_bnd = [4; 4];
Eu_bnd = [0.5; 0.5];

run_test(@() admMACMIMO(H_bnd, Lxu, bu_bnd, Eu_bnd, cb), ...
    bu_bnd, Eu_bnd, 'boundary feasibility');

%% helper functions
function run_test(solver, bu, Eu, test_name, expect_infeasible)

    if nargin < 5
        expect_infeasible = false;
    end

    try
        tic;
        [FEAS_FLAG, bu_a, Rxxs, Eun, theta, w, info] = solver();
        elapsed = toc;

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
            fprintf('  elapsed: %.3f s\n', elapsed);

            % check rates meet targets
            rate_margin = min(bu_a - bu);

            if rate_margin >= -1e-4
                fprintf('  [p] all target rates met (min margin: %.6f)\n', rate_margin);
            else
                fprintf('  [x] some target rates not met (min margin: %.6f)\n', rate_margin);
            end

            % check energy constraints
            if ~isequal(Eun, 0)
                energy_used = sum(Eun, 2);
                energy_margin = Eu - energy_used;

                if min(energy_margin) >= -1e-4
                    fprintf('  [p] all energy constraints satisfied\n');
                else
                    fprintf('  [x] some energy constraints violated\n');
                end

            end

            % check psd constraint for Rxxs
            if ~isequal(Rxxs, 0)
                psd_ok = check_psd(Rxxs);

                if psd_ok
                    fprintf('  [p] covariance matrices are PSD\n');
                else
                    fprintf('  [x] some covariance matrices are not PSD\n');
                end

            end

            % check info structure
            if ~isempty(info)

                if istable(info)
                    fprintf('  [p] info table has %d rows\n', size(info, 1));
                else
                    fprintf('  [p] info structure present\n');
                end

            end

        else
            fprintf('  achieved rates: [%s]\n', num2str(bu_a'));
            fprintf('  elapsed: %.3f s\n', elapsed);
        end

        if expect_infeasible

            if FEAS_FLAG == 0
                fprintf('  [p] infeasible case correctly detected\n');
            else
                fprintf('  [x] expected infeasible but got FEAS_FLAG=%d\n', FEAS_FLAG);
            end

        else

            if FEAS_FLAG > 0
                fprintf('  [p] test passed: %s\n', test_name);
            else
                fprintf('  [x] test failed: expected feasible but got FEAS_FLAG=0\n');
            end

        end

    catch ME
        fprintf('  [x] test crashed: %s\n', ME.message);

        if ~isempty(ME.stack)
            fprintf('  at %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end

    end

    fprintf('\n');
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
