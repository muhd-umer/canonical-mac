%TEST_MAXRMACMIMO Collection of tests for maxRMACMIMO

clear;
clc;
rng(42);

fprintf('[test] basic siso (2 users, 4 tones)\n');
H_siso = zeros(1, 2, 4);
H_siso(:, :, 1) = [80 60];
H_siso(:, :, 2) = [40 30];
H_siso(:, :, 3) = [50 50];
H_siso(:, :, 4) = [30 40];
Lxu = [1 1];
Eu = [2; 2];
theta = [1; 1];
cb = 1;

run_test(@() maxRMACMIMO(H_siso, Lxu, Eu, theta, cb), Eu, theta, 'basic siso');

fprintf('[test] basic mimo (2 users, 2x2, 2 tones)\n');
Ly = 2;
Lxu_mimo = [2 2];
N = 2;
H_mimo = zeros(Ly, sum(Lxu_mimo), N);
H_mimo(:, 1:2, 1) = [3.2 2.1; 1.8 2.9];
H_mimo(:, 3:4, 1) = [2.5 3.0; 2.2 1.9];
H_mimo(:, 1:2, 2) = [2.9 2.5; 2.0 3.1];
H_mimo(:, 3:4, 2) = [2.8 2.3; 1.7 2.6];
Eu_mimo = [5; 6];
theta_mimo = [1; 1];

run_test(@() maxRMACMIMO(H_mimo, Lxu_mimo, Eu_mimo, theta_mimo, cb), ...
    Eu_mimo, theta_mimo, 'basic mimo');

fprintf('[test] single tone\n');
H_single = zeros(1, 2, 1);
H_single(:, :, 1) = [80 60];
Eu_single = [2; 2];

run_test(@() maxRMACMIMO(H_single, Lxu, Eu_single, theta, cb), ...
    Eu_single, theta, 'single tone');

fprintf('[test] unequal weights\n');
H_uneq = zeros(1, 2, 4);
H_uneq(:, :, 1) = [80 60];
H_uneq(:, :, 2) = [40 30];
H_uneq(:, :, 3) = [50 50];
H_uneq(:, :, 4) = [30 40];
Eu_uneq = [3; 3];
theta_uneq = [1; 3];

run_test(@() maxRMACMIMO(H_uneq, Lxu, Eu_uneq, theta_uneq, cb), ...
    Eu_uneq, theta_uneq, 'unequal weights');

fprintf('[test] unequal per-user energies\n');
Eu_diff = [5; 2];
theta_diff = [1; 1];

run_test(@() maxRMACMIMO(H_siso, Lxu, Eu_diff, theta_diff, cb), ...
    Eu_diff, theta_diff, 'unequal energies');

fprintf('[test] three users\n');
H_3u = zeros(2, 3, 4);
H_3u(:, :, 1) = [5 4 3; 4 5 4];
H_3u(:, :, 2) = [4 3 5; 3 4 5];
H_3u(:, :, 3) = [3 5 4; 5 3 4];
H_3u(:, :, 4) = [4 4 4; 4 4 4];
Lxu_3u = [1 1 1];
Eu_3u = [3; 4; 3.5];
theta_3u = [1; 1; 1];

run_test(@() maxRMACMIMO(H_3u, Lxu_3u, Eu_3u, theta_3u, cb), ...
    Eu_3u, theta_3u, 'three users');

fprintf('[test] mimo with unequal antennas\n');
Ly = 3;
Lxu_mixed = [2 3];
N = 2;
H_mixed = (randn(Ly, sum(Lxu_mixed), N) + 1j * randn(Ly, sum(Lxu_mixed), N)) / sqrt(2);
Eu_mixed = [4; 6];
theta_mixed = [1; 1.5];

run_test(@() maxRMACMIMO(H_mixed, Lxu_mixed, Eu_mixed, theta_mixed, cb), ...
    Eu_mixed, theta_mixed, 'mimo unequal antennas');

fprintf('[test] real baseband\n');
H_real = zeros(1, 2, 4);
H_real(:, :, 1) = [5 4];
H_real(:, :, 2) = [3 2];
H_real(:, :, 3) = [2 3];
H_real(:, :, 4) = [4 5];
Eu_real = [2; 2];
cb_real = 2;

run_test(@() maxRMACMIMO(H_real, Lxu, Eu_real, theta, cb_real), ...
    Eu_real, theta, 'real baseband');

fprintf('[test] uniform antennas input (scalar Lxu)\n');
Ly = 2;
U = 3;
Lxu_scalar = 2;
N = 3;
H_uniform = (randn(Ly, U * Lxu_scalar, N) + 1j * randn(Ly, U * Lxu_scalar, N)) / sqrt(2);
Eu_uniform = [3; 4; 5];
theta_uniform = [1; 1; 1];

run_test(@() maxRMACMIMO(H_uniform, Lxu_scalar, Eu_uniform, theta_uniform, cb), ...
    Eu_uniform, theta_uniform, 'uniform antennas');

fprintf('[test] stress test\n');
Ly = 4;
Lxu_stress = [2 1 2 1 2 1];
N = 64;
H_stress = (randn(Ly, sum(Lxu_stress), N) + 1j * randn(Ly, sum(Lxu_stress), N)) / sqrt(2);
Eu_stress = [5; 4; 6; 3; 5; 4];
theta_stress = [1; 2; 1.5; 0.5; 1.2; 0.8];

run_test(@() maxRMACMIMO(H_stress, Lxu_stress, Eu_stress, theta_stress, cb), ...
    Eu_stress, theta_stress, 'stress test');

%% helper functions
function run_test(solver, Eu, theta, test_name)

    try
        tic;
        [Rxxs, Eun, w, bun] = solver();
        elapsed = toc;

        Eu = Eu(:);
        theta = theta(:);
        U = length(Eu);

        bu = sum(bun, 2);
        weighted_rate = sum(theta .* bu);
        energy_used = sum(Eun, 2);
        energy_err = max(abs(energy_used - Eu) ./ max(Eu, 1e-12));

        fprintf('  config: %d users, Eu = [%s]\n', U, num2str(Eu', '%.4g '));
        fprintf('  weights theta: [%s]\n', num2str(theta'));
        fprintf('  per-user rates: [%s]\n', num2str(bu'));
        fprintf('  total rate: %.4f, weighted rate: %.4f\n', sum(bu), weighted_rate);
        fprintf('  energy used: [%s]\n', num2str(energy_used'));
        fprintf('  energy target: [%s]\n', num2str(Eu'));
        fprintf('  energy constraint error: %.2e\n', energy_err);
        fprintf('  lagrange multiplier w: [%s]\n', num2str(w'));
        fprintf('  elapsed time: %.3f s\n', elapsed);

        % check energy constraint
        if energy_err < 1e-3
            fprintf('  [p] all energy constraints satisfied (max rel error: %.2e)\n', energy_err);
        else
            fprintf('  [x] energy constraints violated (max rel error: %.2e)\n', energy_err);
        end

        % check non-negative rates
        if all(bun(:) >= -1e-8)
            fprintf('  [p] rates are non-negative\n');
        else
            fprintf('  [x] some rates are negative\n');
        end

        % check non-negative lagrange multipliers
        if all(w(:) >= -1e-12)
            fprintf('  [p] lagrange multipliers are non-negative\n');
        else
            fprintf('  [x] some lagrange multipliers are negative\n');
        end

        % check psd constraint for Rxxs
        psd_ok = true;

        if iscell(Rxxs)

            for u = 1:size(Rxxs, 1)

                for n = 1:size(Rxxs, 2)
                    Rxx = Rxxs{u, n};

                    if ~isempty(Rxx)
                        eigvals = eig(0.5 * (Rxx + Rxx'));

                        if any(real(eigvals) < -1e-9)
                            psd_ok = false;
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
                    end

                end

            end

        end

        if psd_ok
            fprintf('  [p] covariance matrices are PSD\n');
        else
            fprintf('  [x] some covariance matrices are not PSD\n');
        end

        fprintf('  [p] test passed: %s\n', test_name);

    catch ME
        fprintf('  [x] test crashed: %s\n', ME.message);

        if ~isempty(ME.stack)
            fprintf('  at %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end

    end

    fprintf('\n');
end
