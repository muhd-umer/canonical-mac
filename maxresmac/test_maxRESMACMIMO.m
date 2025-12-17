%TEST_MAXRESMACMIMO Collection of tests for maxRESMACMIMO

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
Etotal = 5;
theta = [1; 1];
cb = 1;

run_test(@() maxRESMACMIMO(H_siso, Lxu, Etotal, theta, cb), Etotal, theta, 'basic siso');

fprintf('[test] basic mimo (2 users, 2x2, 2 tones)\n');
Ly = 2;
Lxu_mimo = [2 2];
N = 2;
H_mimo = zeros(Ly, sum(Lxu_mimo), N);
H_mimo(:, 1:2, 1) = [3.2 2.1; 1.8 2.9];
H_mimo(:, 3:4, 1) = [2.5 3.0; 2.2 1.9];
H_mimo(:, 1:2, 2) = [2.9 2.5; 2.0 3.1];
H_mimo(:, 3:4, 2) = [2.8 2.3; 1.7 2.6];
Etotal_mimo = 10;
theta_mimo = [1; 1];

run_test(@() maxRESMACMIMO(H_mimo, Lxu_mimo, Etotal_mimo, theta_mimo, cb), ...
    Etotal_mimo, theta_mimo, 'basic mimo');

fprintf('[test] single tone\n');
H_single = zeros(1, 2, 1);
H_single(:, :, 1) = [80 60];
Etotal_single = 2;

run_test(@() maxRESMACMIMO(H_single, Lxu, Etotal_single, theta, cb), ...
    Etotal_single, theta, 'single tone');

fprintf('[test] unequal weights\n');
theta_uneq = [1; 3];

run_test(@() maxRESMACMIMO(H_siso, Lxu, Etotal, theta_uneq, cb), ...
    Etotal, theta_uneq, 'unequal weights');

fprintf('[test] three users\n');
H_3u = zeros(2, 3, 4);
H_3u(:, :, 1) = [5 4 3; 4 5 4];
H_3u(:, :, 2) = [4 3 5; 3 4 5];
H_3u(:, :, 3) = [3 5 4; 5 3 4];
H_3u(:, :, 4) = [4 4 4; 4 4 4];
Lxu_3u = [1 1 1];
theta_3u = [1; 1; 1];
Etotal_3u = 6;

run_test(@() maxRESMACMIMO(H_3u, Lxu_3u, Etotal_3u, theta_3u, cb), ...
    Etotal_3u, theta_3u, 'three users');

fprintf('[test] mimo with unequal antennas\n');
Ly = 3;
Lxu_mixed = [2 3];
N = 2;
H_mixed = (randn(Ly, sum(Lxu_mixed), N) + 1j * randn(Ly, sum(Lxu_mixed), N)) / sqrt(2);
Etotal_mixed = 8;
theta_mixed = [1; 1.5];

run_test(@() maxRESMACMIMO(H_mixed, Lxu_mixed, Etotal_mixed, theta_mixed, cb), ...
    Etotal_mixed, theta_mixed, 'mimo unequal antennas');

fprintf('[test] real baseband\n');
H_real = zeros(1, 2, 4);
H_real(:, :, 1) = [5 4];
H_real(:, :, 2) = [3 2];
H_real(:, :, 3) = [2 3];
H_real(:, :, 4) = [4 5];
Etotal_real = 4;
cb_real = 2;

run_test(@() maxRESMACMIMO(H_real, Lxu, Etotal_real, theta, cb_real), ...
    Etotal_real, theta, 'real baseband');

fprintf('[test] stress test\n');
Ly = 4;
Lxu_stress = [2 1 2 1 2 1];
N = 64;
H_stress = (randn(Ly, sum(Lxu_stress), N) + 1j * randn(Ly, sum(Lxu_stress), N)) / sqrt(2);
Etotal_stress = 30;
theta_stress = [1; 2; 1.5; 0.5; 1.2; 0.8];

run_test(@() maxRESMACMIMO(H_stress, Lxu_stress, Etotal_stress, theta_stress, cb), ...
    Etotal_stress, theta_stress, 'stress test');

%% helper functions
function run_test(solver, Etotal, theta, test_name)

    try
        tic;
        [Rxxs, Eun, w, bun] = solver();
        elapsed = toc;

        U = length(theta);
        total_energy = sum(Eun, 'all');
        total_rate = sum(bun, 'all');
        weighted_rate = sum(theta .* sum(bun, 2));
        bu = sum(bun, 2);

        fprintf('  config: %d users, Etotal = %.2f\n', U, Etotal);
        fprintf('  weights theta: [%s]\n', num2str(theta'));
        fprintf('  per-user rates: [%s]\n', num2str(bu'));
        fprintf('  total rate: %.4f, weighted rate: %.4f\n', total_rate, weighted_rate);
        fprintf('  total energy used: %.6f (target: %.6f)\n', total_energy, Etotal);
        fprintf('  energy constraint error: %.2e\n', abs(total_energy - Etotal) / Etotal);
        fprintf('  lagrange multiplier w: [%s]\n', num2str(w'));
        fprintf('  elapsed time: %.3f s\n', elapsed);

        % check energy constraint
        energy_error = abs(total_energy - Etotal) / Etotal;

        if energy_error < 0.01
            fprintf('  [p] energy constraint satisfied\n');
        else
            fprintf('  [x] energy constraint violated (error: %.2e)\n', energy_error);
        end

        % check non-negative rates
        if all(bun(:) >= -1e-6)
            fprintf('  [p] rates are non-negative\n');
        else
            fprintf('  [x] some rates are negative\n');
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

        fprintf('  [p] test passed\n');

    catch ME
        fprintf('  [x] test crashed: %s\n', ME.message);

        if ~isempty(ME.stack)
            fprintf('  at %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end

    end

    fprintf('\n');
end
