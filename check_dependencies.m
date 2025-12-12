function check_dependencies()
    %CHECK_DEPENDENCIES Check for required dependencies for multiuser optimization
    %   CHECK_DEPENDENCIES() verifies the presence of QuaDRiGa,
    %   CVX on the MATLAB path, and MOSEK installation. It notifies the user 
    %   of missing components along with guide.

    fprintf('Checking dependencies...\n');

    % check for QuaDRiGa
    if ~exist('qd_layout', 'file')
        warning('[ERROR] QuaDRiGa not found in MATLAB path\n');
        fprintf('Please install QuaDRiGa and add it to your MATLAB path\n');
        return;
    else
        fprintf('  [✓] QuaDRiGa found\n');
    end

    % check for CVX
    try
        cvx_version;
        fprintf('  [✓] CVX found\n');
    catch
        warning('WARNING: CVX not found; multiuser optimization may fail\n');
        fprintf('Please install CVX from http://cvxr.com/cvx/\n');
    end

    % check for MOSEK
    try
        mosekopt;
        fprintf('  [✓] MOSEK found\n');
    catch
        warning('WARNING: MOSEK not found; multiuser optimization may fail\n');
        fprintf('Please install MOSEK from https://www.mosek.com/\n');
    end
end
