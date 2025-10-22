function check_dependencies()
    %CHECK_DEPENDENCIES Check for required dependencies for the minPMAC demo
    %   CHECK_DEPENDENCIES() verifies the presence of QuaDRiGa, reference
    %   minPMAC functions, and CVX on the MATLAB path, reporting any missing
    %   components along with installation guidance.

    fprintf('Checking dependencies...\n');

    % check for QuaDRiGa
    if ~exist('qd_layout', 'file')
        warning('[ERROR] QuaDRiGa not found in MATLAB path\n');
        fprintf('Please install QuaDRiGa and add it to your MATLAB path\n');
        return;
    else
        fprintf('  [✓] QuaDRiGa found\n');
    end

    % check for minPMAC functions
    if ~exist('minPMACMIMO_reftd', 'file')
        warning('[ERROR] minPMACMIMO_reftd.m not found\n');
        return;
    else
        fprintf('  [✓] minPMACMIMO_reftd found\n');
    end

    if ~exist('minPMAC_reftd', 'file')
        warning('[ERROR] minPMAC_reftd.m not found\n');
        return;
    else
        fprintf('  [✓] minPMAC_reftd found\n');
    end

    % check for CVX
    try
        cvx_version;
        fprintf('  [✓] CVX found\n');
    catch
        warning('WARNING: CVX not found - minPMAC optimization may fail\n');
        fprintf('Please install CVX from http://cvxr.com/cvx/\n');
    end

    fprintf('Dependency check completed.\n\n');
end
