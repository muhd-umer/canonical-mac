function check_dependencies()
    % CHECK_DEPENDENCIES Checks for required dependencies for the minPMAC demo
    %
    % This function checks for the presence of QuaDRiGa, minPMAC functions,
    % and CVX in the MATLAB path. It provides feedback on missing dependencies
    % and suggests installation steps if necessary.

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
