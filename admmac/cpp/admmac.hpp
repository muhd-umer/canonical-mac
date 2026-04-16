/**
 * @file admmac.hpp
 * @brief Feasibility check for multi-user MIMO MAC via ellipsoid method
 *
 * Determines whether target rate vector bu is feasible under per-user
 * energy constraints Eu, using ellipsoid method with maxRMACMIMO as
 * subroutine. Supports time-sharing between multiple decoding orders.
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace admmac {

using Complex = std::complex<double>;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct AdmMACVertex {
    VectorXd bu_v;
    MatrixXd bun;
    std::vector<int> order;
    double frac;
    int cluster_id;
};

struct AdmMACResult {
    int feas_flag;

    VectorXd bu_a;

    std::vector<std::vector<MatrixXcd>> Rxx;

    MatrixXd Eun;

    VectorXd theta;

    VectorXd w;

    std::vector<AdmMACVertex> vertices;

    int iterations;

    double elapsed_sec;
};

struct AdmMACConfig {
    double err = 1e-6;

    double conv_tol = 1e-6;

    int max_iter = 2000;

    int num_threads = 0;

    bool verbose = false;
};

/**
 * @brief Determine feasibility of target rates under energy constraints
 *
 * @param H Channel tensor [Ly x sum(Lxu) x N] as vector of N matrices
 * @param Lxu Antennas per user (length U)
 * @param bu Target rates per user (length U, in bits)
 * @param Eu Per-user energy constraints (length U)
 * @param cb Baseband type: 1 for complex, 2 for real
 * @param config Algorithm configuration
 * @return AdmMACResult with feasibility flag, achieved rates, and details
 */
AdmMACResult admMACMIMO(const std::vector<MatrixXcd>& H,
                         const std::vector<int>& Lxu, const VectorXd& bu,
                         const VectorXd& Eu, int cb,
                         const AdmMACConfig& config = AdmMACConfig());

}  

