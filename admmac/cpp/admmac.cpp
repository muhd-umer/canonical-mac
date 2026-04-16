/**
 * @file admmac.cpp
 * @brief admMACMIMO feasibility solver implementation
 *
 * Ellipsoid method over theta calling maxRMACMIMO, with vertex sharing
 * for equal-theta clusters.
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#include "admmac.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#include "linalg_utils.hpp"
#include "maxrmac.hpp"

namespace admmac {

using maxrmac::MaxRMACConfig;
using maxrmac::MaxRMACResult;
using maxrmac::logdet_spd;
using maxrmac::make_hermitian;

namespace {

struct ClusterInfo {
    std::vector<std::vector<int>> clusters;
    std::vector<int> sizeclus;
    std::vector<int> set_thetaeq;
    int numclus;
};

void update_ellipsoid(VectorXd& theta, MatrixXd& A, const VectorXd& g,
                      int U) {
    double gAg = g.transpose() * A * g;
    double inv_sqrt_gAg = 1.0 / std::sqrt(gAg);
    VectorXd gt = g * inv_sqrt_gAg;
    VectorXd Agt = A * gt;

    const double inv_U1 = 1.0 / (U + 1);
    const double U_sq = static_cast<double>(U * U);
    const double scale_A = U_sq / (U_sq - 1.0);
    const double scale_gt = 2.0 / (U + 1);

    theta -= inv_U1 * Agt;
    A = scale_A * (A - scale_gt * Agt * gt.transpose() * A);

    for (int u = 0; u < U; ++u) {
        if (theta(u) < 0) {
            VectorXd g_proj = VectorXd::Zero(U);
            g_proj(u) = -1.0;
            double gAg_proj = A(u, u);
            double inv_sqrt_proj = 1.0 / std::sqrt(gAg_proj);
            VectorXd gt_proj = g_proj * inv_sqrt_proj;
            VectorXd Agt_proj = A * gt_proj;
            theta -= inv_U1 * Agt_proj;
            A = scale_A * (A - scale_gt * Agt_proj * gt_proj.transpose() * A);
        }
    }

    double theta_norm = theta.norm();
    if (theta_norm < 0.1) {
        theta /= theta_norm;
    }
}

ClusterInfo form_clusters(const std::vector<bool>& spots,
                          const std::vector<int>& Itheta, int U) {
    ClusterInfo info;
    info.numclus = 0;
    bool flagclus = false;

    for (int i = 0; i < U; ++i) {
        if (spots[i] && !flagclus) {
            info.numclus++;
            info.set_thetaeq.push_back(i);
            info.sizeclus.push_back(1);
            flagclus = true;
        } else if (spots[i]) {
            info.sizeclus.back()++;
        } else if (flagclus) {
            info.sizeclus.back()++;
            flagclus = false;
            int start = info.set_thetaeq.back();
            int sz = info.sizeclus.back();
            std::vector<int> cluster(sz);
            for (int k = 0; k < sz; ++k) {
                cluster[k] = Itheta[start + k];
            }
            info.clusters.push_back(cluster);
        }
    }

    if (flagclus && info.numclus > 0) {
        info.sizeclus.back()++;
        int start = info.set_thetaeq.back();
        int sz = info.sizeclus.back();
        std::vector<int> cluster(sz);
        for (int k = 0; k < sz; ++k) {
            cluster[k] = Itheta[start + k];
        }
        info.clusters.push_back(cluster);
    }

    return info;
}

bool check_cluster_feasibility(const std::vector<bool>& spots,
                               const std::vector<int>& Itheta,
                               const VectorXd& bu_v, const VectorXd& bu,
                               double err, int U) {
    bool flagclus = false;
    std::vector<int> set_thetaeq;
    std::vector<int> sizeclus;
    int numclus = 0;

    for (int i = 0; i < U; ++i) {
        if (spots[i] && !flagclus) {
            numclus++;
            set_thetaeq.push_back(i);
            sizeclus.push_back(1);
            flagclus = true;
        } else if (spots[i]) {
            sizeclus.back()++;
        } else if (flagclus) {
            sizeclus.back()++;
            flagclus = false;
            int start = set_thetaeq.back();
            int sz = sizeclus.back();
            double sum_excess = 0.0;
            for (int k = 0; k < sz; ++k) {
                int u = Itheta[start + k];
                sum_excess += bu_v(u) - bu(u);
            }
            if (sum_excess < -err * sz) return false;
        } else {
            if (bu_v(Itheta[i]) - bu(Itheta[i]) < -err) return false;
        }
    }

    if (flagclus && numclus > 0) {
        sizeclus.back()++;
        int start = set_thetaeq.back();
        int sz = sizeclus.back();
        double sum_excess = 0.0;
        for (int k = 0; k < sz; ++k) {
            int u = Itheta[start + k];
            sum_excess += bu_v(u) - bu(u);
        }
        if (sum_excess < -err * sz) return false;
    }

    return true;
}

/**
 * @brief Check if a point is in the convex hull of a set of vertices
 *        using Frank-Wolfe with exact line search
 */
bool in_convex_hull(const MatrixXd& vertices, const VectorXd& target,
                    double tol) {
    int m = vertices.rows();
    int d = vertices.cols();
    if (m == 0) return false;

    VectorXd alpha = VectorXd::Constant(m, 1.0 / m);

    for (int iter = 0; iter < 2000; ++iter) {
        VectorXd point = vertices.transpose() * alpha;
        VectorXd r = point - target;

        if (r.norm() < tol) return true;

        VectorXd grad = vertices * r;

        int j;
        grad.minCoeff(&j);

        VectorXd dv = vertices.row(j).transpose() - point;
        double denom = dv.squaredNorm();
        double gamma;
        if (denom > 1e-15) {
            gamma = std::max(0.0, std::min(1.0, -r.dot(dv) / denom));
        } else {
            gamma = 0.0;
        }

        if (gamma < 1e-15) break;

        alpha *= (1.0 - gamma);
        alpha(j) += gamma;
    }

    VectorXd point = vertices.transpose() * alpha;
    return (point - target).norm() < tol;
}

/**
 * @brief Generate binary matrix (de2bi equivalent)
 */
MatrixXd de2bi(int max_val, int nbits) {
    int nrows = max_val + 1;
    MatrixXd B = MatrixXd::Zero(nrows, nbits);
    for (int i = 0; i < nrows; ++i) {
        int val = i;
        for (int j = 0; j < nbits; ++j) {
            B(i, j) = val % 2;
            val /= 2;
        }
    }
    return B;
}

struct VertexInfo {
    VectorXd bu_v;
    MatrixXd bun;
    std::vector<int> order;
};

struct ProcessResult {
    int feas_flag;
    std::vector<AdmMACVertex> big_info;
    VectorXd bu_a;
};

ProcessResult process_clusters(
    const std::vector<MatrixXcd>& H_tones,
    const std::vector<int>& Lxu, const std::vector<int>& idx_start,
    const std::vector<int>& idx_end,
    const std::vector<std::vector<MatrixXcd>>& Rxxs,
    const VectorXd& bu_v_init, const MatrixXd& bun_init,
    const VectorXd& bu_min,
    const ClusterInfo& cinfo, double err, double conv_tol, int cb,
    int Ly, int N, int U) {

    ProcessResult pr;
    pr.feas_flag = 0;
    pr.bu_a = bu_v_init;

    std::vector<MatrixXcd> Sinit(N);
    VectorXd cumrateinit(N);

    int cumsize_offset = 0;

    // build all permutation orders across clusters
    int total_perms = 0;
    for (int jdx = 0; jdx < cinfo.numclus; ++jdx) {
        total_perms += cinfo.sizeclus[jdx] - 1;
    }

    std::vector<std::vector<int>> all_orders(total_perms, std::vector<int>(U));
    for (int i = 0; i < total_perms; ++i) {
        std::iota(all_orders[i].begin(), all_orders[i].end(), 0);
    }

    std::vector<int> cumsize(cinfo.numclus + 1, 0);
    for (int jdx = 0; jdx < cinfo.numclus; ++jdx) {
        cumsize[jdx + 1] = cumsize[jdx] + cinfo.sizeclus[jdx] - 1;
    }

    for (int jdx = 0; jdx < cinfo.numclus; ++jdx) {
        int start = cinfo.set_thetaeq[jdx];
        int sz = cinfo.sizeclus[jdx];

        for (int i = 0; i < sz - 1; ++i) {
            int row = cumsize[jdx] + i;
            for (int k = 0; k < sz; ++k) {
                all_orders[row][start + k] = start + ((k + i + 1) % sz);
            }
        }
    }

    for (int jdx = 0; jdx < cinfo.numclus; ++jdx) {
        pr.feas_flag = 0;
        int start = cinfo.set_thetaeq[jdx];
        int sz = cinfo.sizeclus[jdx];

        if (jdx == 0) {
            for (int n = 0; n < N; ++n) {
                Sinit[n] = MatrixXcd::Identity(Ly, Ly);
                for (int i = 0; i < start; ++i) {
                    MatrixXcd Hu = H_tones[n].middleCols(
                        idx_start[i], idx_end[i] - idx_start[i] + 1);
                    Sinit[n] += Hu * Rxxs[i][n] * Hu.adjoint();
                }
                Sinit[n] = make_hermitian(Sinit[n]);
                cumrateinit(n) =
                    (1.0 / cb) * logdet_spd(Sinit[n]) / std::log(2.0);
            }
        } else {
            int prev_start = cinfo.set_thetaeq[jdx - 1];
            int prev_sz = cinfo.sizeclus[jdx - 1];
            for (int n = 0; n < N; ++n) {
                for (int i = prev_start; i < prev_start + prev_sz; ++i) {
                    MatrixXcd Hu = H_tones[n].middleCols(
                        idx_start[i], idx_end[i] - idx_start[i] + 1);
                    Sinit[n] += Hu * Rxxs[i][n] * Hu.adjoint();
                    Sinit[n] = make_hermitian(Sinit[n]);
                }
                cumrateinit(n) =
                    (1.0 / cb) * logdet_spd(Sinit[n]) / std::log(2.0);
            }
        }

        // initial vertex info
        VertexInfo init_vi;
        init_vi.bu_v = bu_v_init;
        init_vi.bun = bun_init;
        init_vi.order.resize(U);
        for (int i = 0; i < U; ++i) init_vi.order[i] = U - 1 - i;

        std::vector<VertexInfo> known_vertices;
        known_vertices.push_back(init_vi);

        // extract cluster rates from initial vertex
        VectorXd bd_row0(sz);
        for (int k = 0; k < sz; ++k) {
            bd_row0(k) = bu_v_init(start + k);
        }

        // bd_vertices: each row is a cluster-rate vector from an ordering
        std::vector<VectorXd> bd_vertex_list;
        bd_vertex_list.push_back(bd_row0);

        // check if achievable without time sharing
        double min_excess = 1e30;
        for (int k = 0; k < sz; ++k) {
            min_excess =
                std::min(min_excess, bd_row0(k) - bu_min(start + k));
        }

        if (min_excess >= -conv_tol) {
            AdmMACVertex v;
            v.bu_v = init_vi.bu_v;
            v.bun = init_vi.bun;
            v.order = init_vi.order;
            v.frac = 1.0;
            v.cluster_id = jdx;
            pr.big_info.push_back(v);
            pr.feas_flag = 2;
            continue;
        }

        // build initial full vertex set (binary combinations)
        MatrixXd bin_mask = de2bi((1 << sz) - 1, sz);
        MatrixXd vertices(bin_mask.rows(), sz);
        for (int r = 0; r < bin_mask.rows(); ++r) {
            for (int c = 0; c < sz; ++c) {
                vertices(r, c) = bin_mask(r, c) * bd_row0(c);
            }
        }

        // iterate through permutations within cluster
        for (int idx = cumsize[jdx]; idx < cumsize[jdx] + sz - 1; ++idx) {
            // compute rates under permuted order
            MatrixXd cumrate = MatrixXd::Zero(sz + 1, N);
            cumrate.row(0) = cumrateinit.transpose();

            for (int n = 0; n < N; ++n) {
                int rel_idx = 1;
                MatrixXcd S = Sinit[n];

                for (int k = start; k < start + sz; ++k) {
                    int u_or = all_orders[idx][k];
                    MatrixXcd Hu = H_tones[n].middleCols(
                        idx_start[u_or], idx_end[u_or] - idx_start[u_or] + 1);
                    S = S + Hu * Rxxs[u_or][n] * Hu.adjoint();
                    S = make_hermitian(S);
                    cumrate(rel_idx, n) =
                        (1.0 / cb) * logdet_spd(S) / std::log(2.0);
                    rel_idx++;
                }
            }

            // compute new bun and bu_v
            MatrixXd bun_new = bun_init;
            for (int k = 0; k < sz; ++k) {
                for (int n = 0; n < N; ++n) {
                    bun_new(start + k, n) =
                        cumrate(k + 1, n) - cumrate(k, n);
                }
            }
            // apply permutation: bun_new(order(idx,:),:) = bun_new
            MatrixXd bun_perm = bun_new;
            for (int k = 0; k < U; ++k) {
                bun_perm.row(all_orders[idx][k]) = bun_new.row(k);
            }
            bun_new = bun_perm;

            VectorXd bu_v_new = bun_new.rowwise().sum();

            VertexInfo vi;
            vi.bu_v = bu_v_new;
            vi.bun = bun_new;
            vi.order.resize(U);
            for (int k = 0; k < U; ++k) {
                vi.order[k] = all_orders[idx][U - 1 - k];
            }
            known_vertices.push_back(vi);

            VectorXd bd_new(sz);
            for (int k = 0; k < sz; ++k) {
                bd_new(k) = bu_v_new(start + k);
            }
            bd_vertex_list.push_back(bd_new);

            // build extended vertex set with binary combinations
            MatrixXd new_bin_verts(bin_mask.rows(), sz);
            for (int r = 0; r < bin_mask.rows(); ++r) {
                for (int c = 0; c < sz; ++c) {
                    new_bin_verts(r, c) = bin_mask(r, c) * bd_new(c);
                }
            }

            // concatenate
            int old_rows = vertices.rows();
            vertices.conservativeResize(old_rows + new_bin_verts.rows(), sz);
            vertices.bottomRows(new_bin_verts.rows()) = new_bin_verts;

            // target point for this cluster
            VectorXd target(sz);
            for (int k = 0; k < sz; ++k) {
                target(k) = bu_min(start + k);
            }

            if (in_convex_hull(vertices, target, conv_tol)) {
                pr.feas_flag = 2;

                // build bd_vertices matrix and solve for fractions
                int nv = bd_vertex_list.size();
                MatrixXd bd_mat(nv, sz);
                for (int r = 0; r < nv; ++r) {
                    bd_mat.row(r) = bd_vertex_list[r].transpose();
                }

                // solve bd_mat' * frac = target
                VectorXd frac = bd_mat.transpose()
                                    .colPivHouseholderQr()
                                    .solve(target);

                for (int v = 0; v < nv; ++v) {
                    if (frac(v) >= err) {
                        AdmMACVertex av;
                        av.bu_v = known_vertices[v].bu_v;
                        av.bun = known_vertices[v].bun;
                        av.order = known_vertices[v].order;
                        av.frac = frac(v);
                        av.cluster_id = jdx;
                        pr.big_info.push_back(av);
                    }
                }

                // update bu_a for cluster users
                VectorXd weighted_rates = VectorXd::Zero(sz);
                for (const auto& av : pr.big_info) {
                    if (av.cluster_id == jdx) {
                        for (int k = 0; k < sz; ++k) {
                            weighted_rates(k) +=
                                av.frac * av.bu_v(start + k);
                        }
                    }
                }
                for (int k = 0; k < sz; ++k) {
                    pr.bu_a(start + k) = weighted_rates(k);
                }

                break;
            }
        }

        if (pr.feas_flag == 0) {
            return pr;
        }
    }

    if (pr.big_info.empty()) {
        pr.feas_flag = 0;
    }

    return pr;
}

/**
 * @brief Convert maxRMACResult Rxx (original order) to reordered cell-like storage
 */
std::vector<std::vector<MatrixXcd>> reorder_rxx(
    const MaxRMACResult& mr, const std::vector<int>& Lxu_vec,
    const std::vector<int>& Itheta, int U, int N, bool from_cell) {

    std::vector<std::vector<MatrixXcd>> Rxx_reord(U, std::vector<MatrixXcd>(N));

    for (int u = 0; u < U; ++u) {
        for (int n = 0; n < N; ++n) {
            Rxx_reord[u][n] = mr.Rxx[Itheta[u]][n];
        }
    }

    return Rxx_reord;
}

}  // namespace

AdmMACResult admMACMIMO(const std::vector<MatrixXcd>& H,
                         const std::vector<int>& Lxu, const VectorXd& bu,
                         const VectorXd& Eu, int cb,
                         const AdmMACConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();

    const int Ly = H[0].rows();
    const int Ltot = H[0].cols();
    int N_in = H.size();
    const int U = Lxu.size();

    if (bu.size() != U || Eu.size() != U) {
        throw std::invalid_argument("bu and Eu must have length U");
    }
    if (cb != 1 && cb != 2) {
        throw std::invalid_argument("cb must be 1 or 2");
    }

    int Lxu_sum = 0;
    for (int u = 0; u < U; ++u) Lxu_sum += Lxu[u];
    if (Lxu_sum != Ltot) {
        throw std::invalid_argument("sum(Lxu) must equal H column count");
    }

    AdmMACResult result;
    result.feas_flag = 0;
    result.bu_a = VectorXd::Zero(U);
    result.iterations = 0;

    VectorXd theta = VectorXd::Ones(U);
    for (int u = 0; u < U; ++u) {
        theta(u) += 0.05 * (u + 1);
    }
    MatrixXd A = static_cast<double>(U) * MatrixXd::Identity(U, U);

    const double err = config.err;
    const double conv_tol = config.conv_tol;
    const int max_iter = config.max_iter;

    MaxRMACConfig mr_config;
    mr_config.num_threads = config.num_threads;

    // working copies of channel for reordering
    int N = N_in;
    if (N_in > 1 && cb == 2) {
        N = N_in / 2;
    }

    MaxRMACResult last_mr;
    MatrixXd last_bun;
    VectorXd last_w;

    std::vector<int> idx_end_orig(U), idx_start_orig(U);
    idx_start_orig[0] = 0;
    idx_end_orig[0] = Lxu[0] - 1;
    for (int u = 1; u < U; ++u) {
        idx_start_orig[u] = idx_end_orig[u - 1] + 1;
        idx_end_orig[u] = idx_start_orig[u] + Lxu[u] - 1;
    }

    int count = 0;

    while (count < max_iter) {
        count++;

        MaxRMACResult mr =
            maxrmac::maxRMACMIMO(H, Lxu, Eu, theta, cb, mr_config);

        last_mr = mr;
        last_bun = mr.bun;
        last_w = mr.w;

        VectorXd bu_v = mr.bun.rowwise().sum();
        VectorXd g = bu_v - bu;
        result.bu_a = bu_v;

        // infeasibility check
        if (theta.dot(g) < -err) {
            result.feas_flag = 0;
            result.Eun = MatrixXd::Zero(0, 0);
            result.theta = theta;
            result.w = mr.w;
            result.iterations = count;
            auto end_time = std::chrono::high_resolution_clock::now();
            result.elapsed_sec =
                std::chrono::duration<double>(end_time - start_time).count();
            return result;
        }

        // feasibility: all users meet target
        if (g.minCoeff() >= 0) {
            result.feas_flag = 1;
            result.Rxx = mr.Rxx;
            result.Eun = mr.Eun;
            result.theta = theta;
            result.w = mr.w;
            result.iterations = count;

            AdmMACVertex v;
            v.bu_v = bu_v;
            v.bun = mr.bun;
            v.order.resize(U);
            std::vector<int> sort_idx(U);
            std::iota(sort_idx.begin(), sort_idx.end(), 0);
            std::sort(sort_idx.begin(), sort_idx.end(),
                      [&](int a, int b) { return theta(a) < theta(b); });
            v.order = sort_idx;
            v.frac = 1.0;
            v.cluster_id = 0;
            result.vertices.push_back(v);

            auto end_time = std::chrono::high_resolution_clock::now();
            result.elapsed_sec =
                std::chrono::duration<double>(end_time - start_time).count();
            return result;
        }

        // check for equal thetas (vertex sharing)
        std::vector<int> Itheta(U);
        std::iota(Itheta.begin(), Itheta.end(), 0);
        std::sort(Itheta.begin(), Itheta.end(),
                  [&](int a, int b) { return theta(a) > theta(b); });

        VectorXd stheta(U);
        for (int i = 0; i < U; ++i) stheta(i) = theta(Itheta[i]);

        VectorXd delta(U);
        for (int i = 0; i < U - 1; ++i) {
            delta(i) = stheta(i) - stheta(i + 1);
        }
        delta(U - 1) = stheta(U - 1);

        std::vector<bool> spots(U, false);
        for (int i = 0; i < U; ++i) {
            double rel_d = delta(i) / std::max(stheta(i), err);
            spots[i] = (rel_d < 0.10);
        }
        spots[U - 1] = false;

        bool any_spots = false;
        for (int i = 0; i < U; ++i) {
            if (spots[i]) {
                any_spots = true;
                break;
            }
        }

        if (!any_spots) {
            update_ellipsoid(theta, A, g, U);
            continue;
        }

        ClusterInfo cinfo = form_clusters(spots, Itheta, U);

        if (!check_cluster_feasibility(spots, Itheta, bu_v, bu, err, U)) {
            update_ellipsoid(theta, A, g, U);
            continue;
        }

        // reorder by theta for vertex sharing
        std::vector<int> Lxu_reord(U);
        std::vector<int> hidx;
        for (int i = 0; i < U; ++i) {
            Lxu_reord[i] = Lxu[Itheta[i]];
            for (int j = idx_start_orig[Itheta[i]];
                 j <= idx_end_orig[Itheta[i]]; ++j) {
                hidx.push_back(j);
            }
        }

        std::vector<int> idx_end_reord(U), idx_start_reord(U);
        idx_start_reord[0] = 0;
        idx_end_reord[0] = Lxu_reord[0] - 1;
        for (int u = 1; u < U; ++u) {
            idx_start_reord[u] = idx_end_reord[u - 1] + 1;
            idx_end_reord[u] = idx_start_reord[u] + Lxu_reord[u] - 1;
        }

        // build reordered channel tones
        // we need the working tones (after cb=2 halving)
        std::vector<MatrixXcd> H_tones;
        if (N_in > 1 && cb == 2) {
            H_tones.resize(N_in / 2);
            for (int n = 0; n < N_in / 2; ++n) {
                H_tones[n].resize(Ly, Ltot);
                for (int j = 0; j < Ltot; ++j) {
                    H_tones[n].col(j) = H[n].col(hidx[j]);
                }
            }
        } else {
            H_tones.resize(N_in);
            for (int n = 0; n < N_in; ++n) {
                H_tones[n].resize(Ly, Ltot);
                for (int j = 0; j < Ltot; ++j) {
                    H_tones[n].col(j) = H[n].col(hidx[j]);
                }
            }
        }
        int N_work = H_tones.size();

        // reorder Rxx, bun, bu_v
        auto Rxx_reord = reorder_rxx(mr, Lxu, Itheta, U, N_work, true);

        VectorXd bu_v_reord(U);
        MatrixXd bun_reord(U, N_work);
        VectorXd bu_min_reord(U);
        for (int i = 0; i < U; ++i) {
            bu_v_reord(i) = bu_v(Itheta[i]);
            bu_min_reord(i) = bu(Itheta[i]);
            for (int n = 0; n < N_work; ++n) {
                bun_reord(i, n) = mr.bun(Itheta[i], n);
            }
        }

        ProcessResult pr = process_clusters(
            H_tones, Lxu_reord, idx_start_reord, idx_end_reord, Rxx_reord,
            bu_v_reord, bun_reord, bu_min_reord, cinfo, err, conv_tol, cb,
            Ly, N_work, U);

        if (pr.feas_flag == 0) {
            update_ellipsoid(theta, A, g, U);
            continue;
        }

        result.feas_flag = pr.feas_flag;

        // restore original ordering for bu_a
        result.bu_a = VectorXd::Zero(U);
        for (int i = 0; i < U; ++i) {
            result.bu_a(Itheta[i]) = pr.bu_a(i);
        }

        // restore ordering in vertex info
        for (auto& v : pr.big_info) {
            VectorXd bu_v_restored = VectorXd::Zero(U);
            MatrixXd bun_restored = MatrixXd::Zero(U, N_work);
            for (int i = 0; i < U; ++i) {
                bu_v_restored(Itheta[i]) = v.bu_v(i);
                bun_restored.row(Itheta[i]) = v.bun.row(i);
            }
            v.bu_v = bu_v_restored;
            v.bun = bun_restored;

            std::vector<int> new_order(U);
            for (int i = 0; i < U; ++i) {
                new_order[i] = Itheta[v.order[i]];
            }
            v.order = new_order;
        }

        result.vertices = pr.big_info;

        // restore Rxx ordering
        result.Rxx.resize(U);
        for (int u = 0; u < U; ++u) {
            result.Rxx[Itheta[u]].resize(N_work);
            for (int n = 0; n < N_work; ++n) {
                result.Rxx[Itheta[u]][n] = Rxx_reord[u][n];
            }
        }

        result.Eun = mr.Eun;
        result.theta = theta;
        result.w = mr.w;
        result.iterations = count;

        break;
    }

    if (result.feas_flag == 0) {
        result.Eun = MatrixXd::Zero(0, 0);
        result.theta = theta;
        result.w = last_w;
        result.iterations = count;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_sec =
        std::chrono::duration<double>(end_time - start_time).count();

    if (config.verbose) {
        std::cout << "admMACMIMO: completed in " << result.elapsed_sec
                  << " s (FEAS_FLAG=" << result.feas_flag
                  << ", iters=" << result.iterations << ")" << std::endl;
    }

    return result;
}

}  // namespace admmac
