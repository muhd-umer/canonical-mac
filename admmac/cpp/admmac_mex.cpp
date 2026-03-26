/**
 * @file admmac_mex.cpp
 * @brief MEX gateway for admMACMIMO feasibility solver
 */

#include <complex>
#include <cstring>
#include <vector>

#include "admmac.hpp"
#include "matrix.h"
#include "mex.h"

using namespace admmac;

/**
 * Gateway: [FEAS_FLAG, bu_a, Rxxs, Eun, theta, w, info] =
 *          admmac_mex(H, Lxu, bu, Eu, cb)
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 5) {
        mexErrMsgIdAndTxt("admmac:nrhs",
                          "Five inputs required: H, Lxu, bu, Eu, cb");
    }
    if (nlhs < 7) {
        mexErrMsgIdAndTxt("admmac:nlhs",
                          "Seven outputs required: FEAS_FLAG, bu_a, Rxxs, "
                          "Eun, theta, w, info");
    }

    const mxArray* H_mx = prhs[0];
    if (!mxIsComplex(H_mx) && !mxIsDouble(H_mx)) {
        mexErrMsgIdAndTxt("admmac:invalidInput", "H must be a double array");
    }

    const mwSize* H_dims = mxGetDimensions(H_mx);
    mwSize ndims = mxGetNumberOfDimensions(H_mx);

    int Ly, Ltot, N_in;
    if (ndims == 2) {
        Ly = static_cast<int>(H_dims[0]);
        Ltot = static_cast<int>(H_dims[1]);
        N_in = 1;
    } else if (ndims == 3) {
        Ly = static_cast<int>(H_dims[0]);
        Ltot = static_cast<int>(H_dims[1]);
        N_in = static_cast<int>(H_dims[2]);
    } else {
        mexErrMsgIdAndTxt("admmac:invalidInput", "H must be 2D or 3D");
    }

    std::vector<MatrixXcd> H(N_in);
    if (mxIsComplex(H_mx)) {
        mxComplexDouble* H_data = mxGetComplexDoubles(H_mx);
        for (int n = 0; n < N_in; ++n) {
            H[n].resize(Ly, Ltot);
            for (int j = 0; j < Ltot; ++j) {
                for (int i = 0; i < Ly; ++i) {
                    size_t idx = i + j * Ly + n * Ly * Ltot;
                    H[n](i, j) = Complex(H_data[idx].real, H_data[idx].imag);
                }
            }
        }
    } else {
        double* H_data = mxGetDoubles(H_mx);
        for (int n = 0; n < N_in; ++n) {
            H[n].resize(Ly, Ltot);
            for (int j = 0; j < Ltot; ++j) {
                for (int i = 0; i < Ly; ++i) {
                    size_t idx = i + j * Ly + n * Ly * Ltot;
                    H[n](i, j) = Complex(H_data[idx], 0.0);
                }
            }
        }
    }

    double* Lxu_data = mxGetPr(prhs[1]);
    size_t Lxu_len = mxGetNumberOfElements(prhs[1]);
    std::vector<int> Lxu(Lxu_len);
    for (size_t i = 0; i < Lxu_len; ++i) {
        Lxu[i] = static_cast<int>(Lxu_data[i]);
    }

    double* bu_data = mxGetPr(prhs[2]);
    size_t U = mxGetNumberOfElements(prhs[2]);
    VectorXd bu(U);
    for (size_t i = 0; i < U; ++i) bu(i) = bu_data[i];

    double* Eu_data = mxGetPr(prhs[3]);
    VectorXd Eu(U);
    for (size_t i = 0; i < U; ++i) Eu(i) = Eu_data[i];

    int cb = static_cast<int>(mxGetScalar(prhs[4]));

    if (Lxu.size() == 1) {
        int Lxu_scalar = Lxu[0];
        Lxu.resize(U, Lxu_scalar);
    }

    AdmMACConfig config;
    config.verbose = true;
    AdmMACResult result = admMACMIMO(H, Lxu, bu, Eu, cb, config);

    plhs[0] = mxCreateDoubleScalar(static_cast<double>(result.feas_flag));

    plhs[1] = mxCreateDoubleMatrix(U, 1, mxREAL);
    double* bu_a_out = mxGetPr(plhs[1]);
    for (size_t u = 0; u < U; ++u) bu_a_out[u] = result.bu_a(u);

    if (result.feas_flag > 0 && !result.Rxx.empty()) {
        int N_out = result.Rxx[0].size();
        plhs[2] = mxCreateCellMatrix(U, N_out);
        for (int u = 0; u < static_cast<int>(U); ++u) {
            int Lu = Lxu[u];
            for (int n = 0; n < N_out; ++n) {
                mxArray* Rxx_mx = mxCreateDoubleMatrix(Lu, Lu, mxCOMPLEX);
                mxComplexDouble* Rxx_data = mxGetComplexDoubles(Rxx_mx);
                for (int j = 0; j < Lu; ++j) {
                    for (int i = 0; i < Lu; ++i) {
                        size_t idx = i + j * Lu;
                        Rxx_data[idx].real = std::real(result.Rxx[u][n](i, j));
                        Rxx_data[idx].imag = std::imag(result.Rxx[u][n](i, j));
                    }
                }
                mxSetCell(plhs[2], u + U * n, Rxx_mx);
            }
        }
    } else {
        plhs[2] = mxCreateDoubleScalar(0.0);
    }

    if (result.feas_flag > 0 && result.Eun.size() > 0) {
        int N_out = result.Eun.cols();
        plhs[3] = mxCreateDoubleMatrix(U, N_out, mxREAL);
        double* Eun_out = mxGetPr(plhs[3]);
        for (int n = 0; n < N_out; ++n) {
            for (int u = 0; u < static_cast<int>(U); ++u) {
                Eun_out[u + U * n] = result.Eun(u, n);
            }
        }
    } else {
        plhs[3] = mxCreateDoubleScalar(0.0);
    }

    plhs[4] = mxCreateDoubleMatrix(U, 1, mxREAL);
    double* theta_out = mxGetPr(plhs[4]);
    for (size_t u = 0; u < U; ++u) theta_out[u] = result.theta(u);

    plhs[5] = mxCreateDoubleMatrix(U, 1, mxREAL);
    double* w_out = mxGetPr(plhs[5]);
    for (size_t u = 0; u < U; ++u) w_out[u] = result.w(u);

    size_t nv = result.vertices.size();
    if (nv > 0) {
        const char* field_names[] = {"bu_v", "bun", "order", "frac",
                                     "clusterID"};
        mxArray* info_struct = mxCreateStructMatrix(nv, 1, 5, field_names);

        for (size_t v = 0; v < nv; ++v) {
            const auto& vert = result.vertices[v];

            mxArray* bu_v_mx = mxCreateDoubleMatrix(1, U, mxREAL);
            double* bu_v_out = mxGetPr(bu_v_mx);
            for (size_t u = 0; u < U; ++u) bu_v_out[u] = vert.bu_v(u);
            mxSetField(info_struct, v, "bu_v", bu_v_mx);

            int N_bun = vert.bun.cols();
            mxArray* bun_mx = mxCreateDoubleMatrix(U, N_bun, mxREAL);
            double* bun_out = mxGetPr(bun_mx);
            for (int n = 0; n < N_bun; ++n) {
                for (size_t u = 0; u < U; ++u) {
                    bun_out[u + U * n] = vert.bun(u, n);
                }
            }
            mxSetField(info_struct, v, "bun", bun_mx);

            mxArray* order_mx = mxCreateDoubleMatrix(1, U, mxREAL);
            double* order_out = mxGetPr(order_mx);
            for (size_t u = 0; u < U; ++u) {
                order_out[u] = static_cast<double>(vert.order[u] + 1);
            }
            mxSetField(info_struct, v, "order", order_mx);

            mxSetField(info_struct, v, "frac", mxCreateDoubleScalar(vert.frac));
            mxSetField(
                info_struct, v, "clusterID",
                mxCreateDoubleScalar(static_cast<double>(vert.cluster_id + 1)));
        }

        plhs[6] = info_struct;
    } else {
        plhs[6] = mxCreateCellMatrix(0, 0);
    }
}
