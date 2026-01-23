/**
 * @file minpmac_mex.cpp
 * @brief MEX gateway for minPMAC MIMO solver
 *
 * Compile with CMake (recommended) or:
 *   mex -R2018a minpmac_mex.cpp solve_tone.cpp minpmac.cpp
 *       -I/path/to/eigen3 -lglpk CXXFLAGS='$CXXFLAGS -fopenmp -O3'
 *       LDFLAGS='$LDFLAGS -fopenmp'
 */

#include <complex>
#include <cstring>
#include <vector>

#include "matrix.h"
#include "mex.h"
#include "minpmac.hpp"

using namespace minpmac;

/**
 * Gateway function: [feas_flag, bu_a, bun, frac, Eu_avg] = minpmac_mex(H, Lxu,
 * bu_min, w, cb)
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 5) {
        mexErrMsgIdAndTxt("minpmac:nrhs",
                          "Five inputs required: H, Lxu, bu_min, w, cb");
    }
    if (nlhs < 5) {
        mexErrMsgIdAndTxt("minpmac:nlhs",
                          "Five outputs required: feas_flag, bu_a, bun, frac, "
                          "Eu_avg");
    }

    const mxArray* H_mx = prhs[0];
    if (!mxIsComplex(H_mx) && !mxIsDouble(H_mx)) {
        mexErrMsgIdAndTxt("minpmac:invalidInput", "H must be a double array");
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
        mexErrMsgIdAndTxt("minpmac:invalidInput", "H must be 2D or 3D array");
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

    double* bu_min_data = mxGetPr(prhs[2]);
    size_t U = mxGetNumberOfElements(prhs[2]);
    VectorXd bu_min(U);
    for (size_t i = 0; i < U; ++i) {
        bu_min(i) = bu_min_data[i];
    }

    double* w_data = mxGetPr(prhs[3]);
    VectorXd w(U);
    for (size_t i = 0; i < U; ++i) {
        w(i) = w_data[i];
    }

    int cb = static_cast<int>(mxGetScalar(prhs[4]));

    if (Lxu.size() == 1) {
        int Lxu_scalar = Lxu[0];
        Lxu.resize(U, Lxu_scalar);
    }

    MinPMACConfig config;
    MinPMACResult result = minPMACMIMO(H, Lxu, bu_min, w, cb, config);

    const int N = result.bun.cols();

    plhs[0] = mxCreateDoubleScalar(static_cast<double>(result.feas_flag));

    plhs[1] = mxCreateDoubleMatrix(U, 1, mxREAL);
    double* bu_a_out = mxGetPr(plhs[1]);
    for (size_t u = 0; u < U; ++u) {
        bu_a_out[u] = result.bu_a(u);
    }

    plhs[2] = mxCreateDoubleMatrix(U, N, mxREAL);
    double* bun_out = mxGetPr(plhs[2]);
    for (int n = 0; n < N; ++n) {
        for (size_t u = 0; u < U; ++u) {
            bun_out[u + U * n] = result.bun(u, n);
        }
    }

    size_t num_frac = result.frac.size();
    if (num_frac > 0) {
        plhs[3] = mxCreateDoubleMatrix(num_frac, 1, mxREAL);
        double* frac_out = mxGetPr(plhs[3]);
        for (size_t i = 0; i < num_frac; ++i) {
            frac_out[i] = result.frac(i);
        }
    } else {
        plhs[3] = mxCreateDoubleMatrix(0, 0, mxREAL);
    }

    plhs[4] = mxCreateDoubleMatrix(U, 1, mxREAL);
    double* Eu_avg_out = mxGetPr(plhs[4]);
    for (size_t u = 0; u < U; ++u) {
        Eu_avg_out[u] = result.Eu_avg(u);
    }
}
