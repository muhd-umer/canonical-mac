#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "admmac.hpp"

using namespace admmac;

struct TestResult {
    bool passed;
    std::string name;
    double elapsed_ms;
};

template <typename Func>
TestResult run_test(const std::string& name, Func test_func) {
    TestResult result;
    result.name = name;

    try {
        auto start = std::chrono::high_resolution_clock::now();
        result = test_func();
        auto end = std::chrono::high_resolution_clock::now();
        result.elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        result.name = name;
    } catch (const std::exception& e) {
        result.passed = false;
        result.elapsed_ms = 0;
        std::cout << "  [x] test crashed: " << e.what() << std::endl;
    }

    return result;
}

bool is_psd(const MatrixXcd& M, double tol = -1e-9) {
    MatrixXcd H = 0.5 * (M + M.adjoint());
    Eigen::SelfAdjointEigenSolver<MatrixXcd> es(H);
    return es.eigenvalues().minCoeff() >= tol;
}

TestResult validate_feasible(const AdmMACResult& result, const VectorXd& bu,
                             const VectorXd& Eu,
                             const std::string& name) {
    TestResult test;
    test.passed = true;
    test.name = name;

    const int U = bu.size();

    std::cout << "  config: " << U << " users, bu = [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(2) << bu(u) << (u < U - 1 ? " " : "");
    std::cout << "], Eu = [";
    for (int u = 0; u < U; ++u)
        std::cout << Eu(u) << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  FEAS_FLAG: " << result.feas_flag << std::endl;

    if (result.feas_flag <= 0) {
        std::cout << "  [x] expected feasible but got FEAS_FLAG="
                  << result.feas_flag << std::endl;
        test.passed = false;
        return test;
    }

    std::cout << "  target rates: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4) << bu(u) << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  achieved rates: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4) << result.bu_a(u) << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  rate margin: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4) << result.bu_a(u) - bu(u)
                  << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  theta: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4) << result.theta(u) << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  w: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4) << result.w(u) << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  elapsed: " << std::fixed << std::setprecision(3)
              << result.elapsed_sec * 1000 << " ms" << std::endl;

    double min_margin = 1e30;
    for (int u = 0; u < U; ++u)
        min_margin = std::min(min_margin, result.bu_a(u) - bu(u));

    if (min_margin >= -1e-4) {
        std::cout << "  [p] all target rates met (min margin: " << std::fixed
                  << std::setprecision(6) << min_margin << ")" << std::endl;
    } else {
        std::cout << "  [x] some target rates not met (min margin: " << min_margin
                  << ")" << std::endl;
        test.passed = false;
    }

    if (result.Eun.rows() > 0) {
        VectorXd energy_used = result.Eun.rowwise().sum();
        bool energy_ok = true;
        for (int u = 0; u < U; ++u) {
            if (energy_used(u) - Eu(u) > 1e-4) energy_ok = false;
        }
        if (energy_ok) {
            std::cout << "  [p] all energy constraints satisfied" << std::endl;
        } else {
            std::cout << "  [x] some energy constraints violated" << std::endl;
            test.passed = false;
        }
    }

    if (!result.Rxx.empty() && !result.Rxx[0].empty()) {
        bool psd_ok = true;
        int N = result.Rxx[0].size();
        for (int u = 0; u < U && psd_ok; ++u) {
            for (int n = 0; n < N && psd_ok; ++n) {
                if (!is_psd(result.Rxx[u][n])) psd_ok = false;
            }
        }
        if (psd_ok) {
            std::cout << "  [p] covariance matrices are PSD" << std::endl;
        } else {
            std::cout << "  [x] some covariance matrices are not PSD" << std::endl;
            test.passed = false;
        }
    }

    if (!result.vertices.empty()) {
        std::cout << "  [p] info has " << result.vertices.size() << " vertices" << std::endl;
    }

    if (test.passed) {
        std::cout << "  [p] test passed: " << name << std::endl;
    } else {
        std::cout << "  [x] test failed: " << name << std::endl;
    }

    return test;
}

TestResult validate_infeasible(const AdmMACResult& result, const VectorXd& bu,
                               const VectorXd& Eu,
                               const std::string& name) {
    TestResult test;
    test.passed = true;
    test.name = name;

    const int U = bu.size();

    std::cout << "  config: " << U << " users, bu = [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(2) << bu(u) << (u < U - 1 ? " " : "");
    std::cout << "], Eu = [";
    for (int u = 0; u < U; ++u)
        std::cout << Eu(u) << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  FEAS_FLAG: " << result.feas_flag << std::endl;
    std::cout << "  elapsed: " << std::fixed << std::setprecision(3)
              << result.elapsed_sec * 1000 << " ms" << std::endl;

    if (result.feas_flag == 0) {
        std::cout << "  [p] infeasible case correctly detected" << std::endl;
    } else {
        std::cout << "  [x] expected infeasible but got FEAS_FLAG="
                  << result.feas_flag << std::endl;
        test.passed = false;
    }

    if (test.passed) {
        std::cout << "  [p] test passed: " << name << std::endl;
    } else {
        std::cout << "  [x] test failed: " << name << std::endl;
    }

    return test;
}

TestResult test_basic_siso_feasible() {
    std::cout << "[test] basic siso feasible (2 users, 4 tones)" << std::endl;

    const int N = 4;
    std::vector<int> Lxu = {1, 1};
    VectorXd bu(2); bu << 5, 5;
    VectorXd Eu(2); Eu << 1, 1;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(1, 2); H[0] << Complex(80, 0), Complex(60, 0);
    H[1].resize(1, 2); H[1] << Complex(40, 0), Complex(30, 0);
    H[2].resize(1, 2); H[2] << Complex(50, 0), Complex(50, 0);
    H[3].resize(1, 2); H[3] << Complex(30, 0), Complex(40, 0);

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_feasible(result, bu, Eu, "basic siso feasible");
}

TestResult test_siso_infeasible_low_energy() {
    std::cout << "[test] siso infeasible (low energy)" << std::endl;

    const int N = 4;
    std::vector<int> Lxu = {1, 1};
    VectorXd bu(2); bu << 5, 5;
    VectorXd Eu(2); Eu << 0.001, 0.001;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(1, 2); H[0] << Complex(80, 0), Complex(60, 0);
    H[1].resize(1, 2); H[1] << Complex(40, 0), Complex(30, 0);
    H[2].resize(1, 2); H[2] << Complex(50, 0), Complex(50, 0);
    H[3].resize(1, 2); H[3] << Complex(30, 0), Complex(40, 0);

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_infeasible(result, bu, Eu, "siso infeasible (low energy)");
}

TestResult test_siso_infeasible_high_rate() {
    std::cout << "[test] siso infeasible (high rate)" << std::endl;

    const int N = 4;
    std::vector<int> Lxu = {1, 1};
    VectorXd bu(2); bu << 100, 100;
    VectorXd Eu(2); Eu << 1, 1;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(1, 2); H[0] << Complex(80, 0), Complex(60, 0);
    H[1].resize(1, 2); H[1] << Complex(40, 0), Complex(30, 0);
    H[2].resize(1, 2); H[2] << Complex(50, 0), Complex(50, 0);
    H[3].resize(1, 2); H[3] << Complex(30, 0), Complex(40, 0);

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_infeasible(result, bu, Eu, "siso infeasible (high rate)");
}

TestResult test_siso_vertex_sharing() {
    std::cout << "[test] siso vertex sharing (equal channels)" << std::endl;

    const int N = 2;
    std::vector<int> Lxu = {1, 1, 1};
    VectorXd bu(3); bu << 3, 2.5, 2;
    VectorXd Eu(3); Eu << 1, 1, 1;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(1, 3); H[0] << Complex(80, 0), Complex(80, 0), Complex(80, 0);
    H[1].resize(1, 3); H[1] << Complex(60, 0), Complex(60, 0), Complex(60, 0);

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_feasible(result, bu, Eu, "siso vertex sharing");
}

TestResult test_basic_mimo() {
    std::cout << "[test] basic mimo (2 users, 2x2, 2 tones)" << std::endl;

    const int Ly = 2, N = 2;
    std::vector<int> Lxu = {2, 2};
    VectorXd bu(2); bu << 5, 5;
    VectorXd Eu(2); Eu << 2, 2;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(Ly, 4);
    H[0] << Complex(3.2, 0), Complex(2.1, 0), Complex(2.5, 0), Complex(3.0, 0),
        Complex(1.8, 0), Complex(2.9, 0), Complex(2.2, 0), Complex(1.9, 0);
    H[1].resize(Ly, 4);
    H[1] << Complex(2.9, 0), Complex(2.5, 0), Complex(2.8, 0), Complex(2.3, 0),
        Complex(2.0, 0), Complex(3.1, 0), Complex(1.7, 0), Complex(2.6, 0);

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_feasible(result, bu, Eu, "basic mimo");
}

TestResult test_mimo_multi_tone() {
    std::cout << "[test] mimo multi-tone (8 tones)" << std::endl;

    const int Ly = 2, N = 8;
    std::vector<int> Lxu = {2, 2};
    VectorXd bu(2); bu << 8, 8;
    VectorXd Eu(2); Eu << 3, 3;
    int cb = 1;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(2.0));

    std::vector<MatrixXcd> H(N);
    for (int n = 0; n < N; ++n) {
        H[n].resize(Ly, 4);
        for (int i = 0; i < Ly; ++i)
            for (int j = 0; j < 4; ++j)
                H[n](i, j) = Complex(dist(gen), dist(gen));
    }

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_feasible(result, bu, Eu, "mimo multi-tone");
}

TestResult test_single_tone() {
    std::cout << "[test] single tone" << std::endl;

    std::vector<int> Lxu = {1, 1};
    VectorXd bu(2); bu << 2, 2;
    VectorXd Eu(2); Eu << 0.5, 0.5;
    int cb = 1;

    std::vector<MatrixXcd> H(1);
    H[0].resize(1, 2);
    H[0] << Complex(80, 0), Complex(60, 0);

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_feasible(result, bu, Eu, "single tone");
}

TestResult test_three_users_siso() {
    std::cout << "[test] three users siso" << std::endl;

    const int Ly = 2, N = 4;
    std::vector<int> Lxu = {1, 1, 1};
    VectorXd bu(3); bu << 2, 2, 2;
    VectorXd Eu(3); Eu << 1, 1, 1;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(Ly, 3);
    H[0] << Complex(5, 0), Complex(4, 0), Complex(3, 0),
            Complex(4, 0), Complex(5, 0), Complex(4, 0);
    H[1].resize(Ly, 3);
    H[1] << Complex(4, 0), Complex(3, 0), Complex(5, 0),
            Complex(3, 0), Complex(4, 0), Complex(5, 0);
    H[2].resize(Ly, 3);
    H[2] << Complex(3, 0), Complex(5, 0), Complex(4, 0),
            Complex(5, 0), Complex(3, 0), Complex(4, 0);
    H[3].resize(Ly, 3);
    H[3] << Complex(4, 0), Complex(4, 0), Complex(4, 0),
            Complex(4, 0), Complex(4, 0), Complex(4, 0);

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_feasible(result, bu, Eu, "three users siso");
}

TestResult test_mimo_unequal_antennas() {
    std::cout << "[test] mimo with unequal antennas" << std::endl;

    const int Ly = 3, N = 2;
    std::vector<int> Lxu = {2, 3};
    VectorXd bu(2); bu << 3, 3;
    VectorXd Eu(2); Eu << 2, 3;
    int cb = 1;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(2.0));

    std::vector<MatrixXcd> H(N);
    for (int n = 0; n < N; ++n) {
        H[n].resize(Ly, 5);
        for (int i = 0; i < Ly; ++i)
            for (int j = 0; j < 5; ++j)
                H[n](i, j) = Complex(dist(gen), dist(gen));
    }

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_feasible(result, bu, Eu, "mimo unequal antennas");
}

TestResult test_real_baseband() {
    std::cout << "[test] real baseband" << std::endl;

    const int Ly = 2, N = 4;
    std::vector<int> Lxu = {1, 1};
    VectorXd bu(2); bu << 1, 1;
    VectorXd Eu(2); Eu << 1, 1;
    int cb = 2;

    std::vector<MatrixXcd> H(N);
    H[0].resize(Ly, 2);
    H[0] << Complex(5, 0), Complex(4, 0),
            Complex(4, 0), Complex(5, 0);
    H[1].resize(Ly, 2);
    H[1] << Complex(3, 0), Complex(2, 0),
            Complex(2, 0), Complex(3, 0);
    H[2].resize(Ly, 2);
    H[2] << Complex(2, 0), Complex(3, 0),
            Complex(3, 0), Complex(2, 0);
    H[3].resize(Ly, 2);
    H[3] << Complex(4, 0), Complex(5, 0),
            Complex(5, 0), Complex(4, 0);

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_feasible(result, bu, Eu, "real baseband");
}

TestResult test_stress() {
    std::cout << "[test] stress test" << std::endl;

    const int Ly = 4, N = 64;
    std::vector<int> Lxu = {2, 1, 2, 1, 2, 1};
    const int U = Lxu.size();
    int Ltot = 0;
    for (int u = 0; u < U; ++u) Ltot += Lxu[u];

    VectorXd bu(U); bu << 4, 5, 4, 3, 4, 3;
    VectorXd Eu(U); Eu << 2, 3, 2, 1.5, 2, 1.5;
    int cb = 1;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(2.0));

    std::vector<MatrixXcd> H(N);
    for (int n = 0; n < N; ++n) {
        H[n].resize(Ly, Ltot);
        for (int i = 0; i < Ly; ++i)
            for (int j = 0; j < Ltot; ++j)
                H[n](i, j) = Complex(dist(gen), dist(gen));
    }

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_feasible(result, bu, Eu, "stress test");
}

TestResult test_boundary_feasibility() {
    std::cout << "[test] boundary feasibility" << std::endl;

    const int N = 2;
    std::vector<int> Lxu = {1, 1};
    VectorXd bu(2); bu << 4, 4;
    VectorXd Eu(2); Eu << 0.5, 0.5;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(1, 2); H[0] << Complex(50, 0), Complex(40, 0);
    H[1].resize(1, 2); H[1] << Complex(45, 0), Complex(35, 0);

    auto result = admMACMIMO(H, Lxu, bu, Eu, cb);
    return validate_feasible(result, bu, Eu, "boundary feasibility");
}

int main() {
    std::vector<TestResult> results;

    results.push_back(run_test("basic siso feasible", test_basic_siso_feasible));
    std::cout << std::endl;
    results.push_back(run_test("siso infeasible (low energy)", test_siso_infeasible_low_energy));
    std::cout << std::endl;
    results.push_back(run_test("siso infeasible (high rate)", test_siso_infeasible_high_rate));
    std::cout << std::endl;
    results.push_back(run_test("siso vertex sharing", test_siso_vertex_sharing));
    std::cout << std::endl;
    results.push_back(run_test("basic mimo", test_basic_mimo));
    std::cout << std::endl;
    results.push_back(run_test("mimo multi-tone", test_mimo_multi_tone));
    std::cout << std::endl;
    results.push_back(run_test("single tone", test_single_tone));
    std::cout << std::endl;
    results.push_back(run_test("three users siso", test_three_users_siso));
    std::cout << std::endl;
    results.push_back(run_test("mimo unequal antennas", test_mimo_unequal_antennas));
    std::cout << std::endl;
    results.push_back(run_test("real baseband", test_real_baseband));
    std::cout << std::endl;
    results.push_back(run_test("stress test", test_stress));
    std::cout << std::endl;
    results.push_back(run_test("boundary feasibility", test_boundary_feasibility));
    std::cout << std::endl;

    int passed = 0, failed = 0;
    for (const auto& r : results) {
        if (r.passed) ++passed;
        else ++failed;
    }

    std::cout << std::endl;
    std::cout << "[passed] " << passed << "/" << results.size() << std::endl;
    std::cout << "[failed] " << failed << "/" << results.size() << std::endl;
    std::cout << std::endl;

    std::cout << "[summary]" << std::endl;
    for (const auto& r : results) {
        std::cout << "  " << std::left << std::setw(30) << r.name << ": "
                  << std::fixed << std::setprecision(3) << r.elapsed_ms << " ms"
                  << std::endl;
    }

    return failed > 0 ? 1 : 0;
}
