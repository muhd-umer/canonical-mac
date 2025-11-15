"""Test suite for minPMACMIMO Python implementation."""

import sys
import numpy as np

from minpmac_mimo import minpmac_mimo


def run_test(test_name, H, Lx, bu_min, w, cb, expect_infeasible=False):
    """Run a single test case.

    Args:
        test_name: Name of the test
        H: Channel matrix
        Lx: Antennas per user
        bu_min: Target rates
        w: Weights
        cb: Baseband type
        expect_infeasible: Whether to expect infeasibility

    Returns:
        success: True if test passed
    """
    print(f"[{test_name}]")

    try:
        flag, bu_a, info = minpmac_mimo(H, Lx, bu_min, w, cb)

        num_clusters = len(info["clusters"]) if "clusters" in info else 0
        num_orders = len(info["orderings"]) if "orderings" in info else 0

        print(f"  target rates: {bu_min}")
        print(f"  achieved rates: {bu_a}")
        print(f"  rate error: {bu_a - bu_min}")
        print(f"  feasibility flag: {flag}")
        print(f'  solution status: {info["sol_status"]}')
        print(
            f"  clusters: {num_clusters}, orders: {num_orders}, "
            f'elapsed: {info["elapsed_time"]:.3f} s'
        )

        if expect_infeasible:
            if flag == 0:
                print("  [✓] infeasible case detected")
                success = True
            else:
                print("  [✗] infeasible case not detected")
                success = False
        else:
            if flag > 0:
                # check if rates are satisfied
                rate_satisfied = np.all(bu_a >= bu_min - 1e-3)
                if rate_satisfied:
                    print("  [✓] test passed")
                    success = True
                else:
                    print("  [✗] test failed (rates not satisfied)")
                    success = False
            else:
                print("  [✗] test failed (infeasible)")
                success = False

    except Exception as e:
        print(f"  [✗] test crashed: {e}")
        import traceback

        traceback.print_exc()
        success = False

    print()
    return success


def main():
    """Run all tests."""
    np.random.seed(42)
    results = []

    # reference siso test
    print("[reference] SISO")
    H_siso = np.zeros((1, 2, 4), dtype=complex)
    H_siso[:, :, 0] = [[80, 60]]
    H_siso[:, :, 1] = [[40, 30]]
    H_siso[:, :, 2] = [[50, 50]]
    H_siso[:, :, 3] = [[30, 40]]
    Lx_siso = np.array([1, 1])
    bu_ref_siso = np.array([24.0, 24.0])
    w_ref = np.array([1.0, 1.0])
    cb = 1

    flag, bu_a, info = minpmac_mimo(H_siso, Lx_siso, bu_ref_siso, w_ref, cb)
    total_energy = np.sum(info["Rxx"][0])

    print(f"  users: {len(Lx_siso)}, tones: {H_siso.shape[2]}, rx: {H_siso.shape[0]}")
    print(f"  target rates: {bu_ref_siso}")
    print(f"  achieved rates: {bu_a}")
    print(f"  feasibility flag: {flag}")
    print(f"  total energy: {np.real(total_energy):.4f}")
    print(f'  order: {info["orderings"][0]}')
    print()

    # reference mimo test
    print("[reference] MIMO")
    Ly_ref = 2
    Lx_ref = np.array([2, 2])
    N_ref = 2
    H_mimo = np.zeros((Ly_ref, np.sum(Lx_ref), N_ref), dtype=complex)
    H_mimo[:, 0:2, 0] = [[3.2, 2.1], [1.8, 2.9]]
    H_mimo[:, 2:4, 0] = [[2.5, 3.0], [2.2, 1.9]]
    H_mimo[:, 0:2, 1] = [[2.9, 2.5], [2.0, 3.1]]
    H_mimo[:, 2:4, 1] = [[2.8, 2.3], [1.7, 2.6]]
    bu_ref_mimo = np.array([10.0, 10.0])
    w_ref_mimo = np.array([1.0, 1.0])

    flag, bu_a, info = minpmac_mimo(H_mimo, Lx_ref, bu_ref_mimo, w_ref_mimo, cb)

    print(f"  users: {len(Lx_ref)}, tones: {N_ref}, rx: {Ly_ref}")
    print(f"  target rates: {bu_ref_mimo}")
    print(f"  achieved rates: {bu_a}")
    print(f"  feasibility flag: {flag}")
    print(f'  total energy: {np.real(np.sum(info["Rxx"][0])):.4f}')
    print(f'  order: {info["orderings"][0]}')
    print()

    # randomized and edge-case tests
    print("[test] SISO")
    Ly = 2
    U = 3
    N = 4
    H_siso_rand = (
        np.random.randn(Ly, U, N) + 1j * np.random.randn(Ly, U, N)
    ) / np.sqrt(2)
    Lx = np.ones(U, dtype=int)
    bu_min = np.array([2.0, 1.5, 2.5])
    w = np.ones(U)
    cb = 1

    success = run_test("SISO random", H_siso_rand, Lx, bu_min, w, cb)
    results.append(success)

    print("[test] MIMO")
    Ly = 3
    Lx = np.array([2, 3, 2])
    U = len(Lx)
    Ltot = np.sum(Lx)
    N = 6
    H_mimo_rand = (
        np.random.randn(Ly, Ltot, N) + 1j * np.random.randn(Ly, Ltot, N)
    ) / np.sqrt(2)
    bu_min = np.array([3.0, 4.0, 2.5])
    w = np.array([1.0, 2.0, 1.0])

    success = run_test("MIMO random", H_mimo_rand, Lx, bu_min, w, cb)
    results.append(success)

    print("[test] Stress")
    Ly = 4
    Lx = np.array([3, 2, 4, 2, 1, 2])
    U = len(Lx)
    Ltot = np.sum(Lx)
    N = 64
    H_large = np.zeros((Ly, Ltot, N), dtype=complex)

    for n in range(N):
        H_base = (np.random.randn(Ly, Ltot) + 1j * np.random.randn(Ly, Ltot)) / np.sqrt(
            2
        )
        # add small diagonal component for conditioning
        H_large[:, :, n] = H_base + 0.1 * np.eye(Ly, Ltot)

    bu_min = np.array([2.0, 1.8, 2.2, 1.5, 2.0, 1.8])
    w = np.ones(U)

    success = run_test("Stress", H_large, Lx, bu_min, w, cb)
    results.append(success)

    print("[test] Real baseband")
    H_real = np.random.randn(2, 3, 4)
    Lx = np.ones(3, dtype=int)
    bu_min = np.ones(3)
    w = np.ones(3)
    cb = 2

    success = run_test("Real baseband", H_real, Lx, bu_min, w, cb)
    results.append(success)

    # summary
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("All tests passed! ✓")
        return 0
    else:
        print(f"{total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
