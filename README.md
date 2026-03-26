# multiuser-opt

Solvers for multiple-access channel (MAC) resource allocation problems. Each algorithm has a MATLAB implementation, a C++ implementation (with OpenMP parallelism), and a MEX gateway so the C++ solver can be called directly from MATLAB.

## Algorithms

| Directory | Problem | Method |
|-----------|---------|--------|
| `maxrmac` | Weighted sum-rate maximization subject to per-user energy constraints | Ellipsoid method over dual variables; per-tone waterfilling subproblems |
| `minpmac` | Minimum power for target rates | Ellipsoid method; per-tone LBFGS over Cholesky factors |
| `maxresmac` | Weighted sum-rate maximization subject to a total energy constraint | Bisection on Lagrange multiplier with per-tone LBFGS; post-bisection energy scaling |
| `admmac` | Feasibility / rate-region membership via ADMM | Iterates `maxrmac` with adaptive weights; Frank-Wolfe convex hull test |

Each directory contains:
- `<name>.m` &mdash; MATLAB entry point (accepts `use_mex` flag to select backend)
- `test_<name>.m` &mdash; MATLAB test suite with MATLAB-vs-C++ comparison table
- `cpp/` &mdash; C++ implementation, standalone test, and MEX gateway
- `cvx/` &mdash; CVX reference formulation
- `utils/` or `src/` &mdash; shared helpers

## Dependencies

| Dependency | Required by | Install |
|------------|-------------|---------|
| Eigen 3.3+ | all C++ | `brew install eigen` / `apt install libeigen3-dev` |
| OpenMP (libomp) | all C++ | `brew install libomp` / ships with GCC |
| GLPK | minpmac only | `brew install glpk` / `apt install libglpk-dev` |
| MATLAB (with MEX SDK) | MEX builds | &mdash; |

## Building

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces:
- static libraries (`maxrmac_lib`, `minpmac_lib`, `maxresmac_lib`, `admmac_lib`)
- standalone test binaries (`test_maxrmac`, `test_minpmac`, `test_maxresmac`, `test_admmac`)
- MEX modules (if MATLAB is found): `maxrmac_mex`, `minpmac_mex`, `maxresmac_mex`, `admmac_mex`

MEX files are placed in their respective algorithm directories so MATLAB can find them on the path.

## Running tests

C++ standalone:
```bash
cd build
./test_maxrmac
./test_minpmac
./test_maxresmac
./test_admmac
```

MATLAB (from the repo root):
```matlab
run('maxrmac/test_maxRMACMIMO.m')
run('minpmac/test_minPMACMIMO.m')
run('maxresmac/test_maxRESMACMIMO.m')
run('admmac/test_admMACMIMO.m')
```

When MEX modules are present, the MATLAB tests run both backends and print a comparison table with per-test speedups.

## Todo

- [ ] Add hardware acceleration

## License

See [LICENSE](LICENSE).