// Native sanity check: synthesize data from known Arrhenius parameters,
// fit, and verify recovery. Doubles as the canonical usage example.

#include "phytorch/fit.hpp"
#include "phytorch/models/generic/arrhenius.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main() {
    using namespace phytorch;
    using models::Arrhenius;

    constexpr double ymax_true = 2.5;
    constexpr double Ha_true   = 65.0;

    constexpr int N = 25;
    Eigen::Matrix<double, Eigen::Dynamic, Arrhenius::n_inputs> X(N, 1);
    Eigen::VectorXd y(N);
    for (int i = 0; i < N; ++i) {
        const double T = 283.15 + i * 1.5;  // 10 °C → ~46 °C
        X(i, 0) = T;
        y(i)    = ymax_true * std::exp(
                    Ha_true / (Arrhenius::R * Arrhenius::T_ref)
                  - Ha_true / (Arrhenius::R * T));
    }

    FitOptions opts;
    opts.method = Method::LevenbergMarquardt;
    auto res    = fit<Arrhenius>(X, y, opts);

    std::printf("converged=%d  iters=%d  R²=%.6f  loss=%.3e\n",
                res.converged, res.iterations, res.r_squared, res.loss);
    std::printf("  ymax=%.4f (true %.4f)\n", res.parameters["ymax"], ymax_true);
    std::printf("  Ha  =%.4f (true %.4f)\n", res.parameters["Ha"],   Ha_true);

    const bool ok =
        std::abs(res.parameters["ymax"] - ymax_true) < 1e-3 &&
        std::abs(res.parameters["Ha"]   - Ha_true)   < 1e-2 &&
        res.r_squared > 0.999;
    return ok ? 0 : 1;
}
