#pragma once

#include <Eigen/Core>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace phytorch {

enum class Method {
    Auto,
    LevenbergMarquardt,  // analogue of scipy.optimize.curve_fit (default)
    Adam,                // analogue of torch.optim.Adam
};

// Mirrors the keys that phytorch.fit() accepts in its `options` dict, so a
// JS caller can pass the same shape of object through Embind.
struct FitOptions {
    Method method = Method::Auto;

    // Per-parameter overrides (keyed by parameter name).
    std::unordered_map<std::string, double>                 initial_guess;
    std::unordered_map<std::string, std::pair<double,double>> bounds;
    std::unordered_map<std::string, double>                 fixed_parameters;
    std::unordered_set<std::string>                         fit_parameters;  // empty = fit all unfixed

    // Stopping criteria — names match scipy / phytorch.
    int    max_iterations = 10'000;
    double ftol           = 1e-8;
    double xtol           = 1e-8;
    double gtol           = 1e-8;

    // Adam-only.
    double learning_rate  = 1e-2;
    double beta1          = 0.9;
    double beta2          = 0.999;
    double eps            = 1e-8;

    bool verbose = false;
};

}  // namespace phytorch
