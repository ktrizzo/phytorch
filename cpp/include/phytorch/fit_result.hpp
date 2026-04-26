#pragma once

#include <Eigen/Core>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace phytorch {

// Mirrors phytorch.core.result.FitResult. Stored as plain types so it
// trivially round-trips through Embind to a JS object.
struct FitResult {
    std::unordered_map<std::string, double> parameters;       // fitted (and fixed) values
    Eigen::VectorXd                         predictions;
    Eigen::VectorXd                         residuals;
    double                                  loss          = 0.0;   // Σ residual²
    double                                  r_squared     = 0.0;
    bool                                    converged     = false;
    int                                     iterations    = 0;

    // Covariance over the *fitted* parameters in the same order as
    // `fitted_parameter_order`. Empty for non-converged Adam runs.
    Eigen::MatrixXd          covariance;
    std::vector<std::string> fitted_parameter_order;

    std::string method;          // "levenberg_marquardt" | "adam"
    std::string status_message;  // optimizer-supplied detail
};

}  // namespace phytorch
