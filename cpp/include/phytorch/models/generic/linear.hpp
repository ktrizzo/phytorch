#pragma once

// Linear regression: y(x) = a + b·x.

#include "../../model.hpp"

#include <Eigen/Core>
#include <array>

namespace phytorch::models {

struct Linear : Model<Linear> {
    static constexpr int n_params = 2;
    static constexpr int n_inputs = 1;

    static constexpr std::array<ParameterInfo, n_params> info{{
        {"a", "a", "", "Intercept", 0.0, kNoLowerBound, kNoUpperBound},
        {"b", "b", "", "Slope",     1.0, kNoLowerBound, kNoUpperBound},
    }};
    static constexpr std::array<std::string_view, n_inputs> required_data{{ "x" }};

    template <class T>
    static T forward(const Eigen::Matrix<T, n_inputs, 1>& x,
                     const Eigen::Matrix<T, n_params, 1>& p) {
        return p(0) + p(1) * x(0);
    }

    static Eigen::Matrix<double, n_params, 1>
    initial_guess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        // Closed-form least squares: same heuristic as the Python initial_guess.
        const double n      = static_cast<double>(X.rows());
        const double sum_x  = X.col(0).sum();
        const double sum_y  = y.sum();
        const double sum_xx = X.col(0).cwiseProduct(X.col(0)).sum();
        const double sum_xy = X.col(0).cwiseProduct(y).sum();
        const double denom  = n * sum_xx - sum_x * sum_x;

        Eigen::Matrix<double, n_params, 1> p0;
        if (std::abs(denom) > 1e-10) {
            const double b = (n * sum_xy - sum_x * sum_y) / denom;
            const double a = (sum_y - b * sum_x) / n;
            p0 << a, b;
        } else {
            p0 << y.mean(), 0.0;
        }
        return p0;
    }
};

}  // namespace phytorch::models
