#pragma once

// Rectangular hyperbola (Michaelis–Menten): y(x) = ymax·x / (x50 + x).

#include "../../model.hpp"

#include <Eigen/Core>
#include <array>

namespace phytorch::models {

struct RectangularHyperbola : Model<RectangularHyperbola> {
    static constexpr int n_params = 2;
    static constexpr int n_inputs = 1;

    static constexpr std::array<ParameterInfo, n_params> info{{
        {"ymax", "y_max", "", "Maximum asymptotic y value",   1.0, 0.0, kNoUpperBound},
        {"x50",  "x_50",  "", "Half-saturation constant",     1.0, 0.0, kNoUpperBound},
    }};
    static constexpr std::array<std::string_view, n_inputs> required_data{{ "x" }};

    template <class T>
    static T forward(const Eigen::Matrix<T, n_inputs, 1>& xv,
                     const Eigen::Matrix<T, n_params, 1>& p) {
        return (p(0) * xv(0)) / (p(1) + xv(0));
    }

    static Eigen::Matrix<double, n_params, 1>
    initial_guess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        const double ymax = y.maxCoeff() * 1.1;
        const double half = ymax / 2.0;
        Eigen::Index k;
        (y.array() - half).abs().minCoeff(&k);
        double x50 = X(k, 0);
        if (x50 <= 0) x50 = 1.0;
        Eigen::Matrix<double, n_params, 1> p0;
        p0 << ymax, x50;
        return p0;
    }
};

}  // namespace phytorch::models
