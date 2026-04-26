#pragma once

// Nonrectangular hyperbola — light response curve:
//   y(x) = (αx + ymax - sqrt((αx + ymax)² - 4θαx·ymax)) / (2θ)

#include "../../model.hpp"
#include "../../autodiff.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace phytorch::models {

struct NonrectangularHyperbola : Model<NonrectangularHyperbola> {
    static constexpr int n_params = 3;
    static constexpr int n_inputs = 1;

    static constexpr std::array<ParameterInfo, n_params> info{{
        {"alpha", "α",     "", "Initial slope (quantum yield)", 0.5, 0.0,  1.0},
        {"ymax",  "y_max", "", "Maximum asymptotic y value",    1.0, 0.0,  kNoUpperBound},
        {"theta", "θ",     "", "Curvature (convexity)",         0.7, 0.01, 0.99},
    }};
    static constexpr std::array<std::string_view, n_inputs> required_data{{ "x" }};

    template <class T>
    static T forward(const Eigen::Matrix<T, n_inputs, 1>& xv,
                     const Eigen::Matrix<T, n_params, 1>& p) {
        using std::sqrt; using phytorch::ad::sqrt;
        using std::max;  using phytorch::ad::max;

        const T& alpha = p(0);
        const T& ymax  = p(1);
        const T& theta = p(2);
        const T  ax_y  = alpha * xv(0) + ymax;
        const T  disc  = max(ax_y * ax_y - 4.0 * theta * alpha * xv(0) * ymax, 0.0);
        return (ax_y - sqrt(disc)) / (2.0 * theta);
    }

    static Eigen::Matrix<double, n_params, 1>
    initial_guess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        const double ymax = y.maxCoeff() * 1.05;
        // alpha ≈ initial slope from the lowest 20% of x values
        std::vector<Eigen::Index> idx(X.rows());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](Eigen::Index a, Eigen::Index b){ return X(a,0) < X(b,0); });
        const int n_init = std::max<int>(2, static_cast<int>(X.rows() * 0.2));
        const double dx = X(idx[n_init - 1], 0) - X(idx[0], 0);
        const double dy = y(idx[n_init - 1])    - y(idx[0]);
        double alpha = (dx > 0) ? dy / dx : 0.5;
        alpha = std::clamp(alpha, 0.01, 0.9);
        Eigen::Matrix<double, n_params, 1> p0;
        p0 << alpha, ymax, 0.7;
        return p0;
    }
};

}  // namespace phytorch::models
