#pragma once

// Weibull PDF (location-shifted):
//   y(x) = ymax · (k/λ) · ((x-x0)/λ)^(k-1) · exp(-((x-x0)/λ)^k)   for x > x0
//   y(x) = 0                                                       otherwise
//
// At x ≤ x0 the function is identically zero, so its parameter gradients
// are zero too — the comparison-based branch is correct for AD.

#include "../../model.hpp"
#include "../../autodiff.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>

namespace phytorch::models {

struct Weibull : Model<Weibull> {
    static constexpr int n_params = 4;
    static constexpr int n_inputs = 1;

    static constexpr std::array<ParameterInfo, n_params> info{{
        {"ymax",   "y_max", "", "Amplitude (max height)",          1.0, 0.0,           kNoUpperBound},
        {"x0",     "x_0",   "", "Location/threshold parameter",    0.0, kNoLowerBound, kNoUpperBound},
        {"lambda", "λ",     "", "Scale parameter",                 1.0, 1e-9,          kNoUpperBound},
        {"k",      "k",     "", "Shape parameter",                 2.0, 0.1,           10.0},
    }};
    static constexpr std::array<std::string_view, n_inputs> required_data{{ "x" }};

    template <class T>
    static T forward(const Eigen::Matrix<T, n_inputs, 1>& xv,
                     const Eigen::Matrix<T, n_params, 1>& p) {
        using std::exp; using phytorch::ad::exp;
        using std::log; using phytorch::ad::log;

        const T  shifted = xv(0) - p(1);
        if (shifted <= 0.0) return T(0.0);

        const T& lam = p(2);
        const T& k   = p(3);
        const T  z   = shifted / lam;
        const T  zk  = exp(k * log(z));                  // z^k
        const T  zkm = exp((k - 1.0) * log(z));          // z^(k-1)
        return p(0) * (k / lam) * zkm * exp(-zk);
    }

    static Eigen::Matrix<double, n_params, 1>
    initial_guess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        const double ymax = std::max(y.maxCoeff(), 1.0);
        const double xmin = X.col(0).minCoeff();
        const double xmax = X.col(0).maxCoeff();
        const double x0_g = xmin - (xmax - xmin) * 0.1;
        Eigen::Matrix<double, n_params, 1> p0;
        p0 << ymax, x0_g, std::max((xmax - xmin) / 3.0, 1e-3), 2.0;
        return p0;
    }
};

}  // namespace phytorch::models
