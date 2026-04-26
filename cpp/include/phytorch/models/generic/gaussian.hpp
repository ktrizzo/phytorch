#pragma once

// Gaussian: y(x) = a · exp(-(x-μ)² / (2σ²)).

#include "../../model.hpp"
#include "../../autodiff.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>

namespace phytorch::models {

struct Gaussian : Model<Gaussian> {
    static constexpr int n_params = 3;
    static constexpr int n_inputs = 1;

    static constexpr std::array<ParameterInfo, n_params> info{{
        {"a",     "a", "", "Amplitude (height at peak)",      1.0, 0.0,            kNoUpperBound},
        {"mu",    "μ", "", "Mean (center of distribution)",   0.0, kNoLowerBound,  kNoUpperBound},
        {"sigma", "σ", "", "Standard deviation (width)",      1.0, 1e-9,           kNoUpperBound},
    }};
    static constexpr std::array<std::string_view, n_inputs> required_data{{ "x" }};

    template <class T>
    static T forward(const Eigen::Matrix<T, n_inputs, 1>& xv,
                     const Eigen::Matrix<T, n_params, 1>& p) {
        using std::exp; using phytorch::ad::exp;
        const T  d  = xv(0) - p(1);
        const T& s  = p(2);
        return p(0) * exp(-(d * d) / (2.0 * s * s));
    }

    static Eigen::Matrix<double, n_params, 1>
    initial_guess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        Eigen::Index k;
        y.maxCoeff(&k);
        const double a_g  = y(k) > 0 ? y(k) : 1.0;
        const double mu_g = X(k, 0);
        // FWHM ≈ 2.355σ; fall back to range/6 if no clear peak.
        const double range = X.col(0).maxCoeff() - X.col(0).minCoeff();
        const double sigma_g = std::max(range / 6.0, 1e-6);
        Eigen::Matrix<double, n_params, 1> p0;
        p0 << a_g, mu_g, sigma_g;
        return p0;
    }
};

}  // namespace phytorch::models
