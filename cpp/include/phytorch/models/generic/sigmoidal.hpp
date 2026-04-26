#pragma once

// Rational sigmoid: y(x) = ymax / (1 + |x/x50|^s).
// Used in plant physiology for vulnerability curves and saturation responses.

#include "../../model.hpp"
#include "../../autodiff.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>

namespace phytorch::models {

struct Sigmoidal : Model<Sigmoidal> {
    static constexpr int n_params = 3;
    static constexpr int n_inputs = 1;

    static constexpr std::array<ParameterInfo, n_params> info{{
        {"ymax", "y_max", "", "Maximum y value",               1.0, 0.0,            kNoUpperBound},
        {"x50",  "x_50",  "", "x at half-maximum response",    1.0, kNoLowerBound,  kNoUpperBound},
        {"s",    "s",     "", "Steepness parameter",           2.0, 0.1,            20.0},
    }};
    static constexpr std::array<std::string_view, n_inputs> required_data{{ "x" }};

    template <class T>
    static T forward(const Eigen::Matrix<T, n_inputs, 1>& xv,
                     const Eigen::Matrix<T, n_params, 1>& p) {
        // |x/x50|^s expressed as exp(s · log|·|) so we only need exp/log/abs
        // on Dual (a generic Dual^Dual pow would also work — kept simple).
        using std::abs; using phytorch::ad::abs;
        using std::log; using phytorch::ad::log;
        using std::exp; using phytorch::ad::exp;

        const T r_abs = abs(xv(0) / p(1));
        const T term  = exp(p(2) * log(r_abs + 1e-300));
        return p(0) / (1.0 + term);
    }

    static Eigen::Matrix<double, n_params, 1>
    initial_guess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        const double ymax = y.maxCoeff();
        const double half = ymax / 2.0;
        Eigen::Index k;
        (y.array() - half).abs().minCoeff(&k);
        double x50 = X(k, 0);
        if (std::abs(x50) < 1e-6) x50 = X.col(0).mean();
        Eigen::Matrix<double, n_params, 1> p0;
        p0 << ymax, x50, 2.0;
        return p0;
    }
};

}  // namespace phytorch::models
