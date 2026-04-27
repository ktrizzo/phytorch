#pragma once

// Beta distribution PDF on an arbitrary range [xmin, xmax]:
//   u = (x - xmin) / (xmax - xmin)
//   y = a · u^(α-1) · (1-u)^(β-1) / B(α, β)        for xmin < x < xmax
//   y = 0                                           otherwise
//
// We compute through log-space so AD only needs lgamma + log + exp:
//   log y = log a + (α-1) log u + (β-1) log(1-u)
//         - lgamma(α) - lgamma(β) + lgamma(α+β)

#include "../../model.hpp"
#include "../../autodiff.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>

namespace phytorch::models {

struct Beta : Model<Beta> {
    static constexpr int n_params = 5;
    static constexpr int n_inputs = 1;

    static constexpr std::array<ParameterInfo, n_params> info{{
        {"a",     "a",     "", "Amplitude (height scaling factor)", 1.0, 0.0,           kNoUpperBound},
        {"xmin",  "x_min", "", "Minimum of range",                  0.0, kNoLowerBound, kNoUpperBound},
        {"xmax",  "x_max", "", "Maximum of range",                  1.0, kNoLowerBound, kNoUpperBound},
        {"alpha", "α",     "", "Left shape parameter",              2.0, 0.1,           10.0},
        {"beta",  "β",     "", "Right shape parameter",             2.0, 0.1,           10.0},
    }};
    static constexpr std::array<std::string_view, n_inputs> required_data{{ "x" }};

    template <class T>
    static T forward(const Eigen::Matrix<T, n_inputs, 1>& xv,
                     const Eigen::Matrix<T, n_params, 1>& p) {
        using std::log;    using phytorch::ad::log;
        using std::exp;    using phytorch::ad::exp;
        using std::lgamma; using phytorch::ad::lgamma;

        const T& xmin  = p(1);
        const T& xmax  = p(2);
        const T& alpha = p(3);
        const T& bbeta = p(4);

        if (xv(0) <= xmin || xv(0) >= xmax) return T(0.0);
        const T u  = (xv(0) - xmin) / (xmax - xmin);

        // Tiny numerical clamp to keep log(u), log(1-u) finite at the edges.
        constexpr double eps = 1e-10;
        if (u <= eps || u >= (1.0 - eps)) return T(0.0);

        const T log_y = log(p(0))
                      + (alpha  - 1.0) * log(u)
                      + (bbeta  - 1.0) * log(1.0 - u)
                      - lgamma(alpha) - lgamma(bbeta) + lgamma(alpha + bbeta);
        return exp(log_y);
    }

    static Eigen::Matrix<double, n_params, 1>
    initial_guess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        const double xmin_d = X.col(0).minCoeff();
        const double xmax_d = X.col(0).maxCoeff();
        const double range  = xmax_d - xmin_d;
        const double xmin_g = xmin_d - range * 0.05;
        const double xmax_g = xmax_d + range * 0.05;

        Eigen::Index k;
        y.maxCoeff(&k);
        const double a_g = std::max(y(k) * 1.2, 1.0);

        const double peak = std::clamp((X(k, 0) - xmin_g) / (xmax_g - xmin_g),
                                       0.1, 0.9);
        double alpha_g, beta_g;
        if (peak < 0.4)        { alpha_g = 2.0; beta_g = 3.0; }
        else if (peak > 0.6)   { alpha_g = 3.0; beta_g = 2.0; }
        else                   { alpha_g = 2.0; beta_g = 2.0; }

        Eigen::Matrix<double, n_params, 1> p0;
        p0 << a_g, xmin_g, xmax_g, alpha_g, beta_g;
        return p0;
    }
};

}  // namespace phytorch::models
