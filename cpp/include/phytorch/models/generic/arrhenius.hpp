#pragma once

// Reference model: Arrhenius temperature response.
//   y(T) = ymax · exp( Ha/(R·T_ref) - Ha/(R·T) )
//
// Direct port of phytorch/models/generic/arrhenius.py.

#include "../../model.hpp"
#include "../../autodiff.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>

namespace phytorch::models {

struct Arrhenius : Model<Arrhenius> {
    static constexpr int n_params = 2;
    static constexpr int n_inputs = 1;

    static constexpr double R     = 0.008314;  // kJ/(mol·K)
    static constexpr double T_ref = 298.15;    // K (25 °C)

    static constexpr std::array<ParameterInfo, n_params> info{{
        {"ymax", "y_max", "",       "Value at reference temperature (25 °C)", 1.0,  0.0, kNoUpperBound},
        {"Ha",   "H_a",   "kJ/mol", "Activation energy",                       50.0, 0.0, 200.0},
    }};
    static constexpr std::array<std::string_view, n_inputs> required_data{{ "T" }};

    template <class T>
    static T forward(const Eigen::Matrix<T, n_inputs, 1>& x,
                     const Eigen::Matrix<T, n_params, 1>& p) {
        using std::exp; using phytorch::ad::exp;
        const T& Tk   = x(0);
        const T& ymax = p(0);
        const T& Ha   = p(1);
        return ymax * exp(Ha / (R * T_ref) - Ha / (R * Tk));
    }

    static Eigen::Matrix<double, n_params, 1>
    initial_guess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        Eigen::Index k;
        (X.col(0).array() - T_ref).abs().minCoeff(&k);
        Eigen::Matrix<double, n_params, 1> p0;
        p0 << (y(k) > 0 ? y(k) : y.cwiseMax(0.0).mean()), 50.0;
        return p0;
    }
};

}  // namespace phytorch::models
