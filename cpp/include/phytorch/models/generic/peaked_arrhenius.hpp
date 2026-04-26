#pragma once

// Peaked Arrhenius temperature response with high-temperature deactivation.
//   y(T) = ymax · f_arr(T) · f_peak(T)
//   f_arr  = exp(Ha/(R T_ref) - Ha/(R T))
//   f_peak = (1 + exp(Hd/R · (1/T_opt - 1/T_ref) - log(Hd/Ha - 1)))
//          / (1 + exp(Hd/R · (1/T_opt - 1/T)     - log(Hd/Ha - 1)))

#include "../../model.hpp"
#include "../../autodiff.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>

namespace phytorch::models {

struct PeakedArrhenius : Model<PeakedArrhenius> {
    static constexpr int n_params = 4;
    static constexpr int n_inputs = 1;

    static constexpr double R     = 0.008314;  // kJ/(mol·K)
    static constexpr double T_ref = 298.15;    // K (25 °C)

    static constexpr std::array<ParameterInfo, n_params> info{{
        {"ymax", "y_max", "",       "Maximum value at optimum temperature", 1.0,    0.0,    kNoUpperBound},
        {"Ha",   "H_a",   "kJ/mol", "Activation energy",                    50.0,   0.0,    150.0},
        {"Hd",   "H_d",   "kJ/mol", "Deactivation energy",                  200.0,  150.0,  400.0},
        {"Topt", "T_opt", "K",      "Optimum temperature",                  311.15, 273.15, 333.15},
    }};
    static constexpr std::array<std::string_view, n_inputs> required_data{{ "T" }};

    template <class T>
    static T forward(const Eigen::Matrix<T, n_inputs, 1>& xv,
                     const Eigen::Matrix<T, n_params, 1>& p) {
        using std::exp; using phytorch::ad::exp;
        using std::log; using phytorch::ad::log;
        using std::max; using phytorch::ad::max;

        const T& Tk   = xv(0);
        const T& ymax = p(0);
        const T& Ha   = p(1);
        const T& Hd   = p(2);
        const T& Topt = p(3);

        const T f_arr = exp(Ha / (R * T_ref) - Ha / (R * Tk));

        const T ratio = max(Hd / Ha, 1.0001);
        const T log_term = log(ratio - 1.0);
        const T num = 1.0 + exp(Hd / R * (1.0 / Topt - 1.0 / T_ref) - log_term);
        const T den = 1.0 + exp(Hd / R * (1.0 / Topt - 1.0 / Tk)    - log_term);
        return ymax * f_arr * (num / den);
    }

    static Eigen::Matrix<double, n_params, 1>
    initial_guess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        Eigen::Index k;
        y.maxCoeff(&k);
        const double Topt_g = std::clamp(X(k, 0), 273.15, 333.15);
        const double ymax_g = y(k) > 0 ? y(k) : 1.0;
        Eigen::Matrix<double, n_params, 1> p0;
        p0 << ymax_g, 50.0, 200.0, Topt_g;
        return p0;
    }
};

}  // namespace phytorch::models
