#pragma once

// Ball-Berry-Leuning (1995):
//   gs = gs0 + a1 · max(A,0) / ((Ca - Γ) · (1 + VPD/D0))

#include "../../model.hpp"
#include "../../autodiff.hpp"

#include <Eigen/Core>
#include <array>

namespace phytorch::models {

struct BBL1995 : Model<BBL1995> {
    static constexpr int n_params = 3;
    static constexpr int n_inputs = 2;

    static constexpr double kCa    = 400.0;  // ppm
    static constexpr double kGamma = 40.0;   // ppm

    static constexpr std::array<ParameterInfo, n_params> info{{
        {"gs0", "g_s0", "mol m^-2 s^-1", "Minimum stomatal conductance", 0.01, 0.0, 0.1 },
        {"a1",  "a_1",  "",              "Slope parameter",              10.0, 1.0, 30.0},
        {"D0",  "D_0",  "kPa",           "VPD sensitivity",              1.5,  0.5, 5.0 },
    }};
    static constexpr std::array<std::string_view, n_inputs> required_data{{ "A", "VPD" }};

    template <class T>
    static T forward(const Eigen::Matrix<T, n_inputs, 1>& xv,
                     const Eigen::Matrix<T, n_params, 1>& p) {
        using phytorch::ad::max; using std::max;
        const T A_pos = max(xv(0), 0.0);
        const T denom = max((kCa - kGamma) * (1.0 + xv(1) / p(2)), 1e-10);
        return p(0) + p(1) * A_pos / denom;
    }

    static Eigen::Matrix<double, n_params, 1>
    initial_guess(const Eigen::MatrixXd&, const Eigen::VectorXd& y) {
        double gs0 = 0.01;
        for (Eigen::Index i = 0; i < y.size(); ++i)
            if (y(i) > 0) gs0 = std::min(gs0, y(i));
        Eigen::Matrix<double, n_params, 1> p0;
        p0 << std::min(gs0, 0.05), 10.0, 1.5;
        return p0;
    }
};

}  // namespace phytorch::models
