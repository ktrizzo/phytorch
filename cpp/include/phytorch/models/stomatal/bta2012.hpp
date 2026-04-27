#pragma once

// Buckley, Turnbull & Adams (2012) Model 4:
//   gs = Em (Q + i0) / (k + bQ + (Q + i0) Ds)   then divided by 1000 for mol·m⁻²·s⁻¹

#include "../../model.hpp"
#include "../../autodiff.hpp"

#include <Eigen/Core>
#include <array>

namespace phytorch::models {

struct BTA2012 : Model<BTA2012> {
    static constexpr int n_params = 4;
    static constexpr int n_inputs = 2;

    static constexpr std::array<ParameterInfo, n_params> info{{
        {"Em", "E_m", "mmol m^-2 s^-1",                "Maximum leaf transpiration",       10.0,        0.1, 50.0   },
        {"i0", "i_0", "umol m^-2 s^-1",                "Dark transpiration parameter",     50.0,        0.0, 300.0  },
        {"k",  "k",   "umol m^-2 s^-1 mmol mol^-1",    "Lumped parameter K1/χφ",           1e4,         0.0, 1e6    },
        {"b",  "b",   "mmol mol^-1",                   "Lumped parameter K1/χα0",          20.0/3.0,    0.0, 100.0  },
    }};
    static constexpr std::array<std::string_view, n_inputs> required_data{{ "Q", "Ds" }};

    template <class T>
    static T forward(const Eigen::Matrix<T, n_inputs, 1>& xv,
                     const Eigen::Matrix<T, n_params, 1>& p) {
        using phytorch::ad::max; using std::max;
        const T num = p(0) * (xv(0) + p(1));
        const T den = max(p(2) + p(3) * xv(0) + (xv(0) + p(1)) * xv(1), 1e-10);
        return (num / den) / 1000.0;
    }

    static Eigen::Matrix<double, n_params, 1>
    initial_guess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        // At saturating light (large Q) and assuming i0 ≪ Q,
        //   gs ≈ Em / (b + Ds) / 1000
        // so Em ≈ max(gs) · (b_default + Ds_at_gsmax) · 1000.
        // This dimensionally-grounded estimate puts Em close enough that
        // LM with diagonal scaling reliably converges.
        const double b_def = 20.0 / 3.0;
        Eigen::Index k_max;
        y.maxCoeff(&k_max);
        const double Ds_at_max = X(k_max, 1);
        const double Em_g = std::clamp(
            y.maxCoeff() * (b_def + Ds_at_max) * 1000.0, 1.0, 50.0);

        Eigen::Matrix<double, n_params, 1> p0;
        p0 << Em_g, 50.0, 1e4, b_def;
        return p0;
    }
};

}  // namespace phytorch::models
