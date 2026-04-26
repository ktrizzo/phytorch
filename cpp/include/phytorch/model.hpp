#pragma once

// CRTP base for phytorch models.
//
// Concrete models inherit `Model<Derived>` and provide:
//   static constexpr int n_params  = …;          // number of free parameters
//   static constexpr int n_inputs  = …;          // independent variables
//   static constexpr std::array<ParameterInfo, n_params> info{ … };
//   static constexpr std::array<std::string_view, n_inputs> required_data{ … };
//
//   template <class T>
//   static T forward(const Eigen::Matrix<T, n_inputs, 1>& x,
//                    const Eigen::Matrix<T, n_params, 1>& p);
//
//   static Eigen::Matrix<double, n_params, 1>
//   initial_guess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
//
// Templating `forward` on the scalar lets the same code run with `double`
// (prediction) and `ad::Dual<n_params>` (Jacobian via forward-mode AD).

#include "parameter.hpp"

#include <Eigen/Core>

namespace phytorch {

template <class Derived>
struct Model {};

// Free helper: evaluate a model row-by-row over a batch of inputs.
// Kept as a free template (not a member of `Model<Derived>`) to avoid the
// dependent-name lookup pitfalls of CRTP-inherited member templates.
template <class M, class T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
batch_forward(const Eigen::Matrix<T, Eigen::Dynamic, M::n_inputs>& X,
              const Eigen::Matrix<T, M::n_params, 1>& p) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> y(X.rows());
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        Eigen::Matrix<T, M::n_inputs, 1> row = X.row(i).transpose();
        y(i) = M::template forward<T>(row, p);
    }
    return y;
}

}  // namespace phytorch
