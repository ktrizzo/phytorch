#pragma once

// Top-level fit<Model>(data, options). Mirrors phytorch.fit() — same option
// names, same dispatch logic (auto → LM unless overridden), same shape of
// returned FitResult.
//
// Templated on Model so the optimizer inlines the model's forward(). That's
// what makes WASM execution competitive with native code: no virtual
// dispatch in the inner loop.

#include "fit_options.hpp"
#include "fit_result.hpp"
#include "model.hpp"
#include "optimizers/adam.hpp"
#include "optimizers/levenberg_marquardt.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace phytorch {

template <class M>
FitResult fit(
    const Eigen::Matrix<double, Eigen::Dynamic, M::n_inputs>& X,
    const Eigen::VectorXd& y,
    FitOptions opts = {})
{
    using VecP = Eigen::Matrix<double, M::n_params, 1>;

    if (X.rows() != y.size())
        throw std::invalid_argument("fit(): X and y must have the same number of rows");
    if (X.rows() < M::n_params)
        throw std::invalid_argument("fit(): need at least as many observations as parameters");

    VecP p0 = M::initial_guess(X, y);
    VecP lb, ub;
    for (int j = 0; j < M::n_params; ++j) {
        lb(j) = M::info[j].lower_bound;
        ub(j) = M::info[j].upper_bound;
    }
    for (int j = 0; j < M::n_params; ++j) {
        const std::string name(M::info[j].name);
        if (auto it = opts.initial_guess.find(name); it != opts.initial_guess.end()) p0(j) = it->second;
        if (auto it = opts.bounds.find(name); it != opts.bounds.end()) {
            lb(j) = it->second.first;
            ub(j) = it->second.second;
        }
        if (auto it = opts.fixed_parameters.find(name); it != opts.fixed_parameters.end()) {
            p0(j) = it->second;
            lb(j) = it->second;
            ub(j) = it->second;
        }
        p0(j) = std::clamp(p0(j), lb(j), ub(j));
    }

    Method method = opts.method == Method::Auto ? Method::LevenbergMarquardt : opts.method;

    optim::OptimizerResult<M::n_params> opt = (method == Method::Adam)
        ? optim::adam<M>(X, y, p0, lb, ub, opts)
        : optim::levenberg_marquardt<M>(X, y, p0, lb, ub, opts);

    Eigen::VectorXd y_pred = batch_forward<M, double>(X, opt.p_final);
    const double ss_tot   = (y.array() - y.mean()).square().sum();

    FitResult res;
    res.predictions     = y_pred;
    res.residuals       = opt.residuals;
    res.loss            = opt.loss;
    res.r_squared       = ss_tot > 0.0 ? 1.0 - opt.loss / ss_tot : 0.0;
    res.converged       = opt.converged;
    res.iterations      = opt.iterations;
    res.method          = (method == Method::Adam) ? "adam" : "levenberg_marquardt";
    res.status_message  = opt.status_message;
    res.covariance      = opt.covariance;

    for (int j = 0; j < M::n_params; ++j) {
        const std::string name(M::info[j].name);
        res.parameters[name] = opt.p_final(j);
        res.fitted_parameter_order.emplace_back(name);
    }
    return res;
}

}  // namespace phytorch
