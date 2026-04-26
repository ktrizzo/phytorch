#pragma once

// Adam (Kingma & Ba 2014) on the sum-of-squares loss, with simple projection
// onto box bounds. Mirrors the phytorch torch_optimizer path: useful when the
// loss surface has many local minima or when LM's damped Gauss–Newton step
// gets stuck (most often in coupled biochemical models like FvCB).
//
// Gradient ∇_p L = Jᵀ r, where J is again obtained via forward-mode AD.

#include "../autodiff.hpp"
#include "../fit_options.hpp"
#include "../fit_result.hpp"
#include "../model.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace phytorch::optim {

template <class Model>
FitResult adam(
    const Eigen::Matrix<double, Eigen::Dynamic, Model::n_inputs>& X,
    const Eigen::VectorXd& y,
    const Eigen::Matrix<double, Model::n_params, 1>& p0,
    const Eigen::Matrix<double, Model::n_params, 1>& lb,
    const Eigen::Matrix<double, Model::n_params, 1>& ub,
    const FitOptions& opts)
{
    using Dual = ad::Dual<Model::n_params>;
    using VecP = Eigen::Matrix<double, Model::n_params, 1>;
    using DualV = Eigen::Matrix<Dual, Model::n_params, 1>;

    const Eigen::Index N = y.size();
    const int          P = Model::n_params;

    auto project = [&](VecP p) {
        for (int j = 0; j < P; ++j) p(j) = std::clamp(p(j), lb(j), ub(j));
        return p;
    };

    VecP p = project(p0);
    VecP m = VecP::Zero();
    VecP v = VecP::Zero();

    Eigen::VectorXd r;
    double prev_loss = std::numeric_limits<double>::infinity();
    bool converged = false;
    int  iter      = 0;
    std::string status = "max_iterations";

    for (; iter < opts.max_iterations; ++iter) {
        // Forward + Jacobian via dual numbers.
        DualV pd;
        for (int j = 0; j < P; ++j) pd(j) = Dual::seed(p(j), j);

        r.resize(N);
        VecP grad = VecP::Zero();
        for (Eigen::Index i = 0; i < N; ++i) {
            Eigen::Matrix<Dual, Model::n_inputs, 1> xi;
            for (int k = 0; k < Model::n_inputs; ++k) xi(k) = Dual(X(i, k));
            Dual yi = Model::template forward<Dual>(xi, pd);
            const double ri = yi.value - y(i);
            r(i) = ri;
            grad.noalias() += ri * yi.grad;  // ∇ ½||r||² = Jᵀr
        }

        const double loss = r.squaredNorm();
        if (std::abs(prev_loss - loss) < opts.ftol * std::max(prev_loss, 1.0)) {
            converged = true; status = "ftol"; break;
        }
        if (grad.cwiseAbs().maxCoeff() < opts.gtol) { converged = true; status = "gtol"; break; }
        prev_loss = loss;

        const double t  = static_cast<double>(iter + 1);
        const double bc1 = 1.0 - std::pow(opts.beta1, t);
        const double bc2 = 1.0 - std::pow(opts.beta2, t);

        m = opts.beta1 * m + (1.0 - opts.beta1) * grad;
        v = opts.beta2 * v + (1.0 - opts.beta2) * grad.cwiseAbs2();

        VecP m_hat = m / bc1;
        VecP v_hat = v / bc2;
        VecP step  = (m_hat.array() / (v_hat.array().sqrt() + opts.eps)).matrix();

        VecP p_new = project(p - opts.learning_rate * step);
        if ((p_new - p).norm() < opts.xtol * (p.norm() + opts.xtol)) {
            converged = true; status = "xtol"; p = p_new; break;
        }
        p = p_new;
    }

    FitResult res;
    res.iterations    = iter;
    res.converged     = converged;
    res.method        = "adam";
    res.status_message = status;
    res.residuals     = r;
    res.loss          = r.squaredNorm();
    const double ss_tot = (y.array() - y.mean()).square().sum();
    res.r_squared = ss_tot > 0.0 ? 1.0 - res.loss / ss_tot : 0.0;
    return res;
}

}  // namespace phytorch::optim
