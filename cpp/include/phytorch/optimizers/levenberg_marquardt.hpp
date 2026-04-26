#pragma once

// Bounded Levenberg–Marquardt — the C++ analogue of scipy.optimize.curve_fit
// with method='trf'. Jacobians come from forward-mode autodiff (Dual<N>) so
// the model author never writes a derivative by hand.
//
// Standard LM with Marquardt damping λ:
//   J  = ∂r/∂p, g = Jᵀr, H = JᵀJ + λI
//   solve H Δp = -g; accept if loss decreases (decay λ), else reject (grow λ)
// Box constraints handled by reflecting trial points back into the feasible
// region — close enough to scipy's 'trf' for the well-conditioned problems
// typical of plant physiology fitting.

#include "../autodiff.hpp"
#include "../fit_options.hpp"
#include "optimizer_result.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace phytorch::optim {

template <class Model>
OptimizerResult<Model::n_params> levenberg_marquardt(
    const Eigen::Matrix<double, Eigen::Dynamic, Model::n_inputs>& X,
    const Eigen::VectorXd& y,
    const Eigen::Matrix<double, Model::n_params, 1>& p0,
    const Eigen::Matrix<double, Model::n_params, 1>& lb,
    const Eigen::Matrix<double, Model::n_params, 1>& ub,
    const FitOptions& opts)
{
    using Dual  = ad::Dual<Model::n_params>;
    using VecP  = Eigen::Matrix<double, Model::n_params, 1>;
    using DualV = Eigen::Matrix<Dual,   Model::n_params, 1>;

    const Eigen::Index N = y.size();
    constexpr int      P = Model::n_params;

    auto reflect = [&](VecP p) {
        for (int j = 0; j < P; ++j) {
            if (p(j) < lb(j)) p(j) = lb(j) + (lb(j) - p(j));
            if (p(j) > ub(j)) p(j) = ub(j) - (p(j) - ub(j));
            p(j) = std::clamp(p(j), lb(j), ub(j));
        }
        return p;
    };

    auto eval = [&](const VecP& p,
                    Eigen::VectorXd& r,
                    Eigen::MatrixXd& J)
    {
        DualV pd;
        for (int j = 0; j < P; ++j) pd(j) = Dual::seed(p(j), j);

        r.resize(N);
        J.resize(N, P);
        for (Eigen::Index i = 0; i < N; ++i) {
            Eigen::Matrix<Dual, Model::n_inputs, 1> xi;
            for (int k = 0; k < Model::n_inputs; ++k) xi(k) = Dual(X(i, k));
            Dual yi  = Model::template forward<Dual>(xi, pd);
            r(i)     = yi.value - y(i);
            J.row(i) = yi.grad.transpose();
        }
    };

    VecP p = reflect(p0);
    Eigen::VectorXd r;
    Eigen::MatrixXd J;
    eval(p, r, J);
    double cost = 0.5 * r.squaredNorm();

    double      lambda    = 1e-3;
    bool        converged = false;
    int         iter      = 0;
    std::string status    = "max_iterations";

    for (; iter < opts.max_iterations; ++iter) {
        Eigen::MatrixXd H = J.transpose() * J;
        Eigen::VectorXd g = J.transpose() * r;
        // Marquardt's diagonal scaling: λ·diag(JᵀJ) instead of λ·I makes the
        // step roughly invariant to per-parameter scaling, which matters a
        // lot when parameters span several orders of magnitude (e.g. BTA2012
        // where k ≈ 1e4 and b ≈ 7).
        H.diagonal().array() += lambda * H.diagonal().array().abs();

        VecP dp = H.ldlt().solve(-g);

        if (g.cwiseAbs().maxCoeff() < opts.gtol) { converged = true; status = "gtol"; break; }
        if (dp.norm() < opts.xtol * (p.norm() + opts.xtol)) { converged = true; status = "xtol"; break; }

        VecP p_trial = reflect(p + dp);
        Eigen::VectorXd r_trial;
        Eigen::MatrixXd J_trial;
        eval(p_trial, r_trial, J_trial);
        double cost_trial = 0.5 * r_trial.squaredNorm();

        if (cost_trial < cost) {
            const double rel_red = (cost - cost_trial) / std::max(cost, 1e-300);
            p = p_trial;
            r = std::move(r_trial);
            J = std::move(J_trial);
            cost = cost_trial;
            lambda = std::max(lambda / 3.0, 1e-12);
            if (rel_red < opts.ftol) { converged = true; status = "ftol"; break; }
        } else {
            lambda = std::min(lambda * 5.0, 1e12);
        }
    }

    OptimizerResult<P> out;
    out.p_final        = p;
    out.residuals      = r;
    out.loss           = r.squaredNorm();
    out.iterations     = iter;
    out.converged      = converged;
    out.status_message = status;

    if (N > P) {
        const double sigma2 = out.loss / static_cast<double>(N - P);
        Eigen::MatrixXd JtJ = J.transpose() * J;
        out.covariance = sigma2 * JtJ.completeOrthogonalDecomposition().pseudoInverse();
    }
    return out;
}

}  // namespace phytorch::optim
