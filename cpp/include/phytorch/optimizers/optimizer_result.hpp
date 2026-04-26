#pragma once

#include <Eigen/Core>
#include <string>

namespace phytorch::optim {

// Pure-numeric output of an optimizer. The fit() top-level wraps this into
// a FitResult with named parameters and predictions.
template <int P>
struct OptimizerResult {
    Eigen::Matrix<double, P, 1> p_final;
    Eigen::VectorXd             residuals;
    double                      loss       = 0.0;
    int                         iterations = 0;
    bool                        converged  = false;
    std::string                 status_message;
    Eigen::MatrixXd             covariance;  // P×P, may be empty
};

}  // namespace phytorch::optim
