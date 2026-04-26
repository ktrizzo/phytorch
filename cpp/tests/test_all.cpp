// Synthesize data from known parameters for each ported model, fit, and
// verify recovery. Doubles as a smoke test for the autodiff/optimizer core.

#include "phytorch/fit.hpp"

#include "phytorch/models/generic/arrhenius.hpp"
#include "phytorch/models/generic/gaussian.hpp"
#include "phytorch/models/generic/linear.hpp"
#include "phytorch/models/generic/nonrectangular_hyperbola.hpp"
#include "phytorch/models/generic/peaked_arrhenius.hpp"
#include "phytorch/models/generic/rectangular_hyperbola.hpp"
#include "phytorch/models/generic/sigmoidal.hpp"
#include "phytorch/models/generic/weibull.hpp"

#include "phytorch/models/stomatal/bbl1995.hpp"
#include "phytorch/models/stomatal/bta2012.hpp"
#include "phytorch/models/stomatal/bwb1987.hpp"
#include "phytorch/models/stomatal/med2011.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

template <class M, class Gen>
Eigen::Matrix<double, Eigen::Dynamic, M::n_inputs>
linspace_inputs(int n, double lo, double hi, Gen& /*rng*/) {
    Eigen::Matrix<double, Eigen::Dynamic, M::n_inputs> X(n, M::n_inputs);
    for (int i = 0; i < n; ++i) X(i, 0) = lo + (hi - lo) * i / (n - 1);
    return X;
}

template <class M>
Eigen::VectorXd synth_y(const Eigen::Matrix<double, Eigen::Dynamic, M::n_inputs>& X,
                       const Eigen::Matrix<double, M::n_params, 1>& p_true) {
    return phytorch::batch_forward<M, double>(X, p_true);
}

template <class M>
bool run(const std::string& name,
         const Eigen::Matrix<double, Eigen::Dynamic, M::n_inputs>& X,
         const Eigen::Matrix<double, M::n_params, 1>& p_true,
         double r2_min)
{
    Eigen::VectorXd y = synth_y<M>(X, p_true);
    auto res = phytorch::fit<M>(X, y);
    bool ok = res.r_squared >= r2_min;
    std::printf("[%-26s] r²=%.6f  iters=%d  conv=%d  status=%s   %s\n",
                name.c_str(), res.r_squared, res.iterations,
                static_cast<int>(res.converged),
                res.status_message.c_str(),
                ok ? "OK" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    using namespace phytorch::models;
    std::mt19937 rng(42);
    int failures = 0;

    {
        Eigen::Matrix<double,2,1> p; p << 2.5, 65.0;
        auto X = linspace_inputs<Arrhenius>(25, 283.15, 320.0, rng);
        if (!run<Arrhenius>("Arrhenius",         X, p, 0.999)) ++failures;
    }
    {
        Eigen::Matrix<double,2,1> p; p << 1.5, 0.7;
        auto X = linspace_inputs<Linear>(20, -3.0, 3.0, rng);
        if (!run<Linear>("Linear",               X, p, 0.999)) ++failures;
    }
    {
        Eigen::Matrix<double,3,1> p; p << 4.0, 1.5, 0.6;
        auto X = linspace_inputs<Gaussian>(40, -3.0, 6.0, rng);
        if (!run<Gaussian>("Gaussian",           X, p, 0.999)) ++failures;
    }
    {
        Eigen::Matrix<double,3,1> p; p << 5.0, 2.0, 3.0;
        auto X = linspace_inputs<Sigmoidal>(40, 0.05, 6.0, rng);
        if (!run<Sigmoidal>("Sigmoidal",         X, p, 0.999)) ++failures;
    }
    {
        Eigen::Matrix<double,4,1> p; p << 2.0, 0.5, 1.5, 2.5;
        auto X = linspace_inputs<Weibull>(40, 0.6, 6.0, rng);
        if (!run<Weibull>("Weibull",             X, p, 0.99)) ++failures;
    }
    {
        Eigen::Matrix<double,4,1> p; p << 3.0, 60.0, 200.0, 305.0;
        auto X = linspace_inputs<PeakedArrhenius>(30, 283.15, 318.0, rng);
        if (!run<PeakedArrhenius>("PeakedArrhenius", X, p, 0.99)) ++failures;
    }
    {
        Eigen::Matrix<double,2,1> p; p << 8.0, 3.5;
        auto X = linspace_inputs<RectangularHyperbola>(30, 0.1, 30.0, rng);
        if (!run<RectangularHyperbola>("RectangularHyperbola", X, p, 0.999)) ++failures;
    }
    {
        Eigen::Matrix<double,3,1> p; p << 0.05, 30.0, 0.7;
        auto X = linspace_inputs<NonrectangularHyperbola>(30, 5.0, 1500.0, rng);
        if (!run<NonrectangularHyperbola>("NonrectangularHyperbola", X, p, 0.99)) ++failures;
    }

    // -- Stomatal models: synthesize multi-input data ----------------
    {
        Eigen::Matrix<double, Eigen::Dynamic, 2> X(40, 2);
        for (int i = 0; i < 40; ++i) {
            X(i, 0) = -2.0 + 22.0 * i / 39.0;       // A: -2..20 μmol/m²/s
            X(i, 1) = 0.4 + 0.5 * (i % 5) / 4.0;    // hs: 0.4..0.9
        }
        Eigen::Matrix<double,2,1> p; p << 0.02, 9.0;
        if (!run<BWB1987>("BWB1987", X, p, 0.99)) ++failures;
    }
    {
        Eigen::Matrix<double, Eigen::Dynamic, 2> X(40, 2);
        for (int i = 0; i < 40; ++i) {
            X(i, 0) = 22.0 * i / 39.0;
            X(i, 1) = 0.5 + 2.5 * (i % 5) / 4.0;     // VPD 0.5..3.0
        }
        Eigen::Matrix<double,3,1> p; p << 0.015, 8.5, 1.8;
        if (!run<BBL1995>("BBL1995", X, p, 0.99)) ++failures;
    }
    {
        Eigen::Matrix<double, Eigen::Dynamic, 2> X(40, 2);
        for (int i = 0; i < 40; ++i) {
            X(i, 0) = 22.0 * i / 39.0;
            X(i, 1) = 0.6 + 2.4 * (i % 5) / 4.0;
        }
        Eigen::Matrix<double,2,1> p; p << 0.02, 4.5;
        if (!run<MED2011>("MED2011", X, p, 0.99)) ++failures;
    }
    {
        Eigen::Matrix<double, Eigen::Dynamic, 2> X(40, 2);
        for (int i = 0; i < 40; ++i) {
            X(i, 0) = 50.0 + 1500.0 * i / 39.0;     // Q
            X(i, 1) = 5.0 + 25.0 * (i % 5) / 4.0;   // Ds
        }
        Eigen::Matrix<double,4,1> p; p << 12.0, 60.0, 1.5e4, 7.0;
        if (!run<BTA2012>("BTA2012", X, p, 0.99)) ++failures;
    }

    std::printf("\n%d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
