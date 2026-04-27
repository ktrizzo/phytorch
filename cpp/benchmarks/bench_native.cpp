// Native C++ fitting benchmark. Synthesizes data once per model and times
// repeated `fit<M>()` calls. Reports total time and per-fit median.
//
// Designed to be compared against benchmarks/bench_python.py: identical
// model equations, identical parameter ground truth, identical sample
// counts. The C++ build runs the same algorithm a WASM build would; the
// WASM-vs-native overhead is small (V8 LiftOff/TurboFan execute Wasm at
// roughly 0.7-1.0× native), so this gives a faithful upper bound on the
// browser's expected throughput.

#include "phytorch/fit.hpp"

#include "phytorch/models/generic/arrhenius.hpp"
#include "phytorch/models/generic/beta.hpp"
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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using clock_t = std::chrono::high_resolution_clock;

template <class M>
void bench(const std::string& name,
           const Eigen::Matrix<double, Eigen::Dynamic, M::n_inputs>& X,
           const Eigen::Matrix<double, M::n_params, 1>& p_true,
           int repeats,
           double noise_sigma)
{
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, noise_sigma);

    Eigen::VectorXd y_clean = phytorch::batch_forward<M, double>(X, p_true);
    Eigen::VectorXd y       = y_clean;
    for (Eigen::Index i = 0; i < y.size(); ++i) y(i) += noise(rng);

    // Warm-up.
    auto warm = phytorch::fit<M>(X, y);
    (void)warm;

    std::vector<double> times_us;
    times_us.reserve(repeats);
    double final_r2 = 0.0;
    for (int r = 0; r < repeats; ++r) {
        auto t0 = clock_t::now();
        auto res = phytorch::fit<M>(X, y);
        auto t1 = clock_t::now();
        times_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
        final_r2 = res.r_squared;
    }
    std::sort(times_us.begin(), times_us.end());
    const double median = times_us[times_us.size() / 2];
    const double total  = std::accumulate(times_us.begin(), times_us.end(), 0.0);

    std::printf("%-26s  N=%4lld  reps=%4d  median=%8.2f us  total=%9.2f ms  r²=%.6f\n",
                name.c_str(),
                static_cast<long long>(X.rows()),
                repeats,
                median,
                total / 1000.0,
                final_r2);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace phytorch::models;
    int repeats = (argc > 1) ? std::atoi(argv[1]) : 200;
    int N       = (argc > 2) ? std::atoi(argv[2]) : 60;

    auto linspace = [&](double lo, double hi) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> X(N);
        for (int i = 0; i < N; ++i) X(i) = lo + (hi - lo) * i / (N - 1);
        return X;
    };

    {
        Eigen::Matrix<double, Eigen::Dynamic, 1> X1 = linspace(283.15, 320.0);
        Eigen::Matrix<double, Eigen::Dynamic, 1> X = X1;
        Eigen::Matrix<double,2,1> p; p << 2.5, 65.0;
        bench<Arrhenius>("Arrhenius", X, p, repeats, 0.02);
    }
    {
        auto X1 = linspace(-3.0, 3.0);
        Eigen::Matrix<double,2,1> p; p << 1.5, 0.7;
        bench<Linear>("Linear", X1, p, repeats, 0.02);
    }
    {
        auto X1 = linspace(-3.0, 6.0);
        Eigen::Matrix<double,3,1> p; p << 4.0, 1.5, 0.6;
        bench<Gaussian>("Gaussian", X1, p, repeats, 0.02);
    }
    {
        auto X1 = linspace(0.05, 6.0);
        Eigen::Matrix<double,3,1> p; p << 5.0, 2.0, 3.0;
        bench<Sigmoidal>("Sigmoidal", X1, p, repeats, 0.02);
    }
    {
        auto X1 = linspace(0.6, 6.0);
        Eigen::Matrix<double,4,1> p; p << 2.0, 0.5, 1.5, 2.5;
        bench<Weibull>("Weibull", X1, p, repeats, 0.005);
    }
    {
        auto X1 = linspace(283.15, 318.0);
        Eigen::Matrix<double,4,1> p; p << 3.0, 60.0, 200.0, 305.0;
        bench<PeakedArrhenius>("PeakedArrhenius", X1, p, repeats, 0.02);
    }
    {
        auto X1 = linspace(0.1, 30.0);
        Eigen::Matrix<double,2,1> p; p << 8.0, 3.5;
        bench<RectangularHyperbola>("RectangularHyperbola", X1, p, repeats, 0.05);
    }
    {
        auto X1 = linspace(5.0, 1500.0);
        Eigen::Matrix<double,3,1> p; p << 0.05, 30.0, 0.7;
        bench<NonrectangularHyperbola>("NonrectangularHyperbola", X1, p, repeats, 0.1);
    }
    {
        auto X1 = linspace(0.2, 9.8);
        Eigen::Matrix<double,5,1> p; p << 4.0, 0.0, 10.0, 2.5, 3.0;
        bench<Beta>("Beta", X1, p, repeats, 0.005);
    }
    {
        Eigen::Matrix<double, Eigen::Dynamic, 2> X(N, 2);
        for (int i = 0; i < N; ++i) { X(i,0) = -2.0 + 22.0 * i / (N - 1); X(i,1) = 0.4 + 0.5 * (i%5)/4.0; }
        Eigen::Matrix<double,2,1> p; p << 0.02, 9.0;
        bench<BWB1987>("BWB1987", X, p, repeats, 0.005);
    }
    {
        Eigen::Matrix<double, Eigen::Dynamic, 2> X(N, 2);
        for (int i = 0; i < N; ++i) { X(i,0) = 22.0 * i / (N - 1); X(i,1) = 0.5 + 2.5 * (i%5)/4.0; }
        Eigen::Matrix<double,3,1> p; p << 0.015, 8.5, 1.8;
        bench<BBL1995>("BBL1995", X, p, repeats, 0.005);
    }
    {
        Eigen::Matrix<double, Eigen::Dynamic, 2> X(N, 2);
        for (int i = 0; i < N; ++i) { X(i,0) = 22.0 * i / (N - 1); X(i,1) = 0.6 + 2.4 * (i%5)/4.0; }
        Eigen::Matrix<double,2,1> p; p << 0.02, 4.5;
        bench<MED2011>("MED2011", X, p, repeats, 0.005);
    }
    {
        Eigen::Matrix<double, Eigen::Dynamic, 2> X(N, 2);
        for (int i = 0; i < N; ++i) { X(i,0) = 50.0 + 1500.0 * i / (N - 1); X(i,1) = 5.0 + 25.0 * (i%5)/4.0; }
        Eigen::Matrix<double,4,1> p; p << 12.0, 60.0, 1.5e4, 7.0;
        // gs is order 1e-4 here (small Em, large k, large Ds) — noise must
        // be commensurate or the signal disappears entirely.
        bench<BTA2012>("BTA2012", X, p, repeats, 1e-6);
    }

    return 0;
}
