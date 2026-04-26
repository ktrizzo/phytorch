// Emscripten/Embind layer.
//
// Each registered model gets a `fit_<name>(data, options)` JS function. On
// the JS side, `data` and `options` are plain objects so React code in
// website/ can call into the WASM module without ever touching the heap.
//
// This file is the *only* place where new models need to be enumerated for
// the web build — the C++ fitting core itself is fully generic.

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#include <emscripten/val.h>
#endif

#include "phytorch/fit.hpp"
#include "phytorch/models/generic/arrhenius.hpp"

#include <string>
#include <vector>

namespace phytorch::wasm {

// JS object {x: [...], y: [...]}  →  (X, y) Eigen matrices.
template <class M>
static std::pair<
    Eigen::Matrix<double, Eigen::Dynamic, M::n_inputs>,
    Eigen::VectorXd>
unpack_data(const emscripten::val& data) {
    using emscripten::val;
    using emscripten::vecFromJSArray;

    std::vector<double> y_raw = vecFromJSArray<double>(data["y"]);
    Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(y_raw.data(), y_raw.size());

    Eigen::Matrix<double, Eigen::Dynamic, M::n_inputs> X(y.size(), M::n_inputs);
    for (int k = 0; k < M::n_inputs; ++k) {
        const std::string col(M::required_data[k]);
        std::vector<double> v = vecFromJSArray<double>(data[col]);
        if (static_cast<Eigen::Index>(v.size()) != y.size())
            throw std::invalid_argument("data column '" + col + "' length mismatch with y");
        X.col(k) = Eigen::Map<Eigen::VectorXd>(v.data(), v.size());
    }
    return {X, y};
}

static FitOptions unpack_options(const emscripten::val& opts) {
    FitOptions o;
    if (opts.isUndefined() || opts.isNull()) return o;

    if (opts.hasOwnProperty("method")) {
        std::string m = opts["method"].as<std::string>();
        if      (m == "lm" || m == "scipy") o.method = Method::LevenbergMarquardt;
        else if (m == "adam" || m == "torch") o.method = Method::Adam;
        else                                   o.method = Method::Auto;
    }
    if (opts.hasOwnProperty("max_iterations")) o.max_iterations = opts["max_iterations"].as<int>();
    if (opts.hasOwnProperty("ftol"))           o.ftol           = opts["ftol"].as<double>();
    if (opts.hasOwnProperty("xtol"))           o.xtol           = opts["xtol"].as<double>();
    if (opts.hasOwnProperty("gtol"))           o.gtol           = opts["gtol"].as<double>();
    if (opts.hasOwnProperty("learning_rate"))  o.learning_rate  = opts["learning_rate"].as<double>();
    if (opts.hasOwnProperty("verbose"))        o.verbose        = opts["verbose"].as<bool>();

    auto load_param_map = [&](const char* key, auto& dest) {
        if (!opts.hasOwnProperty(key)) return;
        emscripten::val keys = emscripten::val::global("Object").call<emscripten::val>("keys", opts[key]);
        const unsigned len = keys["length"].as<unsigned>();
        for (unsigned i = 0; i < len; ++i) {
            std::string k = keys[i].as<std::string>();
            if constexpr (std::is_same_v<std::decay_t<decltype(dest)>,
                                         std::unordered_map<std::string,
                                             std::pair<double,double>>>) {
                emscripten::val pair = opts[key][k];
                dest[k] = { pair[0].as<double>(), pair[1].as<double>() };
            } else {
                dest[k] = opts[key][k].as<double>();
            }
        }
    };
    load_param_map("initial_guess",   o.initial_guess);
    load_param_map("fixed_parameters",o.fixed_parameters);
    load_param_map("bounds",          o.bounds);
    return o;
}

static emscripten::val pack_result(const FitResult& r) {
    emscripten::val out = emscripten::val::object();
    emscripten::val params = emscripten::val::object();
    for (auto const& [k, v] : r.parameters) params.set(k, v);
    out.set("parameters",      params);
    out.set("loss",            r.loss);
    out.set("r_squared",       r.r_squared);
    out.set("converged",       r.converged);
    out.set("iterations",      r.iterations);
    out.set("method",          r.method);
    out.set("status_message",  r.status_message);

    emscripten::val preds = emscripten::val::array();
    for (Eigen::Index i = 0; i < r.predictions.size(); ++i) preds.set(i, r.predictions(i));
    out.set("predictions", preds);

    emscripten::val resid = emscripten::val::array();
    for (Eigen::Index i = 0; i < r.residuals.size(); ++i)   resid.set(i, r.residuals(i));
    out.set("residuals",   resid);
    return out;
}

template <class M>
static emscripten::val fit_model(emscripten::val data, emscripten::val options) {
    auto [X, y] = unpack_data<M>(data);
    return pack_result(fit<M>(X, y, unpack_options(options)));
}

}  // namespace phytorch::wasm

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(phytorch_module) {
    using namespace phytorch;
    emscripten::function("fit_arrhenius",
        &wasm::fit_model<models::Arrhenius>);
    // Add additional models here as they are ported:
    //   emscripten::function("fit_medlyn",   &wasm::fit_model<models::Medlyn2011>);
    //   emscripten::function("fit_sigmoid",  &wasm::fit_model<models::HydraulicSigmoid>);
}
#endif
