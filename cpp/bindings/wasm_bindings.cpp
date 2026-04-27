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

#include <string>
#include <vector>

namespace phytorch::wasm {

// Fast path: typed-array → C++ vector via HEAP view.
// Embind's vecFromJSArray<double>() iterates JS↔C++ element-by-element,
// which dominates fit time for typical N=60 datasets (each round-trip is
// ~10 μs of marshalling vs ~5 μs of actual fit). When the caller passes a
// Float64Array, we read its byteOffset/length and Eigen::Map straight onto
// the underlying ArrayBuffer — a single memcpy instead of N V8 calls.
static Eigen::VectorXd to_vec_fast(const emscripten::val& v) {
    using emscripten::val;
    if (v.instanceof(val::global("Float64Array"))) {
        const unsigned len = v["length"].as<unsigned>();
        std::vector<double> tmp(len);
        // Use HEAPF64 view to copy via memcpy; emscripten::val::set on heap
        // pointer would be UB, so we reflect through TypedArray.set instead.
        val heapBuffer = val::module_property("HEAPF64")["buffer"];
        val dest = val::global("Float64Array").new_(
            heapBuffer,
            reinterpret_cast<uintptr_t>(tmp.data()),
            len);
        dest.call<void>("set", v);
        return Eigen::Map<Eigen::VectorXd>(tmp.data(), tmp.size());
    }
    // Fallback: plain JS Array → element-wise (slow path).
    std::vector<double> tmp = emscripten::vecFromJSArray<double>(v);
    return Eigen::Map<Eigen::VectorXd>(tmp.data(), tmp.size());
}

// JS object {x: [...], y: [...]}  →  (X, y) Eigen matrices.
template <class M>
static std::pair<
    Eigen::Matrix<double, Eigen::Dynamic, M::n_inputs>,
    Eigen::VectorXd>
unpack_data(const emscripten::val& data) {
    Eigen::VectorXd y = to_vec_fast(data["y"]);
    Eigen::Matrix<double, Eigen::Dynamic, M::n_inputs> X(y.size(), M::n_inputs);
    for (int k = 0; k < M::n_inputs; ++k) {
        const std::string col(M::required_data[k]);
        Eigen::VectorXd v = to_vec_fast(data[col]);
        if (v.size() != y.size())
            throw std::invalid_argument("data column '" + col + "' length mismatch with y");
        X.col(k) = v;
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

// Build a Float64Array that *copies out* of an Eigen::VectorXd via TypedArray
// .set() (one V8 round-trip + one memcpy on the JS side). This avoids the
// O(N) per-element val::set() loop that previously dominated fit time.
static emscripten::val to_f64_array(const Eigen::VectorXd& v) {
    using emscripten::val;
    val heap = val::module_property("HEAPF64")["buffer"];
    val view = val::global("Float64Array").new_(
        heap,
        reinterpret_cast<uintptr_t>(const_cast<double*>(v.data())),
        static_cast<unsigned>(v.size()));
    // Slice copies into a fresh Float64Array detached from WASM memory, so
    // it survives the next allocation/grow.
    return view.call<val>("slice", 0);
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
    out.set("predictions",     to_f64_array(r.predictions));
    out.set("residuals",       to_f64_array(r.residuals));
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
    emscripten::function("fit_arrhenius",                 &wasm::fit_model<models::Arrhenius>);
    emscripten::function("fit_linear",                    &wasm::fit_model<models::Linear>);
    emscripten::function("fit_gaussian",                  &wasm::fit_model<models::Gaussian>);
    emscripten::function("fit_sigmoidal",                 &wasm::fit_model<models::Sigmoidal>);
    emscripten::function("fit_weibull",                   &wasm::fit_model<models::Weibull>);
    emscripten::function("fit_peaked_arrhenius",          &wasm::fit_model<models::PeakedArrhenius>);
    emscripten::function("fit_rectangular_hyperbola",     &wasm::fit_model<models::RectangularHyperbola>);
    emscripten::function("fit_nonrectangular_hyperbola",  &wasm::fit_model<models::NonrectangularHyperbola>);
    emscripten::function("fit_beta",                      &wasm::fit_model<models::Beta>);
    emscripten::function("fit_bwb1987", &wasm::fit_model<models::BWB1987>);
    emscripten::function("fit_bbl1995", &wasm::fit_model<models::BBL1995>);
    emscripten::function("fit_med2011", &wasm::fit_model<models::MED2011>);
    emscripten::function("fit_bta2012", &wasm::fit_model<models::BTA2012>);
}
#endif
