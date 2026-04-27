# phytorch-wasm — C++→WASM Plant Physiology Fitting Library

A header-mostly C++17 port of phytorch's fitting framework, compiled to
WebAssembly via Emscripten so the same physiological models that run in
the Python package can also run client-side in the browser at near-native
speed.

## Goals

- Match phytorch's Python fitting API surface (`fit(model, data, options) -> FitResult`)
- Run efficiently in the browser via WASM (no Python, no NumPy, no SciPy at runtime)
- Support both scipy-style nonlinear least squares (Levenberg–Marquardt) and
  gradient-based optimization (Adam) — the same two paths phytorch exposes
- Allow models to be written once in C++ and used both natively (CLI/tests)
  and from JavaScript via Embind

## Non-goals (initial release)

- ODE solvers, FvCB's coupled biochemical model — these come later once the
  fitting core is solid
- Reverse-mode autodiff — forward-mode dual numbers are sufficient for the
  small (≤20) parameter spaces typical of plant physiology fits
- Plotting — the website (Docusaurus + React) handles visualization in JS

## Architecture

```
cpp/
├── CMakeLists.txt                      # Native + Emscripten builds
├── include/phytorch/
│   ├── autodiff.hpp                    # Forward-mode dual numbers
│   ├── parameter.hpp                   # ParameterInfo (default/bounds/units)
│   ├── model.hpp                       # CRTP Model<Derived> base
│   ├── fit.hpp                         # fit<M>(data, options) -> FitResult
│   ├── fit_options.hpp                 # FitOptions struct
│   ├── fit_result.hpp                  # FitResult struct
│   ├── optimizers/
│   │   ├── levenberg_marquardt.hpp     # LM (scipy curve_fit equivalent)
│   │   └── adam.hpp                    # Adam (torch optimizer equivalent)
│   └── models/
│       └── generic/
│           └── arrhenius.hpp           # Reference model
├── bindings/
│   └── wasm_bindings.cpp               # Embind: one entry per model
└── tests/
    └── test_arrhenius.cpp              # Native sanity check
```

## Mapping from Python to C++

| Python (phytorch)                                     | C++ (phytorch-wasm)                              |
|-------------------------------------------------------|--------------------------------------------------|
| `class Model(ABC)`                                    | `template<class D> struct Model` (CRTP)          |
| `forward(data: dict, parameters: dict) -> ndarray`    | `template<class T> Vec<T> forward(X, p)` (static)|
| `parameter_info() -> dict`                            | `static constexpr std::array<ParameterInfo,N>`   |
| `initial_guess(data) -> dict`                         | `static Vec<double> initial_guess(X, y)`         |
| `fit(model, data, options)`                           | `fit<Model>(data, options)`                      |
| `scipy.optimize.curve_fit` (Trust Region Reflective)  | `LevenbergMarquardt` (Eigen-based)               |
| `torch.optim.Adam` + autograd                         | `Adam` + dual-number forward-mode AD             |
| `FitResult` (parameters, r², covariance, …)           | `FitResult` (same fields)                        |

Models are templates over the scalar type so the **same** `forward` code
runs in three modes:

1. `forward<double>` — fast prediction
2. `forward<Dual>` — Jacobian via forward-mode AD (for LM and Adam)
3. (later) `forward<Interval>` — uncertainty propagation

## Numerics

- **Linear algebra**: Eigen 3.4 (header-only, WASM-friendly, ~1 MB after dead-code elimination)
- **Autodiff**: Custom `Dual` (header-only, ~150 LOC); avoids dragging in `autodiff.github.io`
- **LM optimizer**: Bounded LM with reflective steps for box constraints (matches scipy's
  `method='trf'` default). Jacobian via dual numbers.
- **Adam**: Standard Kingma & Ba 2014 with optional cosine LR decay.
- **Stopping**: `ftol`, `xtol`, `gtol`, `max_iterations` — same names as scipy/phytorch options.

## JS interop

Embind exposes `fit_<model_name>(data_json, options_json) -> result_json`
for each registered model. Inputs/outputs are plain JS objects so no manual
heap management is required from the website's React code:

```ts
import init from './phytorch_wasm.js';
const phytorch = await init();
const result = phytorch.fit_arrhenius(
  { x: [288, 298, 308, 318], y: [0.5, 1.0, 1.9, 3.4] },
  { method: 'lm', bounds: { Ha: [0, 200] } }
);
// result = { parameters: { ymax, Ha }, r_squared, converged, residuals, ... }
```

## Build

```bash
# Native (for tests + dev)
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j

# WebAssembly
emcmake cmake -S cpp -B cpp/build-wasm -DPHYTORCH_BUILD_WASM=ON -DPHYTORCH_BUILD_TESTS=OFF
cmake --build cpp/build-wasm -j
# -> cpp/build-wasm/phytorch_wasm.{js,wasm}, copied into website/static/wasm/
```

## Benchmarking Python vs Native vs WASM

```bash
python3 cpp/benchmarks/bench_compare.py --repeats 200 --N 60 --wasm
```

This runs three harnesses with the same models, parameter ground truth, and
sample counts — `cpp/benchmarks/bench_native.cpp` (native C++ binary),
`cpp/benchmarks/bench_python.py` (phytorch.fit through SciPy), and
`cpp/benchmarks/bench_wasm.mjs` (Node loads the Emscripten module and times
`fit_<name>(data, options)` calls). The script prints a table with per-fit
median time and a geometric-mean speedup.

Representative numbers from this repo (Linux x86_64, gcc 13, Emscripten
3.1.6, Node 22, N=60, 200 reps per model):

| Model                    | Python (μs) | C++ (μs) | WASM (μs) | Py/WASM |
|--------------------------|-------------|----------|-----------|---------|
| Arrhenius                | 1850        | 6.5      | 61        | 30×     |
| Linear                   | 145         | 2.1      | 38        | 4×      |
| Gaussian                 | 1585        | 13.9     | 52        | 31×     |
| Weibull                  | 5147        | 38.3     | 80        | 64×     |
| Beta                     | 5551        | 62.4     | 123       | 45×     |
| BWB1987 / BBL1995 / MED  | 1500–2100   | 3–12     | 33–36     | 44–59×  |
| BTA2012                  | 2063        | 7.4      | 37        | 56×     |
| **geomean**              |             |          |           | **37×** |

WASM lands at ~5× of native (Embind input/output marshalling per fit call
plus V8's WASM JIT not matching gcc -O3 inlining). For typical N=60
physiology datasets each fit completes in well under 150 μs — interactive
for any plausible web-UI use. Throughput scales near-linearly with N.

## Roadmap

1. **(this PR)** Core fitting design: Model, autodiff, LM, Adam, FitResult, bindings stub, one model
2. Port the 9 generic models (linear, gaussian, sigmoidal, weibull, peaked_arrhenius, …)
3. Port the 4 stomatal models (BWB, Leuning, Medlyn, BMF)
4. Port hydraulics (Sigmoid, SJB2018 pressure–volume)
5. FvCB photosynthesis (requires nonlinear root solving — last)
6. Wire up website demo pages: each model gets an interactive page that fits
   user-pasted data in-browser
