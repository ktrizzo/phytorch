// Browser/Node WASM fitting benchmark.
//
// Mirrors cpp/benchmarks/bench_native.cpp and cpp/benchmarks/bench_python.py
// exactly: same models, same parameter ground truth, same sample counts.
// Designed so its stdout can be parsed by bench_compare.py side by side.
//
// Build first (requires Emscripten):
//   emcmake cmake -S cpp -B cpp/build-wasm -DPHYTORCH_BUILD_WASM=ON
//   cmake --build cpp/build-wasm -j
//
// Run:
//   node cpp/benchmarks/bench_wasm.mjs [repeats] [N]
//
// (For an ES6-modules WASM bundle, Node ≥ 18 is required.)

import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const here   = dirname(fileURLToPath(import.meta.url));
const wasmJs = resolve(here, '..', 'build-wasm', 'phytorch_wasm.js');

if (!existsSync(wasmJs)) {
  console.error(`error: ${wasmJs} not found.`);
  console.error('Build the WASM module first:');
  console.error('  emcmake cmake -S cpp -B cpp/build-wasm -DPHYTORCH_BUILD_WASM=ON');
  console.error('  cmake --build cpp/build-wasm -j');
  process.exit(1);
}

const repeats = parseInt(process.argv[2] ?? '200', 10);
const N       = parseInt(process.argv[3] ?? '60',  10);

// Emscripten emits a CommonJS-style module factory (`createPhytorchModule`).
// Load it via createRequire so this .mjs file can stay an ES module while
// still consuming the generated CJS.
const require = createRequire(import.meta.url);
const createPhytorchModule = require(wasmJs);
// In Node 20+ the global `fetch()` is the undici implementation, which
// refuses raw filesystem paths (ERR_INVALID_URL) — and Emscripten 3.1.x
// reaches for it before falling back to fs. Sidestepping this entirely by
// reading the .wasm bytes ourselves and handing them to the module via
// `wasmBinary`.
const wasmBin = readFileSync(resolve(dirname(wasmJs), 'phytorch_wasm.wasm'));
const phytorch = await createPhytorchModule({ wasmBinary: wasmBin });

// ---- helpers ------------------------------------------------------------

// Float64Array everywhere — Embind's vecFromJSArray<double> takes a fast
// path when given a typed array, dropping per-element JS→C++ overhead from
// O(N · API-call-cost) to a single memcpy.
function linspace(n, lo, hi) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; ++i) out[i] = lo + (hi - lo) * i / (n - 1);
  return out;
}

// Box–Muller with seedable LCG, matching the deterministic noise pattern of
// the native and Python harnesses. (Exact noise sequence cannot match
// numpy's PCG64, but R² values converge to the same plateau — the timing
// is the comparable quantity.)
function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return (s + 0.5) / 4294967296.0;
  };
}
function gauss(rng, sigma) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function synth(model_fn_name, inputs, params, sigma) {
  // We don't have a forward-only export; instead, fit once with ftol=0 and
  // few iterations on the *clean* data, recover predictions, then add noise.
  // Simpler: compute predictions analytically per model below — done by
  // calling the JS implementations of forward() inlined per case.
  throw new Error('use perCase predict()');
}

// ---- benchmark driver ---------------------------------------------------

function bench(label, fitName, data, repeats) {
  // The WASM bindings always read the dependent variable from `data.y`
  // (see cpp/bindings/wasm_bindings.cpp::unpack_data). The .y array length
  // is therefore canonical.
  const N = data.y.length;

  // Warm-up.
  phytorch[fitName](data, {});

  const times = new Array(repeats);
  let r2 = 0;
  for (let r = 0; r < repeats; ++r) {
    const t0 = performance.now();
    const res = phytorch[fitName](data, {});
    const t1 = performance.now();
    times[r] = (t1 - t0) * 1000.0;  // microseconds
    r2 = res.r_squared;
  }
  times.sort((a, b) => a - b);
  const median  = times[times.length >> 1];
  const totalMs = times.reduce((a, b) => a + b, 0) / 1000.0;
  console.log(
    `${label.padEnd(26)}  N=${String(N).padStart(4)}  reps=${String(repeats).padStart(4)}  ` +
    `median=${median.toFixed(2).padStart(9)} us  total=${totalMs.toFixed(2).padStart(9)} ms  ` +
    `r²=${r2.toFixed(6)}`);
}

// ---- forward implementations (clean signal generators) ------------------
// These exist only so we can synthesize y data that matches the native and
// Python harnesses; the *fits* go through phytorch.fit_<name>().

const R = 0.008314, T_ref = 298.15;
const F = {
  arrhenius:  (T,  p) => p.ymax * Math.exp(p.Ha/(R*T_ref) - p.Ha/(R*T)),
  linear:     (x,  p) => p.a + p.b * x,
  gaussian:   (x,  p) => p.a * Math.exp(-((x-p.mu)**2)/(2*p.sigma*p.sigma)),
  sigmoidal:  (x,  p) => p.ymax / (1 + Math.pow(Math.abs(x/p.x50), p.s)),
  weibull:    (x,  p) => {
    const z = (x - p.x0)/p.lambda;
    return z > 0 ? p.ymax * (p.k/p.lambda) * Math.pow(z, p.k-1) * Math.exp(-Math.pow(z, p.k)) : 0;
  },
  peakedArr:  (T,  p) => {
    const farr = Math.exp(p.Ha/(R*T_ref) - p.Ha/(R*T));
    const ratio = Math.max(p.Hd/p.Ha, 1.0001);
    const lt = Math.log(ratio - 1);
    const num = 1 + Math.exp(p.Hd/R*(1/p.Topt - 1/T_ref) - lt);
    const den = 1 + Math.exp(p.Hd/R*(1/p.Topt - 1/T)     - lt);
    return p.ymax * farr * (num/den);
  },
  rectHyp:    (x,  p) => p.ymax*x/(p.x50+x),
  nonRectHyp: (x,  p) => {
    const axy = p.alpha*x + p.ymax;
    const disc = Math.max(axy*axy - 4*p.theta*p.alpha*x*p.ymax, 0);
    return (axy - Math.sqrt(disc))/(2*p.theta);
  },
  beta: (x, p) => {
    if (x <= p.xmin || x >= p.xmax) return 0;
    const u = (x - p.xmin)/(p.xmax - p.xmin);
    if (u <= 1e-10 || u >= 1 - 1e-10) return 0;
    // log Γ via lgamma is not built into Node's Math; use Stirling-ish via series.
    const lgamma = z => {
      // Lanczos approximation (good to ~1e-10)
      const g = 7;
      const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                 771.32342877765313, -176.61502916214059, 12.507343278686905,
                 -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
      if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
      z -= 1;
      let a = c[0];
      const t = z + g + 0.5;
      for (let i = 1; i < g + 2; ++i) a += c[i] / (z + i);
      return 0.5*Math.log(2*Math.PI) + (z+0.5)*Math.log(t) - t + Math.log(a);
    };
    const logy = Math.log(p.a)
               + (p.alpha - 1)*Math.log(u)
               + (p.beta  - 1)*Math.log(1 - u)
               - lgamma(p.alpha) - lgamma(p.beta) + lgamma(p.alpha + p.beta);
    return Math.exp(logy);
  },
  bwb:  (A, hs, p)  => p.gs0 + p.a1 * Math.max(A, 0) * hs / 400.0,
  bbl:  (A, VPD, p) => p.gs0 + p.a1 * Math.max(A, 0) /
                       Math.max((400 - 40) * (1 + VPD/p.D0), 1e-10),
  med:  (A, VPD, p) => p.gs0 + 1.6 * (1 + p.g1/Math.sqrt(VPD)) * Math.max(A, 0) / 400.0,
  bta:  (Q, Ds, p)  => p.Em * (Q + p.i0) /
                       Math.max(p.k + p.b*Q + (Q + p.i0)*Ds, 1e-10) / 1000.0,
};

function noisy(arr, sigma, rng) {
  const out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; ++i) out[i] = arr[i] + gauss(rng, sigma);
  return out;
}
function mapF64(arr, fn) {
  const out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; ++i) out[i] = fn(arr[i], i);
  return out;
}

// ---- run ---------------------------------------------------------------

console.log(`# bench_wasm: repeats=${repeats}  N=${N}`);

let rng = makeRng(7);
{
  const T = linspace(N, 283.15, 320.0);
  const y = noisy(T.map(t => F.arrhenius(t, {ymax: 2.5, Ha: 65})), 0.02, rng);
  bench('Arrhenius', 'fit_arrhenius', { T, y }, repeats);
}
rng = makeRng(7);
{
  const x = linspace(N, -3, 3);
  const y = noisy(x.map(v => F.linear(v, {a: 1.5, b: 0.7})), 0.02, rng);
  bench('Linear', 'fit_linear', { x, y }, repeats);
}
rng = makeRng(7);
{
  const x = linspace(N, -3, 6);
  const y = noisy(x.map(v => F.gaussian(v, {a: 4, mu: 1.5, sigma: 0.6})), 0.02, rng);
  bench('Gaussian', 'fit_gaussian', { x, y }, repeats);
}
rng = makeRng(7);
{
  const x = linspace(N, 0.05, 6);
  const y = noisy(x.map(v => F.sigmoidal(v, {ymax: 5, x50: 2, s: 3})), 0.02, rng);
  bench('Sigmoidal', 'fit_sigmoidal', { x, y }, repeats);
}
rng = makeRng(7);
{
  const x = linspace(N, 0.6, 6);
  const y = noisy(x.map(v => F.weibull(v, {ymax: 2, x0: 0.5, lambda: 1.5, k: 2.5})), 0.005, rng);
  bench('Weibull', 'fit_weibull', { x, y }, repeats);
}
rng = makeRng(7);
{
  const T = linspace(N, 283.15, 318);
  const y = noisy(T.map(t => F.peakedArr(t, {ymax: 3, Ha: 60, Hd: 200, Topt: 305})), 0.02, rng);
  bench('PeakedArrhenius', 'fit_peaked_arrhenius', { T, y }, repeats);
}
rng = makeRng(7);
{
  const x = linspace(N, 0.1, 30);
  const y = noisy(x.map(v => F.rectHyp(v, {ymax: 8, x50: 3.5})), 0.05, rng);
  bench('RectangularHyperbola', 'fit_rectangular_hyperbola', { x, y }, repeats);
}
rng = makeRng(7);
{
  const x = linspace(N, 5, 1500);
  const y = noisy(x.map(v => F.nonRectHyp(v, {alpha: 0.05, ymax: 30, theta: 0.7})), 0.1, rng);
  bench('NonrectangularHyperbola', 'fit_nonrectangular_hyperbola', { x, y }, repeats);
}
rng = makeRng(7);
{
  const x = linspace(N, 0.2, 9.8);
  const y = noisy(x.map(v => F.beta(v, {a: 4, xmin: 0, xmax: 10, alpha: 2.5, beta: 3})), 0.005, rng);
  bench('Beta', 'fit_beta', { x, y }, repeats);
}

// stomatal — multi-input
rng = makeRng(7);
{
  const A  = linspace(N, -2, 20);
  const hs = Float64Array.from({length: N}, (_, i) => 0.4 + 0.5 * (i % 5) / 4);
  const y  = noisy(A.map((a, i) => F.bwb(a, hs[i], {gs0: 0.02, a1: 9})), 0.005, rng);
  bench('BWB1987', 'fit_bwb1987', { A, hs, y }, repeats);
}
rng = makeRng(7);
{
  const A   = linspace(N, 0, 22);
  const VPD = Float64Array.from({length: N}, (_, i) => 0.5 + 2.5 * (i % 5) / 4);
  const y   = noisy(A.map((a, i) => F.bbl(a, VPD[i], {gs0: 0.015, a1: 8.5, D0: 1.8})), 0.005, rng);
  bench('BBL1995', 'fit_bbl1995', { A, VPD, y }, repeats);
}
rng = makeRng(7);
{
  const A   = linspace(N, 0, 22);
  const VPD = Float64Array.from({length: N}, (_, i) => 0.6 + 2.4 * (i % 5) / 4);
  const y   = noisy(A.map((a, i) => F.med(a, VPD[i], {gs0: 0.02, g1: 4.5})), 0.005, rng);
  bench('MED2011', 'fit_med2011', { A, VPD, y }, repeats);
}
rng = makeRng(7);
{
  const Q  = linspace(N, 50, 1550);
  const Ds = Float64Array.from({length: N}, (_, i) => 5 + 25 * (i % 5) / 4);
  const y  = noisy(
    Q.map((q, i) => F.bta(q, Ds[i], {Em: 12, i0: 60, k: 1.5e4, b: 7})), 1e-6, rng);
  bench('BTA2012', 'fit_bta2012', { Q, Ds, y }, repeats);
}
