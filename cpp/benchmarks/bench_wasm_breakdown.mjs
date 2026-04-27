// Attribute the WASM/native gap by timing the floor cost of each layer.
//
//   T_noop      — pure JS↔WASM round-trip, 0 args
//   T_input     — also unpacks {T, y} into Eigen vectors (input marshalling)
//   T_full      — also packs predictions+residuals back as Float64Array
//                 (i.e. a real fit minus the LM kernel)
//   T_fit       — actual fit_arrhenius
//
// Then native compute time T_native lets us back out:
//   compute_wasm  = T_fit  - T_full
//   marshal_wasm  = T_full - T_noop      (input + output combined)
//   roundtrip     = T_noop
//
// Run: node cpp/benchmarks/bench_wasm_breakdown.mjs [N] [reps]

import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const here   = dirname(fileURLToPath(import.meta.url));
const wasmJs = resolve(here, '..', 'build-wasm', 'phytorch_wasm.js');
if (!existsSync(wasmJs)) {
  console.error(`error: ${wasmJs} not found — build cpp/build-wasm first.`);
  process.exit(1);
}
const require = createRequire(import.meta.url);
const createPhytorchModule = require(wasmJs);
const wasmBin = readFileSync(resolve(dirname(wasmJs), 'phytorch_wasm.wasm'));
const m = await createPhytorchModule({ wasmBinary: wasmBin });

const N    = parseInt(process.argv[2] ?? '60',   10);
const reps = parseInt(process.argv[3] ?? '2000', 10);

const T = new Float64Array(N);
for (let i = 0; i < N; ++i) T[i] = 283.15 + (320 - 283.15) * i / (N - 1);
const ymax = 2.5, Ha = 65, R = 0.008314, Tref = 298.15;
const y = new Float64Array(N);
for (let i = 0; i < N; ++i)
  y[i] = ymax * Math.exp(Ha/(R*Tref) - Ha/(R*T[i]));
const data = { T, y };

function time(label, fn) {
  // warm-up
  for (let i = 0; i < 50; ++i) fn();
  const samples = new Array(reps);
  for (let r = 0; r < reps; ++r) {
    const t0 = performance.now();
    fn();
    samples[r] = (performance.now() - t0) * 1000.0;  // µs
  }
  samples.sort((a, b) => a - b);
  const median = samples[reps >> 1];
  return { label, median };
}

const r = [
  time('noop()                       ', () => m.noop()),
  time('noop_data_arrhenius(data)    ', () => m.noop_data_arrhenius(data)),
  time('noop_full_arrhenius(data)    ', () => m.noop_full_arrhenius(data)),
  time('fit_arrhenius(data, {})      ', () => m.fit_arrhenius(data, {})),
];

console.log(`# bench_wasm_breakdown: N=${N}  reps=${reps}\n`);
for (const x of r) console.log(`  ${x.label}  ${x.median.toFixed(2)} µs`);

const t_noop  = r[0].median;
const t_in    = r[1].median;
const t_full  = r[2].median;
const t_fit   = r[3].median;

console.log('\nDecomposition (µs):');
console.log(`  JS↔WASM round trip          ${t_noop.toFixed(2)}`);
console.log(`  + input marshalling          ${(t_in - t_noop).toFixed(2)}`);
console.log(`  + output marshalling         ${(t_full - t_in).toFixed(2)}`);
console.log(`  + actual fit (LM + AD)       ${(t_fit - t_full).toFixed(2)}`);
console.log(`  ────────────────────────────`);
console.log(`  total fit_arrhenius          ${t_fit.toFixed(2)}`);
console.log(`\n  marshalling (round-trip + I/O)  ${(t_full).toFixed(2)} µs`);
console.log(`  fraction marshalling             ${(100 * t_full / t_fit).toFixed(1)}%`);
