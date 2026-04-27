#!/usr/bin/env python3
"""Run the native C++, Python, and (optionally) WASM benchmarks side by side
and print a unified speedup table.

Usage:
    python3 cpp/benchmarks/bench_compare.py [--repeats N] [--N N] [--wasm]

The WASM column requires the Emscripten build to exist:
    emcmake cmake -S cpp -B cpp/build-wasm -DPHYTORCH_BUILD_WASM=ON
    cmake --build cpp/build-wasm -j
"""
import argparse
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

LINE = re.compile(
    r"(?P<name>\S+)\s+N=\s*(?P<N>\d+)\s+reps=\s*(?P<reps>\d+)\s+"
    r"median=\s*(?P<median>[0-9.]+)\s+us\s+total=\s*(?P<total>[0-9.]+)\s+ms\s+"
    r"r²=(?P<r2>[0-9.]+)"
)


def parse(stream):
    out = {}
    for line in stream.splitlines():
        m = LINE.match(line.strip())
        if m:
            out[m.group("name")] = {
                "median_us": float(m.group("median")),
                "total_ms":  float(m.group("total")),
                "r2":        float(m.group("r2")),
            }
    return out


def run(label, cmd):
    print(f"[run] {label}: {' '.join(cmd)}")
    return subprocess.check_output(cmd, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--N",       type=int, default=60)
    ap.add_argument("--wasm",    action="store_true",
                    help="Also run the Emscripten WASM benchmark via Node")
    args = ap.parse_args()

    native = os.path.join(ROOT, "cpp", "build", "bench_native")
    if not os.path.exists(native):
        sys.exit(f"native benchmark not built — expected {native}")

    cpp = parse(run("native",
                    [native, str(args.repeats), str(args.N)]))
    py  = parse(run("python",
                    [sys.executable, os.path.join(HERE, "bench_python.py"),
                     "--repeats", str(args.repeats), "--N", str(args.N)]))

    wasm = {}
    if args.wasm:
        wasm_js = os.path.join(ROOT, "cpp", "build-wasm", "phytorch_wasm.js")
        if not os.path.exists(wasm_js):
            sys.exit(f"WASM build not found at {wasm_js} — see DESIGN.md")
        wasm = parse(run("wasm",
                         ["node", os.path.join(HERE, "bench_wasm.mjs"),
                          str(args.repeats), str(args.N)]))

    # ------------- print combined table ----------------
    print()
    cols = ["Python (us)", "C++ (us)"]
    if args.wasm:
        cols.append("WASM (us)")
    cols += ["Py/C++", *(["Py/WASM", "WASM/C++"] if args.wasm else [])]

    header = f"{'Model':<26} " + " ".join(f"{c:>10}" for c in cols)
    print(header)
    print("-" * len(header))

    py_cpp_speedups = []
    py_wasm_speedups = []
    wasm_cpp_overheads = []
    for name in py:
        if name not in cpp:
            continue
        py_us  = py[name]["median_us"]
        cpp_us = cpp[name]["median_us"]
        row = [f"{py_us:>10.2f}", f"{cpp_us:>10.2f}"]
        if args.wasm:
            wasm_us = wasm.get(name, {}).get("median_us", float("nan"))
            row.append(f"{wasm_us:>10.2f}")
        row.append(f"{py_us/cpp_us:>9.1f}×")
        py_cpp_speedups.append(py_us / cpp_us)
        if args.wasm and name in wasm:
            row.append(f"{py_us/wasm_us:>9.1f}×")
            row.append(f"{wasm_us/cpp_us:>9.2f}×")
            py_wasm_speedups.append(py_us / wasm_us)
            wasm_cpp_overheads.append(wasm_us / cpp_us)
        print(f"{name:<26} " + " ".join(row))

    print("-" * len(header))
    if py_cpp_speedups:
        gmean = math.exp(sum(math.log(s) for s in py_cpp_speedups)
                         / len(py_cpp_speedups))
        print(f"geomean Python/C++ : {gmean:>6.1f}×")
    if py_wasm_speedups:
        gmean = math.exp(sum(math.log(s) for s in py_wasm_speedups)
                         / len(py_wasm_speedups))
        print(f"geomean Python/WASM: {gmean:>6.1f}×")
    if wasm_cpp_overheads:
        gmean = math.exp(sum(math.log(s) for s in wasm_cpp_overheads)
                         / len(wasm_cpp_overheads))
        print(f"geomean WASM/C++   : {gmean:>6.2f}×  (overhead vs native; lower is better)")


if __name__ == "__main__":
    main()
