#!/usr/bin/env python3
"""Run bench_python.py and cpp/build/bench_native side by side and emit a
unified speedup table.

Usage:
    python3 cpp/benchmarks/bench_compare.py [--repeats N] [--N N]

Requires cpp/build/bench_native to be built (see cpp/DESIGN.md for build
instructions).
"""
import argparse
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


def parse(stream: str):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--N", type=int, default=60)
    args = ap.parse_args()

    native = os.path.join(ROOT, "cpp", "build", "bench_native")
    if not os.path.exists(native):
        sys.exit(f"native benchmark not built — expected {native}")

    print(f"[1/2] Native C++:  {native} {args.repeats} {args.N}")
    cpp_out = subprocess.check_output([native, str(args.repeats), str(args.N)],
                                       text=True)
    cpp = parse(cpp_out)

    print(f"[2/2] Python:    bench_python.py --repeats {args.repeats} --N {args.N}")
    py_out = subprocess.check_output(
        [sys.executable, os.path.join(HERE, "bench_python.py"),
         "--repeats", str(args.repeats), "--N", str(args.N)],
        text=True)
    py = parse(py_out)

    print()
    print("Per-fit median time (lower is better) — same model, same data, same N")
    print(f"{'Model':<26} {'Python (us)':>12} {'C++ (us)':>10} {'Speedup':>9}  "
          f"{'Py r²':>8} {'C++ r²':>8}")
    print("-" * 78)
    speedups = []
    for name in py:
        if name not in cpp:
            continue
        ratio = py[name]['median_us'] / cpp[name]['median_us']
        speedups.append(ratio)
        print(f"{name:<26} {py[name]['median_us']:>12.2f} {cpp[name]['median_us']:>10.2f} "
              f"{ratio:>8.1f}× {py[name]['r2']:>8.4f} {cpp[name]['r2']:>8.4f}")
    print("-" * 78)
    if speedups:
        gmean = (
            __import__("math").prod(speedups) ** (1.0 / len(speedups))
        )
        print(f"{'geometric mean speedup':<48} {gmean:>8.1f}×")


if __name__ == "__main__":
    main()
