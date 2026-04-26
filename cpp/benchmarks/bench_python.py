#!/usr/bin/env python3
"""Python phytorch fitting benchmark.

Mirrors cpp/benchmarks/bench_native.cpp exactly: same models, same parameter
ground truth, same sample counts. Reports total time and per-fit median.
"""
import argparse
import os
import sys
import time

import numpy as np

# Ensure we import the *local* phytorch package, not anything pip-installed.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import phytorch  # noqa: E402
from phytorch.models.generic.arrhenius                import Arrhenius
from phytorch.models.generic.linear                   import Linear
from phytorch.models.generic.gaussian                 import Gaussian
from phytorch.models.generic.sigmoidal                import Sigmoidal
from phytorch.models.generic.weibull                  import Weibull
from phytorch.models.generic.peaked_arrhenius        import PeakedArrhenius
from phytorch.models.generic.rectangular_hyperbola    import RectangularHyperbola
from phytorch.models.generic.nonrectangular_hyperbola import NonrectangularHyperbola
from phytorch.models.stomatal.bwb1987 import BWB1987
from phytorch.models.stomatal.bbl1995 import BBL1995
from phytorch.models.stomatal.med2011 import MED2011
from phytorch.models.stomatal.bta2012 import BTA2012


def linspace(N, lo, hi):
    return np.linspace(lo, hi, N)


def synth(model, data, params, sigma):
    rng = np.random.default_rng(7)
    y = model.forward(data, params)
    return y + rng.normal(0.0, sigma, size=y.shape)


def bench(label, model, data, repeats):
    y_field = model.required_data()[-1]
    N = len(data[y_field])

    phytorch.fit(model, data, options={"verbose": False})  # warm-up

    times_us = []
    r2_last = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = phytorch.fit(model, data, options={"verbose": False})
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)
        r2_last = res.r_squared if res.r_squared is not None else 0.0

    times_us.sort()
    median = times_us[len(times_us) // 2]
    total_ms = sum(times_us) / 1000.0
    print(f"{label:<26}  N={N:>4}  reps={repeats:>4}  "
          f"median={median:>9.2f} us  total={total_ms:>9.2f} ms  r²={r2_last:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--N", type=int, default=60)
    args = ap.parse_args()

    R, N = args.repeats, args.N

    # Generic, single-input models.
    cases = [
        ("Arrhenius",                 Arrhenius(),                 linspace(N, 283.15, 320.0),  {"ymax": 2.5, "Ha": 65.0},                                  0.02),
        ("Linear",                    Linear(),                    linspace(N, -3.0, 3.0),      {"a": 1.5, "b": 0.7},                                       0.02),
        ("Gaussian",                  Gaussian(),                  linspace(N, -3.0, 6.0),      {"a": 4.0, "mu": 1.5, "sigma": 0.6},                        0.02),
        ("Sigmoidal",                 Sigmoidal(),                 linspace(N, 0.05, 6.0),      {"ymax": 5.0, "x50": 2.0, "s": 3.0},                        0.02),
        ("Weibull",                   Weibull(),                   linspace(N, 0.6, 6.0),       {"ymax": 2.0, "x0": 0.5, "lambda": 1.5, "k": 2.5},          0.005),
        ("PeakedArrhenius",           PeakedArrhenius(),           linspace(N, 283.15, 318.0),  {"ymax": 3.0, "Ha": 60.0, "Hd": 200.0, "Topt": 305.0},      0.02),
        ("RectangularHyperbola",      RectangularHyperbola(),      linspace(N, 0.1, 30.0),      {"ymax": 8.0, "x50": 3.5},                                  0.05),
        ("NonrectangularHyperbola",   NonrectangularHyperbola(),   linspace(N, 5.0, 1500.0),    {"alpha": 0.05, "ymax": 30.0, "theta": 0.7},                0.1),
    ]

    for label, model, x, params, sigma in cases:
        data = {"x": x, "y": None}
        data["y"] = synth(model, data, params, sigma)
        bench(label, model, data, R)

    # Stomatal models — multi-input data.
    A = np.linspace(-2.0, 20.0, N); hs = 0.4 + 0.5 * (np.arange(N) % 5) / 4.0
    data = {"A": A, "hs": hs, "gs": None}
    data["gs"] = synth(BWB1987(), data, {"gs0": 0.02, "a1": 9.0}, 0.005)
    bench("BWB1987", BWB1987(), data, R)

    A = np.linspace(0.0, 22.0, N); VPD = 0.5 + 2.5 * (np.arange(N) % 5) / 4.0
    data = {"A": A, "VPD": VPD, "gs": None}
    data["gs"] = synth(BBL1995(), data, {"gs0": 0.015, "a1": 8.5, "D0": 1.8}, 0.005)
    bench("BBL1995", BBL1995(), data, R)

    A = np.linspace(0.0, 22.0, N); VPD = 0.6 + 2.4 * (np.arange(N) % 5) / 4.0
    data = {"A": A, "VPD": VPD, "gs": None}
    data["gs"] = synth(MED2011(), data, {"gs0": 0.02, "g1": 4.5}, 0.005)
    bench("MED2011", MED2011(), data, R)

    Q = np.linspace(50.0, 1550.0, N); Ds = 5.0 + 25.0 * (np.arange(N) % 5) / 4.0
    data = {"Q": Q, "Ds": Ds, "gs": None}
    data["gs"] = synth(BTA2012(),
                       data,
                       {"Em": 12.0, "i0": 60.0, "k": 1.5e4, "b": 7.0},
                       0.0005)
    bench("BTA2012", BTA2012(), data, R)


if __name__ == "__main__":
    main()
