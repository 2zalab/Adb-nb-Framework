"""Concept-drift experiment for the dynamic (online) variant.

A stream of 8,000 samples is generated from the conflict design.  At
t = 4,000 the dependence structure changes abruptly: the redundant
block (features 1..6) becomes independent and a previously independent
group (features 7..12) becomes near-duplicated.  OnlineABDNB tracks the
change through its exponentially weighted dependence graph; a
non-adaptive online GNB (identical code, gamma = 0) serves as control.
Prequential (test-then-train) accuracy is reported.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from abdnb import OnlineABDNB
from experiments.common import save_csv

T, SWITCH = 8000, 4000
D = 13  # 1 strong + 2 blocks of 6


def sample(t: int, rng) -> tuple[np.ndarray, int]:
    y = int(rng.integers(0, 2))
    s = 2 * y - 1
    x0 = s * 1.0 + rng.normal()
    zA = s * 0.35 + rng.normal()
    zB = s * 0.35 + rng.normal()
    rho = 0.95
    if t < SWITCH:  # block A redundant, block B independent
        A = [np.sqrt(rho) * zA + np.sqrt(1 - rho) * rng.normal() for _ in range(6)]
        B = [s * 0.35 + rng.normal() for _ in range(6)]
    else:           # roles swapped
        A = [s * 0.35 + rng.normal() for _ in range(6)]
        B = [np.sqrt(rho) * zB + np.sqrt(1 - rho) * rng.normal() for _ in range(6)]
    return np.array([x0] + A + B), y


def run(gamma: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    model = OnlineABDNB(classes=[0, 1], n_features=D, lam=0.995,
                        gamma=gamma, refresh=25)
    correct, rows = [], []
    for t in range(T):
        x, y = sample(t, rng)
        if t > 100:
            correct.append(int(model.predict(x[None])[0] == y))
        model.partial_fit(x, y)
        if t % 100 == 0 and t > 200:
            rows.append({
                "t": t,
                "acc_window": float(np.mean(correct[-500:])),
                "w_strong": float(model.weights_[0, 0]),
                "w_blockA": float(model.weights_[0, 1:7].mean()),
                "w_blockB": float(model.weights_[0, 7:13].mean()),
            })
    return pd.DataFrame(rows)


def main():
    runs = []
    for seed in range(10):
        df = run(gamma=1.0, seed=seed)
        df["model"] = "Online ABD-NB"
        df["seed"] = seed
        runs.append(df)
        df0 = run(gamma=0.0, seed=seed)
        df0["model"] = "Online GNB"
        df0["seed"] = seed
        runs.append(df0)
        print(f"seed {seed} done")
    save_csv(pd.concat(runs), "drift")


if __name__ == "__main__":
    main()
