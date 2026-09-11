"""Operating-conditions study: when does the dependence correction help,
and when does it fail?

The benchmark answers *whether* ABD-NB improves on naive Bayes; this
experiment answers *under which conditions*, which is the question a
practitioner actually faces.  A single controlled generator exposes seven
knobs -- redundancy multiplicity, within-block dependence strength,
sample size, dimensionality, class imbalance, the functional form of the
dependence, and distribution shift between training and test -- and each
knob is swept one at a time around a common base configuration.

For every cell we report the paired difference (ABD-NB minus Gaussian
naive Bayes) in accuracy, log-loss and ECE with its bootstrap confidence
interval over replications, together with the correction strength the
internal selector actually chose.  The resulting map is the empirical
counterpart of the theory: gains concentrate exactly where the
double-counting mechanism predicts them, and the selection layer keeps
the model at parity everywhere else.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import ABDNB
from experiments.common import expected_calibration_error, save_csv

BASE = dict(n=500, d=12, r=6, rho=0.8, imbalance=0.5, shift=0.0,
            nonlinear=False, strong=1.0, weak=0.35, n_noise=10)

GRID = {
    "redundancy r": ("r", [1, 2, 4, 8, 16, 32]),
    "dependence rho": ("rho", [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]),
    "sample size n": ("n", [50, 100, 200, 500, 1000, 4000]),
    "dimension d": ("d", [7, 13, 26, 51, 101, 201]),
    "class imbalance": ("imbalance", [0.5, 0.35, 0.2, 0.1, 0.05, 0.02]),
    "dependence form": ("nonlinear", [False, True]),
    "distribution shift": ("shift", [0.0, 0.25, 0.5, 1.0, 1.5]),
    # -- adverse regimes: the dependent block carries more and more of
    #    the signal, until it carries all of it and discounting it
    #    transfers evidence to pure noise (the Digits pathology)
    "block informativeness": ("weak", [0.1, 0.35, 0.7, 1.2, 2.0]),
    # -- the genuinely adverse regime: as the independent signal
    #    vanishes, *all* the evidence sits inside the dependent block and
    #    discounting it transfers weight to pure noise
    "independent signal": ("strong", [1.0, 0.5, 0.25, 0.0]),
}


def generate(n, d, r, rho, imbalance, shift, nonlinear, strong, weak,
             n_noise=None, seed=0, test_n=4000):
    """One strong independent feature, ``r`` mutually dependent weak ones,
    and ``d - r - 1`` independent noise features; ``shift`` displaces the
    test distribution along the redundant block (covariate shift)."""
    rng = np.random.default_rng(seed)

    def sample(m, offset):
        y = (rng.random(m) < imbalance).astype(int)
        s = 2 * y - 1
        cols = [s * strong + rng.normal(size=m)]
        z = s * weak + rng.normal(size=m)
        for _ in range(r):
            e = rng.normal(size=m)
            v = np.sqrt(rho) * z + np.sqrt(1 - rho) * e
            if nonlinear:                       # non-monotone dependence
                v = np.sign(v) * v ** 2 / (1.0 + abs(v))
            cols.append(v + offset)
        n_extra = (d - r - 1) if n_noise is None else n_noise
        for _ in range(max(0, n_extra)):
            cols.append(rng.normal(size=m))
        return np.column_stack(cols), y

    Xtr, ytr = sample(n, 0.0)
    Xte, yte = sample(test_n, shift)
    return Xtr, ytr, Xte, yte


def _metrics(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    classes = np.unique(ytr)
    yidx = np.searchsorted(classes, yte)
    return {
        "acc": accuracy_score(yte, model.predict(Xte)),
        "ll": log_loss(yidx, np.clip(proba, 1e-12, 1), labels=range(len(classes))),
        "ece": expected_calibration_error(yidx, proba),
    }


def run(n_rep: int = 15) -> pd.DataFrame:
    rows = []
    for factor, (key, values) in GRID.items():
        for val in values:
            cfg = dict(BASE)
            cfg[key] = val
            if key == "d":          # let the dimension sweep set the width
                cfg["n_noise"] = None
            for rep in range(n_rep):
                Xtr, ytr, Xte, yte = generate(**cfg, seed=1000 + rep)
                if len(np.unique(ytr)) < 2:
                    continue
                abd = ABDNB()
                a = _metrics(abd, Xtr, ytr, Xte, yte)
                g = _metrics(GaussianNB(), Xtr, ytr, Xte, yte)
                rows.append({
                    "factor": factor, "value": str(val), "rep": rep,
                    "acc_abd": a["acc"], "acc_gnb": g["acc"],
                    "ll_abd": a["ll"], "ll_gnb": g["ll"],
                    "ece_abd": a["ece"], "ece_gnb": g["ece"],
                    "gamma": abd.gamma_,
                    "class_specific": int(abd.class_specific_),
                    "d_eff": float(np.mean(abd.effective_dimension_)),
                    "d": Xtr.shape[1],
                })
            print(f"  {factor:20s} = {str(val):6s} done", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for (factor, value), g in df.groupby(["factor", "value"], sort=False):
        da = (g["acc_abd"] - g["acc_gnb"]).to_numpy()
        dl = (g["ll_gnb"] - g["ll_abd"]).to_numpy()      # positive = better
        de = (g["ece_gnb"] - g["ece_abd"]).to_numpy()
        boot = rng.choice(da, size=(n_boot, len(da))).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({
            "factor": factor, "value": value,
            "d_acc": da.mean() * 100, "ci_lo": lo * 100, "ci_hi": hi * 100,
            "d_logloss": dl.mean(), "d_ece": de.mean(),
            "gamma_med": float(np.median(g["gamma"])),
            "gamma_zero_frac": float((g["gamma"] == 0).mean()),
            "deff_ratio": float((g["d_eff"] / g["d"]).mean()),
            "verdict": ("help" if lo > 0 else "hurt" if hi < 0 else "parity"),
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = run()
    save_csv(df, "conditions_raw")
    summary = summarise(df)
    save_csv(summary, "conditions_summary")
    print(summary.to_string(index=False))
