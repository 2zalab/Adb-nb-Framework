"""Scalability: full-size datasets, and the empirical cost curves.

The main benchmark caps datasets at 4000 samples so that 22 models can be
evaluated under a 3 x 10-fold protocol.  That cap is a property of the
comparison, not of the method, and this experiment removes it.  Two
studies are reported.

**Full-size evaluation.**  Eighteen datasets are used at their original
size --- up to 20 000 samples and 85 features --- under 5-fold
cross-validation, comparing ABD-NB with naive Bayes and with the
strongest naive-Bayes-family competitor (AODE) and a reference
discriminative learner.

**Cost curves.**  Training time is measured against ``n`` at fixed ``d``
and against ``d`` at fixed ``n``, which is the empirical counterpart of
the complexity statement for Algorithm 1: the dependence stage is
O(d^2 n + d n log n), the spectral step O(d^3), and the selection phase
multiplies only the scoring, not the estimation.  Fitting a power law to
the measured times recovers the exponents directly.
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import ABDNB, AODE
from abdnb.dependence import class_conditional_dependence
from experiments.common import expected_calibration_error, save_csv
from experiments.datasets import equicorrelated_gaussian
from experiments.real_datasets import get_large_datasets

warnings.filterwarnings("ignore")

N_GRID = (500, 1000, 2000, 4000, 8000, 16000, 32000)
D_GRID = (5, 10, 20, 40, 80, 160, 320)


def _models(seed):
    from lightgbm import LGBMClassifier
    return {"GNB": GaussianNB(),
            "ABD-NB": ABDNB(),
            "AODE": AODE(),
            "LGBM": LGBMClassifier(n_estimators=200, max_depth=4, n_jobs=1,
                                   verbose=-1, random_state=seed)}


def _fold(dname, X, y, tr, te, fold):
    rows = []
    classes = np.unique(y[tr])
    yidx = np.searchsorted(classes, y[te])
    for mname, model in _models(fold).items():
        try:
            t0 = time.perf_counter(); model.fit(X[tr], y[tr])
            fit_t = time.perf_counter() - t0
            t0 = time.perf_counter(); yp = model.predict(X[te])
            pred_t = time.perf_counter() - t0
            proba = model.predict_proba(X[te])
            rows.append({"dataset": dname, "model": mname, "fold": fold,
                         "n": X.shape[0], "d": X.shape[1],
                         "accuracy": accuracy_score(y[te], yp),
                         "log_loss": log_loss(yidx, np.clip(proba, 1e-12, 1),
                                              labels=range(len(classes))),
                         "ece": expected_calibration_error(yidx, proba),
                         "fit_time": fit_t, "predict_time": pred_t})
        except Exception as exc:
            print(f"  [warn] {dname}/{mname}: {type(exc).__name__}: {exc}")
    return rows


def full_size(n_jobs: int = 4):
    rows = []
    for dname, (X, y) in get_large_datasets().items():
        X, y = np.asarray(X, float), np.asarray(y)
        cv = StratifiedKFold(5, shuffle=True, random_state=7)
        out = Parallel(n_jobs=n_jobs)(
            delayed(_fold)(dname, X, y, tr, te, f)
            for f, (tr, te) in enumerate(cv.split(X, y)))
        got = [r for c in out for r in c]
        rows.extend(got)
        sub = pd.DataFrame(got)
        acc = sub.groupby("model")["accuracy"].mean()
        ll = sub.groupby("model")["log_loss"].mean()
        t = sub.groupby("model")["fit_time"].mean()
        print(f"== {dname:18s} n={X.shape[0]:6d} d={X.shape[1]:3d} | "
              f"GNB={acc['GNB']:.3f} ABD={acc['ABD-NB']:.3f} "
              f"(LL {ll['GNB']:.3f}->{ll['ABD-NB']:.3f}, "
              f"fit {t['ABD-NB']:.2f}s)", flush=True)
        save_csv(pd.DataFrame(rows), "scalability_full")
    save_csv(pd.DataFrame(rows), "scalability_full")


#: models timed on both axes of the cost grid
def _cost_models():
    return (("ABD-NB", lambda: ABDNB()),
            ("ABD-NB (fixed)", lambda: ABDNB(gamma=1.0, class_specific=False)),
            ("GNB", lambda: GaussianNB()))


def _time_one(mk, X, y):
    """Fit and predict once, recording wall-clock times and peak memory.

    ``tracemalloc`` counts only Python-level allocations, so it under-reports
    what BLAS allocates inside a matrix product.  It is still the right
    quantity here, because the question is whether the model's own state
    grows with ``n`` -- it does not, by construction -- or with ``d``, where
    the K dependence matrices are the term that matters.
    """
    # Timing first, with no instrumentation attached: tracemalloc hooks every
    # allocation and inflates a short fit several-fold, which would corrupt
    # the very exponents this study exists to measure.
    t0 = time.perf_counter()
    model = mk().fit(X, y)
    fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.predict(X)
    pred_t = time.perf_counter() - t0

    # Memory in a separate, untimed pass.
    tracemalloc.start()
    mk().fit(X, y)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return fit_t, pred_t, peak / 1e6


def stage_breakdown(X, y, reps: int = 3):
    """Time the three stages the complexity analysis names, separately.

    Equation (17) attributes the training cost to a dependence-estimation
    term in O(K d^2 n), an eigendecomposition in O(d^3) and the per-
    configuration reweighting and scoring.  Timing them apart is what turns
    the fitted exponent of the total into evidence about the analysis
    rather than about the implementation.
    """
    out = {}
    _, pooled = class_conditional_dependence(X, y, "spearman")   # warm-up
    R = pooled + np.eye(X.shape[1])
    np.linalg.eigvalsh(R)                                        # warm-up
    GaussianNB().fit(X, y)                                       # warm-up

    for name, call in (("dependence",
                        lambda: class_conditional_dependence(X, y, "spearman")),
                       ("eigendecomposition", lambda: np.linalg.eigvalsh(R)),
                       ("gaussian_statistics", lambda: GaussianNB().fit(X, y))):
        t0 = time.perf_counter()
        for _ in range(reps):
            call()
        out[name] = (time.perf_counter() - t0) / reps
    return out


def cost_curves(reps: int = 3):
    rows, stages = [], []
    for n in N_GRID:
        X, y = equicorrelated_gaussian(0.6, n=n, d=20, seed=1)
        for rep in range(reps):
            for label, mk in _cost_models():
                fit_t, pred_t, mem = _time_one(mk, X, y)
                rows.append({"axis": "n", "n": n, "d": 20, "model": label,
                             "rep": rep, "fit_time": fit_t,
                             "predict_time": pred_t, "peak_mb": mem})
        st = stage_breakdown(X, y)
        stages.append({"axis": "n", "n": n, "d": 20, **st})
        print(f"  cost curve n={n} done", flush=True)
    for d in D_GRID:
        X, y = equicorrelated_gaussian(0.6, n=2000, d=d, seed=1)
        for rep in range(reps):
            for label, mk in _cost_models():
                fit_t, pred_t, mem = _time_one(mk, X, y)
                rows.append({"axis": "d", "n": 2000, "d": d, "model": label,
                             "rep": rep, "fit_time": fit_t,
                             "predict_time": pred_t, "peak_mb": mem})
        st = stage_breakdown(X, y)
        stages.append({"axis": "d", "n": 2000, "d": d, **st})
        print(f"  cost curve d={d} done", flush=True)
    df = pd.DataFrame(rows)
    save_csv(df, "scalability_cost")
    sdf = pd.DataFrame(stages)
    for axis, var in (("n", "n"), ("d", "d")):
        sub = sdf[sdf["axis"] == axis]
        for stage in ("dependence", "eigendecomposition", "gaussian_statistics"):
            x = np.log(sub[var].to_numpy(float))
            t = np.log(np.maximum(sub[stage].to_numpy(float), 1e-9))
            half = len(x) // 2
            sdf.loc[sdf["axis"] == axis, stage + "_exponent"] = \
                np.polyfit(x[half:], t[half:], 1)[0]
    save_csv(sdf, "scalability_stages")

    fits = []
    for axis, var in (("n", "n"), ("d", "d")):
        for model in df["model"].unique():
            sub = df[(df["axis"] == axis) & (df["model"] == model)]
            gp = sub.groupby(var)["predict_time"].mean()
            xp, tp = np.log(gp.index.to_numpy()), np.log(gp.to_numpy())
            halfp = len(gp) // 2
            pred_exp = np.polyfit(xp[halfp:], tp[halfp:], 1)[0]
            gm = sub.groupby(var)["peak_mb"].mean()
            g = sub.groupby(var)["fit_time"].mean()
            x, t = np.log(g.index.to_numpy()), np.log(g.to_numpy())
            slope, intercept = np.polyfit(x, t, 1)
            # the small end of each grid is dominated by fixed interpreter
            # overhead, so the asymptotic exponent is read off the upper half
            half = len(g) // 2
            slope_up = np.polyfit(x[half:], t[half:], 1)[0]
            fits.append({"axis": axis, "model": model, "exponent": slope,
                         "exponent_asym": slope_up,
                         "const": float(np.exp(intercept)),
                         "t_min_ms": 1000 * float(g.iloc[0]),
                         "t_max_ms": 1000 * float(g.iloc[-1]),
                         "predict_exponent_asym": pred_exp,
                         "predict_min_ms": 1000 * float(gp.iloc[0]),
                         "predict_max_ms": 1000 * float(gp.iloc[-1]),
                         "peak_mb_min": float(gm.iloc[0]),
                         "peak_mb_max": float(gm.iloc[-1])})
    f = pd.DataFrame(fits)
    save_csv(f, "scalability_exponents")
    print(); print(f.round(3).to_string(index=False))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "full"):
        full_size()
    if what in ("all", "cost"):
        cost_curves()
