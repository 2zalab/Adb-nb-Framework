"""How robust is the redundancy index as a practical diagnostic?

Section "redundancy profile" proposes $R = 1 - M_eff(\\hat D)/d$ as a
data-side predictor of the benefit of the correction.  A diagnostic is
only useful if it is stable, so this experiment probes three things a
practitioner would want to know before relying on it:

1. **Dependence measure.**  $R$ is recomputed with each of the seven
   implemented measures and each version is confronted with the realised
   improvement, giving one correlation per measure.
2. **Sampling stability.**  $R$ is re-estimated on 30 bootstrap resamples
   and on independent random halves of every dataset, yielding a
   coefficient of variation and a split-half agreement.
3. **Sample size.**  $R$ is re-estimated on nested subsamples so that the
   smallest $n$ at which the diagnostic is trustworthy can be read off.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb.dependence import class_conditional_dependence
from abdnb.weights import effective_dimension
from experiments.common import load_csv, save_csv
from experiments.real_datasets import get_real_datasets

warnings.filterwarnings("ignore")

MEASURES = ("spearman", "pearson", "kendall", "mi", "cramersv", "dcor", "hsic")
SIZES = (100, 200, 400, 800, 1600)


def redundancy(X, y, measure="spearman", seed=0):
    _, pooled = class_conditional_dependence(np.asarray(X, float), y, measure,
                                             threshold=True, random_state=seed)
    d = pooled.shape[0]
    return 1.0 - effective_dimension(pooled) / d


def _measure_row(name, X, y, gain_acc, gain_ll):
    row = {"dataset": name, "d_acc": gain_acc, "d_ll_rel": gain_ll}
    for m in MEASURES:
        try:
            row[m] = redundancy(X, y, m)
        except Exception:
            row[m] = np.nan
    return row


def _stability_row(name, X, y, n_boot=30, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    base = redundancy(X, y)
    boots = []
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(redundancy(X[idx], y[idx], seed=b))
    boots = np.array(boots)
    # split-half agreement
    halves = []
    for b in range(10):
        perm = rng.permutation(n)
        a, c = perm[: n // 2], perm[n // 2:]
        if len(np.unique(y[a])) < 2 or len(np.unique(y[c])) < 2:
            continue
        halves.append((redundancy(X[a], y[a], seed=b),
                       redundancy(X[c], y[c], seed=b)))
    halves = np.array(halves) if halves else np.zeros((1, 2))
    sub = {}
    for m in SIZES:
        if n < m + 20:
            continue
        vals = []
        for b in range(10):
            idx = rng.choice(n, m, replace=False)
            if len(np.unique(y[idx])) < 2:
                continue
            vals.append(redundancy(X[idx], y[idx], seed=b))
        if vals:
            sub[f"R_n{m}"] = float(np.mean(vals))
            sub[f"bias_n{m}"] = float(np.mean(vals) - base)
    return {"dataset": name, "R": base,
            "boot_sd": float(boots.std()) if len(boots) else np.nan,
            "boot_cv": float(boots.std() / max(base, 1e-6)) if len(boots) else np.nan,
            "boot_lo": float(np.percentile(boots, 2.5)) if len(boots) else np.nan,
            "boot_hi": float(np.percentile(boots, 97.5)) if len(boots) else np.nan,
            "splithalf_absdiff": float(np.abs(halves[:, 0] - halves[:, 1]).mean()),
            "n": n, **sub}


def main(n_jobs: int = 4):
    prof = load_csv("redundancy_profile").set_index("dataset")
    data = {k: v for k, v in get_real_datasets().items() if k in prof.index}

    rows = Parallel(n_jobs=n_jobs)(
        delayed(_measure_row)(name, X, y, prof.loc[name, "d_acc"],
                              prof.loc[name, "d_ll_rel"])
        for name, (X, y) in data.items())
    df = pd.DataFrame(rows)
    save_csv(df, "redundancy_measures")

    stat = []
    for m in MEASURES:
        ok = df[m].notna()
        ra, pa = stats.spearmanr(df.loc[ok, m], df.loc[ok, "d_acc"])
        rl, pl = stats.spearmanr(df.loc[ok, m], df.loc[ok, "d_ll_rel"])
        # bootstrap CI on the log-loss correlation
        rng = np.random.default_rng(0)
        vals = []
        x, yv = df.loc[ok, m].to_numpy(), df.loc[ok, "d_ll_rel"].to_numpy()
        for _ in range(2000):
            idx = rng.choice(len(x), len(x), replace=True)
            if len(np.unique(x[idx])) < 3:
                continue
            vals.append(stats.spearmanr(x[idx], yv[idx]).statistic)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        stat.append({"measure": m, "n_datasets": int(ok.sum()),
                     "rho_acc": ra, "p_acc": pa,
                     "rho_ll": rl, "p_ll": pl, "ci_lo": lo, "ci_hi": hi,
                     "mean_R": df.loc[ok, m].mean()})
    st = pd.DataFrame(stat)
    save_csv(st, "redundancy_measures_stats")
    print(st.round(3).to_string(index=False))

    rows = Parallel(n_jobs=n_jobs)(
        delayed(_stability_row)(name, np.asarray(X, float), np.asarray(y))
        for name, (X, y) in data.items())
    stab = pd.DataFrame(rows)
    save_csv(stab, "redundancy_stability")
    cols = [c for c in stab.columns if c.startswith("bias_n")]
    print()
    print("median bootstrap CV of R: %.3f" % stab["boot_cv"].median())
    print("median split-half |difference|: %.4f" % stab["splithalf_absdiff"].median())
    print(stab[cols].median().round(4).to_string())


if __name__ == "__main__":
    main()
