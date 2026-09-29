"""Higher-order dependence: where a pairwise correction is structurally blind.

Every dependence measure in Stage 1 is bivariate, so the framework sees
the dependence graph and not the dependence *hypergraph*.  The practical
question is how much that costs, and this experiment answers it by
constructing designs in which the interaction is deliberately invisible
to any pairwise statistic.

Three families are generated, all with the same marginal and pairwise
structure by construction:

- ``parity-k``: ``k`` binary features are pairwise independent but their
  parity carries all the label information (the canonical XOR case);
- ``masked-duplicate``: a duplicated block whose copies are pairwise
  dependent, plus a parity trio that is not, so pairwise and higher-order
  structure coexist and their contributions can be separated;
- ``3-way Gaussian``: a trivariate design with zero pairwise correlation
  and a non-zero third cumulant.

For each we report what the estimated dependence matrix actually detects,
what ABD-NB then does, and how the result compares with a model that can
represent the interaction (AODE, a depth-limited tree ensemble).
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import ABDNB, AODE
from abdnb.dependence import class_conditional_dependence
from experiments.common import save_csv

warnings.filterwarnings("ignore")


def parity(n=2000, k=3, n_noise=6, flip=0.05, seed=0):
    """``k`` pairwise-independent bits whose parity is the label."""
    rng = np.random.default_rng(seed)
    B = rng.integers(0, 2, size=(n, k))
    y = B.sum(axis=1) % 2
    flip_mask = rng.random(n) < flip
    y = np.where(flip_mask, 1 - y, y)
    noise = rng.integers(0, 2, size=(n, n_noise))
    X = np.hstack([B, noise]).astype(float) + 0.05 * rng.normal(size=(n, k + n_noise))
    return X, y


def masked_duplicate(n=2000, r=6, k=3, flip=0.05, seed=0):
    """A redundant block (pairwise visible) next to a parity trio (invisible)."""
    rng = np.random.default_rng(seed)
    B = rng.integers(0, 2, size=(n, k))
    par = B.sum(axis=1) % 2
    y = np.where(rng.random(n) < flip, 1 - par, par)
    s = 2 * y - 1
    z = s * 0.45 + rng.normal(size=n)
    block = [np.sqrt(0.95) * z + np.sqrt(0.05) * rng.normal(size=n) for _ in range(r)]
    X = np.column_stack([B.astype(float) + 0.05 * rng.normal(size=(n, k))] + block)
    return X, y


def three_way_gaussian(n=2000, n_noise=6, seed=0):
    """Zero pairwise correlation, non-zero three-way interaction."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    c = a * b                       # uncorrelated with a and b, dependent on both
    y = (c > 0).astype(int)
    noise = rng.normal(size=(n, n_noise))
    return np.column_stack([a, b, c, noise]), y


DESIGNS = {"parity-3": lambda s: parity(k=3, seed=s),
           "parity-4": lambda s: parity(k=4, seed=s),
           "masked-duplicate": lambda s: masked_duplicate(seed=s),
           "3-way Gaussian": lambda s: three_way_gaussian(seed=s)}


def _eval(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    classes = np.unique(ytr)
    proba = model.predict_proba(Xte)
    return (accuracy_score(yte, model.predict(Xte)),
            log_loss(np.searchsorted(classes, yte), np.clip(proba, 1e-12, 1),
                     labels=range(len(classes))))


def main(reps: int = 10):
    rows = []
    for name, gen in DESIGNS.items():
        for rep in range(reps):
            Xtr, ytr = gen(rep)
            Xte, yte = gen(1000 + rep)
            _, pooled = class_conditional_dependence(Xtr, ytr, "spearman")
            _, pooled_mi = class_conditional_dependence(Xtr, ytr, "mi")
            abd = ABDNB()
            a_acc, a_ll = _eval(abd, Xtr, ytr, Xte, yte)
            g_acc, g_ll = _eval(GaussianNB(), Xtr, ytr, Xte, yte)
            o_acc, o_ll = _eval(AODE(), Xtr, ytr, Xte, yte)
            r_acc, r_ll = _eval(RandomForestClassifier(n_estimators=200,
                                                       random_state=rep), Xtr, ytr,
                                Xte, yte)
            rows.append({"design": name, "rep": rep,
                         "max_pairwise_spearman": float(pooled.max()),
                         "mean_pairwise_spearman": float(
                             pooled[np.triu_indices_from(pooled, 1)].mean()),
                         "max_pairwise_mi": float(pooled_mi.max()),
                         "gamma": float(abd.gamma_),
                         "d_eff_ratio": float(np.mean(abd.effective_dimension_)
                                              / Xtr.shape[1]),
                         "acc_nb": g_acc, "acc_abd": a_acc,
                         "acc_aode": o_acc, "acc_rf": r_acc,
                         "ll_nb": g_ll, "ll_abd": a_ll,
                         "ll_aode": o_ll, "ll_rf": r_ll})
        print(f"== {name} done", flush=True)
    df = pd.DataFrame(rows)
    save_csv(df, "higher_order")
    summary = df.groupby("design").agg(
        max_rho=("max_pairwise_spearman", "mean"),
        max_mi=("max_pairwise_mi", "mean"),
        gamma=("gamma", "median"),
        deff_ratio=("d_eff_ratio", "mean"),
        nb=("acc_nb", "mean"), abd=("acc_abd", "mean"),
        aode=("acc_aode", "mean"), rf=("acc_rf", "mean"),
        d_acc=("acc_abd", "mean")).reset_index()
    summary["d_acc"] = 100 * (df.groupby("design")["acc_abd"].mean()
                              - df.groupby("design")["acc_nb"].mean()).values
    save_csv(summary, "higher_order_summary")
    print(); print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
