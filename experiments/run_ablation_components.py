"""Component ablation: what does each stage contribute on its own?

The framework has three separable ingredients beyond the naive Bayes
factorisation --- the dependence weighting of Stage 2, the spectral
rescaling of Stage 3, and the validation-based selection of Stage 4 ---
and the main benchmark reports only their combination.  This experiment
switches them on one at a time over the whole suite, on the folds of the
main benchmark, so that each row differs from its neighbour in exactly
one component:

    (a) gamma = 0                      : plain Gaussian naive Bayes
    (b) harmonic, no rescaling         : Stage 2 only
    (c) harmonic, rescale to d         : Stage 2 + a naive scale restore
    (d) harmonic, rescale to M_eff     : Stages 2 + 3          (ABD-NB-G)
    (e) (b) with validated gamma       : Stages 2 + 4
    (f) (c) with validated gamma       : Stages 2 + 3' + 4
    (g) (d) with validated gamma       : Stages 2 + 3 + 4      (flagship)

Comparing (d) against (b) isolates the spectral rescaling; (g) against
(d) isolates the selection layer; (b) against (a) isolates the weighting
rule itself.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import ABDNB
from experiments.common import expected_calibration_error, save_csv
from experiments.datasets import dataset_stratum, get_all_datasets

warnings.filterwarnings("ignore")

#: label -> (weighting?, rescale, selection?)
CONFIGS = {
    "(a) NB":                 (False, "none", False),
    "(b) W":                  (True, "none", False),
    "(c) W+scale$d$":         (True, "dim", False),
    "(d) W+scale$M$":         (True, "meff", False),
    "(e) W+sel":              (True, "none", True),
    "(f) W+scale$d$+sel":     (True, "dim", True),
    "(g) W+scale$M$+sel":     (True, "meff", True),
}


def _make(label):
    weighting, rescale, select = CONFIGS[label]
    if not weighting:
        return GaussianNB()
    return ABDNB(weight_fn="harmonic", rescale=rescale,
                 gamma="auto" if select else 1.0,
                 class_specific="auto" if select else False)


def _fold(dname, X, y, tr, te, fold):
    rows = []
    classes = np.unique(y[tr])
    yidx = np.searchsorted(classes, y[te])
    for label in CONFIGS:
        try:
            m = _make(label).fit(X[tr], y[tr])
            proba = m.predict_proba(X[te])
            yp = m.predict(X[te])
            rows.append({
                "dataset": dname, "stratum": dataset_stratum(dname),
                "config": label, "fold": fold,
                "accuracy": accuracy_score(y[te], yp),
                "macro_f1": f1_score(y[te], yp, average="macro"),
                "log_loss": log_loss(yidx, np.clip(proba, 1e-12, 1),
                                     labels=range(len(classes))),
                "ece": expected_calibration_error(yidx, proba),
                "d_eff": float(np.mean(getattr(m, "effective_dimension_",
                                               [X.shape[1]]))),
                "d": X.shape[1],
            })
        except Exception as exc:
            print(f"  [warn] {dname}/{label}: {type(exc).__name__}")
    return rows


def main(n_splits: int = 10, n_repeats: int = 3, seed: int = 7, n_jobs: int = 4):
    rows = []
    for dname, (X, y) in get_all_datasets().items():
        X, y = np.asarray(X, float), np.asarray(y)
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        out = Parallel(n_jobs=n_jobs)(
            delayed(_fold)(dname, X, y, tr, te, f)
            for f, (tr, te) in enumerate(cv.split(X, y)))
        rows.extend([r for c in out for r in c])
        sub = pd.DataFrame([r for r in rows if r["dataset"] == dname])
        acc = sub.groupby("config")["accuracy"].mean()
        print(f"== {dname:18s} " +
              "  ".join(f"{k.split(')')[0]})={v:.3f}" for k, v in acc.items()),
              flush=True)
        save_csv(pd.DataFrame(rows), "ablation_components")
    df = pd.DataFrame(rows)
    save_csv(df, "ablation_components")

    # aggregate: mean ranks and paired deltas against plain NB
    acc = df.pivot_table("accuracy", "dataset", "config")
    ll = df.pivot_table("log_loss", "dataset", "config")
    real = [d for d in acc.index if dataset_stratum(d) == "real"]
    summary = []
    for label in CONFIGS:
        da = (acc[label] - acc["(a) NB"]) * 100
        dl = 100 * (ll["(a) NB"] - ll[label]) / ll["(a) NB"]
        summary.append({
            "config": label,
            "acc_rank": acc.rank(axis=1, ascending=False).mean()[label],
            "ll_rank": ll.rank(axis=1).mean()[label],
            "d_acc_mean": da.mean(), "d_acc_worst": da.min(),
            "d_ll_median": dl.median(),
            "d_acc_real": da[real].mean(), "d_acc_worst_real": da[real].min(),
            "wins": int((da > 0.05).sum()), "losses": int((da < -0.05).sum()),
        })
    s = pd.DataFrame(summary)
    save_csv(s, "ablation_components_summary")
    print(); print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
