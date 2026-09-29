"""Add models to an existing benchmark without recomputing it.

The cross-validation splitter of ``run_benchmark`` is seeded, so the
folds are reproducible exactly.  This script evaluates a subset of models
on those same folds and merges the rows into ``results/benchmark.csv``,
which keeps every comparison paired without paying for a full re-run.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import CFWNB, HNB
from experiments.common import save_csv
from experiments.datasets import dataset_stratum, get_all_datasets
from experiments.run_benchmark import evaluate

warnings.filterwarnings("ignore")

EXTRA = {"CFW-NB": lambda seed: CFWNB(random_state=seed),
         "HNB": lambda seed: HNB()}


def _fold(dname, X, y, tr, te, fold):
    rows = []
    for mname, mk in EXTRA.items():
        try:
            res = evaluate(mk(fold), X[tr], y[tr], X[te], y[te])
        except Exception as exc:
            print(f"  [warn] {dname}/{mname}/fold{fold}: {type(exc).__name__}")
            continue
        rows.append({"dataset": dname, "stratum": dataset_stratum(dname),
                     "model": mname, "fold": fold, **res})
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
        got = [r for c in out for r in c]
        rows.extend(got)
        acc = pd.DataFrame(got).groupby("model")["accuracy"].mean()
        print(f"== {dname:18s} " + "  ".join(f"{m}={v:.3f}" for m, v in acc.items()),
              flush=True)
    extra = pd.DataFrame(rows)
    save_csv(extra, "benchmark_extra")

    base = pd.read_csv(os.path.join(os.path.dirname(__file__), "..",
                                    "results", "benchmark.csv"))
    base = base[~base["model"].isin(EXTRA)]
    merged = pd.concat([base, extra], ignore_index=True)
    save_csv(merged, "benchmark")
    print(f"\nmerged benchmark: {merged.shape[0]} rows, "
          f"{merged['model'].nunique()} models, {merged['dataset'].nunique()} datasets")


if __name__ == "__main__":
    main()
