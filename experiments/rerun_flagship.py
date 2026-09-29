"""Re-evaluate only the flagship after a change to its selection grid.

The folds of ``run_benchmark`` are seeded, so the flagship can be
re-evaluated on exactly the same splits and merged back, keeping every
paired comparison valid without re-running the other 21 models.
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
from abdnb import ABDNB
from experiments.common import save_csv
from experiments.datasets import dataset_stratum, get_all_datasets
from experiments.run_benchmark import evaluate

warnings.filterwarnings("ignore")


def _fold(dname, X, y, tr, te, fold):
    res = evaluate(ABDNB(), X[tr], y[tr], X[te], y[te])
    return {"dataset": dname, "stratum": dataset_stratum(dname),
            "model": "ABD-NB", "fold": fold, **res}


def main(n_splits: int = 10, n_repeats: int = 3, seed: int = 7, n_jobs: int = 4):
    rows = []
    for dname, (X, y) in get_all_datasets().items():
        X, y = np.asarray(X, float), np.asarray(y)
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        out = Parallel(n_jobs=n_jobs)(
            delayed(_fold)(dname, X, y, tr, te, f)
            for f, (tr, te) in enumerate(cv.split(X, y)))
        rows.extend(out)
        acc = np.mean([r["accuracy"] for r in out])
        print(f"== {dname:18s} ABD-NB={acc:.4f}", flush=True)
    new = pd.DataFrame(rows)
    base = pd.read_csv(os.path.join(os.path.dirname(__file__), "..",
                                    "results", "benchmark.csv"))
    base = base[base["model"] != "ABD-NB"]
    save_csv(pd.concat([base, new], ignore_index=True), "benchmark")
    print("merged")


if __name__ == "__main__":
    main()
