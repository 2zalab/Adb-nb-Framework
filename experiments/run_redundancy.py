"""Duplication stress test.

For each real dataset, ``r`` noisy copies of one randomly chosen
feature are appended (relative noise 5%), and the 5-fold CV accuracy of
each classifier is recorded as a function of ``r``.  Naive Bayes counts
the duplicated evidence ``r + 1`` times; ABD-NB detects the duplication
and shares the weight across the copies.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from abdnb import ABDNB
from experiments.common import save_csv

R_VALUES = [0, 1, 2, 4, 8, 16, 32]
N_TRIALS = 10

#: eight real datasets from eight different application domains, so the
#: duplication stress test is no longer read off three datasets only
STRESS_SUITE = ["breast-cancer", "credit-ger", "vehicle", "spambase",
                "yeast", "segmentation", "phoneme", "contraceptive"]


def main():
    from experiments.real_datasets import get_real_datasets
    real = get_real_datasets()
    datasets = {
        "iris": load_iris(return_X_y=True),
        "wine": load_wine(return_X_y=True),
    }
    datasets.update({name: real[name] for name in STRESS_SUITE
                     if name in real})
    models = {
        "GNB": lambda: GaussianNB(),
        "ABD-NB": lambda: ABDNB(),
        "LR": lambda: make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000)),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for dname, (X, y) in datasets.items():
        X = np.asarray(X, dtype=float)
        print(f"== {dname}", flush=True)
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(100 + trial)
            j = int(rng.integers(0, X.shape[1]))
            sd = X[:, j].std()
            for r in R_VALUES:
                copies = [
                    (X[:, j] + rng.normal(scale=0.05 * sd, size=len(X)))[:, None]
                    for _ in range(r)
                ]
                Xa = np.hstack([X] + copies) if r else X
                for mname, mk in models.items():
                    acc = cross_val_score(mk(), Xa, y, cv=cv).mean()
                    rows.append({"dataset": dname, "trial": trial, "r": r,
                                 "model": mname, "accuracy": acc})
        df = pd.DataFrame([x for x in rows if x["dataset"] == dname])
        print(df.groupby(["model", "r"])["accuracy"].mean().round(3).unstack().to_string())
    save_csv(pd.DataFrame(rows), "redundancy")


if __name__ == "__main__":
    main()
