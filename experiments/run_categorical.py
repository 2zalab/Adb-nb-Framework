"""Categorical, text-derived and sequence benchmarks.

The earlier version of this study evaluated Gaussian factors only.  Here
the same dependence-adaptive weighting is applied to *multinomial*
factors (:class:`abdnb.CategoricalABDNB`) and compared with categorical
naive Bayes and with the semi-naive classifiers on the discrete datasets
of the suite: categorical medical and social data, a DNA-sequence
dataset, and a text-derived spam corpus represented by word and character
frequencies.  This removes the "continuous factors only" limitation of
the previous evaluation.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import AODE, TAN, CategoricalABDNB, CategoricalNB_
from experiments.common import expected_calibration_error, save_csv
from experiments.real_datasets import DOMAINS, get_real_datasets

#: discrete / text-derived subset of the real suite
DISCRETE_SUITE = ["dermatology", "mushroom", "soybean", "splice-dna",
                  "titanic", "nursery", "spambase", "credit-ger",
                  "contraceptive"]


def make_models():
    return {
        "Cat-NB": CategoricalNB_(),
        "Cat-ABD-G": CategoricalABDNB(gamma=1.0, class_specific=False),
        "Cat-ABD": CategoricalABDNB(),
        "TAN": TAN(),
        "AODE": AODE(),
    }


def main(n_splits: int = 10, n_repeats: int = 3, seed: int = 7):
    data = get_real_datasets()
    names = [n for n in DISCRETE_SUITE if n in data]
    rows = []
    for dname in names:
        X, y = data[dname]
        X = np.asarray(X, float)
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        t0 = time.perf_counter()
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            for mname, model in make_models().items():
                try:
                    model.fit(X[tr], y[tr])
                    proba = model.predict_proba(X[te])
                    classes = np.unique(y[tr])
                    yidx = np.searchsorted(classes, y[te])
                    rows.append({
                        "dataset": dname, "domain": DOMAINS.get(dname, "-"),
                        "model": mname, "fold": fold,
                        "accuracy": accuracy_score(y[te], model.predict(X[te])),
                        "macro_f1": f1_score(y[te], model.predict(X[te]),
                                             average="macro"),
                        "log_loss": log_loss(yidx, np.clip(proba, 1e-12, 1),
                                             labels=range(len(classes))),
                        "ece": expected_calibration_error(yidx, proba),
                        "gamma": getattr(model, "gamma_", np.nan),
                    })
                except Exception as exc:
                    print(f"  [warn] {dname}/{mname}: {type(exc).__name__}")
        df = pd.DataFrame([r for r in rows if r["dataset"] == dname])
        acc = df.groupby("model")["accuracy"].mean()
        ll = df.groupby("model")["log_loss"].mean()
        print(f"== {dname:14s} [{time.perf_counter()-t0:6.1f}s] "
              f"NB={acc['Cat-NB']:.3f}/{ll['Cat-NB']:.3f}  "
              f"ABD={acc['Cat-ABD']:.3f}/{ll['Cat-ABD']:.3f}", flush=True)
        save_csv(pd.DataFrame(rows), "categorical")
    save_csv(pd.DataFrame(rows), "categorical")


if __name__ == "__main__":
    main()
