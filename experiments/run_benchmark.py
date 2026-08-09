"""Main benchmark: 10 classifiers x 13 datasets x (3 x 10-fold) CV.

Produces ``results/benchmark.csv`` with one row per (dataset, model,
repeat, fold) containing accuracy, macro-F1, log-loss, ECE and timing.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import ABDNB, MIWeightedNB
from experiments.common import expected_calibration_error, save_csv
from experiments.datasets import get_datasets


def make_models(seed: int = 0) -> dict:
    return {
        "GNB": GaussianNB(),
        "MI-WNB": MIWeightedNB(random_state=seed),
        "ABD-NB-L": ABDNB(weight_fn="linear", gamma=1.0, class_specific=False),
        "ABD-NB-G": ABDNB(weight_fn="harmonic", gamma=1.0, class_specific=False),
        "ABD-NB-C": ABDNB(weight_fn="harmonic", gamma=1.0, class_specific=True),
        "ABD-NB": ABDNB(),  # flagship: gamma and structure tuned internally
        "LR": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=seed)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "CART": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed)),
    }


def evaluate(model, Xtr, ytr, Xte, yte) -> dict:
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    yp = model.predict(Xte)
    pred_t = time.perf_counter() - t0
    proba = model.predict_proba(Xte)
    classes = np.unique(ytr)
    yte_idx = np.searchsorted(classes, yte)
    return {
        "accuracy": accuracy_score(yte, yp),
        "macro_f1": f1_score(yte, yp, average="macro"),
        "log_loss": log_loss(yte_idx, np.clip(proba, 1e-12, 1), labels=range(len(classes))),
        "ece": expected_calibration_error(yte_idx, proba),
        "fit_time": fit_t,
        "predict_time": pred_t,
    }


def main(n_splits: int = 10, n_repeats: int = 3, seed: int = 7):
    rows = []
    datasets = get_datasets()
    for dname, (X, y) in datasets.items():
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        print(f"== {dname}  n={X.shape[0]} d={X.shape[1]}")
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            models = make_models(seed=fold)
            for mname, model in models.items():
                res = evaluate(model, X[tr], y[tr], X[te], y[te])
                rows.append({"dataset": dname, "model": mname, "fold": fold, **res})
        df = pd.DataFrame([r for r in rows if r["dataset"] == dname])
        print(df.groupby("model")["accuracy"].mean().round(4).to_string())
    save_csv(pd.DataFrame(rows), "benchmark")


if __name__ == "__main__":
    main()
