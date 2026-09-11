"""Main benchmark: 20 classifiers x 48 datasets x (3 x 10-fold) CV.

The comparison set spans four families -- plain and weighted naive Bayes,
semi-naive Bayesian network classifiers (TAN, KDB, AODE), post-hoc
calibrated naive Bayes, and contemporary discriminative learners
(gradient boosting, random forests, SVM, logistic regression) -- so that
the dependence correction of ABD-NB is measured against the strongest
current alternatives rather than against classical baselines only.

Produces ``results/benchmark.csv`` with one row per (dataset, model,
fold) containing accuracy, macro-F1, log-loss, ECE and timing.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import ABDNB, MIWeightedNB
from abdnb.baselines import AODE, KDB, TAN, WANBIA, CorrelationWeightedNB
from experiments.common import expected_calibration_error, save_csv
from experiments.datasets import dataset_stratum, get_all_datasets

warnings.filterwarnings("ignore")

#: models grouped by family (the grouping drives the result tables)
NB_FAMILY = ["GNB", "MI-WNB", "CW-NB", "WANBIA", "ABD-NB-L", "ABD-NB-G",
             "ABD-NB-C", "ABD-NB"]
SEMI_NAIVE = ["TAN", "KDB", "AODE"]
CALIBRATED = ["GNB-Platt", "GNB-Iso"]
DISCRIMINATIVE = ["LR", "kNN", "CART", "RF", "XGB", "LGBM", "SVM"]
ALL_MODELS = NB_FAMILY + SEMI_NAIVE + CALIBRATED + DISCRIMINATIVE


def make_models(seed: int = 0) -> dict:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        # -- naive Bayes family -------------------------------------
        "GNB": GaussianNB(),
        "MI-WNB": MIWeightedNB(random_state=seed),
        "CW-NB": CorrelationWeightedNB(random_state=seed),
        "WANBIA": WANBIA(),
        "ABD-NB-L": ABDNB(weight_fn="linear", gamma=1.0, class_specific=False),
        "ABD-NB-G": ABDNB(weight_fn="harmonic", gamma=1.0, class_specific=False),
        "ABD-NB-C": ABDNB(weight_fn="harmonic", gamma=1.0, class_specific=True),
        "ABD-NB": ABDNB(),  # flagship: gamma and structure tuned internally
        # -- semi-naive Bayesian network classifiers ----------------
        "TAN": TAN(),
        "KDB": KDB(k=2),
        "AODE": AODE(),
        # -- post-hoc calibrated naive Bayes ------------------------
        "GNB-Platt": CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=3),
        "GNB-Iso": CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=3),
        # -- contemporary discriminative learners -------------------
        "LR": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=seed)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "CART": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "XGB": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             tree_method="hist", n_jobs=1, verbosity=0,
                             random_state=seed),
        "LGBM": LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                               n_jobs=1, verbose=-1, random_state=seed),
        "SVM": make_pipeline(StandardScaler(),
                             SVC(probability=True, random_state=seed)),
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
        "log_loss": log_loss(yte_idx, np.clip(proba, 1e-12, 1),
                             labels=range(len(classes))),
        "ece": expected_calibration_error(yte_idx, proba),
        "fit_time": fit_t,
        "predict_time": pred_t,
    }


def _one_fold(dname, X, y, tr, te, fold):
    rows = []
    models = make_models(seed=fold)
    for mname, model in models.items():
        try:
            res = evaluate(model, X[tr], y[tr], X[te], y[te])
        except Exception as exc:  # a model may fail on a degenerate fold
            print(f"  [warn] {dname}/{mname}/fold{fold}: {type(exc).__name__}")
            continue
        rows.append({"dataset": dname, "stratum": dataset_stratum(dname),
                     "model": mname, "fold": fold, **res})
    return rows


def main(n_splits: int = 10, n_repeats: int = 3, seed: int = 7, n_jobs: int = 4):
    rows = []
    datasets = get_all_datasets()
    for dname, (X, y) in datasets.items():
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        t0 = time.perf_counter()
        folds = list(cv.split(X, y))
        out = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one_fold)(dname, X, y, tr, te, f)
            for f, (tr, te) in enumerate(folds))
        got = [r for chunk in out for r in chunk]
        rows.extend(got)
        acc = pd.DataFrame(got).groupby("model")["accuracy"].mean()
        best = acc.idxmax()
        print(f"== {dname:18s} n={X.shape[0]:5d} d={X.shape[1]:3d} "
              f"K={len(np.unique(y)):2d} [{time.perf_counter()-t0:6.1f}s]  "
              f"GNB={acc.get('GNB', float('nan')):.3f} "
              f"ABD-NB={acc.get('ABD-NB', float('nan')):.3f} "
              f"best={best}({acc[best]:.3f})", flush=True)
        save_csv(pd.DataFrame(rows), "benchmark")
    save_csv(pd.DataFrame(rows), "benchmark")


if __name__ == "__main__":
    main()
