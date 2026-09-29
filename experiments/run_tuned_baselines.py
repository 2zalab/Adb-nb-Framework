"""Fairness of the comparison: every model gets the same tuning budget.

In the main benchmark ABD-NB selects its correction strength on internal
validation while the baselines use library defaults.  That asymmetry
favours ABD-NB, and a comparison that leaves it unaddressed cannot
support any ranking claim.  This experiment removes it: on each outer
training fold, *every* model --- ABD-NB included --- is given an
identical randomised search budget (``N_ITER`` candidates, 3-fold inner
CV, training data only), so the model-selection procedure is the same
across the board and nothing is tuned on test data.

The subset of datasets is fixed in advance and chosen to span the
redundancy range and the application domains, because a nested search
over all 48 settings and 22 models is not affordable; the selection rule
is stated in the paper and the list is frozen here.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import AODE, ABDNB, WANBIA
from experiments.common import expected_calibration_error, save_csv
from experiments.real_datasets import SUBSAMPLE_SEED, stratified_subsample

warnings.filterwarnings("ignore")

N_ITER = 10
INNER_CV = 3
#: this study caps the sample size: a nested search over ten model
#: families is far more expensive than a single fit, and the question it
#: answers -- does the ranking survive a matched protocol -- does not
#: need the full sample
MAX_N = 2000

#: frozen subset: four per redundancy tercile plus the two augmented and
#: two synthetic designs that carry the mechanism claims
SUBSET = ["breast-cancer", "vehicle", "page-blocks", "texture",       # high R
          "hypothyroid", "spectf",                                    # high/med R
          "credit-ger", "titanic", "segmentation",                    # medium R
          "heart-cleveland", "phoneme",                               # low R
          "synth-conflict"]                                           # mechanism


def _spaces(d, seed):
    """(estimator, search space) per model; identical budget for all."""
    return {
        "GNB": (GaussianNB(),
                {"var_smoothing": loguniform(1e-12, 1e-3)}),
        "WANBIA": (WANBIA(), {"l2": loguniform(1e-5, 1e0),
                              "var_smoothing": loguniform(1e-12, 1e-3)}),
        "AODE": (AODE(), {"n_bins": randint(3, 9), "m": randint(1, 6)}),
        # the variance floor is a parameter of the shared Gaussian
        # likelihood, not of the correction, so it must be searchable by
        # every Gaussian model or the matched protocol is not matched
        "ABD-NB": (ABDNB(),
                   {"kappa": uniform(1.0, 2.0),
                    "alpha": uniform(0.01, 0.19),
                    "measure": ["spearman", "pearson", "kendall", "mi"],
                    "var_smoothing": loguniform(1e-12, 1e-3),
                    "tune_cv": randint(3, 6)}),
        "LR": (Pipeline([("s", StandardScaler()),
                         ("m", LogisticRegression(max_iter=3000, random_state=seed))]),
               {"m__C": loguniform(1e-3, 1e3)}),
        "kNN": (Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier())]),
                {"m__n_neighbors": randint(1, 40),
                 "m__weights": ["uniform", "distance"]}),
        "SVM": (Pipeline([("s", StandardScaler()),
                          ("m", SVC(probability=True, random_state=seed))]),
                {"m__C": loguniform(1e-2, 1e3), "m__gamma": loguniform(1e-4, 1e1)}),
        "RF": (RandomForestClassifier(random_state=seed, n_jobs=1),
               {"n_estimators": randint(100, 400), "max_depth": randint(3, 25),
                "max_features": uniform(0.1, 0.9),
                "min_samples_leaf": randint(1, 10)}),
    }


def _boosting(seed):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        "XGB": (XGBClassifier(tree_method="hist", n_jobs=1, verbosity=0,
                              random_state=seed),
                {"n_estimators": randint(100, 400), "max_depth": randint(2, 9),
                 "learning_rate": loguniform(0.01, 0.3),
                 "subsample": uniform(0.6, 0.4)}),
        "LGBM": (LGBMClassifier(n_jobs=1, verbose=-1, random_state=seed),
                 {"n_estimators": randint(100, 400), "max_depth": randint(2, 9),
                  "learning_rate": loguniform(0.01, 0.3),
                  "num_leaves": randint(8, 64)}),
    }


def _fold(dname, X, y, tr, te, fold):
    rows = []
    spaces = {**_spaces(X.shape[1], fold), **_boosting(fold)}
    classes = np.unique(y[tr])
    yidx = np.searchsorted(classes, y[te])
    for mname, (est, space) in spaces.items():
        for tuned in (False, True):
            try:
                t0 = time.perf_counter()
                if tuned:
                    model = RandomizedSearchCV(
                        est, space, n_iter=N_ITER, cv=INNER_CV, n_jobs=1,
                        random_state=fold, scoring="accuracy", refit=True)
                else:
                    model = est
                model.fit(X[tr], y[tr])
                fit_t = time.perf_counter() - t0
                proba = model.predict_proba(X[te])
                rows.append({
                    "dataset": dname, "model": mname,
                    "protocol": "tuned" if tuned else "default",
                    "fold": fold,
                    "accuracy": accuracy_score(y[te], model.predict(X[te])),
                    "log_loss": log_loss(yidx, np.clip(proba, 1e-12, 1),
                                         labels=range(len(classes))),
                    "ece": expected_calibration_error(yidx, proba),
                    "fit_time": fit_t})
            except Exception as exc:
                print(f"  [warn] {dname}/{mname}/{tuned}: {type(exc).__name__}")
    return rows


def main(n_splits: int = 5, n_repeats: int = 1, seed: int = 7, n_jobs: int = 2):
    from experiments.datasets import get_all_datasets
    data = get_all_datasets()
    rows = []
    for dname in SUBSET:
        if dname not in data:
            print(f"  [skip] {dname} absent")
            continue
        X, y = np.asarray(data[dname][0], float), np.asarray(data[dname][1])
        if len(y) > MAX_N:
            X, y = stratified_subsample(X, y, MAX_N, seed=SUBSAMPLE_SEED)
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        t0 = time.perf_counter()
        out = Parallel(n_jobs=n_jobs)(
            delayed(_fold)(dname, X, y, tr, te, f)
            for f, (tr, te) in enumerate(cv.split(X, y)))
        got = [r for c in out for r in c]
        rows.extend(got)
        sub = pd.DataFrame(got)
        piv = sub.pivot_table("accuracy", "model", "protocol")
        print(f"== {dname:16s} [{time.perf_counter()-t0:6.0f}s] " +
              "  ".join(f"{m}:{r['default']:.3f}->{r['tuned']:.3f}"
                        for m, r in piv.iterrows()), flush=True)
        save_csv(pd.DataFrame(rows), "tuned_baselines")
    df = pd.DataFrame(rows)
    save_csv(df, "tuned_baselines")

    summary = []
    for proto in ("default", "tuned"):
        sub = df[df["protocol"] == proto]
        acc = sub.pivot_table("accuracy", "dataset", "model")
        ll = sub.pivot_table("log_loss", "dataset", "model")
        ra = acc.rank(axis=1, ascending=False).mean()
        rl = ll.rank(axis=1).mean()
        for m in acc.columns:
            summary.append({"protocol": proto, "model": m,
                            "acc_mean": 100 * acc[m].mean(),
                            "acc_rank": ra[m], "ll_rank": rl[m],
                            "fit_time": sub[sub["model"] == m]["fit_time"].mean()})
    s = pd.DataFrame(summary)
    save_csv(s, "tuned_baselines_summary")
    print()
    print(s.pivot_table(index="model", columns="protocol",
                        values=["acc_mean", "acc_rank"]).round(2).to_string())


if __name__ == "__main__":
    main()
