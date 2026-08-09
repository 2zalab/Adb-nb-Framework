"""Sensitivity and ablation studies.

(a) Correction strength gamma: accuracy / log-loss across a gamma grid
    on four representative datasets.
(b) Dependence measure: Spearman vs Pearson, Kendall, MI, dCor, HSIC,
    Cramer's V -- including the non-monotone design where rank measures
    are blind.
(c) Weight function: linear vs exponential vs inverse vs harmonic.
"""

from __future__ import annotations

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from abdnb import ABDNB
from abdnb.dependence import MEASURES
from abdnb.weights import WEIGHT_FUNCTIONS
from experiments.common import expected_calibration_error, save_csv
from experiments.datasets import conflict_block, equicorrelated_gaussian, nonlinear_dependent


def cv_eval(model, X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc, ll, ece, t = [], [], [], []
    for tr, te in cv.split(X, y):
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        t.append(time.perf_counter() - t0)
        p = model.predict_proba(X[te])
        yte = np.searchsorted(np.unique(y[tr]), y[te])
        acc.append(float((p.argmax(1) == yte).mean()))
        ll.append(log_loss(yte, np.clip(p, 1e-12, 1)))
        ece.append(expected_calibration_error(yte, p))
    return {"accuracy": np.mean(acc), "log_loss": np.mean(ll),
            "ece": np.mean(ece), "fit_time": np.mean(t)}


def get_data():
    return {
        "wine": load_wine(return_X_y=True),
        "breast-cancer": load_breast_cancer(return_X_y=True),
        "synth-conflict": conflict_block(n=800),
        "synth-rho0.6": equicorrelated_gaussian(0.6, n=800, delta=2.0),
        "synth-nonlin": nonlinear_dependent(n=800),
    }


def study_gamma():
    rows = []
    for dname, (X, y) in get_data().items():
        for gamma in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            m = ABDNB(gamma=gamma, class_specific=False)
            rows.append({"dataset": dname, "gamma": gamma,
                         **cv_eval(m, X, y)})
        print(f"gamma study: {dname} done")
    save_csv(pd.DataFrame(rows), "sensitivity_gamma")


def study_measure():
    rows = []
    for dname, (X, y) in get_data().items():
        for measure in MEASURES:
            m = ABDNB(measure=measure, gamma=1.0, class_specific=False)
            rows.append({"dataset": dname, "measure": measure,
                         **cv_eval(m, X, y)})
        print(f"measure study: {dname} done")
    save_csv(pd.DataFrame(rows), "sensitivity_measure")


def study_weight_fn():
    rows = []
    for dname, (X, y) in get_data().items():
        for wf in WEIGHT_FUNCTIONS:
            m = ABDNB(weight_fn=wf, gamma=1.0, class_specific=False)
            rows.append({"dataset": dname, "weight_fn": wf,
                         **cv_eval(m, X, y)})
        print(f"weight-fn study: {dname} done")
    save_csv(pd.DataFrame(rows), "sensitivity_weightfn")


if __name__ == "__main__":
    study_gamma()
    study_weight_fn()
    study_measure()
