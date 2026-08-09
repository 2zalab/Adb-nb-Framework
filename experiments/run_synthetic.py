"""Controlled synthetic studies.

(a) Equicorrelated Gaussians: accuracy / log-loss / ECE as a function
    of the true within-class correlation ``rho``.
(b) Conflict blocks: metrics as a function of the block size ``r``
    (number of near-duplicated weak features competing with one strong
    independent feature).
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from abdnb import ABDNB
from experiments.common import expected_calibration_error, save_csv
from experiments.datasets import conflict_block, equicorrelated_gaussian


def cv_metrics(model_factory, X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc, ll, ece, deff = [], [], [], []
    for tr, te in cv.split(X, y):
        m = model_factory().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        yte = np.searchsorted(np.unique(y[tr]), y[te])
        acc.append(float((p.argmax(1) == yte).mean()))
        ll.append(log_loss(yte, np.clip(p, 1e-12, 1)))
        ece.append(expected_calibration_error(yte, p))
        if hasattr(m, "effective_dimension_"):
            deff.append(float(m.effective_dimension_.mean()))
    return {
        "accuracy": np.mean(acc), "log_loss": np.mean(ll), "ece": np.mean(ece),
        "accuracy_sd": np.std(acc), "log_loss_sd": np.std(ll), "ece_sd": np.std(ece),
        "d_eff": np.mean(deff) if deff else np.nan,
    }


MODELS = {
    "GNB": lambda: GaussianNB(),
    "ABD-NB": lambda: ABDNB(),
    "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
}


def study_rho():
    rows = []
    for rho in [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.95]:
        for rep in range(5):
            X, y = equicorrelated_gaussian(rho, n=800, d=12, delta=2.0,
                                           seed=1000 + rep)
            for name, mk in MODELS.items():
                rows.append({"rho": rho, "rep": rep, "model": name,
                             **cv_metrics(mk, X, y, seed=rep)})
        print(f"rho={rho} done")
    save_csv(pd.DataFrame(rows), "synthetic_rho")


def study_conflict():
    rows = []
    for r in [1, 2, 4, 8, 12, 16, 24]:
        for rep in range(5):
            X, y = conflict_block(n=800, r=r, seed=2000 + rep)
            for name, mk in MODELS.items():
                rows.append({"r": r, "rep": rep, "model": name,
                             **cv_metrics(mk, X, y, seed=rep)})
        print(f"r={r} done")
    save_csv(pd.DataFrame(rows), "synthetic_conflict")


if __name__ == "__main__":
    study_rho()
    study_conflict()
