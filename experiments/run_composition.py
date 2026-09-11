"""Is the dependence correction redundant with post-hoc recalibration?

Post-hoc calibration (Platt scaling, isotonic regression) is the standard
answer to naive Bayes overconfidence, and on the benchmark it is a strong
opponent on log-loss.  It is, however, a *monotone* transformation of the
score: it can repair the reported confidence but it cannot change which
class the model picks.  ABD-NB acts before the score is formed and can
therefore change the decision as well.

This experiment separates the two effects with a 2 x 2 design --
{naive Bayes, ABD-NB} x {raw, Platt, isotonic} -- over the whole
benchmark suite.  If the mechanisms were the same, the calibrated
variants would coincide; if they are complementary, the gains compose.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import ABDNB
from experiments.common import expected_calibration_error, save_csv
from experiments.datasets import dataset_stratum, get_all_datasets

warnings.filterwarnings("ignore")


def make_models():
    return {
        "GNB": GaussianNB(),
        "GNB+Platt": CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=3),
        "GNB+Iso": CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=3),
        "ABD-NB": ABDNB(),
        "ABD-NB+Platt": CalibratedClassifierCV(ABDNB(), method="sigmoid", cv=3),
        "ABD-NB+Iso": CalibratedClassifierCV(ABDNB(), method="isotonic", cv=3),
    }


def _fold(dname, X, y, tr, te, fold):
    rows = []
    for mname, model in make_models().items():
        try:
            model.fit(X[tr], y[tr])
            proba = model.predict_proba(X[te])
            classes = np.unique(y[tr])
            yidx = np.searchsorted(classes, y[te])
            rows.append({"dataset": dname, "stratum": dataset_stratum(dname),
                         "model": mname, "fold": fold,
                         "accuracy": accuracy_score(y[te], model.predict(X[te])),
                         "log_loss": log_loss(yidx, np.clip(proba, 1e-12, 1),
                                              labels=range(len(classes))),
                         "ece": expected_calibration_error(yidx, proba)})
        except Exception as exc:
            print(f"  [warn] {dname}/{mname}: {type(exc).__name__}")
    return rows


def main(n_splits: int = 10, n_repeats: int = 2, seed: int = 7, n_jobs: int = 3):
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
        ll = sub.groupby("model")["log_loss"].mean()
        ac = sub.groupby("model")["accuracy"].mean()
        print(f"== {dname:18s} LL: NB={ll['GNB']:.3f} NB+P={ll['GNB+Platt']:.3f} "
              f"ABD={ll['ABD-NB']:.3f} ABD+P={ll['ABD-NB+Platt']:.3f} | "
              f"acc ABD+P={ac['ABD-NB+Platt']:.3f}", flush=True)
        save_csv(pd.DataFrame(rows), "composition")
    save_csv(pd.DataFrame(rows), "composition")


if __name__ == "__main__":
    main()
