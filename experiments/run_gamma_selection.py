"""What does the selector actually choose?

The adaptivity claim of ABD-NB rests on the internal selection layer, so
the distribution of the configurations it selects is itself a result and
not an implementation detail.  This script refits the flagship on every
cross-validation training fold of the benchmark and records the chosen
(gamma, structure) pair, giving three views:

- the marginal distribution of gamma over all (dataset, fold) fits;
- the same distribution stratified by the redundancy index of the
  dataset, which tests whether the selector reacts to the diagnostic of
  Section "redundancy profile" in the direction the mechanism predicts;
- the per-dataset modal choice, reported alongside the realised gain.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import ABDNB
from experiments.common import load_csv, save_csv
from experiments.datasets import dataset_stratum, get_all_datasets

GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _fold(dname, X, y, tr, fold):
    clf = ABDNB().fit(X[tr], y[tr])
    return {"dataset": dname, "stratum": dataset_stratum(dname), "fold": fold,
            "gamma": float(clf.gamma_), "class_specific": int(clf.class_specific_),
            "rescale": clf.rescale_, "d": X.shape[1],
            "d_eff": float(np.mean(clf.effective_dimension_))}


def main(n_splits: int = 10, n_repeats: int = 3, seed: int = 7, n_jobs: int = 4):
    rows = []
    for dname, (X, y) in get_all_datasets().items():
        X, y = np.asarray(X, float), np.asarray(y)
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        out = Parallel(n_jobs=n_jobs)(
            delayed(_fold)(dname, X, y, tr, f)
            for f, (tr, _) in enumerate(cv.split(X, y)))
        rows.extend(out)
        g = pd.Series([r["gamma"] for r in out])
        print(f"== {dname:18s} modal gamma={g.mode().iloc[0]:.2f}  "
              f"zero={100*(g == 0).mean():4.0f}%  "
              f"cls-spec={100*np.mean([r['class_specific'] for r in out]):4.0f}%",
              flush=True)
    df = pd.DataFrame(rows)
    save_csv(df, "gamma_selection")

    # marginal distribution over all fits, and by stratum
    tab = []
    for g in GRID:
        row = {"gamma": g, "all": 100 * (df["gamma"] == g).mean()}
        for st in ("real", "augmented", "synthetic"):
            sub = df[df["stratum"] == st]
            row[st] = 100 * (sub["gamma"] == g).mean()
        row["n_fits"] = int((df["gamma"] == g).sum())
        tab.append(row)
    marg = pd.DataFrame(tab)
    save_csv(marg, "gamma_selection_marginal")

    resc = pd.DataFrame([{
        "rule": r,
        "all": 100 * (df["rescale"] == r).mean(),
        "real": 100 * (df[df["stratum"] == "real"]["rescale"] == r).mean(),
        "n_fits": int((df["rescale"] == r).sum()),
    } for r in ("meff", "none")])
    save_csv(resc, "gamma_selection_rescale")
    print(); print(resc.round(1).to_string(index=False))

    # stratified by the redundancy index of the dataset
    prof = load_csv("redundancy_profile").set_index("dataset")
    real = df[df["dataset"].isin(prof.index)].copy()
    real["R"] = real["dataset"].map(prof["redundancy"])
    q = prof["redundancy"].quantile([1 / 3, 2 / 3]).to_numpy()
    real["band"] = np.where(real["R"] <= q[0], "low",
                            np.where(real["R"] <= q[1], "medium", "high"))
    bands = []
    for band in ("low", "medium", "high"):
        sub = real[real["band"] == band]
        row = {"band": band, "n_datasets": sub["dataset"].nunique(),
               "mean_gamma": sub["gamma"].mean(),
               "zero_pct": 100 * (sub["gamma"] == 0).mean(),
               "ge1_pct": 100 * (sub["gamma"] >= 1).mean(),
               "cls_pct": 100 * sub["class_specific"].mean(),
               "meff_pct": 100 * (sub["rescale"] == "meff").mean(),
               "deff_ratio": (sub["d_eff"] / sub["d"]).mean()}
        bands.append(row)
    band_df = pd.DataFrame(bands)
    save_csv(band_df, "gamma_selection_bands")
    print(); print(marg.round(1).to_string(index=False))
    print(); print(band_df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
