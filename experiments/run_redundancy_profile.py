"""Does the correction pay off where the theory says it should?

The mechanism of ABD-NB predicts that the benefit over naive Bayes is
governed by a single observable quantity: how much of the nominal feature
dimension is genuinely independent evidence.  We measure it on every real
dataset of the suite by the *redundancy index*

    R  =  1 - M_eff(hat D) / d,

computed from the pooled within-class dependence matrix, and confront it
with the realised improvement of ABD-NB over Gaussian naive Bayes taken
from the main benchmark.

This turns the paper's central claim into a falsifiable, dataset-level
prediction -- and gives practitioners a diagnostic that can be computed
*before* fitting anything: a dataset with a small redundancy index has
nothing for the correction to remove.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb.dependence import class_conditional_dependence
from abdnb.weights import effective_dimension
from experiments.common import load_csv, save_csv
from experiments.real_datasets import DOMAINS, get_real_datasets


def redundancy_index(X, y) -> tuple[float, float, int]:
    """(index, M_eff, d) from the pooled within-class dependence matrix."""
    _, pooled = class_conditional_dependence(np.asarray(X, float), y,
                                             "spearman", threshold=True)
    d = pooled.shape[0]
    meff = effective_dimension(pooled)
    return 1.0 - meff / d, meff, d


def main():
    bench = load_csv("benchmark")
    per = (bench.groupby(["dataset", "model"])[["accuracy", "log_loss", "ece"]]
           .mean().reset_index())
    rows = []
    for name, (X, y) in get_real_datasets().items():
        sub = per[per["dataset"] == name].set_index("model")
        if "GNB" not in sub.index or "ABD-NB" not in sub.index:
            continue
        idx, meff, d = redundancy_index(X, y)
        rows.append({
            "dataset": name, "domain": DOMAINS.get(name, "-"),
            "n": len(y), "d": d, "K": int(len(np.unique(y))),
            "M_eff": meff, "redundancy": idx,
            "acc_gnb": sub.loc["GNB", "accuracy"] * 100,
            "acc_abd": sub.loc["ABD-NB", "accuracy"] * 100,
            "d_acc": (sub.loc["ABD-NB", "accuracy"] - sub.loc["GNB", "accuracy"]) * 100,
            "ll_gnb": sub.loc["GNB", "log_loss"],
            "ll_abd": sub.loc["ABD-NB", "log_loss"],
            "d_ll_rel": 100 * (sub.loc["GNB", "log_loss"] - sub.loc["ABD-NB", "log_loss"])
                        / max(sub.loc["GNB", "log_loss"], 1e-12),
            "ece_gnb": sub.loc["GNB", "ece"],
            "ece_abd": sub.loc["ABD-NB", "ece"],
        })
        print(f"  {name:18s} R={idx:.3f}  d_acc={rows[-1]['d_acc']:+.2f}  "
              f"d_ll={rows[-1]['d_ll_rel']:+.1f}%", flush=True)
    df = pd.DataFrame(rows)
    save_csv(df, "redundancy_profile")

    stats_rows = []
    for target, label in (("d_acc", "accuracy gain (pts)"),
                          ("d_ll_rel", "log-loss reduction (%)")):
        r, p = stats.spearmanr(df["redundancy"], df[target])
        rp, pp = stats.pearsonr(df["redundancy"], df[target])
        stats_rows.append({"target": label, "spearman_rho": r, "spearman_p": p,
                           "pearson_r": rp, "pearson_p": pp, "n_datasets": len(df)})
    # stratified view: low / medium / high redundancy
    q = df["redundancy"].quantile([1 / 3, 2 / 3]).to_numpy()
    df["band"] = np.where(df["redundancy"] <= q[0], "low",
                          np.where(df["redundancy"] <= q[1], "medium", "high"))
    band = df.groupby("band")[["redundancy", "d_acc", "d_ll_rel"]].agg(["mean", "size"])
    save_csv(pd.DataFrame(stats_rows), "redundancy_profile_stats")
    save_csv(band.reset_index(), "redundancy_profile_bands")
    print(pd.DataFrame(stats_rows).to_string(index=False))
    print(band.round(3).to_string())


if __name__ == "__main__":
    main()
