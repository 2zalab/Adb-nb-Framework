"""Statistical comparison of the benchmarked classifiers.

Following Demsar (2006):
- Friedman test (and Iman-Davenport correction) on the mean-accuracy
  ranks across datasets;
- Nemenyi post-hoc critical difference;
- pairwise Wilcoxon signed-rank tests of ABD-NB against every baseline,
  with Holm correction;
- win / tie / loss counts.

Outputs ``results/stats_*.csv`` consumed by ``make_figures.py``.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats

from experiments.common import load_csv, save_csv


def average_ranks(table: pd.DataFrame) -> pd.Series:
    """table: datasets x models mean accuracies -> average ranks
    (rank 1 = best)."""
    return table.rank(axis=1, ascending=False).mean(axis=0)


def friedman(table: pd.DataFrame):
    k = table.shape[1]
    N = table.shape[0]
    chi2, p = stats.friedmanchisquare(*[table[c] for c in table.columns])
    # Iman-Davenport F correction
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_ff = 1 - stats.f.cdf(ff, k - 1, (k - 1) * (N - 1))
    return chi2, p, ff, p_ff


def nemenyi_cd(k: int, N: int, alpha: float = 0.05) -> float:
    """Critical difference for the Nemenyi test.  q_alpha values are the
    Studentized-range-based constants of Demsar (2006), Table 5."""
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
               13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458,
               18: 3.489, 19: 3.517, 20: 3.544, 21: 3.569, 22: 3.593}[k]
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))


def main(metric: str = "accuracy", flagship: str = "ABD-NB",
         higher_is_better: bool = True, suffix: str = ""):
    df = load_csv("benchmark")
    table = (
        df.groupby(["dataset", "model"])[metric].mean().unstack()
    )
    # a model that failed on some dataset would bias the rank analysis
    table = table.dropna(axis=1, how="any")
    if not higher_is_better:
        table = -table
    ranks = average_ranks(table)
    chi2, p, ff, p_ff = friedman(table)
    cd = nemenyi_cd(table.shape[1], table.shape[0])
    print(f"Friedman chi2={chi2:.2f} (p={p:.2e});  "
          f"Iman-Davenport F={ff:.2f} (p={p_ff:.2e});  Nemenyi CD={cd:.3f}")
    print(ranks.sort_values().round(3).to_string())
    save_csv(ranks.rename("avg_rank").reset_index(), f"stats_ranks{suffix}")
    pd.DataFrame({"chi2": [chi2], "p": [p], "ff": [ff], "p_ff": [p_ff],
                  "cd": [cd], "k": [table.shape[1]], "N": [table.shape[0]]}
                 ).to_csv(os.path.join(os.path.dirname(__file__), "..",
                                       f"results/stats_friedman{suffix}.csv"),
                          index=False)

    # pairwise Wilcoxon (flagship vs all), Holm-corrected
    rows = []
    others = [m for m in table.columns if m != flagship]
    pvals = []
    for m in others:
        d = table[flagship] - table[m]
        try:
            w, pv = stats.wilcoxon(d, zero_method="pratt")
        except ValueError:
            w, pv = np.nan, 1.0
        pvals.append(pv)
        rows.append({"opponent": m, "median_diff": float(d.median()),
                     "wins": int((d > 1e-12).sum()), "ties": int((d.abs() <= 1e-12).sum()),
                     "losses": int((d < -1e-12).sum()), "p_raw": pv})
    order = np.argsort(pvals)
    holm = np.empty(len(pvals))
    for rank_i, idx in enumerate(order):
        holm[idx] = min(1.0, pvals[idx] * (len(pvals) - rank_i))
    for r, h in zip(rows, holm):
        r["p_holm"] = float(h)
    out = pd.DataFrame(rows).sort_values("p_holm")
    print(out.round(4).to_string(index=False))
    save_csv(out, f"stats_wilcoxon{suffix}")

    # summary table (mean +/- std per dataset/model) for the paper
    summary = df.groupby(["dataset", "model"]).agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1_mean=("macro_f1", "mean"), ll_mean=("log_loss", "mean"),
        ece_mean=("ece", "mean"), fit_time=("fit_time", "mean"),
    ).reset_index()
    save_csv(summary, "benchmark_summary")


if __name__ == "__main__":
    print("=== accuracy ===")
    main("accuracy")
    print("=== log-loss ===")
    main("log_loss", higher_is_better=False, suffix="_logloss")
    print("=== ECE ===")
    main("ece", higher_is_better=False, suffix="_ece")
