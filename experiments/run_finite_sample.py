"""Finite-sample behaviour of the weight estimator and of the selector.

The asymptotic statements of the paper (consistency, root-n normality,
never-worse selection) are supplemented here by their *finite-sample*
counterparts, which are the statements a practitioner can act on:

1. **Deviation bound.**  For the harmonic weights before rescaling the
   paper proves a bound of the form

       P( max_i |hat w_i - w*_i| > eps )  <=  2 d^2 exp(-c n eps^2 / (gamma kappa d)^2)

   obtained by combining a Hoeffding bound for the bounded-kernel
   U-statistics that define the rank dependence estimates with the
   Lipschitz constant of the map D -> w(D).  This script measures the
   empirical exceedance probability against the bound over two decades of
   sample size and reports the realised constant.

2. **Selection regret.**  The validation-based selector is compared with
   the *oracle* configuration (the grid point with the lowest test risk),
   giving the empirical regret whose finite-sample bound the paper
   states.  The bound decays as sqrt(log(|Gamma|/delta) / n_val); the
   measured regret is compared with it directly.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import ABDNB
from abdnb.dependence import class_conditional_dependence
from abdnb.weights import weights_from_matrix
from experiments.common import save_csv
from experiments.datasets import conflict_block, equicorrelated_gaussian

GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
STRUCTURES = (False, True)


# ---------------------------------------------------------------------------
# 1. deviation bound for the weights
# ---------------------------------------------------------------------------

def spearman_population(rho: float) -> float:
    """Population Spearman correlation of a bivariate Gaussian pair.

    The estimand of the plug-in weights is the dependence matrix *on the
    scale of the chosen measure*, not the Pearson correlation of the
    generator: for Gaussian margins the population Spearman value is
    ``(6/pi) arcsin(rho/2)``.  Using ``rho`` itself as the reference
    would introduce a fixed bias of about 0.018 at rho = 0.6 and make
    the measured deviation plateau instead of vanishing.
    """
    return float(6.0 / np.pi * np.arcsin(rho / 2.0))


def population_weights(rho: float, d: int, gamma: float = 1.0,
                       kappa: float = 2.0) -> np.ndarray:
    """Weights of the exact equicorrelated population matrix."""
    D = np.full((d, d), spearman_population(rho))
    np.fill_diagonal(D, 0.0)
    return weights_from_matrix(D, "harmonic", gamma, rescale="none", kappa=kappa)


def deviation_study(rho: float = 0.6, d: int = 12, gamma: float = 1.0,
                    reps: int = 300, sizes=(50, 100, 200, 400, 800, 1600, 3200),
                    eps_grid=(0.02, 0.05, 0.10)) -> pd.DataFrame:
    w_star = population_weights(rho, d, gamma)
    rows = []
    for n in sizes:
        dev = np.empty(reps)
        for rep in range(reps):
            X, y = equicorrelated_gaussian(rho, n=n, d=d, seed=7000 + rep)
            _, pooled = class_conditional_dependence(X, y, "spearman",
                                                     threshold=False)
            w = weights_from_matrix(pooled, "harmonic", gamma, rescale="none")
            dev[rep] = np.max(np.abs(w - w_star))
        for eps in eps_grid:
            emp = float((dev > eps).mean())
            # Theorem "finite-sample deviation of the weights":
            #   P(||w_hat - w*||_inf > eps) <= K d^2 exp(-(n_min - 1) eps^2 / (4 L^2))
            # with L = gamma * kappa * (d - 1) the Lipschitz constant of
            # the harmonic rule (Lemma "Lipschitz continuity").  The
            # dependence matrices are estimated per class, so n_min is
            # the smallest class size (n/2 for the balanced design).
            L = gamma * 2.0 * (d - 1)          # kappa = 2
            n_min = n / 2.0
            bound = min(1.0, 2 * d ** 2 * np.exp(-(n_min - 1) * eps ** 2 / (4 * L ** 2)))
            q95 = float(np.quantile(dev, 0.95))
            # constant that the observed 95th percentile actually implies
            # in the shape of the bound, for comparison with L
            implied = q95 / (2 * np.sqrt(np.log(2 * d ** 2 / 0.05) / (n_min - 1)))
            rows.append({"n": n, "eps": eps, "empirical": emp,
                         "bound": bound, "mean_dev": float(dev.mean()),
                         "q95_dev": q95, "L_theory": L, "L_implied": implied})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. selection regret against the grid oracle
# ---------------------------------------------------------------------------

def _fit_eval(X, y, Xte, yte, gamma, cs):
    clf = ABDNB().fit_configured(X, y, gamma, cs)
    return accuracy_score(yte, clf.predict(Xte))


def regret_study(reps: int = 30,
                 sizes=(60, 120, 250, 500, 1000, 2000)) -> pd.DataFrame:
    rows = []
    for n in sizes:
        for rep in range(reps):
            Xtr, ytr = conflict_block(n=n, seed=9000 + rep)
            Xte, yte = conflict_block(n=4000, seed=99000 + rep)
            scores = {}
            for g in GRID:
                for s in STRUCTURES:
                    if g == 0.0 and s:
                        continue
                    scores[(g, s)] = _fit_eval(Xtr, ytr, Xte, yte, g, s)
            sel = ABDNB().fit(Xtr, ytr)
            chosen = accuracy_score(yte, sel.predict(Xte))
            oracle = max(scores.values())
            n_val = n // 3            # one held-out internal fold
            n_cfg = len(scores)
            bound = np.sqrt(np.log(2 * n_cfg / 0.05) / (2 * n_val)) + 0.01
            rows.append({"n": n, "rep": rep, "oracle": oracle,
                         "selected": chosen, "regret": oracle - chosen,
                         "nb": scores[(0.0, False)], "bound": bound,
                         "gamma": sel.gamma_})
        print(f"  regret n={n} done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    dev = deviation_study()
    save_csv(dev, "finite_sample_deviation")
    print(dev.to_string(index=False))
    reg = regret_study()
    save_csv(reg, "finite_sample_regret")
    summary = reg.groupby("n").agg(
        regret_mean=("regret", "mean"), regret_q95=("regret", lambda s: s.quantile(0.95)),
        bound=("bound", "mean"), vs_nb=("nb", "mean"), selected=("selected", "mean"))
    save_csv(summary.reset_index(), "finite_sample_regret_summary")
    print(summary.round(4).to_string())
