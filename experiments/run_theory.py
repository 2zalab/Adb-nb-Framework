"""Empirical verification of the theoretical properties.

(a) Consistency and root-n convergence of the weight estimator:
    ||w_hat(n) - w*|| on the equicorrelated design, where the population
    weight vector w* is computed from the true correlation matrix.
(b) Bias / variance / MSE decomposition of the weight estimator.
(c) Duplicate-recovery property: with r exact copies of a feature each
    copy receives weight 1/r (gamma = 1) and d_eff = d - r + 1.
(d) Learning curves (accuracy vs training-set size) on the conflict
    design: excess error of ABD-NB vs GNB relative to the Bayes rate.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.naive_bayes import GaussianNB

from abdnb import ABDNB
from abdnb.weights import weights_from_matrix, effective_dimension
from experiments.common import save_csv
from experiments.datasets import conflict_block, equicorrelated_gaussian


def spearman_population(rho: float) -> float:
    """Population Spearman correlation of a bivariate Gaussian with
    Pearson correlation ``rho``: rho_S = (6/pi) arcsin(rho/2)."""
    return 6.0 / np.pi * np.arcsin(rho / 2.0)


def population_weights(rho: float, d: int) -> np.ndarray:
    """Population weights w* for the equicorrelated design (gamma=1,
    kappa=2, spectral rescaling).  The dependence estimator is the
    Spearman correlation, so the population matrix is expressed on the
    Spearman scale."""
    D = np.full((d, d), spearman_population(rho))
    np.fill_diagonal(D, 0.0)
    return weights_from_matrix(D, "harmonic", 1.0, rescale="meff", kappa=2.0)


def study_consistency():
    """||w_hat - w*||_2 as a function of n (with sqrt(n) reference),
    plus bias/variance/MSE of the raw (pre-rescaling) harmonic weight
    of the first coordinate, whose population value is
    w~* = 1/(1 + (d-1) rho_S^2)."""
    rho, d = 0.6, 12
    wstar = population_weights(rho, d)
    rho_s = spearman_population(rho)
    wtilde_star = 1.0 / (1.0 + (d - 1) * rho_s**2)
    rows = []
    for n in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
        for rep in range(30):
            X, y = equicorrelated_gaussian(rho, n=n, d=d, delta=2.0,
                                           seed=3000 + rep * 7 + n)
            m = ABDNB(gamma=1.0, class_specific=False, threshold=False)
            m.fit(X, y)
            w = m.weights_[0]
            D = m.pooled_dependence_
            wtilde_1 = float(1.0 / (1.0 + (D[0] ** 2).sum()))
            rows.append({
                "n": n, "rep": rep,
                "err_l2": float(np.linalg.norm(w - wstar)),
                "err_max": float(np.abs(w - wstar).max()),
                "w_tilde_1": wtilde_1,
                "d_eff": float(m.effective_dimension_[0]),
            })
        print(f"n={n} done")
    save_csv(pd.DataFrame(rows), "consistency")
    # bias / variance / mse of the raw first-coordinate weight
    df = pd.DataFrame(rows)
    bv = []
    for n, g in df.groupby("n"):
        bias = g["w_tilde_1"].mean() - wtilde_star
        var = g["w_tilde_1"].var()
        bv.append({"n": n, "bias": bias, "variance": var,
                   "mse": bias**2 + var})
    save_csv(pd.DataFrame(bv), "bias_variance")


def study_duplicates():
    """Weight of each of r exact copies, as a function of r."""
    rows = []
    rng = np.random.default_rng(0)
    n, base_d = 3000, 6
    for r in [1, 2, 3, 4, 5, 6, 8, 10]:
        X0 = rng.normal(size=(n, base_d))
        y = (X0[:, :2].sum(1) + 0.5 * rng.normal(size=n) > 0).astype(int)
        copies = [X0[:, [0]] + 1e-8 * rng.normal(size=(n, 1)) for _ in range(r - 1)]
        X = np.hstack([X0] + copies)
        m = ABDNB(gamma=1.0, class_specific=False, threshold=False).fit(X, y)
        w = m.weights_[0]
        rows.append({
            "r": r,
            "w_copy_mean": float(np.mean(w[[0] + list(range(base_d, base_d + r - 1))])),
            "w_theory": 1.0 / r,
            "d_eff": float(m.effective_dimension_[0]),
            "d_eff_theory": base_d,  # 6 features, copies collapse to one
        })
        print(f"r={r}: w_copy={rows[-1]['w_copy_mean']:.4f} vs 1/r={1/r:.4f}")
    save_csv(pd.DataFrame(rows), "duplicates")


def study_learning_curve():
    rows = []
    for n in [60, 120, 250, 500, 1000, 2000, 4000]:
        for rep in range(20):
            Xtr, ytr = conflict_block(n=n, seed=4000 + rep)
            Xte, yte = conflict_block(n=4000, seed=9999 + rep)
            for name, mk in [("GNB", GaussianNB), ("ABD-NB", ABDNB)]:
                m = mk().fit(Xtr, ytr)
                rows.append({"n": n, "rep": rep, "model": name,
                             "accuracy": float((m.predict(Xte) == yte).mean())})
        print(f"n={n} done")
    # Bayes rate of the conflict design, by Monte Carlo with the true model
    save_csv(pd.DataFrame(rows), "learning_curve")


if __name__ == "__main__":
    study_duplicates()
    study_consistency()
    study_learning_curve()
