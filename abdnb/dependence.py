"""Dependence estimation for ABD-NB.

This module implements the pairwise dependence estimators used by the
Adaptive Bayesian Dependence Naive Bayes (ABD-NB) classifier.  Every
estimator returns a symmetric matrix ``D`` with entries in ``[0, 1]``
where ``D[i, j]`` quantifies the strength of the statistical dependence
between features ``X_i`` and ``X_j`` (0 = independent, 1 = perfectly
dependent).  The diagonal is set to zero by convention because a feature
is never penalised for depending on itself.

Available measures
------------------
- ``pearson``   : absolute Pearson product-moment correlation
- ``spearman``  : absolute Spearman rank correlation (default; robust to
                  monotone transformations and moderately robust to
                  outliers)
- ``kendall``   : absolute Kendall tau-b
- ``mi``        : normalised mutual information on quantile-discretised
                  features
- ``dcor``      : distance correlation (detects non-linear, non-monotone
                  dependence); computed on a subsample for tractability
- ``hsic``      : normalised Hilbert-Schmidt Independence Criterion with
                  RBF kernels (median heuristic); subsampled
- ``cramersv``  : Cramer's V on quantile-discretised features

All estimators support an optional *shrinkage threshold*: entries whose
magnitude is not statistically distinguishable from zero at the
asymptotic level ``t_n = z_{1-alpha/2} / sqrt(n)`` are soft-thresholded.
This reduces the variance of the plug-in weight estimator in small
samples (see the paper, Section on the thresholded estimator).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

__all__ = [
    "dependence_matrix",
    "class_conditional_dependence",
    "MEASURES",
    "DISCRETE_MEASURES",
    "discrete_dependence_matrix",
    "class_conditional_discrete_dependence",
]

MEASURES = ("pearson", "spearman", "kendall", "mi", "dcor", "hsic", "cramersv")


# ---------------------------------------------------------------------------
# Elementary pairwise measures
# ---------------------------------------------------------------------------

def _safe_corr(fun, x, y) -> float:
    """Evaluate a scipy correlation function, mapping NaN -> 0."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r = fun(x, y)[0]
    return 0.0 if np.isnan(r) else float(abs(r))


def _pearson_matrix(X: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    D = np.abs(C)
    np.fill_diagonal(D, 0.0)
    return D


def _spearman_matrix(X: np.ndarray) -> np.ndarray:
    # Spearman = Pearson on ranks; computed directly so that constant
    # columns degrade gracefully to zero dependence.
    R = np.apply_along_axis(stats.rankdata, 0, X)
    return _pearson_matrix(R)


def _kendall_matrix(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    D = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            D[i, j] = D[j, i] = _safe_corr(stats.kendalltau, X[:, i], X[:, j])
    return D


def _discretise(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile discretisation with degenerate-column handling."""
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(np.unique(qs), x)


def _mi_matrix(X: np.ndarray, n_bins: int = 8) -> np.ndarray:
    n, d = X.shape
    n_bins = max(2, min(n_bins, int(np.sqrt(n))))
    Xd = np.column_stack([_discretise(X[:, j], n_bins) for j in range(d)])
    D = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            D[i, j] = D[j, i] = _normalised_mi(Xd[:, i], Xd[:, j])
    return D


def _normalised_mi(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information normalised by min entropy -> [0, 1]."""
    joint = {}
    for u, v in zip(a, b):
        joint[(u, v)] = joint.get((u, v), 0) + 1
    n = len(a)
    pa = {u: c / n for u, c in zip(*np.unique(a, return_counts=True))}
    pb = {v: c / n for v, c in zip(*np.unique(b, return_counts=True))}
    mi = 0.0
    for (u, v), c in joint.items():
        p = c / n
        mi += p * np.log(p / (pa[u] * pb[v]))
    ha = -sum(p * np.log(p) for p in pa.values())
    hb = -sum(p * np.log(p) for p in pb.values())
    hmin = min(ha, hb)
    if hmin <= 0:
        return 0.0
    return float(np.clip(mi / hmin, 0.0, 1.0))


def _cramersv_matrix(X: np.ndarray, n_bins: int = 6) -> np.ndarray:
    n, d = X.shape
    n_bins = max(2, min(n_bins, int(np.sqrt(n))))
    Xd = np.column_stack([_discretise(X[:, j], n_bins) for j in range(d)])
    D = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            D[i, j] = D[j, i] = _cramers_v(Xd[:, i], Xd[:, j])
    return D


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    r, k = len(ua), len(ub)
    if r < 2 or k < 2:
        return 0.0
    table = np.zeros((r, k))
    np.add.at(table, (ia, ib), 1)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = len(a)
    # bias-corrected Cramer's V (Bergsma, 2013)
    phi2 = max(0.0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    k_c = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_c - 1, k_c - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2 / denom))


def _centred_distance(x: np.ndarray) -> np.ndarray:
    Dm = squareform(pdist(x.reshape(-1, 1)))
    return Dm - Dm.mean(0, keepdims=True) - Dm.mean(1, keepdims=True) + Dm.mean()


def _dcor_matrix(X: np.ndarray, max_n: int = 400, rng=None) -> np.ndarray:
    n, d = X.shape
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(n, min(n, max_n), replace=False)
    Xs = X[idx]
    A = [_centred_distance(Xs[:, j]) for j in range(d)]
    var = [max((Ai * Ai).mean(), 1e-12) for Ai in A]
    D = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            cov = (A[i] * A[j]).mean()
            D[i, j] = D[j, i] = float(
                np.sqrt(max(cov, 0.0) / np.sqrt(var[i] * var[j]))
            )
    return D


def _rbf_gram(x: np.ndarray) -> np.ndarray:
    Dm = squareform(pdist(x.reshape(-1, 1), "sqeuclidean"))
    med = np.median(Dm[Dm > 0]) if np.any(Dm > 0) else 1.0
    return np.exp(-Dm / max(med, 1e-12))


def _hsic_matrix(X: np.ndarray, max_n: int = 300, rng=None) -> np.ndarray:
    n, d = X.shape
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(n, min(n, max_n), replace=False)
    Xs = X[idx]
    m = len(idx)
    H = np.eye(m) - np.ones((m, m)) / m
    K = [H @ _rbf_gram(Xs[:, j]) @ H for j in range(d)]
    norm = [max(np.sqrt((Kj * Kj).sum()), 1e-12) for Kj in K]
    D = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            # normalised HSIC = centred-kernel alignment in [0, 1]
            D[i, j] = D[j, i] = float(
                np.clip((K[i] * K[j]).sum() / (norm[i] * norm[j]), 0.0, 1.0)
            )
    return D


_MATRIX_FUN = {
    "pearson": _pearson_matrix,
    "spearman": _spearman_matrix,
    "kendall": _kendall_matrix,
    "mi": _mi_matrix,
    "dcor": _dcor_matrix,
    "hsic": _hsic_matrix,
    "cramersv": _cramersv_matrix,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dependence_matrix(
    X: np.ndarray,
    measure: str = "spearman",
    threshold: bool = True,
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> np.ndarray:
    """Estimate the pairwise dependence matrix of the columns of ``X``.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    measure : one of :data:`MEASURES`
    threshold : if True, soft-threshold entries below the asymptotic
        significance level ``t_n = z_{1-alpha/2}/sqrt(n-3)`` (Fisher
        z-scale for correlations, used as a generic noise floor for all
        measures).  This yields the *thresholded* estimator analysed in
        the paper.
    alpha : significance level of the threshold.

    Returns
    -------
    D : (d, d) symmetric matrix with zero diagonal and entries in [0, 1].
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if measure not in _MATRIX_FUN:
        raise ValueError(f"unknown measure {measure!r}; choose from {MEASURES}")
    kwargs = {}
    if measure in ("dcor", "hsic"):
        kwargs["rng"] = np.random.default_rng(random_state)
    D = _MATRIX_FUN[measure](X, **kwargs)
    D = np.clip(D, 0.0, 1.0)
    if threshold and n > 4:
        t_n = stats.norm.ppf(1 - alpha / 2) / np.sqrt(max(n - 3, 1))
        t_n = min(t_n, 0.5)
        # soft-thresholding: shrink toward zero, keep continuity
        D = np.where(D > t_n, (D - t_n) / (1.0 - t_n), 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def class_conditional_dependence(
    X: np.ndarray,
    y: np.ndarray,
    measure: str = "spearman",
    threshold: bool = True,
    alpha: float = 0.05,
    min_class_size: int = 8,
    random_state: int | None = 0,
) -> tuple[dict, np.ndarray]:
    """Class-conditional and pooled within-class dependence.

    The relevant estimand for naive Bayes is the dependence of the
    features *given* the class: marginal association between two
    informative features is induced by the label itself and does not
    violate the conditional-independence assumption.  This function
    therefore estimates one matrix ``D^(c)`` per class and the pooled
    within-class matrix ``D = sum_c pi_c D^(c)`` (prior-weighted
    average).  Classes with fewer than ``min_class_size`` samples fall
    back to a matrix computed on the class-centred residuals of the
    whole sample, which acts as a shrinkage target.

    Returns
    -------
    (mats, pooled) : ``mats[c]`` is the matrix of class ``c``;
        ``pooled`` is the prior-weighted average.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    priors = counts / counts.sum()
    # shrinkage target: dependence of the class-centred residuals
    Xc_centred = X.copy()
    for c in classes:
        m = y == c
        Xc_centred[m] -= Xc_centred[m].mean(axis=0)
    fallback = dependence_matrix(Xc_centred, measure, threshold, alpha, random_state)
    mats = {}
    for c, nc in zip(classes, counts):
        if nc < min_class_size:
            mats[c] = fallback.copy()
        else:
            mats[c] = dependence_matrix(
                X[y == c], measure, threshold, alpha, random_state
            )
    pooled = np.sum([p * mats[c] for p, c in zip(priors, classes)], axis=0)
    return mats, pooled


# ---------------------------------------------------------------------------
# Discrete (categorical) features
# ---------------------------------------------------------------------------

DISCRETE_MEASURES = ("cramersv", "mi")


def discrete_dependence_matrix(
    Xd: np.ndarray,
    measure: str = "cramersv",
    threshold: bool = True,
    alpha: float = 0.05,
) -> np.ndarray:
    """Pairwise dependence between *already discrete* columns.

    Unlike :func:`dependence_matrix`, no quantile binning is applied: the
    integer codes are used as they are, which is the correct estimand for
    genuinely categorical features (multinomial or Bernoulli factors).
    """
    Xd = np.asarray(Xd)
    n, d = Xd.shape
    fun = _cramers_v if measure == "cramersv" else _normalised_mi
    D = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            D[i, j] = D[j, i] = fun(Xd[:, i], Xd[:, j])
    D = np.clip(D, 0.0, 1.0)
    if threshold and n > 4:
        t_n = min(stats.norm.ppf(1 - alpha / 2) / np.sqrt(max(n - 3, 1)), 0.5)
        D = np.where(D > t_n, (D - t_n) / (1.0 - t_n), 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def class_conditional_discrete_dependence(
    Xd: np.ndarray,
    y: np.ndarray,
    measure: str = "cramersv",
    threshold: bool = True,
    alpha: float = 0.05,
    min_class_size: int = 12,
) -> tuple[dict, np.ndarray]:
    """Class-conditional and pooled dependence for discrete features."""
    Xd = np.asarray(Xd)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    priors = counts / counts.sum()
    fallback = discrete_dependence_matrix(Xd, measure, threshold, alpha)
    mats = {}
    for c, nc in zip(classes, counts):
        if nc < min_class_size:
            mats[c] = fallback.copy()
        else:
            mats[c] = discrete_dependence_matrix(Xd[y == c], measure,
                                                 threshold, alpha)
    pooled = np.sum([p * mats[c] for p, c in zip(priors, classes)], axis=0)
    return mats, pooled
