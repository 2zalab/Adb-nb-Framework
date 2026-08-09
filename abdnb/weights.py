"""Weighting schemes for ABD-NB.

Given a pairwise dependence matrix ``D`` (see :mod:`abdnb.dependence`),
this module converts it into per-feature adaptive weights ``w_i`` in
``(0, 1]``.  Two families are implemented:

1. *Scalar* schemes first aggregate the row ``D[i, :]`` into a single
   dependence degree ``bar_D_i`` (mean or max of the off-diagonal
   entries), then apply a decreasing link function ``w_i = f(bar_D_i)``:

   - ``linear``      : ``w = 1 - D``
   - ``exponential`` : ``w = exp(-gamma * D)``
   - ``inverse``     : ``w = 1 / (1 + gamma * D)``

2. The *graph* (harmonic-redundancy) scheme operates directly on the
   dependence graph without collapsing it first:

   - ``harmonic``    : ``w_i = 1 / (1 + beta * sum_{j != i} D[i, j])``

   With ``beta = 1`` this is the flagship ABD-NB weighting: if a feature
   is duplicated ``r`` times (``D[i, j] = 1`` within the group, 0
   outside), every copy receives weight exactly ``1/r``, so the group
   contributes a *single* effective likelihood factor and the classical
   double-counting pathology of naive Bayes disappears (Theorem 2 in the
   paper).  The quantity ``d_eff = sum_i w_i`` is the *effective number
   of independent features*.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "aggregate_dependence",
    "weights_from_matrix",
    "effective_dimension",
    "WEIGHT_FUNCTIONS",
]

WEIGHT_FUNCTIONS = ("linear", "exponential", "inverse", "harmonic")


def effective_dimension(D: np.ndarray) -> float:
    """Spectral effective number of independent features.

    Following the effective-number-of-tests literature (Li & Ji, 2005),
    the eigenvalues ``lambda_k`` of ``R = D + I`` (the dependence matrix
    with unit diagonal restored) are mapped through
    ``f(lambda) = 1(lambda >= 1) + (lambda - floor(lambda))`` and summed:

        M_eff = sum_k f(lambda_k),   with  1 <= M_eff <= d.

    ``M_eff = d`` for mutually independent features and
    ``M_eff = d - r + 1`` when one feature is duplicated ``r`` times.
    """
    d = D.shape[0]
    R = D.copy()
    np.fill_diagonal(R, 1.0)
    lam = np.clip(np.linalg.eigvalsh(R), 0.0, None)
    m = np.sum((lam >= 1).astype(float) + (lam - np.floor(lam)))
    return float(np.clip(m, 1.0, d))


def aggregate_dependence(D: np.ndarray, how: str = "mean") -> np.ndarray:
    """Collapse a dependence matrix into per-feature degrees in [0, 1]."""
    d = D.shape[0]
    if d == 1:
        return np.zeros(1)
    off = D.copy()
    np.fill_diagonal(off, 0.0)
    if how == "mean":
        return off.sum(axis=1) / (d - 1)
    if how == "max":
        return off.max(axis=1)
    raise ValueError(f"unknown aggregation {how!r}")


def weights_from_matrix(
    D: np.ndarray,
    weight_fn: str = "harmonic",
    gamma: float = 1.0,
    aggregation: str = "mean",
    rescale: str = "meff",
    kappa: float = 2.0,
    w_min: float = 1e-3,
) -> np.ndarray:
    """Map a pairwise dependence matrix to per-feature weights.

    The construction has two stages:

    1. *Relative* weights ``w~_i = f(D)`` (one of the four link
       functions) encode how each feature's contribution should shrink
       relative to the others.
    2. A *spectral rescaling* (``rescale='meff'``, default) multiplies
       the vector by a common factor so that ``sum_i w_i = M_eff(D)``,
       the effective number of independent features.  This keeps the
       total evidence mass of the weighted log-likelihood equal to that
       of ``M_eff`` truly independent features: redundancy is removed
       without crushing the likelihood scale.  Under exact duplication
       the rescaling factor is 1 and the harmonic weights ``1/r`` are
       returned unchanged.

    Parameters
    ----------
    D : (d, d) dependence matrix with entries in [0, 1].
    weight_fn : one of :data:`WEIGHT_FUNCTIONS`.
    gamma : decay strength for ``exponential``/``inverse``, redundancy
        strength ``beta`` for ``harmonic``.  Ignored by ``linear``.
    aggregation : row aggregation used by the scalar schemes.
    rescale : 'meff' (spectral, default), 'dim' (sum to d, i.e. the
        same scale as unweighted NB) or 'none'.
    w_min : numerical floor so no feature is fully discarded.

    Returns
    -------
    w : (d,) array of positive weights, bounded by 1 before rescaling.
    """
    D = np.asarray(D, dtype=float)
    d = D.shape[0]
    if gamma == 0.0:
        return np.ones(d)
    if weight_fn == "harmonic":
        # Squared dependences measure *shared information*: for
        # bivariate Gaussians MI = -log(1 - rho^2)/2 ~ rho^2/2, so
        # moderate correlations contribute quadratically less than
        # near-duplicates.  Exact duplication (D = 1) is unaffected and
        # keeps the 1/r recovery property at gamma = 1.
        D2 = D**kappa
        degree = D2.sum(axis=1) - np.diag(D2)
        w = 1.0 / (1.0 + gamma * degree)
    else:
        bar = aggregate_dependence(D, aggregation)
        if weight_fn == "linear":
            w = 1.0 - bar
        elif weight_fn == "exponential":
            w = np.exp(-gamma * bar)
        elif weight_fn == "inverse":
            w = 1.0 / (1.0 + gamma * bar)
        else:
            raise ValueError(
                f"unknown weight_fn {weight_fn!r}; choose from {WEIGHT_FUNCTIONS}"
            )
    w = np.clip(w, w_min, 1.0)
    if rescale == "meff":
        w = w * (effective_dimension(D) / w.sum())
    elif rescale == "dim":
        w = w * (d / w.sum())
    elif rescale != "none":
        raise ValueError(f"unknown rescale {rescale!r}")
    return w
