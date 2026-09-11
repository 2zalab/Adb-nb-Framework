"""Adaptive Bayesian Dependence Naive Bayes (ABD-NB).

The classifier keeps the factorised architecture of Gaussian naive
Bayes but replaces the implicit unit exponents of the class-conditional
likelihood with data-driven *dependence-adaptive* exponents:

    log P(c | x)  proportional to
        log pi_c + sum_i  w_{ic} * log N(x_i ; mu_{ic}, sigma_{ic}^2)

where ``w_{ic} in (0, 1]`` decreases with the estimated dependence of
feature ``i`` on the remaining features (optionally within class ``c``).
Setting all weights to one recovers Gaussian naive Bayes exactly.

Three estimators are provided:

- :class:`ABDNB` : the batch classifier (global, class-specific, or
  graph/harmonic weights; any dependence measure of
  :mod:`abdnb.dependence`).
- :class:`MIWeightedNB` : a label-relevance weighted naive Bayes
  baseline (weights proportional to the mutual information between each
  feature and the class label), representing the classical
  feature-weighted NB family.
- :class:`OnlineABDNB` : a streaming variant with exponential
  forgetting, whose weights track *time-varying* dependence structure
  (used in the concept-drift experiment).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from .dependence import dependence_matrix, class_conditional_dependence
from .weights import weights_from_matrix

__all__ = ["ABDNB", "MIWeightedNB", "OnlineABDNB"]


class _GaussianWeightedNBBase(ClassifierMixin, BaseEstimator):
    """Shared Gaussian machinery for weighted NB variants."""

    var_smoothing = 1e-9

    def _fit_gaussians(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        n, d = X.shape
        self.n_features_in_ = d
        self.class_prior_ = counts / n
        eps = self.var_smoothing * X.var(axis=0).max()
        self.theta_ = np.zeros((len(self.classes_), d))
        self.var_ = np.zeros((len(self.classes_), d))
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[k] = Xc.mean(axis=0)
            self.var_[k] = Xc.var(axis=0) + eps + 1e-12

    def _feature_log_likelihood(self, X):
        """(n, K, d) array of per-feature Gaussian log-densities."""
        Xe = X[:, None, :]
        return -0.5 * (
            np.log(2.0 * np.pi * self.var_)[None]
            + (Xe - self.theta_[None]) ** 2 / self.var_[None]
        )

    def _joint_log_likelihood(self, X):
        ll = self._feature_log_likelihood(X)  # (n, K, d)
        W = self.weights_  # (K, d)
        return np.log(self.class_prior_)[None] + np.einsum("nkd,kd->nk", ll, W)

    def predict_log_proba(self, X):
        check_is_fitted(self, "theta_")
        X = check_array(X)
        jll = self._joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        check_is_fitted(self, "theta_")
        X = check_array(X)
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]


class ABDNB(_GaussianWeightedNBBase):
    """Adaptive Bayesian Dependence Naive Bayes.

    Parameters
    ----------
    measure : dependence measure ('spearman', 'pearson', 'kendall',
        'mi', 'dcor', 'hsic', 'cramersv').  Dependence is always
        estimated *within class* (pooled or per class), because the
        naive Bayes assumption concerns conditional -- not marginal --
        independence.
    weight_fn : 'harmonic' (graph weights, default), 'linear',
        'exponential' or 'inverse'.
    gamma : correction strength.  ``gamma = 0`` recovers Gaussian naive
        Bayes exactly; ``gamma = 1`` is the canonical value for which
        the duplicate-recovery property holds; ``'auto'`` selects gamma
        on an internal stratified CV over ``tune_grid``, which makes
        ABD-NB adaptive: the data decide how much correction the
        independence violation warrants.
    kappa : redundancy exponent of the harmonic degree
        ``sum_j D_ij^kappa`` (default 2: shared-variance scale).
    class_specific : True, False, or 'auto' (jointly tuned with gamma).
    aggregation : row aggregation for the scalar weight functions.
    threshold : apply the asymptotic soft-threshold to the dependence
        estimates (variance reduction in small samples).
    rescale : weight rescaling rule -- 'meff' (spectral effective
        dimension, default), 'dim' (sum to d) or 'none' (raw weights).
    tune_grid : gamma grid for 'auto' (default (0, 0.25, 0.5, 1, 2, 4)).
    tune_cv : internal folds for 'auto' (default 3).
    """

    _DEFAULT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

    def __init__(
        self,
        measure: str = "spearman",
        weight_fn: str = "harmonic",
        gamma="auto",
        kappa: float = 2.0,
        class_specific="auto",
        aggregation: str = "mean",
        threshold: bool = True,
        alpha: float = 0.05,
        rescale: str = "meff",
        tune_grid=None,
        tune_cv: int = 3,
        var_smoothing: float = 1e-9,
        random_state: int | None = 0,
    ):
        self.measure = measure
        self.weight_fn = weight_fn
        self.gamma = gamma
        self.kappa = kappa
        self.class_specific = class_specific
        self.aggregation = aggregation
        self.threshold = threshold
        self.alpha = alpha
        self.rescale = rescale
        self.tune_grid = tune_grid
        self.tune_cv = tune_cv
        self.var_smoothing = var_smoothing
        self.random_state = random_state

    # -- internals ---------------------------------------------------

    def _dependence(self, X, y):
        return class_conditional_dependence(
            X, y, self.measure, self.threshold, self.alpha,
            random_state=self.random_state,
        )

    def _make_weights(self, mats, pooled, classes, prior, gamma, class_specific):
        K = len(classes)
        if class_specific:
            W = np.vstack(
                [
                    weights_from_matrix(
                        mats[c], self.weight_fn, gamma,
                        self.aggregation, self.rescale, self.kappa,
                    )
                    for c in classes
                ]
            )
            # Common evidence scale across classes: the per-feature
            # *shape* stays class-specific, but every row is rescaled
            # to the prior-weighted mean total weight.  Log-densities
            # are negative, so unequal row sums would mechanically
            # favour the class with the smaller effective dimension.
            target = float(prior @ W.sum(axis=1))
            W = W * (target / W.sum(axis=1, keepdims=True))
        else:
            w = weights_from_matrix(pooled, self.weight_fn, gamma,
                                    self.aggregation, self.rescale, self.kappa)
            W = np.tile(w, (K, 1))
        return W

    def _candidate_configs(self):
        grid = tuple(self.tune_grid) if self.tune_grid is not None else self._DEFAULT_GRID
        gammas = grid if self.gamma == "auto" else (float(self.gamma),)
        if self.class_specific == "auto":
            structures = (False, True)
        else:
            structures = (bool(self.class_specific),)
        return [(g, s) for g in gammas for s in structures if not (g == 0.0 and s)]

    def _tune(self, X, y, configs, acc_tol: float = 0.01):
        """Internal CV selection with a one-percentage-point tolerance
        rule: among the configurations whose validation accuracy is
        within ``acc_tol`` of the best, pick the one with the smallest
        validation log-loss (a strictly proper score, hence sensitive
        to the calibration gains the weights provide); remaining ties
        go to the smallest gamma, i.e. the model closest to plain NB.
        """
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(self.tune_cv, shuffle=True,
                              random_state=self.random_state)
        acc = {c: 0.0 for c in configs}
        nll = {c: 0.0 for c in configs}
        n_folds = 0
        for tr, va in skf.split(X, y):
            n_folds += 1
            sub = ABDNB(**{**self.get_params(), "gamma": 0.0,
                           "class_specific": False})
            sub.fit_configured(X[tr], y[tr], 0.0, False)  # fits Gaussians + dependence
            mats, pooled = sub.dependence_, sub.pooled_dependence_
            ll = sub._feature_log_likelihood(X[va])
            yva = np.searchsorted(sub.classes_, y[va])
            logprior = np.log(sub.class_prior_)
            for cfg in configs:
                W = sub._make_weights(mats, pooled, sub.classes_,
                                      sub.class_prior_, *cfg)
                jll = logprior[None] + np.einsum("nkd,kd->nk", ll, W)
                lp = jll - logsumexp(jll, axis=1, keepdims=True)
                acc[cfg] += float((jll.argmax(1) == yva).mean())
                nll[cfg] += float(-lp[np.arange(len(yva)), yva].mean())
        best_acc = max(acc.values())
        admissible = [c for c in configs
                      if acc[c] >= best_acc - acc_tol * n_folds]
        return min(admissible, key=lambda c: (nll[c], c[0]))

    # -- public API ---------------------------------------------------

    def fit_configured(self, X, y, gamma: float, class_specific: bool):
        """Fit with an explicit (gamma, class_specific) configuration."""
        self._fit_gaussians(X, y)
        mats, pooled = self._dependence(X, y)
        self.dependence_ = mats
        self.pooled_dependence_ = pooled
        self.weights_ = self._make_weights(
            mats, pooled, self.classes_, self.class_prior_, gamma, class_specific
        )
        self.gamma_ = gamma
        self.class_specific_ = class_specific
        self.effective_dimension_ = self.weights_.sum(axis=1)
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        configs = self._candidate_configs()
        if len(configs) > 1:
            gamma, cs = self._tune(X, y, configs)
        else:
            gamma, cs = configs[0]
        return self.fit_configured(X, y, gamma, cs)


class MIWeightedNB(_GaussianWeightedNBBase):
    """Feature-weighted NB baseline: weights from label relevance.

    ``w_i = MI(X_i; Y) / max_j MI(X_j; Y)`` -- the classical
    'correlation-with-class' weighting used in the feature-weighted NB
    literature.  It measures *relevance to the label*, not *redundancy
    between features*, and therefore serves as the natural control for
    isolating the contribution of the dependence-based weights of
    ABD-NB.
    """

    def __init__(self, var_smoothing: float = 1e-9, random_state: int | None = 0):
        self.var_smoothing = var_smoothing
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._fit_gaussians(X, y)
        mi = mutual_info_classif(X, y, random_state=self.random_state)
        w = mi / mi.max() if mi.max() > 0 else np.ones_like(mi)
        self.weights_ = np.tile(np.clip(w, 1e-3, 1.0), (len(self.classes_), 1))
        return self


class OnlineABDNB(_GaussianWeightedNBBase):
    """Streaming ABD-NB with exponential forgetting.

    Class-conditional means/variances and the pairwise dependence
    statistics are maintained as exponentially weighted moments with
    forgetting factor ``lam`` (effective memory ``1/(1-lam)`` samples).
    Dependence is tracked through the exponentially weighted Pearson
    correlation of the streamed values -- an O(d^2) update per sample --
    and mapped to harmonic graph weights after every ``refresh`` steps.
    """

    def __init__(self, classes, n_features, lam: float = 0.995,
                 gamma: float = 1.0, refresh: int = 25):
        self.classes = classes
        self.n_features = n_features
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = n_features
        self.lam = lam
        self.gamma = gamma
        self.refresh = refresh
        K, d = len(self.classes_), n_features
        self._count = np.full(K, 1e-8)
        self.theta_ = np.zeros((K, d))
        self._m2 = np.ones((K, d))
        # global EW moments for the correlation graph
        self._mean = np.zeros(d)
        self._cov = np.eye(d) * 1e-6
        self._wsum = 1e-8
        self._t = 0
        self.class_prior_ = np.full(K, 1.0 / K)
        self.var_ = np.ones((K, d))
        self.weights_ = np.ones((K, d))
        self.weight_history_ = []

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        jll = self._joint_log_likelihood(X)
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))

    def partial_fit(self, x, y):
        """Update with one sample ``x`` (shape (d,)) of class ``y``."""
        x = np.asarray(x, dtype=float)
        k = int(np.searchsorted(self.classes_, y))
        lam = self.lam
        # class-conditional EW mean / variance
        self._count *= lam
        self._count[k] += 1.0
        eta = 1.0 / self._count[k]
        delta = x - self.theta_[k]
        self.theta_[k] += eta * delta
        self._m2[k] = (1 - eta) * (self._m2[k] + eta * delta**2)
        self.var_[k] = self._m2[k] + 1e-9
        self.class_prior_ = self._count / self._count.sum()
        # global EW covariance for the dependence graph
        self._wsum = lam * self._wsum + 1.0
        eta_g = 1.0 / self._wsum
        dg = x - self._mean
        self._mean += eta_g * dg
        self._cov = (1 - eta_g) * self._cov + eta_g * np.outer(dg, x - self._mean)
        self._t += 1
        if self._t % self.refresh == 0:
            self._refresh_weights()
        return self

    def _refresh_weights(self):
        sd = np.sqrt(np.clip(np.diag(self._cov), 1e-12, None))
        C = self._cov / np.outer(sd, sd)
        D = np.clip(np.abs(C), 0.0, 1.0)
        np.fill_diagonal(D, 0.0)
        w = weights_from_matrix(D, "harmonic", self.gamma)
        self.weights_ = np.tile(w, (len(self.classes_), 1))
        self.weight_history_.append((self._t, w.copy()))
