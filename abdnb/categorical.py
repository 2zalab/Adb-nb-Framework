"""Categorical (multinomial / Bernoulli) instantiation of ABD-NB.

Stages 1--4 of the framework only consume the per-feature log-densities
``log p(x_i | c)``, so the dependence-adaptive exponents transfer
verbatim from Gaussian to discrete factors.  This module provides that
instantiation and is what the paper uses for the categorical, text-derived
and sequence benchmarks:

    log P(c | x)  =  log pi_c + sum_i w_{ic} log P(X_i = x_i | c) + const,

with multinomial factors estimated by Laplace-smoothed frequencies and
class-conditional dependence measured by bias-corrected Cramer's V on the
categorical codes (rank correlations are not meaningful for unordered
levels).  Setting ``gamma = 0`` recovers the ordinary categorical naive
Bayes exactly, so the safeguard of the continuous model carries over.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .dependence import class_conditional_discrete_dependence
from .discretize import EqualFrequencyDiscretizer
from .weights import weights_from_matrix

__all__ = ["CategoricalABDNB", "CategoricalNB_"]


class CategoricalABDNB(ClassifierMixin, BaseEstimator):
    """Dependence-weighted categorical naive Bayes.

    Parameters mirror :class:`abdnb.ABDNB`; ``n_bins`` controls the
    equal-frequency discretisation applied to any continuous column
    (already-categorical columns pass through unchanged) and ``alpha`` is
    the Laplace smoothing constant of the multinomial factors.
    """

    _DEFAULT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

    def __init__(self, n_bins: int = 5, alpha: float = 1.0,
                 measure: str = "cramersv", weight_fn: str = "harmonic",
                 gamma="auto", kappa: float = 2.0, class_specific="auto",
                 aggregation: str = "mean", threshold: bool = True,
                 dep_alpha: float = 0.05, rescale: str = "meff",
                 tune_grid=None, tune_cv: int = 3, random_state: int | None = 0):
        self.n_bins = n_bins
        self.alpha = alpha
        self.measure = measure
        self.weight_fn = weight_fn
        self.gamma = gamma
        self.kappa = kappa
        self.class_specific = class_specific
        self.aggregation = aggregation
        self.threshold = threshold
        self.dep_alpha = dep_alpha
        self.rescale = rescale
        self.tune_grid = tune_grid
        self.tune_cv = tune_cv
        self.random_state = random_state

    # -- factors ------------------------------------------------------

    def _fit_factors(self, Xd, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        K, d = len(self.classes_), Xd.shape[1]
        self.n_features_in_ = d
        self.class_prior_ = counts / counts.sum()
        self.log_prob_ = []
        for j in range(d):
            L = int(self.n_levels_[j])
            cnt = np.zeros((K, L))
            np.add.at(cnt, (np.searchsorted(self.classes_, y),
                            np.clip(Xd[:, j], 0, L - 1)), 1.0)
            cnt += self.alpha
            self.log_prob_.append(np.log(cnt / cnt.sum(axis=1, keepdims=True)))

    def _feature_log_likelihood(self, Xd):
        n, d = Xd.shape
        out = np.empty((n, len(self.classes_), d))
        for j in range(d):
            tab = self.log_prob_[j]                      # (K, L)
            out[:, :, j] = tab[:, np.clip(Xd[:, j], 0, tab.shape[1] - 1)].T
        return out

    # -- weights ------------------------------------------------------

    def _make_weights(self, mats, pooled, gamma, class_specific):
        K = len(self.classes_)
        if class_specific:
            W = np.vstack([
                weights_from_matrix(mats[c], self.weight_fn, gamma,
                                    self.aggregation, self.rescale, self.kappa)
                for c in self.classes_])
            target = float(self.class_prior_ @ W.sum(axis=1))
            W = W * (target / W.sum(axis=1, keepdims=True))
        else:
            w = weights_from_matrix(pooled, self.weight_fn, gamma,
                                    self.aggregation, self.rescale, self.kappa)
            W = np.tile(w, (K, 1))
        return W

    def _candidate_configs(self):
        grid = tuple(self.tune_grid) if self.tune_grid is not None else self._DEFAULT_GRID
        gammas = grid if self.gamma == "auto" else (float(self.gamma),)
        structures = (False, True) if self.class_specific == "auto" \
            else (bool(self.class_specific),)
        return [(g, s) for g in gammas for s in structures if not (g == 0.0 and s)]

    def _tune(self, X, y, configs, acc_tol: float = 0.01):
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(self.tune_cv, shuffle=True,
                              random_state=self.random_state)
        acc = {c: 0.0 for c in configs}
        nll = {c: 0.0 for c in configs}
        n_folds = 0
        for tr, va in skf.split(X, y):
            n_folds += 1
            sub = CategoricalABDNB(**{**self.get_params(), "gamma": 0.0,
                                      "class_specific": False})
            sub.fit_configured(X[tr], y[tr], 0.0, False)
            ll = sub._feature_log_likelihood(sub.disc_.transform(X[va]))
            yva = np.searchsorted(sub.classes_, y[va])
            logprior = np.log(sub.class_prior_)
            for cfg in configs:
                W = sub._make_weights(sub.dependence_, sub.pooled_dependence_, *cfg)
                jll = logprior[None] + np.einsum("nkd,kd->nk", ll, W)
                lp = jll - logsumexp(jll, axis=1, keepdims=True)
                acc[cfg] += float((jll.argmax(1) == yva).mean())
                nll[cfg] += float(-lp[np.arange(len(yva)), yva].mean())
        best = max(acc.values())
        admissible = [c for c in configs if acc[c] >= best - acc_tol * n_folds]
        return min(admissible, key=lambda c: (nll[c], c[0]))

    # -- public API ---------------------------------------------------

    def fit_configured(self, X, y, gamma: float, class_specific: bool):
        self.disc_ = EqualFrequencyDiscretizer(self.n_bins)
        Xd = self.disc_.fit_transform(X)
        self.n_levels_ = self.disc_.n_levels_
        self._fit_factors(Xd, y)
        mats, pooled = class_conditional_discrete_dependence(
            Xd, y, self.measure, self.threshold, self.dep_alpha)
        self.dependence_, self.pooled_dependence_ = mats, pooled
        self.weights_ = self._make_weights(mats, pooled, gamma, class_specific)
        self.gamma_, self.class_specific_ = gamma, class_specific
        self.effective_dimension_ = self.weights_.sum(axis=1)
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        configs = self._candidate_configs()
        gamma, cs = self._tune(X, y, configs) if len(configs) > 1 else configs[0]
        return self.fit_configured(X, y, gamma, cs)

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "weights_")
        Xd = self.disc_.transform(check_array(X))
        ll = self._feature_log_likelihood(Xd)
        return np.log(self.class_prior_)[None] + np.einsum("nkd,kd->nk", ll,
                                                           self.weights_)

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))


class CategoricalNB_(CategoricalABDNB):
    """Plain categorical naive Bayes (``gamma = 0``), the matched control."""

    def __init__(self, n_bins: int = 5, alpha: float = 1.0):
        super().__init__(n_bins=n_bins, alpha=alpha, gamma=0.0,
                         class_specific=False)
