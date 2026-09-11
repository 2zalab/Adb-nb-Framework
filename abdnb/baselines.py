"""Contemporary and semi-naive baselines for the ABD-NB study.

The comparison set of the paper deliberately spans the three families
that a dependence-correcting naive Bayes must be measured against:

*Structure-extending Bayesian classifiers* -- they model the dependence
explicitly instead of correcting for it:

- :class:`TAN`  : tree-augmented naive Bayes (Friedman, Geiger & Goldszmidt, 1997)
- :class:`KDB`  : k-dependence Bayesian classifier (Sahami, 1996)
- :class:`AODE` : averaged one-dependence estimators (Webb, Boughton & Wang, 2005)

*Feature-weighted naive Bayes* -- they keep the factorisation and learn
one exponent per feature:

- :class:`WANBIA` : discriminative weight optimisation of the conditional
  log-likelihood (Zaidi, Cerquides, Carman & Webb, 2013)
- :class:`CorrelationWeightedNB` : the relevance/redundancy heuristic of
  the correlation-based weighting literature (a filter counterpart of
  ABD-NB whose redundancy term is a *marginal* discount)

All three structure-extending models operate on equal-frequency
discretised features with Laplace smoothing, following the standard
protocol of the semi-naive Bayes literature.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .discretize import EqualFrequencyDiscretizer

__all__ = ["TAN", "KDB", "AODE", "WANBIA", "CorrelationWeightedNB"]


# ---------------------------------------------------------------------------
# Shared discrete machinery
# ---------------------------------------------------------------------------

class _DiscreteBayesBase(ClassifierMixin, BaseEstimator):
    """Discretisation, one-hot encoding and pairwise class-conditional counts."""

    def _prepare(self, X, y):
        self.disc_ = EqualFrequencyDiscretizer(self.n_bins)
        Xd = self.disc_.fit_transform(X)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.n_features_in_ = Xd.shape[1]
        self.n_levels_ = self.disc_.n_levels_
        self.class_prior_ = counts / counts.sum()
        self.offsets_ = np.concatenate([[0], np.cumsum(self.n_levels_)[:-1]])
        self.total_levels_ = int(self.n_levels_.sum())
        return Xd, np.searchsorted(self.classes_, y)

    def _flat(self, Xd):
        """Column-offset encoding: level ``a`` of feature ``j`` -> a single id."""
        return Xd + self.offsets_[None, :]

    def _pair_counts(self, Xd, yi):
        """(K, L, L) class-conditional co-occurrence counts (L = total levels)."""
        F = self._flat(Xd)
        K, L = len(self.classes_), self.total_levels_
        out = np.zeros((K, L, L))
        for k in range(K):
            m = yi == k
            if not np.any(m):
                continue
            oh = np.zeros((int(m.sum()), L))
            rows = np.repeat(np.arange(int(m.sum())), Xd.shape[1])
            oh[rows, F[m].ravel()] = 1.0
            out[k] = oh.T @ oh
        return out

    def _marginal_counts(self, pair):
        """(K, L) single-feature counts read off the diagonal of ``pair``."""
        return np.diagonal(pair, axis1=1, axis2=2).copy()

    def _cond_mutual_information(self, pair, single, n):
        """I(X_i ; X_j | Y) for every feature pair, from the count tensors."""
        d = self.n_features_in_
        K = len(self.classes_)
        cmi = np.zeros((d, d))
        for k in range(K):
            nk = single[k].sum() / d
            if nk <= 1:
                continue
            for i in range(d):
                si = slice(self.offsets_[i], self.offsets_[i] + self.n_levels_[i])
                pi = single[k, si] / nk
                for j in range(i + 1, d):
                    sj = slice(self.offsets_[j],
                               self.offsets_[j] + self.n_levels_[j])
                    joint = pair[k, si, sj] / nk
                    outer = np.outer(pi, single[k, sj] / nk)
                    mask = joint > 0
                    if not np.any(mask):
                        continue
                    term = np.sum(joint[mask] *
                                  np.log(joint[mask] / np.maximum(outer[mask], 1e-300)))
                    cmi[i, j] += (nk / n) * term
        return cmi + cmi.T

    def predict(self, X):
        jll = self._joint_log_likelihood(check_array(X))
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(check_array(X))
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(check_array(X))
        return jll - logsumexp(jll, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Tree-augmented naive Bayes and the k-dependence generalisation
# ---------------------------------------------------------------------------

class KDB(_DiscreteBayesBase):
    """k-dependence Bayesian classifier (Sahami, 1996).

    Features are ordered by decreasing mutual information with the class;
    each feature receives up to ``k`` parents chosen among the already
    ordered features by decreasing conditional mutual information.
    ``k = 1`` gives a TAN-like ordered structure, ``k = 2`` the usual KDB
    configuration reported in the semi-naive literature.
    """

    def __init__(self, k: int = 2, n_bins: int = 5, alpha: float = 1.0):
        self.k = k
        self.n_bins = n_bins
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        Xd, yi = self._prepare(X, y)
        n, d = Xd.shape
        pair = self._pair_counts(Xd, yi)
        single = self._marginal_counts(pair)
        cmi = self._cond_mutual_information(pair, single, n)
        # mutual information with the class, for the feature ordering
        mi_y = np.zeros(d)
        for j in range(d):
            sj = slice(self.offsets_[j], self.offsets_[j] + self.n_levels_[j])
            joint = single[:, sj] / n
            px = joint.sum(0)
            mask = joint > 0
            mi_y[j] = np.sum(joint[mask] * np.log(
                joint[mask] / np.maximum(
                    (self.class_prior_[:, None] * px[None, :])[mask], 1e-300)))
        order = np.argsort(-mi_y)
        self.parents_ = {}
        for pos, j in enumerate(order):
            cand = order[:pos]
            if len(cand) == 0:
                self.parents_[j] = []
            else:
                best = cand[np.argsort(-cmi[j, cand])][: self.k]
                self.parents_[j] = list(best)
        self._fit_tables(Xd, yi)
        return self

    def _fit_tables(self, Xd, yi):
        K = len(self.classes_)
        self.tables_ = {}
        for j, pa in self.parents_.items():
            shape = [K] + [int(self.n_levels_[p]) for p in pa] + [int(self.n_levels_[j])]
            cnt = np.zeros(shape)
            idx = tuple([yi] + [Xd[:, p] for p in pa] + [Xd[:, j]])
            np.add.at(cnt, idx, 1.0)
            cnt += self.alpha
            self.tables_[j] = np.log(cnt / cnt.sum(axis=-1, keepdims=True))

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "tables_")
        Xd = self.disc_.transform(X)
        jll = np.tile(np.log(self.class_prior_), (Xd.shape[0], 1))
        for j, pa in self.parents_.items():
            tab = self.tables_[j]
            idx = tuple([slice(None)] + [Xd[:, p] for p in pa] + [Xd[:, j]])
            jll += np.moveaxis(tab[idx], 0, -1) if len(pa) else tab[:, Xd[:, j]].T
        return jll


class TAN(KDB):
    """Tree-augmented naive Bayes (Friedman, Geiger & Goldszmidt, 1997).

    Each feature receives exactly one augmenting parent, obtained from
    the maximum-weight spanning tree of the conditional-mutual-information
    graph (Chow--Liu construction conditioned on the class).
    """

    def __init__(self, n_bins: int = 5, alpha: float = 1.0):
        super().__init__(k=1, n_bins=n_bins, alpha=alpha)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        Xd, yi = self._prepare(X, y)
        n, d = Xd.shape
        pair = self._pair_counts(Xd, yi)
        single = self._marginal_counts(pair)
        cmi = self._cond_mutual_information(pair, single, n)
        # Prim's maximum spanning tree on the CMI graph, rooted at 0
        in_tree = np.zeros(d, dtype=bool)
        in_tree[0] = True
        self.parents_ = {0: []}
        best_w = cmi[0].copy()
        best_p = np.zeros(d, dtype=int)
        for _ in range(d - 1):
            cand = np.where(~in_tree, best_w, -np.inf)
            j = int(np.argmax(cand))
            in_tree[j] = True
            self.parents_[j] = [int(best_p[j])]
            upd = cmi[j] > best_w
            best_w = np.where(upd, cmi[j], best_w)
            best_p = np.where(upd, j, best_p)
        self._fit_tables(Xd, yi)
        return self


# ---------------------------------------------------------------------------
# Averaged one-dependence estimators
# ---------------------------------------------------------------------------

class AODE(_DiscreteBayesBase):
    """Averaged one-dependence estimators (Webb, Boughton & Wang, 2005).

    Instead of searching for a structure, AODE averages every model in
    which a single feature is the super-parent of all the others, keeping
    only super-parents whose value occurs at least ``m`` times in the
    training sample.
    """

    def __init__(self, n_bins: int = 5, m: int = 1, chunk: int = 128):
        self.n_bins = n_bins
        self.m = m
        self.chunk = chunk

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        Xd, yi = self._prepare(X, y)
        self.n_ = Xd.shape[0]
        self.pair_ = self._pair_counts(Xd, yi)
        self.single_ = self._marginal_counts(self.pair_)
        return self

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "pair_")
        Xd = self.disc_.transform(X)
        F = self._flat(Xd)
        n_test, d = Xd.shape
        K = len(self.classes_)
        # per-feature level counts, used for the Laplace corrections
        lev = self.n_levels_.astype(float)
        out = np.empty((n_test, K))
        for start in range(0, n_test, self.chunk):
            Fc = F[start:start + self.chunk]
            T = Fc.shape[0]
            # counts of the (class, super-parent value, child value) cells
            cnt = self.pair_[:, Fc[:, :, None], Fc[:, None, :]]  # (K, T, d, d)
            cnt = np.moveaxis(cnt, 0, 0)
            sp = self.single_[:, Fc]                             # (K, T, d)
            # P(y, x_i) with the standard AODE Laplace correction
            log_joint = np.log(sp + 1.0 / (K * lev[None, None, :])) - np.log(self.n_ + 1.0)
            # P(x_j | y, x_i)
            log_cond = (np.log(cnt + 1.0 / lev[None, None, None, :])
                        - np.log(sp[:, :, :, None] + 1.0))
            # a super-parent never conditions on itself
            eye = np.eye(d, dtype=bool)
            log_cond[:, :, eye] = 0.0
            model = log_joint + log_cond.sum(axis=3)             # (K, T, d)
            # frequency filter on the super-parents
            valid = (sp.sum(axis=0) >= self.m)                   # (T, d)
            if not valid.any():
                valid = np.ones_like(valid)
            model = np.where(valid[None, :, :], model, -np.inf)
            out[start:start + T] = logsumexp(model, axis=2).T
        return out


# ---------------------------------------------------------------------------
# Feature-weighted naive Bayes baselines
# ---------------------------------------------------------------------------

class _GaussianCore:
    """Gaussian sufficient statistics shared by the weighted NB baselines."""

    def _fit_gaussians(self, X, y, var_smoothing=1e-9):
        self.classes_, counts = np.unique(y, return_counts=True)
        n, d = X.shape
        self.n_features_in_ = d
        self.class_prior_ = counts / n
        eps = var_smoothing * X.var(axis=0).max()
        self.theta_ = np.zeros((len(self.classes_), d))
        self.var_ = np.zeros((len(self.classes_), d))
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[k] = Xc.mean(axis=0)
            self.var_[k] = Xc.var(axis=0) + eps + 1e-12

    def _feature_log_likelihood(self, X):
        Xe = X[:, None, :]
        return -0.5 * (np.log(2.0 * np.pi * self.var_)[None]
                       + (Xe - self.theta_[None]) ** 2 / self.var_[None])

    def _jll(self, X):
        ll = self._feature_log_likelihood(X)
        return np.log(self.class_prior_)[None] + np.einsum("nkd,kd->nk", ll, self.weights_)


class WANBIA(ClassifierMixin, BaseEstimator, _GaussianCore):
    """Discriminatively weighted naive Bayes (WANBIA-C, Zaidi et al., 2013).

    The generative parameters are the naive Bayes ones; the per-feature
    exponents are obtained by maximising the *conditional* log-likelihood
    with an L-BFGS solver and an optional ridge penalty, which is the
    strongest weighting baseline in the feature-weighted NB literature.
    """

    def __init__(self, l2: float = 1e-3, max_iter: int = 200,
                 var_smoothing: float = 1e-9):
        self.l2 = l2
        self.max_iter = max_iter
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._fit_gaussians(X, y, self.var_smoothing)
        ll = self._feature_log_likelihood(X)              # (n, K, d)
        yi = np.searchsorted(self.classes_, y)
        logprior = np.log(self.class_prior_)
        n, K, d = ll.shape
        rows = np.arange(n)

        def obj(w):
            jll = logprior[None] + np.einsum("nkd,d->nk", ll, w)
            lse = logsumexp(jll, axis=1)
            nll = -(jll[rows, yi] - lse).mean() + self.l2 * np.sum((w - 1.0) ** 2)
            p = np.exp(jll - lse[:, None])
            grad_n = ll[rows, yi, :] - np.einsum("nk,nkd->nd", p, ll)
            grad = -grad_n.mean(axis=0) + 2.0 * self.l2 * (w - 1.0)
            return nll, grad

        res = minimize(obj, np.ones(d), jac=True, method="L-BFGS-B",
                       bounds=[(0.0, None)] * d,
                       options={"maxiter": self.max_iter})
        self.weights_ = np.tile(np.clip(res.x, 1e-3, None), (K, 1))
        self.effective_dimension_ = self.weights_.sum(axis=1)
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self._jll(check_array(X)), axis=1)]

    def predict_proba(self, X):
        jll = self._jll(check_array(X))
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))


class CorrelationWeightedNB(ClassifierMixin, BaseEstimator, _GaussianCore):
    """Relevance/redundancy weighted naive Bayes (filter family).

    ``w_i  proportional to  MI(X_i; Y) / (1 + mean_j |rho_ij|)`` -- the
    heuristic shape used by correlation-based weight-adjustment schemes.
    It is the closest published relative of ABD-NB and differs from it in
    three respects that the paper isolates: the redundancy term is
    *marginal* rather than class-conditional, it enters linearly rather
    than on the shared-information scale, and it carries neither an
    evidence-mass constraint nor a fallback to plain naive Bayes.
    """

    def __init__(self, var_smoothing: float = 1e-9, random_state: int | None = 0):
        self.var_smoothing = var_smoothing
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.feature_selection import mutual_info_classif
        X, y = check_X_y(X, y)
        self._fit_gaussians(X, y, self.var_smoothing)
        mi = mutual_info_classif(X, y, random_state=self.random_state)
        rel = mi / mi.max() if mi.max() > 0 else np.ones_like(mi)
        C = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0))
        np.fill_diagonal(C, 0.0)
        red = C.sum(axis=1) / max(X.shape[1] - 1, 1)
        w = rel / (1.0 + red)
        self.weights_ = np.tile(np.clip(w / w.max(), 1e-3, 1.0),
                                (len(self.classes_), 1))
        self.effective_dimension_ = self.weights_.sum(axis=1)
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self._jll(check_array(X)), axis=1)]

    def predict_proba(self, X):
        jll = self._jll(check_array(X))
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))
