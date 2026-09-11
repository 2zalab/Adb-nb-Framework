"""Equal-frequency discretisation shared by the discrete-factor models.

Semi-naive Bayesian network classifiers (TAN, AODE, KDB) and the
categorical variant of ABD-NB all operate on discrete features.  The
transformation used throughout the paper is the standard equal-frequency
(quantile) binning with ``n_bins`` bins, with degenerate columns
collapsed to a single level so that constant or near-constant features
degrade gracefully instead of producing empty cells.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EqualFrequencyDiscretizer", "is_discrete"]


def is_discrete(X: np.ndarray, max_levels: int = 12) -> bool:
    """True when every column takes at most ``max_levels`` distinct values."""
    X = np.asarray(X)
    return all(len(np.unique(X[:, j])) <= max_levels for j in range(X.shape[1]))


class EqualFrequencyDiscretizer:
    """Quantile binning that also passes already-discrete columns through."""

    def __init__(self, n_bins: int = 5, max_levels: int = 12):
        self.n_bins = n_bins
        self.max_levels = max_levels

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        n_bins = max(2, min(self.n_bins, int(np.sqrt(n))))
        self.edges_ = []
        self.levels_ = []
        for j in range(d):
            col = X[:, j]
            uniq = np.unique(col)
            if len(uniq) <= self.max_levels:
                # already categorical: keep the observed levels
                self.edges_.append(None)
                self.levels_.append(uniq)
            else:
                q = np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1])
                self.edges_.append(np.unique(q))
                self.levels_.append(None)
        self.n_levels_ = np.array(
            [len(e) + 1 if e is not None else max(len(l), 1)
             for e, l in zip(self.edges_, self.levels_)]
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        cols = []
        for j in range(X.shape[1]):
            if self.edges_[j] is not None:
                cols.append(np.searchsorted(self.edges_[j], X[:, j]))
            else:
                lev = self.levels_[j]
                # nearest observed level (robust to unseen test values)
                pos = np.abs(X[:, j][:, None] - lev[None, :]).argmin(axis=1)
                cols.append(pos)
        return np.column_stack(cols).astype(np.int64)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
