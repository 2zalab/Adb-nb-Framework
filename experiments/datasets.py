"""Benchmark datasets for the ABD-NB study.

The evaluation suite is organised in three clearly separated strata, so
that no conclusion rests on artificially constructed evidence:

1. **Real-world benchmarks** (:mod:`experiments.real_datasets`): 37
   publicly available classification datasets from 14 application
   domains.  These carry the empirical claims of the paper.
2. **Redundancy-augmented real datasets** (2): real data with duplicated
   columns appended, used as a *stress test* of the duplication theory
   on non-synthetic signal.
3. **Controlled synthetic families** (9): generators whose dependence
   structure is known exactly, used to *isolate the mechanism* rather
   than to demonstrate general performance.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)

RNG = 12345


def _add_redundant_copies(X, n_copies: int = 2, noise: float = 0.1, seed: int = 0):
    """Append ``n_copies`` noisy duplicates of every feature (a realistic
    model of redundant measurements: same signal, independent sensor
    noise of relative magnitude ``noise``)."""
    rng = np.random.default_rng(seed)
    sd = X.std(axis=0, keepdims=True)
    blocks = [X]
    for _ in range(n_copies):
        blocks.append(X + rng.normal(scale=noise * sd, size=X.shape))
    return np.hstack(blocks)


def equicorrelated_gaussian(rho: float, n: int = 600, d: int = 12,
                            n_classes: int = 2, delta: float = 1.0,
                            seed: int = RNG):
    """Class-conditional Gaussians with equicorrelation ``rho``.

    Sigma = (1 - rho) I + rho 11';  class c has mean  c * delta / sqrt(d) * 1.
    The Bayes-optimal rule depends only on the projection onto 1, so the
    problem difficulty is held constant while the redundancy grows with
    rho -- exactly the regime in which naive Bayes double-counts.
    """
    rng = np.random.default_rng(seed)
    Sigma = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
    L = np.linalg.cholesky(Sigma)
    y = rng.integers(0, n_classes, size=n)
    mu = (np.arange(n_classes)[:, None] * delta / np.sqrt(d)) * np.ones((1, d))
    X = mu[y] + rng.normal(size=(n, d)) @ L.T
    return X, y


def block_correlated(n: int = 600, n_blocks: int = 4, block_size: int = 4,
                     rho: float = 0.85, seed: int = RNG):
    """Blocks of strongly correlated features; one informative signal per
    block plus independent noise features."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    cols = []
    for b in range(n_blocks):
        z = (2 * y - 1) * 0.9 + rng.normal(size=n)
        for _ in range(block_size):
            cols.append(np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.normal(size=n))
    for _ in range(4):  # independent weak features
        cols.append((2 * y - 1) * 0.25 + rng.normal(size=n))
    return np.column_stack(cols), y


def nonlinear_dependent(n: int = 600, seed: int = RNG):
    """Non-monotone (quadratic/absolute-value) dependence between
    features: invisible to Pearson/Spearman, visible to dCor/HSIC/MI."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    s = 2 * y - 1
    x1 = s * 0.8 + rng.normal(size=n)
    x2 = x1**2 + 0.4 * rng.normal(size=n)          # nonlinear copy of x1
    x3 = np.abs(x1) + 0.4 * rng.normal(size=n)     # nonlinear copy of x1
    x4 = s * 0.8 + rng.normal(size=n)              # independent signal
    x5 = rng.normal(size=n)                        # pure noise
    x6 = np.cos(x4) + 0.4 * rng.normal(size=n)     # nonlinear copy of x4
    return np.column_stack([x1, x2, x3, x4, x5, x6]), y


def conflict_block(n: int = 600, r: int = 12, rho: float = 0.95,
                   strong: float = 1.0, weak: float = 0.35, seed: int = RNG):
    """The classical naive Bayes double-counting pathology (Hand & Yu,
    2001): one strong independent feature against a block of ``r``
    near-duplicated weak features.  NB counts the block ``r`` times and
    lets it out-vote the strong feature; the Bayes rule weights the
    block as a single piece of evidence."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    s = 2 * y - 1
    x0 = s * strong + rng.normal(size=n)
    z = s * weak + rng.normal(size=n)
    block = [
        np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.normal(size=n) for _ in range(r)
    ]
    return np.column_stack([x0] + block), y


def class_asymmetric(n: int = 600, d: int = 12, rho: float = 0.9,
                     delta: float = 1.2, seed: int = RNG):
    """Class-dependent dependence structure: within class 0 the first
    half of the features is equicorrelated (rho) and the second half is
    independent; within class 1 the roles are swapped.  Class-specific
    weights w_{ic} can adapt to each block; a single global weight
    vector cannot."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    h = d // 2
    X = np.zeros((n, d))
    Sig = (1 - rho) * np.eye(h) + rho * np.ones((h, h))
    L = np.linalg.cholesky(Sig)
    for c, corr_first in ((0, True), (1, False)):
        m = y == c
        nc = int(m.sum())
        corr = rng.normal(size=(nc, h)) @ L.T
        indep = rng.normal(size=(nc, h))
        X[m] = np.hstack([corr, indep] if corr_first else [indep, corr])
        X[m] += c * delta / np.sqrt(d)
    return X, y


def sklearn_redundant(n: int = 600, seed: int = RNG):
    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=5, n_redundant=10,
        n_repeated=0, n_clusters_per_class=2, class_sep=1.0,
        flip_y=0.02, random_state=seed,
    )
    return X, y


def get_synthetic_datasets() -> dict:
    """name -> (X, y) for the nine controlled synthetic families."""
    return {
        "synth-rho0.0": equicorrelated_gaussian(0.0),
        "synth-rho0.3": equicorrelated_gaussian(0.3),
        "synth-rho0.6": equicorrelated_gaussian(0.6),
        "synth-rho0.9": equicorrelated_gaussian(0.9),
        "synth-blocks": block_correlated(),
        "synth-conflict": conflict_block(),
        "synth-classdep": class_asymmetric(),
        "synth-nonlin": nonlinear_dependent(),
        "synth-sklearn": sklearn_redundant(),
    }


def get_augmented_datasets() -> dict:
    """name -> (X, y) for the two redundancy-augmented real datasets."""
    iris, wine = load_iris(), load_wine()
    return {
        "iris-red": (_add_redundant_copies(iris.data, 2, 0.10, 1), iris.target),
        "wine-red": (_add_redundant_copies(wine.data, 2, 0.10, 2), wine.target),
    }


def get_all_datasets() -> dict:
    """The complete evaluation suite: real, augmented and synthetic."""
    from experiments.real_datasets import get_real_datasets
    data = dict(get_real_datasets())
    data.update(get_augmented_datasets())
    data.update(get_synthetic_datasets())
    return data


def dataset_stratum(name: str) -> str:
    """'real', 'augmented' or 'synthetic' -- used by every results table."""
    if name.startswith("synth-"):
        return "synthetic"
    if name.endswith("-red"):
        return "augmented"
    return "real"


def get_datasets() -> dict:
    """Legacy 15-setting suite kept so earlier results remain reproducible."""
    iris = load_iris()
    wine = load_wine()
    bc = load_breast_cancer()
    dig = load_digits()
    data = {
        "iris": (iris.data, iris.target),
        "wine": (wine.data, wine.target),
        "breast-cancer": (bc.data, bc.target),
        "digits": (dig.data, dig.target),
        "iris-red": (_add_redundant_copies(iris.data, 2, 0.10, 1), iris.target),
        "wine-red": (_add_redundant_copies(wine.data, 2, 0.10, 2), wine.target),
        "synth-rho0.0": equicorrelated_gaussian(0.0),
        "synth-rho0.3": equicorrelated_gaussian(0.3),
        "synth-rho0.6": equicorrelated_gaussian(0.6),
        "synth-rho0.9": equicorrelated_gaussian(0.9),
        "synth-blocks": block_correlated(),
        "synth-conflict": conflict_block(),
        "synth-classdep": class_asymmetric(),
        "synth-nonlin": nonlinear_dependent(),
        "synth-sklearn": sklearn_redundant(),
    }
    return data


if __name__ == "__main__":
    for name, (X, y) in get_datasets().items():
        print(f"{name:16s} n={X.shape[0]:5d}  d={X.shape[1]:3d}  K={len(np.unique(y))}")
