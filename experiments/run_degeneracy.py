"""Why the Gaussian instantiation collapses on some datasets.

The full-size study (``run_scalability.py full``) shows ABD-NB and naive
Bayes failing together on a handful of datasets while a discretising
competitor does well.  The explanation is not the dependence correction
and not the sample size: it is that the Gaussian factor is degenerate.
This script measures the two quantities that establish that, per dataset,
so the failure analysis in the paper rests on a released measurement
rather than on an assertion:

* how discrete the features are (how many are binary, how many have at
  most ten distinct values, the median number of levels), and
* how degenerate the class-conditional Gaussians are (the smallest
  within-class variance as a fraction of the feature's global variance,
  and how many features are constant within some class).

A feature that is constant within a class has zero class-conditional
variance, so its density is governed entirely by the variance floor
``var_smoothing * max_j Var(X_j)`` rather than by the data.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.common import save_csv
from experiments.real_datasets import LARGE_REGISTRY, PMLB_REGISTRY, load_pmlb

#: the datasets the full-size study reports as failures, plus the one the
#: matched-tuning study rescues, plus two controls that do not fail
FOCUS = ["ann-thyroid", "kr-vs-kp", "nursery", "coil2000", "hypothyroid",
         "letter", "magic-full", "texture-full", "pendigits"]


def _stats(name, X, y):
    d = X.shape[1]
    levels = np.array([len(np.unique(X[:, j])) for j in range(d)])
    gvar = X.var(axis=0)
    scale = max(gvar.max(), 1e-300)
    ratios = []
    constant = 0
    for c in np.unique(y):
        Xc = X[y == c]
        if len(Xc) < 2:
            continue
        wv = Xc.var(axis=0)
        ratios.append(wv / scale)
        constant = max(constant, int((wv <= 0).sum()))
    R = np.vstack(ratios) if ratios else np.zeros((1, d))
    return {"dataset": name, "n": len(y), "d": d, "K": int(len(np.unique(y))),
            "binary": int((levels == 2).sum()),
            "at_most_10_levels": int((levels <= 10).sum()),
            "median_levels": float(np.median(levels)),
            "min_within_class_var_ratio": float(R.min()),
            "features_constant_in_some_class": constant,
            "frac_features_constant": constant / d}


def main():
    registry = {slug: name for slug, name, *_ in PMLB_REGISTRY}
    registry.update({slug: name for slug, name, *_ in LARGE_REGISTRY})
    inverse = {v: k for k, v in registry.items()}
    rows = []
    for name in FOCUS:
        slug = inverse.get(name, name.replace("-", "_"))
        try:
            X, y = load_pmlb(slug)
        except Exception as exc:  # pragma: no cover - missing cache
            print(f"  [skip] {name}: {exc}", flush=True)
            continue
        rows.append(_stats(name, np.asarray(X, float), np.asarray(y)))
        r = rows[-1]
        print(f"  {name:14s} binary {r['binary']:2d}/{r['d']:2d}  "
              f"median levels {r['median_levels']:4.0f}  "
              f"min within-class var ratio {r['min_within_class_var_ratio']:.1e}  "
              f"constant-in-a-class {r['features_constant_in_some_class']}", flush=True)
    save_csv(pd.DataFrame(rows), "degeneracy")


if __name__ == "__main__":
    main()
