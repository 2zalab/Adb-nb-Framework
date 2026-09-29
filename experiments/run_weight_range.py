"""Range of the fitted likelihood exponents across the real suite.

The model equation constrains the exponents only to be positive.  The raw
harmonic weights of Stage 2 do lie in ``(0, 1]``, but two later steps
multiply them by factors that can exceed one --- the evidence-budget
rescaling of Stage 3 and the class common-scale renormalisation --- so the
fitted exponents are not bounded by one.  This script records the observed
range for each of the four combinations, so the statement in the paper is
a measurement rather than an assumption.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from abdnb import ABDNB
from experiments.common import save_csv
from experiments.real_datasets import get_real_datasets

warnings.filterwarnings("ignore")


def main(gamma: float = 1.0):
    rows = []
    for name, (X, y) in get_real_datasets().items():
        for cs in (False, True):
            for resc in ("none", "meff"):
                W = ABDNB(gamma=gamma, class_specific=cs,
                          rescale=resc).fit(X, y).weights_
                rows.append({"dataset": name, "class_specific": cs,
                             "rescale": resc, "wmax": float(W.max()),
                             "wmin": float(W.min()),
                             "frac_above_1": float((W > 1.0).mean())})
    df = pd.DataFrame(rows)
    agg = (df.groupby(["rescale", "class_specific"])
             .agg(wmax=("wmax", "max"), wmin=("wmin", "min"),
                  frac_above_1=("frac_above_1", "mean"),
                  datasets_with_w_above_1=("frac_above_1", lambda s: int((s > 0).sum())),
                  argmax_dataset=("wmax", lambda s: df.loc[s.idxmax(), "dataset"]))
             .reset_index())
    for _, r in agg.iterrows():
        print(f"  {r['rescale']:5s} {'class-specific' if r['class_specific'] else 'global':14s}"
              f"  max {r['wmax']:.3f} ({r['argmax_dataset']})  min {r['wmin']:.3f}"
              f"  datasets with any w>1: {r['datasets_with_w_above_1']}/37", flush=True)
    save_csv(agg, "weight_range")
    save_csv(df, "weight_range_per_dataset")


if __name__ == "__main__":
    main()
