"""Real-world benchmark suite for the ABD-NB study.

The suite deliberately privileges *real* data over synthetic constructions:
thirty publicly available classification benchmarks drawn from eight
application domains (medicine, biology/agriculture, finance and business,
social science, industry and sensing, signal/image, text, physics).

Twenty-six datasets are obtained from the Penn Machine Learning Benchmarks
(PMLB) distribution of the UCI/OpenML repositories; four (iris, wine,
breast-cancer, digits) ship with scikit-learn and are kept for continuity
with the earlier literature.  Files are cached under ``implementation/data``
so the experiments are reproducible offline after a first download.

Datasets whose sample size exceeds ``MAX_N`` are reduced by a *stratified*
subsample with a fixed seed, so that the full suite can be evaluated under
the 3 x 10-fold protocol in reasonable time; the reduction is recorded in
the dataset table of the paper.
"""

from __future__ import annotations

import gzip
import io
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

#: Git-LFS media endpoint of the PMLB repository (raw pointers otherwise).
PMLB_URL = ("https://media.githubusercontent.com/media/EpistasisLab/pmlb/"
            "master/datasets/{name}/{name}.tsv.gz")

MAX_N = 4000
SUBSAMPLE_SEED = 20250

#: (pmlb name, display name, domain, feature type)
PMLB_REGISTRY = [
    # -- medicine -----------------------------------------------------
    ("heart_disease_cleveland", "heart-cleveland", "medicine", "mixed"),
    ("hepatitis", "hepatitis", "medicine", "mixed"),
    ("dermatology", "dermatology", "medicine", "categorical"),
    ("hypothyroid", "hypothyroid", "medicine", "mixed"),
    ("saheart", "saheart", "medicine", "mixed"),
    ("spectf", "spectf", "medicine", "numeric"),
    ("new_thyroid", "new-thyroid", "medicine", "numeric"),
    ("breast_cancer_wisconsin_original", "breast-w", "medicine", "numeric"),
    ("horse_colic_surgery", "horse-colic", "medicine", "mixed"),
    # -- biology / agriculture ---------------------------------------
    ("ecoli", "ecoli", "biology", "numeric"),
    ("yeast", "yeast", "biology", "numeric"),
    ("penguins", "penguins", "biology", "mixed"),
    ("agaricus_lepiota", "mushroom", "biology", "categorical"),
    ("soybean", "soybean", "agriculture", "categorical"),
    ("splice", "splice-dna", "genomics", "categorical"),
    # -- finance / business ------------------------------------------
    ("credit_approval_australia", "credit-aus", "finance", "mixed"),
    ("credit_approval_germany", "credit-ger", "finance", "mixed"),
    ("churn", "churn", "business", "mixed"),
    ("profb", "profb", "business", "mixed"),
    # -- social science ----------------------------------------------
    ("contraceptive_method", "contraceptive", "social", "mixed"),
    ("titanic", "titanic", "social", "categorical"),
    ("nursery", "nursery", "social", "categorical"),
    # -- industry / sensing / signal ---------------------------------
    ("segmentation", "segmentation", "imaging", "numeric"),
    ("vehicle", "vehicle", "industry", "numeric"),
    ("satimage", "satimage", "remote-sensing", "numeric"),
    ("phoneme", "phoneme", "speech", "numeric"),
    ("page_blocks", "page-blocks", "document", "numeric"),
    ("ionosphere", "ionosphere", "physics", "numeric"),
    ("sonar", "sonar", "sensing", "numeric"),
    ("texture", "texture", "imaging", "numeric"),
    ("mfeat_karhunen", "mfeat-karhunen", "imaging", "numeric"),
    # -- text --------------------------------------------------------
    ("spambase", "spambase", "text", "numeric"),
    # -- physics -----------------------------------------------------
    ("magic", "magic", "physics", "numeric"),
]

#: large datasets used *without* subsampling by the scalability study
#: (pmlb name, display name, domain, feature type)
LARGE_REGISTRY = [
    ("letter", "letter", "imaging", "numeric"),
    ("magic", "magic-full", "physics", "numeric"),
    ("nursery", "nursery-full", "social", "categorical"),
    ("pendigits", "pendigits", "imaging", "numeric"),
    ("coil2000", "coil2000", "insurance", "mixed"),
    ("ann_thyroid", "ann-thyroid", "medicine", "mixed"),
    ("ring", "ring", "physics", "numeric"),
    ("twonorm", "twonorm", "physics", "numeric"),
    ("optdigits", "optdigits", "imaging", "numeric"),
    ("agaricus_lepiota", "mushroom-full", "biology", "categorical"),
    ("satimage", "satimage-full", "remote-sensing", "numeric"),
    ("texture", "texture-full", "imaging", "numeric"),
    ("spambase", "spambase-full", "text", "numeric"),
    ("phoneme", "phoneme-full", "speech", "numeric"),
    ("page_blocks", "page-blocks-full", "document", "numeric"),
    ("churn", "churn-full", "business", "mixed"),
    ("kr_vs_kp", "kr-vs-kp", "games", "categorical"),
    ("waveform_40", "waveform-40", "signal", "numeric"),
]

SKLEARN_REGISTRY = [
    ("iris", "biology", "numeric"),
    ("wine", "chemistry", "numeric"),
    ("breast-cancer", "medicine", "numeric"),
    ("digits", "imaging", "numeric"),
]

DOMAINS = {name: dom for _, name, dom, _ in PMLB_REGISTRY}
DOMAINS.update({name: dom for name, dom, _ in SKLEARN_REGISTRY})
FEATURE_TYPE = {name: ft for _, name, _, ft in PMLB_REGISTRY}
FEATURE_TYPE.update({name: ft for name, _, ft in SKLEARN_REGISTRY})

#: datasets whose features are (mostly) discrete -- used by the
#: categorical/multinomial experiments.
CATEGORICAL_DATASETS = [name for name, ft in FEATURE_TYPE.items()
                        if ft == "categorical"]


def _cache_path(pmlb_name: str) -> str:
    return os.path.join(DATA, f"{pmlb_name}.tsv.gz")


def download_pmlb(pmlb_name: str, force: bool = False) -> str:
    """Download one PMLB dataset into the local cache and return its path."""
    path = _cache_path(pmlb_name)
    if os.path.exists(path) and not force and os.path.getsize(path) > 200:
        return path
    url = PMLB_URL.format(name=pmlb_name)
    with urllib.request.urlopen(url, timeout=120) as resp:
        blob = resp.read()
    # sanity: must be a gzip stream, not an LFS pointer or a 404 page
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as fh:
        fh.read(64)
    with open(path, "wb") as fh:
        fh.write(blob)
    return path


def load_pmlb(pmlb_name: str):
    """Return ``(X, y)`` of one cached PMLB dataset as float arrays."""
    df = pd.read_csv(_cache_path(pmlb_name), sep="\t", compression="gzip")
    y = df["target"].to_numpy()
    X = df.drop(columns=["target"]).to_numpy(dtype=float)
    # PMLB encodes missing values as NaN in a handful of medical sets
    if np.isnan(X).any():
        col_median = np.nanmedian(X, axis=0)
        col_median = np.where(np.isnan(col_median), 0.0, col_median)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_median, idx[1])
    # drop constant columns (they carry no evidence and break rank measures)
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    _, y = np.unique(y, return_inverse=True)
    return X, y


def stratified_subsample(X, y, max_n: int = MAX_N, seed: int = SUBSAMPLE_SEED):
    """Class-proportional subsample used for the largest datasets."""
    n = X.shape[0]
    if n <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(2, np.round(counts / n * max_n).astype(int))
    for c, q in zip(classes, quota):
        pool = np.flatnonzero(y == c)
        idx.append(rng.choice(pool, size=min(q, pool.size), replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]


def get_large_datasets() -> dict:
    """name -> (X, y) for the scalability study, with **no** subsampling."""
    out = {}
    for pmlb_name, disp, _, _ in LARGE_REGISTRY:
        X, y = load_pmlb(pmlb_name)
        out[disp] = (X, y)
    return out


def fetch_all(verbose: bool = True) -> None:
    """Populate the local cache (one network round-trip per dataset)."""
    registry = PMLB_REGISTRY + [r for r in LARGE_REGISTRY
                                if r[0] not in {x[0] for x in PMLB_REGISTRY}]
    for pmlb_name, disp, _, _ in registry:
        try:
            path = download_pmlb(pmlb_name)
            if verbose:
                size = os.path.getsize(path) / 1024
                print(f"  [ok]   {disp:18s} <- {pmlb_name} ({size:.0f} kB)")
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  [FAIL] {disp:18s} <- {pmlb_name}: {exc}")


def get_real_datasets(max_n: int = MAX_N) -> dict:
    """name -> (X, y) for the thirty-five real-world benchmarks."""
    out = {}
    for loader, name in (
        (load_iris, "iris"),
        (load_wine, "wine"),
        (load_breast_cancer, "breast-cancer"),
        (load_digits, "digits"),
    ):
        bunch = loader()
        out[name] = (np.asarray(bunch.data, dtype=float), np.asarray(bunch.target))
    for pmlb_name, disp, _, _ in PMLB_REGISTRY:
        X, y = load_pmlb(pmlb_name)
        out[disp] = stratified_subsample(X, y, max_n)
    return out


if __name__ == "__main__":
    fetch_all()
    print()
    for name, (X, y) in get_real_datasets().items():
        print(f"{name:18s} n={X.shape[0]:6d}  d={X.shape[1]:4d}  "
              f"K={len(np.unique(y)):3d}  domain={DOMAINS[name]}")
