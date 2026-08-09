# ABD-NB — Implementation

Reference implementation of **Adaptive Bayesian Dependence Naive Bayes**
(scikit-learn compatible) and all experiments of the paper.

## Installation

```bash
pip install -r requirements.txt
```

## Package layout

| Module | Contents |
|---|---|
| `abdnb/dependence.py` | Pairwise dependence estimators (Pearson, Spearman, Kendall, normalised MI, bias-corrected Cramér's V, distance correlation, HSIC), asymptotic soft-thresholding, class-conditional + pooled within-class matrices |
| `abdnb/weights.py` | Weight link functions (linear, exponential, inverse, harmonic graph rule with squared-dependence degree), spectral effective dimension `M_eff` (Li & Ji), rescaling |
| `abdnb/classifier.py` | `ABDNB` (batch, with internal (γ, structure) selection), `MIWeightedNB` (relevance-weighted baseline), `OnlineABDNB` (exponentially weighted streaming variant) |

## Quick start

```python
from abdnb import ABDNB

clf = ABDNB()                          # flagship: gamma='auto', structure='auto'
clf.fit(X, y)
proba = clf.predict_proba(X_new)

clf.gamma_                # selected correction strength
clf.class_specific_       # selected structure
clf.weights_              # (n_classes, n_features) adaptive exponents
clf.effective_dimension_  # sum of weights per class (d_eff)
```

Fixed configurations used as ablations in the paper:

```python
ABDNB(gamma=1.0, class_specific=False)            # ABD-NB-G
ABDNB(gamma=1.0, class_specific=True)             # ABD-NB-C
ABDNB(weight_fn="linear", gamma=1.0,
      class_specific=False)                       # ABD-NB-L
ABDNB(measure="dcor", gamma=1.0)                  # swap the dependence measure
```

## Reproducing the paper

Each script writes CSVs to `results/`; `make_figures.py` renders every PDF
figure of the paper into `figures/`.

```bash
python experiments/run_benchmark.py     # 15 datasets x 11 models x 3x10-fold CV
python experiments/run_synthetic.py     # equicorrelation & conflict sweeps
python experiments/run_theory.py        # duplicate recovery, consistency, learning curves
python experiments/run_sensitivity.py   # gamma / weight-function / measure ablations
python experiments/run_redundancy.py    # duplication stress on real data
python experiments/run_drift.py         # dependence-drift stream experiment
python experiments/run_stats.py         # Friedman / Nemenyi / Wilcoxon-Holm analyses
python experiments/make_figures.py      # all 20 figures (PDF)
```

Total runtime on a 4-core machine: roughly 30–40 minutes, dominated by the
benchmark.

## Mapping figures to scripts

| Figure (paper) | File | Produced by |
|---|---|---|
| Dependence heatmaps | `fig_dependence_heatmaps.pdf` | `make_figures.py` (direct) |
| Weight links / profile | `fig_weight_functions.pdf`, `fig_weights_profile.pdf` | `make_figures.py` (direct) |
| ρ / conflict sweeps | `fig_rho_study.pdf`, `fig_conflict_study.pdf` | `run_synthetic.py` |
| Duplication stress | `fig_redundancy.pdf` | `run_redundancy.py` |
| Theory checks | `fig_duplicates.pdf`, `fig_consistency.pdf`, `fig_learning_curve.pdf` | `run_theory.py` |
| Benchmark | `fig_benchmark_heatmap.pdf`, `fig_nb_family_bars.pdf`, `fig_boxplots.pdf`, `fig_cd_diagram.pdf` | `run_benchmark.py` + `run_stats.py` |
| Calibration | `fig_calibration_benchmark.pdf`, `fig_reliability.pdf` | `run_benchmark.py` (direct) |
| Ablations | `fig_gamma_sensitivity.pdf`, `fig_weightfn.pdf`, `fig_measures.pdf` | `run_sensitivity.py` |
| Streams | `fig_drift.pdf` | `run_drift.py` |
| Runtime | `fig_runtime.pdf` | `run_benchmark.py` |
