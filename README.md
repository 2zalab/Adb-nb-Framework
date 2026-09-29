# ABD-NB — Implementation

Reference implementation of **Adaptive Bayesian Dependence Naive Bayes**
(scikit-learn compatible) and all experiments of the paper
*"Correcting Redundant Evidence in Naïve Bayes: Dependence-Adaptive
Likelihood Weighting with Finite-Sample Analysis"*
(Knowledge-Based Systems, KNOSYS-D-26-21554).

## Installation

```bash
pip install -r requirements.txt
```

The real-world benchmark suite is downloaded once and cached under `data/`:

```bash
python experiments/real_datasets.py     # fetch + summarise the 37 real datasets
```

## Package layout

| Module | Contents |
|---|---|
| `abdnb/dependence.py` | Pairwise dependence estimators (Pearson, Spearman, Kendall, normalised MI, bias-corrected Cramér's V, distance correlation, HSIC), asymptotic soft-thresholding, class-conditional + pooled within-class matrices, and their discrete counterparts |
| `abdnb/weights.py` | Weight link functions (linear, exponential, inverse, harmonic graph rule with dependence degree exponent κ, default 2), spectral effective dimension `M_eff` (Li & Ji), rescaling |
| `abdnb/classifier.py` | `ABDNB` (batch, with internal (γ, structure, rescaling) selection over 21 configurations), `MIWeightedNB` (relevance-weighted baseline), `OnlineABDNB` (exponentially weighted streaming variant) |
| `abdnb/categorical.py` | `CategoricalABDNB` / `CategoricalNB_`: the multinomial instantiation used for the categorical, sequence and text-derived benchmarks |
| `abdnb/baselines.py` | `TAN`, `KDB`, `AODE`, `HNB` (semi-naive Bayesian network classifiers), `WANBIA` (discriminative weight optimisation), `CorrelationWeightedNB` and `CFWNB` (relevance/redundancy filters) |
| `abdnb/discretize.py` | Equal-frequency discretisation shared by the discrete-factor models |

## Quick start

```python
from abdnb import ABDNB

clf = ABDNB()             # flagship: gamma, structure and rescaling all 'auto'
clf.fit(X, y)
proba = clf.predict_proba(X_new)

clf.gamma_                # selected correction strength
clf.class_specific_       # selected weight structure
clf.rescale_              # selected evidence-budget rule ('meff' or 'none')
clf.weights_              # (n_classes, n_features) fitted exponents
clf.effective_dimension_  # sum of weights per class (d_eff)
```

Before enabling the correction, the **redundancy index** tells you whether it can
help at all (Section 7.5 of the paper); below `R ≈ 0.05` it will not:

```python
from abdnb.dependence import class_conditional_dependence
from abdnb.weights import effective_dimension

_, pooled = class_conditional_dependence(X, y, "spearman")
R = 1.0 - effective_dimension(pooled) / X.shape[1]
```

Fixed configurations used as ablations in the paper:

```python
ABDNB(gamma=1.0, class_specific=False)            # ABD-NB-G
ABDNB(gamma=1.0, class_specific=True)             # ABD-NB-C
ABDNB(weight_fn="linear", gamma=1.0,
      class_specific=False)                       # ABD-NB-L
ABDNB(measure="dcor", gamma=1.0)                  # swap the dependence measure
ABDNB(rescale="none")                             # never restore the budget
ABDNB(rescale="meff")                             # always restore it
CategoricalABDNB()                                # multinomial factors
```

## Reproducing the paper

Each script writes CSVs to `results/`; `make_tables.py` emits every LaTeX table
body to `results/tables/` and `make_figures.py` renders every PDF figure into
`figures/`. Both are consumed verbatim by the manuscript, so no number in the
paper is typed by hand.

```bash
python experiments/run_benchmark.py          # 48 settings x 22 models x 3x10-fold CV
python experiments/run_composition.py        # {NB, ABD-NB} x {raw, Platt, isotonic}
python experiments/run_categorical.py        # categorical / sequence / text-derived data
python experiments/run_redundancy_profile.py # redundancy index vs realised gain (needs benchmark)
python experiments/run_conditions.py         # operating conditions: helps / parity / hurts
python experiments/run_finite_sample.py      # deviation bound + selection regret
python experiments/run_synthetic.py          # equicorrelation & conflict sweeps
python experiments/run_theory.py             # duplicate recovery, consistency, learning curves
python experiments/run_sensitivity.py        # gamma / weight-function / measure ablations
python experiments/run_redundancy.py         # duplication stress on ten real datasets
python experiments/run_drift.py              # dependence-drift stream experiment
python experiments/run_stats.py              # Friedman / Nemenyi / Wilcoxon-Holm analyses
python experiments/run_ablation_components.py # per-stage ablation (changed the method)
python experiments/run_tuned_baselines.py    # matched tuning protocol for every model
python experiments/run_redundancy_robustness.py # R across 7 measures, bootstrap, n
python experiments/run_higher_order.py       # parity / masked-duplicate / 3-way designs
python experiments/run_scalability.py        # cost curves;  `full` for the 18 full-size sets
python experiments/run_degeneracy.py         # why the Gaussian factor fails where it fails
python experiments/run_weight_range.py       # observed range of the fitted exponents
python experiments/run_gamma_selection.py    # what the selector chooses, per fold
python experiments/make_tables.py            # all LaTeX tables
python experiments/make_figures.py           # all figures (PDF + 300 dpi PNG)
python experiments/verify_claims.py          # recompute every number in the paper
python check_api.py                          # scikit-learn estimator contract
```

`verify_claims.py` is the guard that keeps the manuscript honest: it recomputes
every numeric claim in the paper from the CSVs in `results/` and exits non-zero
on any mismatch. It currently checks 402 claims. If you change an experiment,
run it before touching the text.

`run_benchmark.py` parallelises over folds (`n_jobs=4` by default) and dominates
the total runtime; the full pipeline takes roughly 90 minutes on a 4-core machine.

## Mapping figures to scripts

| Figure (paper) | File | Produced by |
|---|---|---|
| Dependence heatmaps | `fig_dependence_heatmaps.pdf` | `make_figures.py` (direct) |
| Weight links / profile | `fig_weight_functions.pdf`, `fig_weights_profile.pdf` | `make_figures.py` (direct) |
| Benchmark | `fig_benchmark_heatmap.pdf`, `fig_nb_family_bars.pdf`, `fig_rank_summary.pdf`, `fig_boxplots.pdf`, `fig_cd_diagram.pdf` | `run_benchmark.py` + `run_stats.py` |
| Calibration | `fig_calibration_benchmark.pdf`, `fig_reliability.pdf` | `run_benchmark.py` |
| Redundancy profile | `fig_redundancy_profile.pdf` | `run_redundancy_profile.py` |
| Operating conditions | `fig_conditions.pdf` | `run_conditions.py` |
| Finite-sample | `fig_finite_sample.pdf` (the paper's Figure 25; `fig_selection_regret.pdf` is also produced but the paper reports that study as a table) | `run_finite_sample.py` |
| Categorical | `fig_categorical.pdf` | `run_categorical.py` |
| ρ / conflict sweeps | `fig_rho_study.pdf`, `fig_conflict_study.pdf` | `run_synthetic.py` |
| Duplication stress | `fig_redundancy.pdf` | `run_redundancy.py` |
| Theory checks | `fig_duplicates.pdf`, `fig_consistency.pdf`, `fig_learning_curve.pdf` | `run_theory.py` |
| Ablations | `fig_gamma_sensitivity.pdf`, `fig_weightfn.pdf`, `fig_measures.pdf` | `run_sensitivity.py` |
| Streams | `fig_drift.pdf` | `run_drift.py` |
| Runtime | `fig_runtime.pdf` | `run_benchmark.py` |

Figure 1 of the paper (the architecture diagram) is drawn in TikZ inside the
manuscript and rasterised by `adb-elsevier/make_architecture_png.py`, so it does
not pass through `make_figures.py`.
