"""Adaptive Bayesian Dependence Naive Bayes (ABD-NB).

Reference implementation accompanying the paper
"Adaptive Bayesian Dependence Naive Bayes: Dependence-Adaptive
Likelihood Weighting for Robust Probabilistic Classification".
"""

from .classifier import ABDNB, MIWeightedNB, OnlineABDNB
from .dependence import dependence_matrix, class_conditional_dependence, MEASURES
from .weights import weights_from_matrix, aggregate_dependence, WEIGHT_FUNCTIONS

__version__ = "1.0.0"

__all__ = [
    "ABDNB",
    "MIWeightedNB",
    "OnlineABDNB",
    "dependence_matrix",
    "class_conditional_dependence",
    "weights_from_matrix",
    "aggregate_dependence",
    "MEASURES",
    "WEIGHT_FUNCTIONS",
]
