"""Adaptive Bayesian Dependence Naive Bayes (ABD-NB).

Reference implementation accompanying the paper
"Adaptive Bayesian Dependence Naive Bayes: Dependence-Adaptive
Likelihood Weighting for Robust Probabilistic Classification".
"""

from .baselines import TAN, KDB, AODE, HNB, WANBIA, CorrelationWeightedNB, CFWNB
from .categorical import CategoricalABDNB, CategoricalNB_
from .classifier import ABDNB, MIWeightedNB, OnlineABDNB
from .discretize import EqualFrequencyDiscretizer
from .dependence import dependence_matrix, class_conditional_dependence, MEASURES
from .weights import weights_from_matrix, aggregate_dependence, WEIGHT_FUNCTIONS

__version__ = "2.0.0"

__all__ = [
    "ABDNB",
    "CategoricalABDNB",
    "CategoricalNB_",
    "TAN",
    "KDB",
    "AODE",
    "HNB",
    "CFWNB",
    "WANBIA",
    "CorrelationWeightedNB",
    "EqualFrequencyDiscretizer",
    "MIWeightedNB",
    "OnlineABDNB",
    "dependence_matrix",
    "class_conditional_dependence",
    "weights_from_matrix",
    "aggregate_dependence",
    "MEASURES",
    "WEIGHT_FUNCTIONS",
]
