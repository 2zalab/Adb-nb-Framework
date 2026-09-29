"""Check that every estimator satisfies scikit-learn's estimator contract.

The paper claims a scikit-learn compatible implementation, so the claim is
checked rather than asserted: ``check_estimator`` runs the full contract
suite (cloning, parameter round-tripping, fitted-attribute conventions,
input validation, error messages) on each estimator.

``OnlineABDNB`` is excluded deliberately: it is a streaming model whose
constructor requires the class list and the feature count up front, so it
is not a plain estimator and ``check_estimator`` cannot instantiate it.

Run with ``python3 check_api.py``; exits non-zero on any failure.
"""

from __future__ import annotations

import sys
import warnings

from sklearn.utils.estimator_checks import check_estimator

from abdnb import ABDNB, AODE, CFWNB, HNB, KDB, TAN, WANBIA, CategoricalABDNB
from abdnb.baselines import CorrelationWeightedNB
from abdnb.classifier import MIWeightedNB

ESTIMATORS = [ABDNB, MIWeightedNB, WANBIA, CorrelationWeightedNB, CFWNB,
              AODE, HNB, TAN, KDB, CategoricalABDNB]


def main() -> int:
    warnings.filterwarnings("ignore")
    failed = []
    for cls in ESTIMATORS:
        try:
            check_estimator(cls())
            print(f"  {cls.__name__:22s} PASS")
        except Exception as exc:
            first = str(exc).split("\n")[0]
            print(f"  {cls.__name__:22s} FAIL  {first}")
            failed.append(cls.__name__)
    print()
    if failed:
        print(f"{len(failed)} of {len(ESTIMATORS)} estimators fail the contract: "
              + ", ".join(failed))
        return 1
    print(f"all {len(ESTIMATORS)} estimators satisfy the scikit-learn contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
