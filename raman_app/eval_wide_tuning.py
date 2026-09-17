"""
eval_wide_tuning.py -- L12: wide hyperparameter tuning, INSIDE the
nested protocol.  The app's grids are deliberately tiny (RF/ET: 2x2,
n_estimators fixed); this wraps the RF/ET layers in GridSearchCV with
a meaningfully wider grid, grouped inner CV, so the tuning happens
inside every outer fold of validate_arch (no leakage).
SAFETY: pure experiment script -> experiments/ only.
"""

from __future__ import annotations

import os

import modeling
import sequential
from exp_common import OUT_DIR, load_data, oof_f1, record, stamp
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold


class _GroupedGS(BaseEstimator, ClassifierMixin):
    """GridSearchCV whose inner CV is PATIENT-GROUPED: sklearn 1.9's
    metadata routing silently drops `groups` before the internal
    splitter (which then saw ONE pseudo-group and crashed), so this
    wrapper builds explicit grouped index splits in fit() instead."""

    def __init__(self, est, grid):
        self.est = est
        self.grid = grid

    def fit(self, X, y=None, groups=None, **params):
        if groups is None:
            cv = 3
        else:
            cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True,
                                           random_state=0
                                           ).split(X, y, groups))
        self.gs_ = GridSearchCV(clone(self.est), self.grid,
                                scoring="f1_macro", n_jobs=1, cv=cv,
                                refit=True).fit(X, y)
        self.classes_ = self.gs_.classes_
        return self

    def predict(self, X):
        return self.gs_.predict(X)

    def predict_proba(self, X):
        return self.gs_.predict_proba(X)


def _gs_factory(est, grid):
    def make():
        return _GroupedGS(clone(est), grid)
    return make


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    arch, X, y, groups, wn, meta = load_data()
    print(f"[tune] {stamp()} arch {' -> '.join(arch)} · {X.shape[0]} "
          "rows", flush=True)
    facs = sequential._factories(wn)

    rf = modeling.clone(next(s for s in modeling.model_specs()
                             if s["name"] == "Random Forest")["estimator"])
    et = modeling.clone(next(s for s in modeling.model_specs()
                             if s["name"] == "Extra Trees")["estimator"])
    wide = {
        "clf__n_estimators": [250, 500],
        "clf__max_depth": [None, 8, 16],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features": ["sqrt", 0.3],
    }
    for name, est in (("Random Forest", rf), ("Extra Trees", et)):
        grid = {k.replace("clf__", ""): v for k, v in wide.items()}
        if hasattr(est, "named_steps"):
            grid = {f"clf__{k}": v for k, v in grid.items()}
        facs[name] = _gs_factory(est, grid)

    m = sequential.validate_arch(arch, facs, X, y, groups, seed=42,
                                 k_outer=5, wn=wn)
    f1 = oof_f1(m)
    print(f"[tune] wide-tuned F1 {f1:.3f} (baseline 0.753)", flush=True)
    record("L12 wide-tuning", "RF/ET wide grids in-fold", f1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
