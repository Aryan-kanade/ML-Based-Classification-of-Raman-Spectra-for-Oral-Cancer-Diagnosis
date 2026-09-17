"""
eval_augment.py -- L13: physics-safe augmentation for the tree chain
layers.  modeling.lorentzian_synthesize (convex same-class blends +
smooth gain/tilt) is wrapped in an in-fold augmenting estimator: fit()
appends ~60% synthetic same-class rows, then fits the base model.
Leakage-free by construction (augmentation happens inside fit, i.e.
inside every outer fold of validate_arch).
SAFETY: pure experiment script -> experiments/ only.
"""

from __future__ import annotations

import os

import numpy as np

import modeling
import sequential
from exp_common import OUT_DIR, load_data, oof_f1, record, stamp
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class _AugWrap(BaseEstimator, ClassifierMixin):
    """Fit `est` on the training rows + lorentzian_synthesize blends."""

    def __init__(self, est=None, frac=0.6, seed=0):
        self.est = est
        self.frac = frac
        self.seed = seed

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_new = int(len(X) * self.frac)
        pieces = [X]
        piece_y = [y]
        rng = np.random.default_rng(self.seed)
        for c in np.unique(y):
            Xc = X[y == c]
            k = int(round(n_new * len(Xc) / max(len(X), 1)))
            if k > 0 and len(Xc) >= 2:
                pieces.append(modeling.lorentzian_synthesize(
                    Xc, k, seed=int(rng.integers(1 << 30))))
                piece_y.append(np.full(k, c))
        Xa = np.vstack(pieces)
        ya = np.concatenate(piece_y)
        self.est_ = clone(self.est)
        self.est_.fit(Xa, ya)
        self.classes_ = self.est_.classes_
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(X)

    def predict(self, X):
        return self.est_.predict(X)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    arch, X, y, groups, wn, meta = load_data()
    print(f"[aug] {stamp()} arch {' -> '.join(arch)}", flush=True)
    facs = sequential._factories(wn)
    for name in ("Random Forest", "Extra Trees"):
        base = facs[name]
        facs[name] = (lambda b: lambda: _AugWrap(b(), frac=0.6))(base)
    m = sequential.validate_arch(arch, facs, X, y, groups, seed=42,
                                 k_outer=5, wn=wn)
    f1 = oof_f1(m)
    print(f"[aug] augmented F1 {f1:.3f} (baseline 0.753)", flush=True)
    record("L13 augmentation", "lorentzian blends 60% in RF/ET layers",
           f1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
