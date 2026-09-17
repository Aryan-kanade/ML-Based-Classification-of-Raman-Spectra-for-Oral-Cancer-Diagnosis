"""
eval_closers.py -- L28 SAM + cosine-kNN, L31 keratin/site
residualization, L33 chain-input t-test gate (SAFETY: pure
experiment script -> experiments/ only).  Cheap arms closing the
classical-method space; all scored by the same grouped protocol.
"""

from __future__ import annotations

import os

import numpy as np

import sequential
from exp_common import OUT_DIR, load_data, record, stamp
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier


class SAMClassifier(BaseEstimator, ClassifierMixin):
    """Spectral Angle Mapper: class prototypes by mean spectrum;
    classify by smallest spectral angle."""

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        self.protos_ = np.vstack(
            [np.asarray(X)[np.asarray(y) == c].mean(axis=0)
             for c in self.classes_])
        return self

    def predict(self, X):
        X = np.asarray(X)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        Pn = self.protos_ / (np.linalg.norm(self.protos_, axis=1,
                                            keepdims=True) + 1e-9)
        cos = Xn @ Pn.T
        return self.classes_[np.argmax(cos, axis=1)]


class FilterGate(BaseEstimator, ClassifierMixin):
    """L33: keep top-k columns by train-fold t-statistic, then fit
    the wrapped estimator on the sliced matrix."""

    def __init__(self, est=None, k=100):
        self.est = est
        self.k = k

    def fit(self, X, y, groups=None):
        X = np.asarray(X)
        y = np.asarray(y)
        pos = X[y == 1].mean(axis=0)
        neg = X[y == 0].mean(axis=0)
        v1 = X[y == 1].var(axis=0) + 1e-9
        v0 = X[y == 0].var(axis=0) + 1e-9
        n1, n0 = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
        t = np.abs(pos - neg) / np.sqrt(v1 / n1 + v0 / n0)
        self.cols_ = np.argsort(t)[::-1][:self.k]
        self.est_ = clone(self.est)
        self.est_.fit(X[:, self.cols_], y)
        self.classes_ = self.est_.classes_
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(np.asarray(X)[:, self.cols_])

    def predict(self, X):
        return self.est_.predict(np.asarray(X)[:, self.cols_])


def residualize_in_fold(F_tr, F_te, groups_tr):
    """L31: regress each column on [keratin proxy, patient-cluster
    mean-P proxy]; train on residuals.  Keratin proxy = amplitude in
    the 930-945 cm-1 window of the deviation itself; covariate 2 =
    row norm (coarse site/patient severity proxy)."""
    ker_tr = np.abs(F_tr).mean(axis=1)
    ker_te = np.abs(F_te).mean(axis=1)
    C_tr = np.column_stack([np.ones(len(F_tr)), ker_tr,
                            np.linalg.norm(F_tr, axis=1)])
    C_te = np.column_stack([np.ones(len(F_te)), ker_te,
                            np.linalg.norm(F_te, axis=1)])
    B, *_ = np.linalg.lstsq(C_tr, F_tr, rcond=None)
    return F_tr - C_tr @ B, F_te - C_te @ B


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    arch, X, y, groups, wn, _meta = load_data()
    X = np.asarray(X, dtype=np.float32)
    y_arr = np.searchsorted(sorted(set(y)), np.asarray(y))
    groups = np.asarray(groups)
    classes = sorted(set(y))
    factories = sequential._factories(wn)
    print(f"[clo] {stamp()} arch {' -> '.join(arch)}", flush=True)

    def chain_predict(F_tr, F_te, y_tr, g_tr, seed, facs):
        # facs is a parameter (audit fix 2026-09-16): the old closure
        # ignored run_arm's est_swap, so the t-gate arm silently
        # re-evaluated the plain winner chain.
        k_in = max(2, min(3, int(min(np.bincount(y_tr)))))
        k_in = min(k_in, len(set(g_tr.tolist())))
        for pos, name in enumerate(arch[:-1]):
            P_tr = sequential._oof_proba(
                sequential._layer_est(name, facs, pos, wn),
                F_tr, list(y_tr), g_tr, k_in, seed)
            est = sequential.fit_maybe_grouped(
                sequential._layer_est(name, facs, pos, wn),
                F_tr, y_tr, g_tr)
            F_tr = np.hstack([F_tr, P_tr])
            F_te = np.hstack([F_te, est.predict_proba(F_te)])
        last = sequential.fit_maybe_grouped(
            sequential._layer_est(arch[-1], facs, len(arch) - 1,
                                  wn), F_tr, y_tr, g_tr)
        return last.predict_proba(F_te)

    def run_arm(name, transform=None, est_swap=None):
        outer = sequential._splitter(5, 42, y_arr, groups)
        oof = np.full((len(y_arr), len(classes)), np.nan)
        facs = dict(factories)
        if est_swap:
            facs.update(est_swap)
        for tr, te in outer.split(X, y_arr, groups):
            F_tr, F_te = X[tr].copy(), X[te].copy()
            if transform is not None:
                F_tr, F_te = transform(F_tr, F_te, groups[tr])
            oof[te] = chain_predict(F_tr, F_te, y_arr[tr], groups[tr],
                                    42, facs)
        v = ~np.isnan(oof).any(axis=1)
        f1 = float(f1_score(y_arr[v], oof[v].argmax(axis=1),
                            average="macro"))
        print(f"[clo] {name}: F1 {f1:.3f}", flush=True)
        record("closers", name, f1)
        return f1

    # L31 residualization
    run_arm("L31 residualize (keratin+norm)", residualize_in_fold)

    # L33 t-gate: FilterGate-wrapped RF substituted as the final layer
    # (t-test feature gating inside every fit)
    rf = next(s for s in __import__("modeling").model_specs()
              if s["name"] == "Random Forest")["estimator"]
    gate_facs = dict(factories)
    gate_facs["Extra Trees"] = lambda: FilterGate(
        clone(rf), k=200)
    run_arm("L33 t-gate k=200 (gated RF final)", None, gate_facs)

    # L28 SAM + cosine-kNN standalone (not chained — direct arms)
    for nm, est in (("L28 SAM", SAMClassifier()),
                    ("L28 cosine-kNN k=7",
                     KNeighborsClassifier(n_neighbors=7,
                                          metric="cosine"))):
        outer = sequential._splitter(5, 42, y_arr, groups)
        preds = np.full(len(y_arr), -1)
        for tr, te in outer.split(X, y_arr, groups):
            e = clone(est)
            e.fit(X[tr], y_arr[tr])
            preds[te] = e.predict(X[te])
        f1 = float(f1_score(y_arr[preds >= 0], preds[preds >= 0],
                            average="macro"))
        print(f"[clo] {nm}: F1 {f1:.3f}", flush=True)
        record("closers", nm, f1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
