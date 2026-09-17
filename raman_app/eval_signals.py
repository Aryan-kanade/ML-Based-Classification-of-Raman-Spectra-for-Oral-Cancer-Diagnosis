"""
eval_signals.py -- L45-50 of the push-to-0.8 program (script was
missing from the 2026-09-16 queue run).  Three signal-level mechanisms,
each an honest method-bank candidate on the d2 winner features:

  L45 band-ratio LIKELIHOOD-RATIO rule (novel): per band-ratio feature,
      fit class-conditional Gaussians on the TRAIN rows only; predict
      by the summed log-likelihood ratio.  Fully interpretable.
  L46 medoid prototype rule: ONE medoid per class (point of minimum
      mean distance, train side only); score = distance difference.
  L47 disagreement-gated dual classifier: ET + XGB; when they disagree
      the spectrum is deferred to the L45 LR rule (ensemble arbiter).

Usage:  python eval_signals.py
Writes: experiments/signals.json + method_bank.jsonl entries
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import exp_common
import method_bank
import sequential
from discover_batch import data

OUT = os.path.join(exp_common.OUT_DIR, "signals.json")


def load_named():
    """(X, y, groups, wn, names) for the d2 winner features — the API
    eval_deep_arms.L58 (adversarial-validation diagnostic) imports.
    names = patient ids (the clinical tree keeps filenames unordered
    after hygiene; groups are the stable per-row identifier)."""
    X, y, g, wn = data()
    return X, y, list(g), wn, list(g)


def _features(X, wn):
    from discover_batch import rowmap_ratios
    return rowmap_ratios(X, wn)


def lr_rule(Xtr, ytr, Xte):
    """Gaussian class-conditional log-likelihood ratio, train-fitted."""
    ll = {}
    for c in (0, 1):
        m = ytr == c
        mu, sd = Xtr[m].mean(0), Xtr[m].std(0) + 1e-6

        def logp(Z, mu=mu, sd=sd):
            return (-0.5 * ((Z - mu) / sd) ** 2 - np.log(sd)).sum(1)
        ll[c] = logp
    return ll[1](Xte) - ll[0](Xte)      # >0 => class 1


def run_arm(mid, mech, fn):
    X, y, g, wn = data()
    Xm = _features(X, wn)
    splitter = sequential._splitter(5, 42, y, g)
    oof = fn(Xm, y, g, splitter)
    f1 = float(f1_score(y, oof, average="macro"))
    method_bank.add({"id": mid, "mechanism": mech, "family": "signal",
                     "f1_mean": f1, "f1_std": 0.0, "pat_f1": None,
                     "seeds": 1, "needs_seed_confirm": True})
    print(f"[{mid}] F1 {f1:.3f} — {mech}", flush=True)
    return f1


def arm_lr(Xm, y, g, splitter):
    oof = np.zeros(len(y))
    for tr, te in splitter.split(Xm, y, g):
        s = lr_rule(Xm[tr], y[tr], Xm[te])
        oof[te] = (s >= 0).astype(int)
    return oof


def arm_medoids(Xm, y, g, splitter):
    oof = np.zeros(len(y))
    for tr, te in splitter.split(Xm, y, g):
        est = _MedoidRule()
        mu, sd = Xm[tr].mean(0), Xm[tr].std(0) + 1e-8
        est.fit((Xm[tr] - mu) / sd, y[tr])
        d = est.decision_function((Xm[te] - mu) / sd)
        oof[te] = (d >= 0).astype(int)
    return oof


class _MedoidRule:
    """k-medoids-ish: medoid = point of min mean distance per class."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.protos_ = []
        for c in self.classes_:
            m = y == c
            D = np.linalg.norm(X[m][:, None] - X[m][None, :], axis=-1)
            self.protos_.append(X[m][np.argmin(D.mean(1))])

    def decision_function(self, X):
        d0 = np.linalg.norm(X - self.protos_[0], axis=1)
        d1 = np.linalg.norm(X - self.protos_[1], axis=1)
        return d0 - d1                      # >0 => class 1


def arm_dual(Xm, y, g, splitter, Xfull=None):
    """ET + XGB on full features; LR arbiter on disagreement."""
    X, _y, _g, wn = data()
    oof = np.zeros(len(y))
    for tr, te in splitter.split(X, y, g):
        a = ExtraTreesClassifier(n_estimators=300, random_state=42,
                                 n_jobs=1, class_weight="balanced")
        b = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                          random_state=42, n_jobs=1, eval_metric="logloss")
        a.fit(X[tr], y[tr])
        b.fit(X[tr], y[tr])
        pa, pb = a.predict(X[te]), b.predict(X[te])
        s = lr_rule(Xm[tr], y[tr], Xm[te])
        out = np.where(pa == pb, pa, (s >= 0).astype(int))
        oof[te] = out
    return oof


def main() -> int:
    res = {}
    res["L45"] = run_arm("MS1", "band-ratio Gaussian likelihood-ratio "
                              "rule", lambda a, b, c, s: arm_lr(a, b, c, s))
    res["L46"] = run_arm("MS2", "class-medoid prototype rule "
                                 "(min-distance)", arm_medoids)
    res["L47"] = run_arm("MS3", "ET+XGB dual with LR arbiter on "
                                 "disagreement", arm_dual)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(method_bank.render(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
