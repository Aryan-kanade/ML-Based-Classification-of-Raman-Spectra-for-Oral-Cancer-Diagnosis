"""
pat_common.py -- shared cache of the d2 winner's OOF predictions for the
patient-level repair arms (Stage 1 of the push-to-0.8 program v10).

Running sequential.validate_arch once costs minutes; every patient-level
arm (threshold tuning, GLMM, cascade) reuses experiments/oof_cache.npz.
SAFETY: writes experiments/ only. Do NOT run while the program queue is
running (one loky pool at a time — Brain gotcha #16).
"""

from __future__ import annotations

import os

import numpy as np

import exp_common
import sequential

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(exp_common.OUT_DIR, "oof_cache.npz")


def winner_oof(seed: int = 42, k: int = 5):
    """(y_true, oof_proba, groups) for the d2 winner; cached on disk."""
    if os.path.isfile(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["y_true"], z["oof"], z["groups"]
    arch, X, y, groups, wn, _meta = exp_common.load_data()
    factories = sequential._factories(wn)
    m = sequential.validate_arch(arch, factories, X, y, groups,
                                 seed=seed, k_outer=k, wn=wn)
    proba = np.asarray(m["oof_proba"], dtype=np.float64)
    ye = np.asarray(m["y_true"], dtype=np.int64)
    keep = ~np.isnan(proba).any(axis=1)
    ye, proba, g = ye[keep], proba[keep], np.asarray(groups)[keep]
    np.savez_compressed(CACHE, y_true=ye, oof=proba, groups=g)
    return ye, proba, g


def patient_table(y_true, proba, groups, agg: str = "mean"):
    """Per-patient (label, aggregated score).  agg: mean|median|trimmed|
    logit (mean of logits).  Label = dominant class (paired-design
    convention, sequential.py)."""
    eps = 1e-6
    proba = np.asarray(proba)
    if proba.ndim == 1:                 # raw scores already
        score_rows = proba.astype(np.float64)
    elif agg == "logit":
        p = np.clip(proba[:, 1], eps, 1 - eps)
        score_rows = np.log(p / (1 - p))
    else:
        score_rows = proba[:, 1]
    pats = sorted(set(groups.tolist()))
    ys, ps = [], []
    for pat in pats:
        msk = groups == pat
        s = score_rows[msk]
        if agg == "median":
            v = float(np.median(s))
        elif agg == "trimmed":
            lo = max(1, int(0.1 * s.size)) if s.size > 5 else 0
            v = float(np.sort(s)[lo:s.size - lo].mean()) \
                if s.size - 2 * lo > 0 else float(s.mean())
        else:
            v = float(s.mean())
        vals, cnts = np.unique(y_true[msk], return_counts=True)
        ys.append(int(vals[np.argmax(cnts)]))
        ps.append(v)
    return np.asarray(ys), np.asarray(ps), np.asarray(pats)
