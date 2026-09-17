"""
eval_agg_batch.py -- Batch E of the discovery loop: patient-aggregation
variants evaluated honestly (LOO-tuned thresholds) on the cached winner
OOF.  Pure numpy on experiments/oof_cache.npz — no training, seconds.

Usage:  python eval_agg_batch.py
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

import method_bank
import pat_common


def _loo(y, p):
    pred = np.empty_like(y)
    for i in range(y.size):
        tr = np.arange(y.size) != i
        best_t, best_f1 = 0.5, -1.0
        for t in np.unique(p[tr]):
            f1 = f1_score(y[tr], (p[tr] >= t).astype(int), average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        pred[i] = int(p[i] >= best_t)
    return f1_score(y, pred, average="macro")


def main() -> int:
    ye, proba, g = pat_common.winner_oof()
    p1 = proba[:, 1]

    def table(score_fn):
        pats = sorted(set(g.tolist()))
        ys, ps = [], []
        for pat in pats:
            m = g == pat
            ps.append(score_fn(p1[m]))
            vals, cnts = np.unique(ye[m], return_counts=True)
            ys.append(int(vals[np.argmax(cnts)]))
        return np.asarray(ys), np.asarray(ps)

    arms = {
        "ME1": ("median-of-logit aggregation", lambda v: float(
            np.median(np.log(np.clip(v, 1e-6, 1 - 1e-6)
                              / np.clip(1 - v, 1e-6, 1))))),
        "ME2": ("winsorized-mean aggregation (clip .1-.9)", lambda v:
                float(np.clip(v, 0.1, 0.9).mean())),
        "ME3": ("mean of 3 most-confident replicates", lambda v: float(
            v[np.argsort(-np.abs(v - 0.5))[:3]].mean())),
        "ME4": ("mean of 3 least-confident replicates", lambda v: float(
            v[np.argsort(np.abs(v - 0.5))[:3]].mean())),
        "ME5": ("geometric-mean aggregation", lambda v: float(
            np.exp(np.log(np.clip(v, 1e-6, 1)).mean()))),
        "ME6": ("midhinge aggregation (q25+q75)/2", lambda v: float(
            (np.quantile(v, .25) + np.quantile(v, .75)) / 2)),
    }
    for mid, (mech, fn) in arms.items():
        if mid in method_bank.bank_ids():
            continue
        y, p = table(fn)
        pf1 = _loo(y, p)
        method_bank.add({"id": mid, "mechanism": mech + " (LOO-tuned)",
                         "family": "aggregation", "f1_mean": None,
                         "f1_std": None, "pat_f1": float(pf1),
                         "seeds": 1, "needs_seed_confirm": False,
                         "note": "cache-based, honest LOO thresholds"})
        print(f"[{mid}] patient F1 {pf1:.3f} — {mech}", flush=True)
    print(method_bank.render(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
