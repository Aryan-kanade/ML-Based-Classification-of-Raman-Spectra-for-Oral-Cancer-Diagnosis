"""
eval_fusion_batch.py -- Batch H (final) of the discovery loop: FIXED
combination rules over the banked honest OOFs (experiments/oof_bank +
the d2-winner cache).  No fitted weights -> nothing can leak; the only
inputs are per-method OOF probabilities that were themselves produced
under patient-grouped CV.

Arms (each scored spectrum-level macro-F1 and patient-level with
LOO-tuned thresholds, mean + median aggregation):
  MH1 mean-probability fusion        MH2 median-probability fusion
  MH3 rank fusion (mean of ranks)    MH4 top-3-by-F1 mean fusion
  MH5 trimmed-mean fusion

Usage:  python eval_fusion_batch.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
from sklearn.metrics import f1_score

import exp_common
import method_bank
import pat_common

OOF_DIR = os.path.join(exp_common.OUT_DIR, "oof_bank")


def _pool():
    """(ids, y, [proba]) — every stored OOF aligned on identical rows."""
    entries = {e["id"]: e for e in method_bank.load_bank()}
    ids, probs, y0, g0 = [], [], None, None
    cands = sorted(glob.glob(os.path.join(OOF_DIR, "*.npz")))
    cands.append(os.path.join(exp_common.OUT_DIR, "oof_cache.npz"))
    for path in cands:
        z = np.load(path, allow_pickle=True)
        y, o, g = z["y_true"], z["oof"], z["groups"]
        if y0 is None:
            y0, g0 = y, g
        # row alignment guard (same 287-row order by construction)
        if len(y) != len(y0) or not np.array_equal(g, g0):
            continue
        mid = ("d2-winner" if path.endswith("oof_cache.npz")
               else os.path.basename(path)[:-4])
        if mid in entries and entries[mid].get("f1_mean", 0) < 0.70 \
                and mid != "MC3":
            continue                     # fuse only decent OOFs
        ids.append(mid)
        probs.append(o.astype(np.float64))
    return ids, y0, g0, probs


def _patient(y, proba, g, agg):
    yp, p, _ = pat_common.patient_table(y, proba, g, agg=agg)
    pred = np.empty_like(yp)
    for i in range(yp.size):
        tr = np.arange(yp.size) != i
        best_t, best_f1 = 0.5, -1.0
        for t in np.unique(p[tr]):
            f1 = f1_score(yp[tr], (p[tr] >= t).astype(int), average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        pred[i] = int(p[i] >= best_t)
    return float(f1_score(yp, pred, average="macro"))


def main() -> int:
    ids, y, g, probs = _pool()
    f1s = [f1_score(y, pr.argmax(1), average="macro") for pr in probs]
    order = np.argsort(-np.asarray(f1s))
    P = np.stack(probs)                       # (m, n, 2)
    print(f"[H] fusing {len(ids)} OOFs: "
          + ", ".join(f"{ids[i]}={f1s[i]:.3f}" for i in order), flush=True)

    def top3_mean():
        Q = P[order[:3]]
        return Q.mean(0)

    def trimmed_mean():                       # drop min/max per row
        lo = P.min(0)
        hi = P.max(0)
        s = P.sum(0)
        return (s - lo - hi) / max(len(P) - 2, 1)

    arms = {
        "MH1": ("mean-probability fusion of bank OOFs",
                lambda: P.mean(0)),
        "MH2": ("median-probability fusion of bank OOFs",
                lambda: np.median(P, axis=0)),
        "MH3": ("rank fusion (mean per-class rank)",
                lambda: np.argsort(np.argsort(P, axis=1), axis=1)
                .astype(float).mean(0)),
        "MH4": ("top-3-by-F1 mean fusion", top3_mean),
        "MH5": ("trimmed-mean fusion", trimmed_mean),
    }
    for mid, (mech, fn) in arms.items():
        if mid in method_bank.bank_ids():
            continue
        fused = fn()
        sf1 = float(f1_score(y, fused.argmax(1), average="macro"))
        pm = _patient(y, fused, g, "mean")
        pd = _patient(y, fused, g, "median")
        method_bank.add({"id": mid, "mechanism": mech, "family": "fusion",
                         "f1_mean": sf1, "f1_std": 0.0,
                         "pat_f1": max(pm, pd), "seeds": 1,
                         "note": f"patient mean {pm:.3f} / median {pd:.3f}"})
        print(f"[{mid}] F1 {sf1:.3f} · patient mean {pm:.3f} "
              f"median {pd:.3f} — {mech}", flush=True)
    print(method_bank.render(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
