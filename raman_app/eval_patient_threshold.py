"""
eval_patient_threshold.py -- L9 of the push-to-0.8 program (SAFETY:
pure experiment script; writes experiments/ only, light compute).

The clinic operates on PATIENTS, but the deployed threshold was tuned
at SPECTRUM level.  This script aggregates the winner's OOF
probabilities per patient (mean P), sweeps the decision threshold at
the PATIENT level, and reports the best operating point (patient F1 /
Youden) next to the default spectrum-level threshold.

Usage:  python eval_patient_threshold.py
Output: experiments/patient_threshold.json
"""

from __future__ import annotations

import json
import os

import numpy as np

import sequential
from run_3sse_d2 import _prepare_winner

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "experiments")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]
    X, y, groups, wn, _grid, _params, _meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    factories = sequential._factories(wn)
    m = sequential.validate_arch(arch, factories, X, list(y),
                                 list(groups), seed=42, k_outer=5, wn=wn)
    ye = np.asarray(m["y_true"])
    proba = np.asarray(m["oof_proba"])
    g = np.asarray(groups)[~np.isnan(proba).any(axis=1)]

    # per-patient mean P(positive)
    pats = sorted(set(g.tolist()))
    y_pat, p_pat = [], []
    for pat in pats:
        msk = g == pat
        p_pat.append(proba[msk, 1].mean())
        vals, cnts = np.unique(ye[msk], return_counts=True)
        y_pat.append(int(vals[np.argmax(cnts)]))
    P = np.asarray(p_pat)
    Y = np.asarray(y_pat)

    # spectrum-level deployed threshold for reference (winner.json;
    # null -> 0.5 default)
    from sklearn.metrics import f1_score
    thr_raw = json.load(open(
        os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
        encoding="utf-8"))["threshold"]
    spec_thr = (0.5 if thr_raw is None else float(thr_raw))
    pred_spec = (P >= spec_thr).astype(int)
    spec_pt = {"threshold": spec_thr,
               "f1": float(f1_score(Y, pred_spec, average="macro")),
               "acc": float((Y == pred_spec).mean())}

    # sweep patient-level thresholds
    grid = np.unique(np.concatenate([P, [0.05, 0.95]]))
    best_f1, best_thr, best_pt = -1.0, 0.5, None
    for t in grid:
        pred = (P >= t).astype(int)
        f1 = f1_score(Y, pred, average="macro")
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
            sens = float(((pred == 1) & (Y == 1)).sum() / max((Y == 1).sum(), 1))
            spec = float(((pred == 0) & (Y == 0)).sum() / max((Y == 0).sum(), 1))
            best_pt = {"threshold": float(t), "f1": float(f1),
                       "acc": float((Y == pred).mean()),
                       "sens": sens, "spec": spec,
                       "n_correct": int((Y == pred).sum()),
                       "n_patients": int(len(Y))}
    out = {"arch": arch, "n_patients": int(len(Y)),
           "spectrum_threshold_deployed": spec_pt,
           "best_patient_operating_point": best_pt}
    out["verdict"] = (
        f"patient-level threshold {best_thr:.3f} gives patient F1 "
        f"{best_f1:.3f} vs {spec_pt['f1']:.3f} at the spectrum-tuned "
        f"threshold {spec_thr:.3f} "
        f"({best_f1 - spec_pt['f1']:+.3f})")
    print(f"[pt] {out['verdict']}", flush=True)
    print(f"[pt] at best point: {best_pt['n_correct']}/{len(Y)} "
          f"patients correct · sens {best_pt['sens']:.3f} · "
          f"spec {best_pt['spec']:.3f}", flush=True)
    path = os.path.join(OUT_DIR, "patient_threshold.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[pt] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
