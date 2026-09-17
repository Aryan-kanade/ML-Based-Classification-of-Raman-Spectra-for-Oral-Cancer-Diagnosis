"""
eval_patient_nested.py -- Stage 1a of the push-to-0.8 program v10.

LEAKAGE FIX over eval_patient_threshold.py: the old script swept the
patient threshold on the SAME patients it scored (in-sample sweep).
Here the threshold for each patient is tuned on the OTHER patients only
(leave-one-patient-out tuning), so the reported patient F1/sens/spec
are honest.  Aggregation sweep: mean / median / trimmed / mean-logit.

Usage:  python eval_patient_nested.py
Output: experiments/patient_nested.json  (+ console verdict)
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import f1_score

import exp_common
import pat_common

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-6


def _metrics(y, pred):
    sens = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    spec = float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return {"f1": float(f1_score(y, pred, average="macro")),
            "acc": float((y == pred).mean()),
            "sens": sens, "spec": spec,
            "n_correct": int((y == pred).sum()), "n_patients": int(y.size)}


def _loo_tuned(y, p):
    """Honest per-patient predictions: threshold for patient i tuned on
    all patients != i (maximize macro-F1)."""
    pred = np.empty_like(y)
    for i in range(y.size):
        tr = np.arange(y.size) != i
        best_t, best_f1 = 0.5, -1.0
        for t in np.unique(p[tr]):
            f1 = f1_score(y[tr], (p[tr] >= t).astype(int), average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        pred[i] = int(p[i] >= best_t)
    return pred


def main() -> int:
    ye, proba, g = pat_common.winner_oof()
    deployed_thr = json.load(open(
        os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
        encoding="utf-8"))["threshold"]
    deployed_thr = 0.5 if deployed_thr is None else float(deployed_thr)
    # CAVEAT (audit finding 9): winner.json thresholds live in
    # Platt-calibrated space but are applied here to RAW validate_arch
    # OOF probabilities.  Benign while the stored threshold is None
    # (0.5); if a calibrated threshold is ever stored, calibrate the
    # OOF first before trusting the "deployed_threshold" rows.

    results = {}
    for agg in ("mean", "median", "trimmed", "logit"):
        y, p, _pats = pat_common.patient_table(ye, proba, g, agg=agg)
        # deployed threshold in this aggregation's scale
        if agg == "logit":
            t_dep = float(np.log(deployed_thr / (1 - deployed_thr)))
        else:
            t_dep = deployed_thr
        base = _metrics(y, (p >= t_dep).astype(int))
        honest = _metrics(y, _loo_tuned(y, p))
        results[agg] = {"deployed_threshold": base, "loo_tuned": honest,
                        "verdict": (f"agg={agg}: LOO-tuned patient F1 "
                                    f"{honest['f1']:.3f} vs "
                                    f"{base['f1']:.3f} at deployed thr "
                                    f"({honest['f1'] - base['f1']:+.3f})")}
        print(f"[1a] {results[agg]['verdict']}", flush=True)

    best_agg = max(results, key=lambda a: results[a]["loo_tuned"]["f1"])
    out = {"n_patients": int(np.unique(g).size),
           "deployed_threshold": deployed_thr,
           "aggregations": results,
           "best_agg": best_agg,
           "verdict": results[best_agg]["verdict"]}
    path = os.path.join(exp_common.OUT_DIR, "patient_nested.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[1a] BEST aggregation = {best_agg} -> {path}", flush=True)
    exp_common.record("patient-level", f"LOO-tuned thr ({best_agg})",
                      results[best_agg]["loo_tuned"]["f1"],
                      extra=f"patient F1 {results[best_agg]['loo_tuned']['f1']:.3f} "
                            f"sens {results[best_agg]['loo_tuned']['sens']:.3f} "
                            f"spec {results[best_agg]['loo_tuned']['spec']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
