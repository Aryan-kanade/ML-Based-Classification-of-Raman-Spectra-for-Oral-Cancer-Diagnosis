"""
eval_cascade.py -- Stage 1e of the push-to-0.8 program v10.

Two-stage clinical cascade on patient-level scores: a rule-OUT arm
(sens >= 0.95) and a rule-IN arm (spec >= 0.90) around an
INDETERMINATE band (the existing triage concept, now measured).
Reports sens/spec on DECIDED patients (what the clinic gains by
re-sampling the middle) next to the all-patient numbers.

Usage:  python eval_cascade.py
Output: experiments/cascade.json
"""

from __future__ import annotations

import json
import os

import numpy as np

import exp_common
import pat_common


def _op_point(y, p, mode, target):
    """Rule-out: HIGHEST threshold whose positive-call set still catches
    >= target of positives (scores sorted descending; cum-tpr is
    monotone, so the first qualifying index is the highest threshold).
    Rule-in: LOWEST threshold that still keeps fpr <= 1-target (the
    last qualifying index).  None when unreachable.  (2026-09-16 audit
    fix: the extremes were inverted, making the band degenerate.)"""
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tprs = np.cumsum(ys == 1) / max((ys == 1).sum(), 1)
    fprs = np.cumsum(ys == 0) / max((ys == 0).sum(), 1)
    if mode == "rule_out":
        ok = np.where(tprs >= target)[0]
        return float(ps[ok[0]]) if ok.size else None
    ok = np.where(fprs <= 1 - target)[0]
    return float(ps[ok[-1]]) if ok.size else None


def main() -> int:
    ye, proba, g = pat_common.winner_oof()
    agg = json.load(open(os.path.join(
        exp_common.OUT_DIR, "patient_nested.json"),
        encoding="utf-8"))["best_agg"] \
        if os.path.isfile(os.path.join(
            exp_common.OUT_DIR, "patient_nested.json")) else "mean"
    y, p, pats = pat_common.patient_table(ye, proba, g, agg=agg)

    thr_out = _op_point(y, p, "rule_out", 0.95)
    thr_in = _op_point(y, p, "rule_in", 0.90)
    out = {"agg": agg, "n_patients": int(y.size),
           "rule_out_thr": thr_out, "rule_in_thr": thr_in}

    if thr_out is not None and thr_in is not None and thr_out < thr_in:
        dec = (p <= thr_out) | (p >= thr_in)
        pred = (p[dec] >= thr_in).astype(int)
        yd = y[dec]
        sens = float(((pred == 1) & (yd == 1)).sum() / max((yd == 1).sum(), 1))
        spec = float(((pred == 0) & (yd == 0)).sum() / max((yd == 0).sum(), 1))
        out["decided"] = {
            "n_decided": int(dec.sum()),
            "coverage": float(dec.mean()),
            "sens": sens, "spec": spec,
            "correct": int((pred == yd).sum())}
        out["verdict"] = (f"cascade [{thr_out:.3f}, {thr_in:.3f}]: "
                          f"{dec.sum()}/{y.size} patients decided "
                          f"(cov {dec.mean():.0%}) with sens {sens:.3f} / "
                          f"spec {spec:.3f} on decided; "
                          f"{(~dec).sum()} -> re-sample/biopsy")
    else:
        out["verdict"] = ("operating points unreachable or inverted — "
                          "patient-level score separation insufficient")
    print(f"[1e] {out['verdict']}", flush=True)
    path = os.path.join(exp_common.OUT_DIR, "cascade.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[1e] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
