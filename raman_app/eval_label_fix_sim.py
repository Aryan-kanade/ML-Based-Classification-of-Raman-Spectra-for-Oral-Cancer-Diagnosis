"""
eval_label_fix_sim.py -- does fixing the flagged labels give 0.8+?
(user question, 2026-09-17).  SIMULATION ONLY: nothing on disk is
changed; the source Data folder is untouched.

Arms (all: proven d2 chain, paired winner params, honest nested
validate_arch, k=10, seed 42 — the best combination from
eval_best_combo):
  A baseline            current labels, no change
  B exclude-flagged     the flagged spectra REMOVED (as if confirmed
                        unusable)
  C flip-flagged        the flagged spectra's labels FLIPPED (as if
                        the patient record had contradicted the
                        given label)

Flag detection: modeling.label_error_report on arm A's OOF — row
indices are exact (no name-mapping needed for the ablation).

Output: experiments/label_fix_sim.json + console verdict.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("RAMAN_DEVICE", "cpu")

import numpy as np

import modeling
import sequential
from run_3sse_d2 import _prepare_winner

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(exp_out := os.path.join(HERE, "experiments"),
                   "label_fix_sim.json")
K, SEED = 10, 42


def run_chain(X, y, groups, wn, tag):
    factories = sequential._factories(wn)
    m = sequential.validate_arch(
        ["PLS + XGBoost", "Random Forest", "Extra Trees"],
        factories, X, y, groups, seed=SEED, k_outer=K, wn=wn)
    f1 = float(m.get("f1", m.get("f1_mean", float("nan"))))
    print(f"[label-fix] {tag}: F1 {f1:.3f}", flush=True)
    return f1


def main() -> int:
    X, y, groups, wn, _grid, _params, meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    print(f"[label-fix] tree: {meta['n_spectra']} rows / "
          f"{meta['n_patients']} patients", flush=True)

    # ---- arm A: baseline + flag detection on its own OOF ----------
    factories = sequential._factories(wn)
    m = sequential.validate_arch(
        ["PLS + XGBoost", "Random Forest", "Extra Trees"],
        factories, X, y, groups, seed=SEED, k_outer=K, wn=wn)
    base_f1 = float(m.get("f1", m.get("f1_mean", float("nan"))))
    proba = np.asarray(m["oof_proba"], dtype=float)
    ye = np.asarray(m["y_true"])
    keep = ~np.isnan(proba).any(axis=1)
    rows = modeling.label_error_report(proba[keep], ye[keep])
    # map flag rows back to full-matrix indices
    valid_idx = np.flatnonzero(keep)
    flagged = sorted(int(valid_idx[r["i"]]) for r in rows)
    print(f"[label-fix] baseline F1 {base_f1:.3f}; flagged rows: "
          f"{flagged} "
          f"({[(r['tier'], round(r['p_self'], 3)) for r in rows]})",
          flush=True)

    # ---- arm B: exclude flagged rows ------------------------------
    mask = np.ones(len(y), dtype=bool)
    mask[flagged] = False
    f1_b = run_chain(X[mask], [y[i] for i in np.flatnonzero(mask)],
                     [groups[i] for i in np.flatnonzero(mask)], wn,
                     "B exclude-flagged") if flagged else base_f1

    # ---- arm C: flip flagged labels -------------------------------
    classes = sorted(set(y))
    flip = {classes[0]: classes[1], classes[1]: classes[0]}
    y_flip = list(y)
    for i in flagged:
        y_flip[i] = flip.get(y[i], y[i])
    f1_c = run_chain(X, y_flip, groups, wn,
                     "C flip-flagged") if flagged else base_f1

    out = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": f"proven chain, k={K}, seed={SEED}, "
                    "honest nested, current tree",
        "flagged_rows": flagged,
        "flags": [{"tier": r["tier"], "p_self": r["p_self"],
                   "p_other": r["p_other"]} for r in rows],
        "f1_baseline": base_f1,
        "f1_exclude": f1_b,
        "f1_flip": f1_c,
        "verdict": ("0.8+ NOT reached by label fixing alone"
                    if max(base_f1, f1_b, f1_c) < 0.80 else
                    "0.8+ reached"),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[label-fix] VERDICT: {out['verdict']} — "
          f"A {base_f1:.3f} / B {f1_b:.3f} / C {f1_c:.3f} — "
          f"saved: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
