"""
eval_d3_adjudicate.py -- 3-seed + McNemar adjudication of the d3
search winner against the proven d2 chain, on identical outer folds.

Run AFTER run_3sse_d3.py completes.  For seeds 42/43/44:
  - d3 winner chain: honest nested validate_arch (k=5, recovered tree)
  - d2 chain:        the SAME splits (same seed/k/data)
McNemar b/c on the pooled seed-42 OOF predictions (d3 vs d2).

Verdict rule (pre-registered, §60 discipline): d3 wins only if its
3-seed mean exceeds the d2 3-seed mean AND McNemar p < 0.05; a
single-seed high draw alone is selection optimism.

Output: experiments/d3_adjudication.json + console verdict.
SAFETY: writes experiments/ only.  RAMAN_DEVICE=cpu.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("RAMAN_DEVICE", "cpu")

import numpy as np

import exp_common
import sequential
from run_3sse_d2 import _prepare_winner

HERE = os.path.dirname(os.path.abspath(__file__))
D2_ARCH = ["PLS + XGBoost", "Random Forest", "Extra Trees"]
SEEDS = (42, 43, 44)
K = 5
OUT = os.path.join(exp_common.OUT_DIR, "d3_adjudication.json")


def oof_arrays(arch, factories, X, y, groups, wn, seed):
    m = sequential.validate_arch(arch, factories, X, y, groups,
                                 seed=seed, k_outer=K, wn=wn)
    proba = np.asarray(m["oof_proba"], dtype=np.float64)
    ye = np.asarray(m["y_true"], dtype=np.int64)
    keep = ~np.isnan(proba).any(axis=1)
    f1 = float(m.get("f1", m.get("f1_mean", float("nan"))))
    return ye[keep], proba[keep], f1


def mcnemar(ye, pa, pb):
    """Exact-ish McNemar (no continuity correction) between two
    predictors' errors on identical rows."""
    from scipy.stats import binomtest
    a_wrong = pa.argmax(axis=1) != ye
    b_wrong = pb.argmax(axis=1) != ye
    b = int((a_wrong & ~b_wrong).sum())   # A wrong, B right
    c = int((~a_wrong & b_wrong).sum())   # B wrong, A right
    p = (binomtest(b, b + c, 0.5).pvalue
         if (b + c) else 1.0)
    return b, c, float(p)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    w3 = os.path.join(here, "study_run_3sse_d3", "winner.json")
    if not os.path.isfile(w3):
        print("[adjudicate] study_run_3sse_d3/winner.json missing — "
              "run run_3sse_d3.py to completion first", flush=True)
        return 1
    with open(w3, encoding="utf-8") as fh:
        arch3 = json.load(fh)["arch"]
    X, y, groups, wn, _grid, _params, meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    factories = sequential._factories(wn)
    print(f"[adjudicate] d3 winner: {' -> '.join(arch3)}", flush=True)

    d3_f1s, d2_f1s = [], []
    paired42 = None
    for seed in SEEDS:
        ye3, p3, f13 = oof_arrays(arch3, factories, X, y, groups, wn,
                                  seed)
        ye2, p2, f12 = oof_arrays(D2_ARCH, factories, X, y, groups, wn,
                                  seed)
        d3_f1s.append(f13)
        d2_f1s.append(f12)
        if seed == 42:
            paired42 = (ye3, p3, p2)
        print(f"[adjudicate] seed {seed}: d3 {f13:.3f} vs d2 "
              f"{f12:.3f}", flush=True)

    ye42, p3, p2 = paired42
    b, c, p = mcnemar(ye42, p3, p2)
    mean3, mean2 = float(np.mean(d3_f1s)), float(np.mean(d2_f1s))
    win = bool(mean3 > mean2 and p < 0.05)
    out = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "d3_arch": arch3,
        "d2_arch": D2_ARCH,
        "d3_f1s": d3_f1s, "d2_f1s": d2_f1s,
        "d3_mean": mean3, "d2_mean": mean2,
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
        "verdict": ("d3 ADOPT (mean better AND McNemar p<0.05)"
                    if win else
                    "d3 NOT adopted (selection optimism / tie)"),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[adjudicate] VERDICT: {out['verdict']} — d3 mean "
          f"{mean3:.3f} vs d2 mean {mean2:.3f} (McNemar b={b} c={c} "
          f"p={p:.3f}) — saved: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
