"""
eval_best_combo.py -- find the best (preprocessing, folds, seed)
combination for the proven winner chain on the CURRENT data tree.

Protocol: paired §52 winner preprocessing (no-crop, d2, db6 L2,
SG 11/4, norm none) via run_3sse_d2._prepare_winner; architecture =
the d2 proven chain (PLS + XGBoost -> Random Forest -> Extra Trees);
honest nested grouped CV (sequential.validate_arch) — OOF chain
features regenerated inside every outer training fold.

Sweep: k_outer in {5, 8, 10} x seeds {42..46}.  The per-k MEAN over
seeds is the trustworthy number; the best single seed is reported as
a draw, not as the estimate.

Output: experiments/best_combo.json + console table.
SAFETY: writes experiments/ only.  RAMAN_DEVICE=cpu (loky children).
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
OUT = os.path.join(exp_common.OUT_DIR, "best_combo.json")
KS = (5, 8, 10)
SEEDS = (42, 43, 44, 45, 46)


def main() -> int:
    with open(os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]
    X, y, groups, wn, _grid, _params, meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    print(f"[best-combo] chain: {' -> '.join(arch)} | tree: "
          f"{meta['n_spectra']} rows / {meta['n_patients']} patients",
          flush=True)
    factories = sequential._factories(wn)

    rows = []
    t0 = time.time()
    for k in KS:
        for seed in SEEDS:
            m = sequential.validate_arch(arch, factories, X, y, groups,
                                         seed=seed, k_outer=k, wn=wn)
            f1 = float(m.get("f1", m.get("f1_mean", float("nan"))))
            rows.append({"k": k, "seed": seed, "f1": f1})
            print(f"[best-combo] k={k:>2} seed={seed}: F1 {f1:.3f}",
                  flush=True)

    # aggregate per k: mean over seeds is the honest estimate
    per_k = []
    for k in KS:
        f1s = np.array([r["f1"] for r in rows if r["k"] == k])
        best = max((r for r in rows if r["k"] == k),
                   key=lambda r: r["f1"])
        per_k.append({
            "k": k,
            "mean_f1": float(f1s.mean()),
            "std_f1": float(f1s.std()),
            "best_seed": best["seed"],
            "best_seed_f1": float(best["f1"]),
        })
        print(f"[best-combo] k={k}: MEAN {f1s.mean():.3f} "
              f"± {f1s.std():.3f} | best single-seed draw "
              f"{best['seed']} {best['f1']:.3f}", flush=True)

    best_k = max(per_k, key=lambda e: e["mean_f1"])
    out = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arch": arch,
        "tree": {"n_spectra": meta["n_spectra"],
                 "n_patients": meta["n_patients"]},
        "preprocessing": ("paired §52 winner: no-crop, d2, db6 L2, "
                          "SG 11/4, norm none"),
        "rows": rows,
        "per_k": per_k,
        "best_k": best_k["k"],
        "best_mean_f1": best_k["mean_f1"],
        "minutes": round((time.time() - t0) / 60.0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[best-combo] BEST: k={best_k['k']} "
          f"(mean F1 {best_k['mean_f1']:.3f} ± {best_k['std_f1']:.3f}; "
          f"best single-seed draw seed {best_k['best_seed']} "
          f"{best_k['best_seed_f1']:.3f}) — saved: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
