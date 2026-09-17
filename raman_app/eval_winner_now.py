"""
eval_winner_now.py -- re-baseline the proven d2 winner chain on the
CURRENT data tree (2026-09-17).

The 0.757/0.791 headline was measured 2026-09-13 on the then-tree
(287 rows / 64 patients).  The Data folder has since changed
(286 CSVs -> 262 loaded spectra / 62 subjects after reference +
cross-class-duplicate exclusion), so the recorded OOF is no longer
regenerable (Brain.md §63 finding 3).  This script measures the SAME
chain (PLS + XGBoost -> Random Forest -> Extra Trees), SAME protocol
(paired §52 winner params, honest nested grouped 5-fold, OOF chain
features regenerated inside each outer training fold) on TODAY's
tree, across seeds 42/43/44.

Output: experiments/winner_chain_now.json (+ console table).
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
OUT = os.path.join(exp_common.OUT_DIR, "winner_chain_now.json")
SEEDS = (42, 43, 44)
D2_F1 = 0.7568                     # 2026-09-13 recorded headline


def main() -> int:
    import modeling
    from sklearn.metrics import confusion_matrix

    with open(os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]
    X, y, groups, wn, _grid, _params, meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    print(f"[winner-now] arch: {' -> '.join(arch)}", flush=True)
    print(f"[winner-now] current tree: {meta['n_spectra']} rows / "
          f"{meta['n_patients']} patients "
          f"(d2 recorded 287/64 on 2026-09-13)", flush=True)
    factories = sequential._factories(wn)

    per_seed = []
    t0 = time.time()
    for seed in SEEDS:
        m = sequential.validate_arch(arch, factories, X, y, groups,
                                     seed=seed, k_outer=5, wn=wn)
        proba = np.asarray(m["oof_proba"], dtype=np.float64)
        ye = np.asarray(m["y_true"], dtype=np.int64)
        keep = ~np.isnan(proba).any(axis=1)
        ye_k, pr_k = ye[keep], proba[keep]
        pred = pr_k.argmax(axis=1)
        cm = confusion_matrix(ye_k, pred, labels=[0, 1])
        f1 = float(m.get("f1", m.get("f1_mean", float("nan"))))
        entry = {
            "seed": seed,
            "f1": f1,
            "f1_mean_fold": float(m.get("f1_mean", f1)),
            "sens": float(cm[1, 1]) / max(1, int(cm[1].sum())),
            "spec": float(cm[0, 0]) / max(1, int(cm[0].sum())),
            "auc": (float(modeling.roc_points(
                ye_k, pr_k[:, 1])[2])
                if len(set(ye_k.tolist())) == 2 else None),
            "n": int(ye_k.size),
        }
        per_seed.append(entry)
        print(f"[winner-now] seed {seed}: pooled F1 {f1:.3f} "
              f"(sens {entry['sens']:.3f} / spec {entry['spec']:.3f}"
              + (f", AUC {entry['auc']:.3f}" if entry["auc"] else "")
              + ")", flush=True)

    f1s = np.array([e["f1"] for e in per_seed], dtype=float)
    out = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arch": arch,
        "tree": {"n_spectra": meta["n_spectra"],
                 "n_patients": meta["n_patients"]},
        "d2_record": {"f1": D2_F1, "auc": 0.7914,
                      "tree": {"n_spectra": 287, "n_patients": 64},
                      "when": "2026-09-13"},
        "per_seed": per_seed,
        "mean_f1": float(f1s.mean()),
        "std_f1": float(f1s.std()),
        "delta_vs_d2": float(f1s.mean()) - D2_F1,
        "minutes": round((time.time() - t0) / 60.0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[winner-now] MEAN pooled F1 {out['mean_f1']:.3f} "
          f"± {out['std_f1']:.3f} over {len(SEEDS)} seeds "
          f"— d2 record {D2_F1:.3f} "
          f"({out['delta_vs_d2']:+.3f} on the current tree)", flush=True)
    print(f"[winner-now] saved: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
