"""
eval_winner_now.py -- re-baseline the proven d2 winner chain on the
CURRENT data tree.  v2 (2026-09-17, post-recovery): the 55 misplaced
spectra / 20 patients were restored (tree back to 317 loaded / ~307
paired rows / 72 patients), so this run measures the chain on the
RECOVERED tree at both k=5 and k=10, 3 seeds, and computes the
selective-prediction numbers (rule-out/rule-in, decided, screen) on
the best OOF.

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

import clinical as clin
import exp_common
import modeling
import sequential
from run_3sse_d2 import _prepare_winner

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(exp_common.OUT_DIR, "winner_chain_now.json")
SEEDS = (42, 43, 44)
KS = (5, 10)
D2_F1 = 0.7568                     # 2026-09-13 recorded headline


def eval_k(arch, factories, X, y, groups, wn, k):
    per_seed = []
    for seed in SEEDS:
        m = sequential.validate_arch(arch, factories, X, y, groups,
                                     seed=seed, k_outer=k, wn=wn)
        proba = np.asarray(m["oof_proba"], dtype=np.float64)
        ye = np.asarray(m["y_true"], dtype=np.int64)
        keep = ~np.isnan(proba).any(axis=1)
        ye_k, pr_k = ye[keep], proba[keep]
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(ye_k, pr_k.argmax(axis=1), labels=[0, 1])
        f1 = float(m.get("f1", m.get("f1_mean", float("nan"))))
        entry = {
            "seed": seed,
            "f1": f1,
            "sens": float(cm[1, 1]) / max(1, int(cm[1].sum())),
            "spec": float(cm[0, 0]) / max(1, int(cm[0].sum())),
            "auc": (float(modeling.roc_points(ye_k, pr_k[:, 1])[2])
                    if len(set(ye_k.tolist())) == 2 else None),
            "n": int(ye_k.size),
        }
        per_seed.append(entry)
        print(f"[winner-now] k={k} seed {seed}: pooled F1 {f1:.3f} "
              f"(sens {entry['sens']:.3f} / spec {entry['spec']:.3f}"
              + (f", AUC {entry['auc']:.3f}" if entry["auc"] else "")
              + ")", flush=True)
    return per_seed


def main() -> int:
    with open(os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]
    X, y, groups, wn, _grid, _params, meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    print(f"[winner-now] arch: {' -> '.join(arch)}", flush=True)
    print(f"[winner-now] RECOVERED tree: {meta['n_spectra']} rows / "
          f"{meta['n_patients']} patients", flush=True)
    factories = sequential._factories(wn)
    t0 = time.time()

    per_k = {}
    for k in KS:
        per_seed = eval_k(arch, factories, X, y, groups, wn, k)
        f1s = np.array([e["f1"] for e in per_seed], dtype=float)
        per_k[k] = {"per_seed": per_seed,
                    "mean_f1": float(f1s.mean()),
                    "std_f1": float(f1s.std())}
        print(f"[winner-now] k={k}: MEAN {f1s.mean():.3f} "
              f"± {f1s.std():.3f}", flush=True)

    best_k = max(per_k, key=lambda k: per_k[k]["mean_f1"])
    best_entry = max(per_k[best_k]["per_seed"], key=lambda e: e["f1"])

    # selective numbers on the best (k, seed) OOF, recompute once
    m = sequential.validate_arch(arch, factories, X, y, groups,
                                 seed=best_entry["seed"],
                                 k_outer=best_k, wn=wn)
    proba = np.asarray(m["oof_proba"], dtype=np.float64)
    ye = np.asarray(m["y_true"], dtype=np.int64)
    keep = ~np.isnan(proba).any(axis=1)
    yv, pv = ye[keep], proba[keep, 1]
    lo, hi, s_at, sp_at = clin.operating_points(yv, pv)
    conf = clin.decided_case(yv, pv, lo, hi)
    screen = clin.decided_case(yv, pv, 0.30, 0.70)

    f1s = np.array([per_k[best_k]["per_seed"][i]["f1"]
                    for i in range(len(SEEDS))], dtype=float)
    out = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arch": arch,
        "tree": {"n_spectra": meta["n_spectra"],
                 "n_patients": meta["n_patients"],
                 "note": "post-recovery (55 spectra / 20 patients "
                         "restored 2026-09-17)"},
        "d2_record": {"f1": D2_F1, "auc": 0.7914,
                      "tree": {"n_spectra": 287, "n_patients": 64},
                      "when": "2026-09-13"},
        "per_k": {str(k): {kk: vv for kk, vv in d.items()}
                  for k, d in per_k.items()},
        "best_k": best_k,
        "best_seed": best_entry["seed"],
        "best_f1": best_entry["f1"],
        "mean_f1_best_k": float(f1s.mean()),
        "std_f1_best_k": float(f1s.std()),
        "selective": {
            "ruleout_sens": s_at, "ruleout_p": lo,
            "rulein_spec": sp_at, "rulein_p": hi,
            "decided_f1": conf["f1"], "decided_coverage": conf["coverage"],
            "screen_f1": screen["f1"], "screen_coverage": screen["coverage"],
        },
        "minutes": round((time.time() - t0) / 60.0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[winner-now] BEST: k={best_k} seed={best_entry['seed']} "
          f"F1 {best_entry['f1']:.3f} | mean at best k "
          f"{out['mean_f1_best_k']:.3f} ± {out['std_f1_best_k']:.3f}",
          flush=True)
    print(f"[winner-now] SELECTIVE: rule-out sens {s_at:.3f} / "
          f"rule-in spec {sp_at:.3f} | decided F1 {conf['f1']:.3f} "
          f"on {conf['coverage']:.0%} | screen F1 {screen['f1']:.3f} "
          f"on {screen['coverage']:.0%}", flush=True)
    print(f"[winner-now] saved: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
