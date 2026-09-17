"""
eval_feature_expand.py -- L5 of the push-to-0.8 program (SAFETY: pure
experiment script; writes experiments/ only).

Idea: the chain layers see deriv-2 deviation SHAPE only.  Append
per-band integrated-intensity features (biochemistry.BANDS windows on
the same deviation spectra) so the tree layers also see local
AMPLITUDE.  Keep only if the winner arch scores higher on the enriched
matrix under IDENTICAL folds (validate_arch, seed 42, k=5).

Usage:  python eval_feature_expand.py
Output: experiments/feature_expand.json
"""

from __future__ import annotations

import json
import os

import numpy as np

import sequential
from run_3sse_d2 import _prepare_winner

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "experiments")


def band_integrals(X: np.ndarray, wn: np.ndarray) -> np.ndarray:
    """One integrated-intensity feature per literature band window."""
    import biochemistry as bio
    feats = []
    for center, hw, *_rest in bio.BANDS:
        m = (wn >= center - hw) & (wn <= center + hw)
        if m.sum() < 2:
            continue
        feats.append(np.trapezoid(X[:, m], wn[m], axis=1))
    if not feats:
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    F = np.vstack(feats).T
    # No rescaling: every consumer layer (RF/ET/XGB) thresholds each
    # feature independently, so a global block scale is a no-op for
    # trees — and the previous X.std() global scalar was fitted on all
    # rows (audit finding 14: mild test-row influence, removed).
    return F.astype(np.float32)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]
    X, y, groups, wn, _grid, _params, meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    print(f"[feat] arch {' -> '.join(arch)} · {X.shape[0]} rows · "
          f"{meta['n_patients']} patients", flush=True)
    factories = sequential._factories(wn)

    base = sequential.validate_arch(arch, factories, X, list(y),
                                    list(groups), seed=42, k_outer=5,
                                    wn=wn)
    F = band_integrals(np.asarray(X, dtype=np.float32), wn)
    Xp = np.hstack([np.asarray(X, dtype=np.float32), F])
    print(f"[feat] enriched matrix: {Xp.shape[1]} features "
          f"(+{F.shape[1]} band integrals)", flush=True)
    exp = sequential.validate_arch(arch, factories, Xp, list(y),
                                   list(groups), seed=42, k_outer=5,
                                   wn=None)   # WN models sliced: skip

    out = {
        "arch": arch,
        "base": {k: base.get(k) for k in
                 ("f1", "sens", "spec", "auc", "pat_f1")},
        "enriched": {k: exp.get(k) for k in
                     ("f1", "sens", "spec", "auc", "pat_f1")},
    }
    d = out["enriched"]["f1"] - out["base"]["f1"]
    out["delta_f1"] = d
    out["verdict"] = ("KEEP (enriched wins)" if d > 0.005 else
                      "DROP (no gain)" if d > -0.005 else
                      "DROP (enriched worse)")
    print(f"[feat] base F1 {out['base']['f1']:.3f} · enriched F1 "
          f"{out['enriched']['f1']:.3f} · delta {d:+.3f}\n"
          f"[feat] {out['verdict']}", flush=True)
    path = os.path.join(OUT_DIR, "feature_expand.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[feat] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
