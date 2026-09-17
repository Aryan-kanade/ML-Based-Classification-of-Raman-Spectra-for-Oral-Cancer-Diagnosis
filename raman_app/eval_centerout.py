"""
eval_centerout.py -- Stage 2f of the push-to-0.8 program v10.

Center-out external validation with RSClass (leave-one-center-out
protocol, Radiology:AI 2023): a standard-mode model (Extra Trees on the
§52 preprocessing chain — the strongest non-paired configuration) is
trained on ALL our data and tested on the external center, and vice
versa.  Paired mode cannot transfer (external corpus has no patient
pairing), so this is the honest ceiling of cross-center deployment of
a non-paired model.  Direction "ours->ext" is the deployment question;
"ext->ours" measures how much external data alone knows about our
cohort.

Usage:  python eval_centerout.py
Output: experiments/centerout.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score

import exp_common
from eval_combat_pooled import ext_features, our_standard_features


def _report(name, y, p):
    pred = (p >= 0.5).astype(int)
    sens = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    spec = float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return {"direction": name, "n": int(len(y)),
            "f1": float(f1_score(y, pred, average="macro")),
            "sens": sens, "spec": spec,
            "auc": float(roc_auc_score(y, p))}


def main() -> int:
    X, y_str, groups, grid = our_standard_features()
    cls = sorted(set(y_str))
    y = np.asarray([cls.index(v) for v in y_str])
    Xe, ye_str = ext_features(grid)
    ye = np.asarray([cls.index(v) for v in ye_str])

    model = ExtraTreesClassifier(n_estimators=300, random_state=42,
                                 n_jobs=1, class_weight="balanced")
    model.fit(X, y)
    p_ext = model.predict_proba(Xe)[:, 1]
    ours2ext = _report("ours->RSClass", ye, p_ext)

    model2 = ExtraTreesClassifier(n_estimators=300, random_state=42,
                                  n_jobs=1, class_weight="balanced")
    model2.fit(Xe, ye)
    p_ours = model2.predict_proba(X)[:, 1]
    ext2ours = _report("RSClass->ours", y, p_ours)

    out = {"n_ours": int(len(y)), "n_ext": int(len(Xe)),
           "ours_to_external": ours2ext, "external_to_ours": ext2ours,
           "verdict": (f"ours->RSClass F1 {ours2ext['f1']:.3f} "
                       f"(sens {ours2ext['sens']:.3f} / "
                       f"spec {ours2ext['spec']:.3f} / "
                       f"AUC {ours2ext['auc']:.3f}); "
                       f"RSClass->ours F1 {ext2ours['f1']:.3f} "
                       f"(AUC {ext2ours['auc']:.3f})")}
    print(f"[2f] {out['verdict']}", flush=True)
    path = os.path.join(exp_common.OUT_DIR, "centerout.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[2f] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
