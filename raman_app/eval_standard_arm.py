"""
eval_standard_arm.py -- L8 of the push-to-0.8 program (SAFETY: pure
experiment script; writes experiments/ only).

The 0.75-family scores come from PAIRED features (deviation vs the
patient's own normal, 287 rows).  This arm scores the same winning
architectures on STANDARD features (every labeled spectrum vs the
population baseline -- ~2x more rows, same patient-grouped CV).
Deployment trade-off if it wins: clinically the paired comparison is
stronger, so a standard-mode win changes the operating definition.

Usage:  python eval_standard_arm.py
Output: experiments/standard_arm.json
"""

from __future__ import annotations

import json
import os

import numpy as np

import dataset as ds
import preprocessing as pp
import sequential
from clinical_data import load_clinical_dataset
from run_3sse_d2 import WINNER

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "experiments")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]

    cd = load_clinical_dataset(sequential.DEFAULT_DATA)
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)
    labels = [s.label for s in cd.spectra]
    flagged = cd.flagged or [False] * len(labels)
    keep = [i for i, lab in enumerate(labels)
            if lab.strip() and (WINNER.despike or not flagged[i])]
    y = [labels[i].strip() for i in keep]
    g = [cd.groups[i] for i in keep]
    X = pp.preprocess_matrix(X_raw[keep], WINNER, wn=grid)
    print(f"[std] standard features {X.shape[0]} rows · "
          f"{len(set(g))} patients · {X.shape[1]} features", flush=True)
    factories = sequential._factories(grid)

    m = sequential.validate_arch(arch, factories, X, y, g, seed=42,
                                 k_outer=5, wn=np.asarray(grid))
    out = {"arch": arch, "n_rows": int(X.shape[0]),
           "n_patients": len(set(g)),
           "f1": float(m["f1"]), "sens": float(m["sens"]),
           "spec": float(m["spec"]),
           "auc": (float(m["auc"])
                   if isinstance(m.get("auc"), (int, float)) else None),
           "pat_f1": (float(m["pat_f1"])
                      if isinstance(m.get("pat_f1"), (int, float))
                      else None),
           "paired_reference_f1": 0.757}
    d = out["f1"] - out["paired_reference_f1"]
    out["delta_vs_paired"] = d
    out["verdict"] = ("KEEP (standard wins)" if d > 0.005 else
                      "DROP (no gain)" if d > -0.005 else
                      "DROP (paired wins)")
    print(f"[std] standard F1 {out['f1']:.3f} · sens {out['sens']:.3f} "
          f"· spec {out['spec']:.3f} · delta vs paired {d:+.3f}\n"
          f"[std] {out['verdict']}", flush=True)
    path = os.path.join(OUT_DIR, "standard_arm.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[std] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
