"""
eval_combat_pooled.py -- Stage 2b of the push-to-0.8 program v10.

Pooled training with RSClass in STANDARD (unpaired) feature space: our
winner features are PAIRED deviations (no external equivalent), so
pooling happens on per-spectrum features preprocessed with the same
§52 chain.  Domain shift is removed by a location-scale ComBat-style
correction of the external block toward ours, fitted INSIDE each outer
training fold only (leakage-safe; arXiv 2410.19643 shows most reported
ComBat gains come from fitting it on all data).

Gate: compare against our standard-mode baseline (~0.62) and the
program baseline 0.753.  Model = Extra Trees (winner's layer-3).

Usage:  python eval_combat_pooled.py [--seed 42]
Output: experiments/combat_pooled.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import dataset as ds
import exp_common
import pat_common
import preprocessing as pp
from clinical_data import load_clinical_dataset
from run_3sse_d2 import WINNER

EXT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "_external", "processed",
    "oral_binary.npz")


def our_standard_features():
    cd = load_clinical_dataset(ds_find_root())
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)
    labels = [s.label for s in cd.spectra]
    flagged = cd.flagged or [False] * len(labels)
    keep = [i for i, lab in enumerate(labels)
            if lab.strip() and not flagged[i]]
    X = pp.preprocess_matrix(X_raw[keep], WINNER, grid)
    y = [labels[i].strip() for i in keep]
    g = [cd.groups[i] for i in keep]
    return np.asarray(X, dtype=np.float32), y, g, grid


def ds_find_root():
    from clinical_data import find_data_root
    return find_data_root()


def ext_features(grid):
    z = np.load(EXT)
    Xe_raw, wne = z["X"], z["wn"]
    X = np.stack([np.interp(grid, wne, row) for row in Xe_raw])
    X = pp.preprocess_matrix(X, WINNER, grid)
    y = np.asarray(["Tumor" if v == 1 else "Normal" for v in z["y"]])
    return np.asarray(X, dtype=np.float32), y


def combat_correct(tr_ours, tr_ext, te_ext):
    """Location-scale correction of ext toward ours, per feature.
    Fitted on training blocks only; applied to the ext test block."""
    mu_o, sd_o = tr_ours.mean(0), tr_ours.std(0) + 1e-8
    mu_e, sd_e = tr_ext.mean(0), tr_ext.std(0) + 1e-8
    return (te_ext - mu_e) / sd_e * sd_o + mu_o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y_str, groups, grid = our_standard_features()
    cls = sorted(set(y_str))
    y = np.asarray([cls.index(v) for v in y_str])
    groups = np.asarray(groups)
    Xe, ye_str = ext_features(grid)
    ye = np.asarray([cls.index(v) for v in ye_str])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                random_state=args.seed)
    oof = np.full(len(y), np.nan)
    for f, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        ext_idx = np.random.default_rng(args.seed + f).choice(
            len(Xe), size=min(len(Xe), 1500), replace=False)
        ext_tr = ext_idx[:int(0.8 * len(ext_idx))]
        Xe_cor = combat_correct(X[tr], Xe[ext_tr], Xe)
        model = ExtraTreesClassifier(n_estimators=300, random_state=42,
                                     n_jobs=1, class_weight="balanced")
        model.fit(np.vstack([X[tr], Xe_cor[ext_idx]]),
                  np.concatenate([y[tr], ye[ext_idx]]))
        oof[te] = model.predict_proba(X[te])[:, 1]
        print(f"[2b] fold {f + 1}/5 done", flush=True)

    m = ~np.isnan(oof)
    sf1 = float(f1_score(y[m], (oof[m] >= 0.5).astype(int),
                         average="macro"))
    yp, ppat, _ = pat_common.patient_table(
        y, np.c_[1 - oof, oof], groups, "mean")
    pred = np.empty_like(yp)
    for i in range(yp.size):                     # LOO-tuned threshold
        trm = np.arange(yp.size) != i
        best_t, best_f1 = 0.5, -1.0
        for t in np.unique(ppat[trm]):
            f1 = f1_score(yp[trm], (ppat[trm] >= t).astype(int),
                          average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        pred[i] = int(ppat[i] >= best_t)
    pf1 = float(f1_score(yp, pred, average="macro"))

    out = {"seed": args.seed, "n_ext": int(len(Xe)),
           "spectrum_f1_at_0.5": sf1, "patient_f1": pf1,
           "verdict": f"ComBat-pooled standard-arm spectrum F1 {sf1:.3f} "
                      f"· patient F1 {pf1:.3f} (std baseline ~0.62, "
                      f"program baseline 0.753)"}
    print(f"[2b] {out['verdict']}", flush=True)
    path = os.path.join(exp_common.OUT_DIR, "combat_pooled.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    exp_common.record("external-data", "ComBat pooled (standard arm)",
                      sf1, extra=f"patient F1 {pf1:.3f}")
    print(f"[2b] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
