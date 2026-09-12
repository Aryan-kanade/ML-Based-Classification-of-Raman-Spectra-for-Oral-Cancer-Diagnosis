"""
prep_locked_check.py -- LOCKED one-shot holdout: does the §52 winner
preprocessing actually PREDICT better on never-seen patients?

The deep search (§52) reported nested-CV F1 — an honest ESTIMATE.
This is the FINAL-EXAM version of the question: split patients
70/15/15 once (clinical_data.patient_split, seed 42), train on the
70% (evaluate_models tunes hyperparams on train patients only and
refits the winner on all train rows), then predict the 15% test
patients ONCE.  Defaults vs §52 winner, paired mode.

Usage:  python prep_locked_check.py
Output: vit_outputs/prep_locked_check.json + console table
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

os.environ.setdefault("RAMAN_DEVICE", "cpu")

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)

import dataset as ds
import modeling
import paired as pmod
import preprocessing as pp
from clinical_data import find_data_root, load_clinical_dataset, patient_split
from prep_deep_search import describe

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vit_outputs", "prep_locked_check.json")

MODELS = [
    "Random Forest", "Extra Trees", "LightGBM", "CatBoost",
    "PCA + SVM (RBF)", "Ensemble (top-3)",
]

DEFAULTS = pp.PreprocessParams().validate()          # crop 500-2000 d0 vector
WINNER = pp.PreprocessParams(                        # §52 paired winner
    crop_min=0.0, crop_max=0.0, sg_deriv=2, norm="none",
    sg_window=11, sg_poly=4, wavelet_name="db6", wavelet_level=2,
).validate()


def locked_eval(tag: str, params: pp.PreprocessParams, cd, grid, X_raw,
                seed: int = 42) -> dict:
    flagged = cd.flagged or [False] * len(cd.spectra)
    keep = [i for i, s in enumerate(cd.spectra)
            if s.label.strip() and (params.despike or not flagged[i])]
    y = [cd.spectra[i].label.strip() for i in keep]
    g = [cd.groups[i] for i in keep]
    pd_ = pmod.paired_features(X_raw[keep], y, g, grid, params)
    X, yy, gg, wn = pd_.X, pd_.y, pd_.groups, pd_.wn
    tr, _va, te = patient_split(gg, yy, seed=seed)
    print(f"\n=== {tag}: {describe(params)}", flush=True)
    print(f"    paired rows {X.shape[0]}x{X.shape[1]} | "
          f"train {len(tr)} / test {len(te)} rows "
          f"({len(set(np.asarray(gg)[te]))} patients held out)",
          flush=True)
    t0 = time.time()
    results, winner = modeling.evaluate_models(
        X[tr], list(np.asarray(yy)[tr]), model_names=MODELS, k_folds=5,
        seed=seed, groups=list(np.asarray(gg)[tr]), wavenumbers=wn)
    y_te = np.asarray(yy)[te]
    g_te = np.asarray(gg)[te]
    proba = winner.pipeline.predict_proba(X[te])
    classes = list(winner.classes)
    pred = [classes[int(np.argmax(p))] for p in proba]
    pos = classes[1] if len(classes) == 2 else classes[-1]
    p_pos = (proba[:, classes.index(pos)] if len(classes) == 2
             else proba.max(axis=1))
    f1 = f1_score(y_te, pred, average="macro")
    acc = accuracy_score(y_te, pred)
    try:
        auc = roc_auc_score([1 if v == pos else 0 for v in y_te], p_pos)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_te, pred, labels=classes).tolist()
    # patient-level verdict: mean P(pos) per patient, >=0.5 rule
    pat_pred, pat_true = [], []
    for gp in sorted(set(g_te)):
        m = g_te == gp
        pat_pred.append(pos if p_pos[m].mean() >= 0.5 else classes[0])
        vals, cnts = np.unique(y_te[m], return_counts=True)
        pat_true.append(vals[cnts.argmax()])
    pat_acc = accuracy_score(pat_true, pat_pred)
    print(f"    CV-winner on train: {winner.name} "
          f"F1 {winner.macro_f1():.3f} ({time.time() - t0:.0f}s)",
          flush=True)
    print(f"    LOCKED one-shot on {len(set(g_te))} unseen patients: "
          f"macro-F1 {f1:.3f} | acc {acc:.3f} | AUC {auc:.3f} | "
          f"patient-level acc {pat_acc:.3f} "
          f"({len(pat_true)} patients)", flush=True)
    print(f"    confusion (rows=true {classes}): {cm}", flush=True)
    return {"params": asdict(params), "train_winner": winner.name,
            "train_cv_f1": winner.macro_f1(),
            "n_test_rows": int(len(te)),
            "n_test_patients": int(len(set(g_te))),
            "locked_f1": float(f1), "locked_acc": float(acc),
            "locked_auc": float(auc),
            "patient_acc": float(pat_acc), "confusion": cm,
            "classes": classes}


def main():
    root = find_data_root()
    if not root:
        raise SystemExit("no data root")
    cd = load_clinical_dataset(root)
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)

    out = {"seed": 42}
    out["defaults"] = locked_eval("defaults", DEFAULTS, cd, grid, X_raw)
    out["winner"] = locked_eval("winner", WINNER, cd, grid, X_raw)
    d, w = out["defaults"], out["winner"]
    print("\n=== LOCKED one-shot: defaults -> winner ===", flush=True)
    for k in ("locked_f1", "locked_acc", "locked_auc", "patient_acc"):
        print(f"  {k:<12} {d[k]:.3f} -> {w[k]:.3f}  ({w[k] - d[k]:+.3f})",
              flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
