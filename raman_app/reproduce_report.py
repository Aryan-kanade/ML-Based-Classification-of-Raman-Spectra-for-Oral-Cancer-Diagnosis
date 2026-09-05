"""
reproduce_report.py — head-to-head replication of the internship report.

Bidipta Rana, "Machine Learning Based Classification of Raman Spectra for
Oral Cancer Diagnosis" (BARC / IIT Madras, July 2026) reports, on a small
spectrum-level holdout: PLS+XGBoost best (acc 72.4%, sens 73.3%, spec
71.4%, AUC 0.721).  This script runs the report's EXACT analytical
pipeline —

    crop 700-1800 -> ASLS baseline -> vector normalization ->
    SURE wavelet denoise (coif5, level 5, soft) ->
    Welch-t + Cohen's-d feature filter (p<0.05, |d|>0.1) ->
    PCA(5) / PLS(5) -> its 7 models

under TWO protocols:

    A  report protocol : spectrum-level StratifiedKFold(5)  (no patient
                         grouping — the report's implicit scheme)
    B  honest protocol : patient-grouped StratifiedGroupKFold(5) x R
                         repeats + Friedman test across models

and prints/writes the side-by-side table.  Where the honest numbers land
vs the report's is the useful scientific output of this script.

Usage:
    python reproduce_report.py               # auto-detected dataset root
    python reproduce_report.py --mini        # 3-model fast subset
    python reproduce_report.py --repeats 1   # quick single repeat
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
from scipy.stats import friedmanchisquare
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modeling                                 # noqa: E402
import preprocessing as pp                      # noqa: E402
from clinical_data import find_data_root        # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = find_data_root() or os.path.join(APP_DIR, "..", "Data")

# the report's published test-set numbers (Table 4.2), for side-by-side
REPORT_NUMBERS = {  # model: (acc, sens, spec, f1, auc)
    "PCA+SVM":     (0.655, 0.714, 0.600, 0.667, 0.662),
    "PLS-DA+SVM":  (0.655, 0.643, 0.667, 0.643, 0.762),
    "PLS+QDA":     (0.517, 0.286, 0.733, 0.364, 0.555),
    "PCA+XGB":     (0.586, 0.571, 0.600, 0.571, 0.548),
    "PCA+RF":      (0.517, 0.429, 0.600, 0.462, 0.533),
    "PLS+XGB":     (0.724, 0.733, 0.714, 0.733, 0.721),
    "PLS+RF":      (0.690, 0.714, 0.667, 0.690, 0.629),
}


def report_params() -> pp.PreprocessParams:
    """The report's preprocessing, expressed in this app's parameters."""
    return pp.PreprocessParams(
        crop_min=700.0, crop_max=1800.0,
        despike=False,
        wavelet=True, wavelet_name="coif5", wavelet_level=5,
        wavelet_threshold="sure", wavelet_mode="soft",
        sg_window=11, sg_poly=3, sg_deriv=0,     # app default (report
        baseline_method="als",                   # specifies no smoothing)
        als_lambda=1e5, als_p=0.01, als_niter=10,
        norm="vector",
    )


def report_models() -> "OrderedDict[str, Pipeline]":
    """The report's 7 models: t-test filter -> PCA(5)/PLS(5) -> classifier.
    XGBoost hyperparameters = the report's tuned values (§3.5.2)."""
    xgb_kwargs = dict(n_estimators=112, max_depth=2, learning_rate=0.017,
                      colsample_bytree=0.638, gamma=0.289,
                      min_child_weight=2, subsample=0.993,
                      eval_metric="mlogloss",
                      random_state=42, n_jobs=-1)
    svm = dict(kernel="rbf", probability=True, random_state=42)
    out = OrderedDict()

    def make(dim, clf):
        return Pipeline([
            ("sel", modeling.TTestSelect(p_max=0.05, d_min=0.1)),
            ("dim", dim), ("sc", StandardScaler()), ("clf", clf)])

    out["PCA+SVM"] = make(PCA(n_components=5, random_state=42),
                          SVC(**svm))
    out["PLS-DA+SVM"] = make(modeling.PLSScores(n_components=5),
                             SVC(**svm))
    out["PLS+QDA"] = make(modeling.PLSScores(n_components=5),
                          QuadraticDiscriminantAnalysis())
    out["PCA+XGB"] = make(PCA(n_components=5, random_state=42),
                          modeling.XGBClassifier(**xgb_kwargs)
                          if modeling.HAS_XGB else None)
    out["PCA+RF"] = make(PCA(n_components=5, random_state=42),
                         RandomForestClassifier(n_estimators=200,
                                                random_state=42, n_jobs=-1))
    out["PLS+XGB"] = make(modeling.PLSScores(n_components=5),
                          modeling.XGBClassifier(**xgb_kwargs)
                          if modeling.HAS_XGB else None)
    out["PLS+RF"] = make(modeling.PLSScores(n_components=5),
                         RandomForestClassifier(n_estimators=200,
                                                random_state=42, n_jobs=-1))
    if not modeling.HAS_XGB:                    # graceful degradation
        for k in [k for k, v in out.items() if v is None]:
            del out[k]
    return out


def _binary_metrics(y_true, proba, classes, pos_idx: int = 1) -> dict:
    """acc/sens/spec/F1/AUC with classes[pos_idx] as the positive class."""
    pred = classes[np.argmax(proba, axis=1)]
    p = proba[:, pos_idx]
    tn, fp, fn, tp = confusion_matrix(
        (y_true == classes[pos_idx]).astype(int),
        (pred == classes[pos_idx]).astype(int)).ravel()
    return {
        "acc": accuracy_score(y_true, pred),
        "sens": tp / max(tp + fn, 1),
        "spec": tn / max(tn + fp, 1),
        "f1": f1_score(y_true, pred, average="macro"),
        "auc": roc_auc_score((y_true == classes[pos_idx]).astype(int), p),
    }


def pooled_oof(pipe, X, y, groups, folds: int, seed: int,
               grouped: bool) -> np.ndarray:
    """Pooled out-of-fold probabilities under one protocol."""
    classes = np.unique(y)
    if grouped:
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                  random_state=seed)
        kw = dict(groups=groups)
    else:
        cv = StratifiedKFold(n_splits=folds, shuffle=True,
                             random_state=seed)
        kw = {}
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba",
                             **kw), classes


def run(X: np.ndarray, y, groups, folds: int = 5, repeats: int = 3,
        seed: int = 42, mini: bool = False) -> dict:
    """
    The full head-to-head.  X = raw intensities on the common grid (the
    report pipeline preprocesses internally here).  Returns a dict of
    per-model rows: report numbers, protocol-A metrics, protocol-B
    mean +- std, plus the Friedman p.
    """
    models = report_models()
    if mini:
        keep = [k for k in models if k in ("PCA+SVM", "PLS+XGB", "PCA+RF")]
        models = OrderedDict((k, models[k]) for k in keep)

    n_sel = modeling.TTestSelect().fit(X, np.asarray(y)).n_selected_
    print(f"[report] t-test filter selects {n_sel} wavenumbers "
          f"(the report: 31)")

    res = {"n_selected": n_sel, "models": {}}
    for name, pipe in models.items():
        row = {"report": REPORT_NUMBERS.get(name)}
        # A: report protocol (spectrum-level)
        proba, cls = pooled_oof(pipe, X, y, groups, folds, seed,
                                grouped=False)
        row["A"] = _binary_metrics(np.asarray(y), proba, cls)
        # B: patient-grouped x repeats
        per = []
        for r in range(repeats):
            proba, cls = pooled_oof(pipe, X, y, groups, folds,
                                    seed + 100 * r, grouped=True)
            per.append(_binary_metrics(np.asarray(y), proba, cls))
        row["B"] = {k: (float(np.mean([m[k] for m in per])),
                        float(np.std([m[k] for m in per])))
                    for k in per[0]}
        row["B_repeats"] = per
        res["models"][name] = row
        print(f"[report] {name:11s} A: F1 {row['A']['f1']:.3f}  "
              f"B: F1 {row['B']['f1'][0]:.3f}±{row['B']['f1'][1]:.3f}")
    if len(models) >= 3 and repeats >= 2:
        stat, p = friedmanchisquare(*[
            [m["f1"] for m in res["models"][n]["B_repeats"]]
            for n in models])
        res["friedman_p"] = float(p)
        print(f"[report] Friedman across models (grouped CV): "
              f"chi2={stat:.2f} p={p:.4f}")
    return res


def write_markdown(res: dict, path: str) -> None:
    lines = [
        "# Internship-report replication — report protocol vs "
        "patient-grouped CV",
        "",
        f"t-test filter: **{res['n_selected']} wavenumbers** selected "
        "(report: 31).  Pipeline: crop 700–1800 · ASLS · vector norm · "
        "coif5/SURE/soft · Welch-t+Cohen-d filter · PCA(5)/PLS(5).",
        "",
        "| Model | Report acc/sens/spec/F1/AUC | A: spectrum-CV F1 · AUC "
        "| B: grouped-CV F1 · AUC |",
        "|---|---|---|---|",
    ]
    for name, row in res["models"].items():
        rep = row["report"]
        rep_s = "/".join(f"{v:.3f}" for v in rep) if rep else "—"
        a, b = row["A"], row["B"]
        lines.append(
            f"| {name} | {rep_s} | {a['f1']:.3f} · {a['auc']:.3f} "
            f"| {b['f1'][0]:.3f}±{b['f1'][1]:.3f} · "
            f"{b['auc'][0]:.3f}±{b['auc'][1]:.3f} |")
    if "friedman_p" in res:
        lines += ["", f"Friedman across models (grouped CV): "
                      f"**p = {res['friedman_p']:.4f}**"]
    lines += [
        "",
        "Deviations from the report (honesty notes): the app pipeline "
        "adds Savitzky–Golay smoothing (11/3, app default; the report "
        "specifies none); metrics are pooled out-of-fold over ALL spectra "
        "(the report used a 29-spectrum holdout); protocol B groups by "
        "patient, which the report does not.",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--mini", action="store_true",
                    help="fast subset: PCA+SVM / PLS+XGB / PCA+RF")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(
        APP_DIR, "report_replication.md"))
    args = ap.parse_args(argv)

    if not os.path.isdir(args.data):
        print(f"[report] data folder not found: {args.data}")
        return 2
    import dataset as ds
    from clinical_data import load_clinical_dataset
    cd = load_clinical_dataset(args.data)
    spectra = cd.spectra
    labels = [s.label for s in spectra]
    groups = cd.groups or [f"S{i}" for i in range(len(spectra))]
    grid = ds.common_grid(spectra)
    X_raw, _ = ds.to_matrix(spectra, grid)
    keep = [i for i, lab in enumerate(labels) if lab.strip()]
    if cd.flagged and not pp.PreprocessParams().despike:
        keep = [i for i in keep if not cd.flagged[i]]
    y = np.array([labels[i].strip() for i in keep])
    g = [groups[i] for i in keep]
    print(f"[report] {len(y)} spectra / {len(set(g))} patients "
          f"/ classes {sorted(set(y))}")

    Xp = pp.preprocess_matrix(X_raw[keep], report_params(), wn=grid)
    print(f"[report] preprocessed matrix {Xp.shape}")

    res = run(Xp, y, g, folds=args.folds, repeats=args.repeats,
              seed=args.seed, mini=args.mini)
    write_markdown(res, args.out)
    print(f"[report] written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
