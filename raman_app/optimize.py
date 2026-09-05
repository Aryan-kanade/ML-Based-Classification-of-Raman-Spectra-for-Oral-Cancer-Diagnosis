"""
optimize.py — automatically tune the preprocessing pipeline on YOUR data.

Grid-searches a small set of pilot-validated preprocessing configurations
(crop range, Savitzky-Golay derivative, normalization) x three fast models
(PCA+SVM, RandomForest, PCA+LogReg) under patient-grouped cross-validation,
and reports the best combination by macro-F1.

The winner is written to vit_outputs/preprocess_best.json, which
vit_train.py picks up automatically (and the GUI can apply).

Usage:
    python optimize.py                     # default: ../Data clinical set
    python optimize.py --data <folder>     # any spectra folder
    python optimize.py --mode paired       # sweep on the margin deviations
    python optimize.py --folds 3           # faster, rougher
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import clone

import dataset as ds
import preprocessing as pp
from clinical_data import (is_clinical_layout,
                           load_clinical_dataset)

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_PARAMS_PATH = os.path.join(HERE, "vit_outputs", "preprocess_best.json")

# pilot-validated configuration grid (see project history): crop+deriv1
# was the single biggest win; entries vary one axis at a time.  arPLS
# (Baek 2015) variants included when pybaselines is available.
GRID: list[dict] = [
    dict(crop_min=0.0, crop_max=0.0, sg_deriv=0, norm="vector"),
    dict(crop_min=400.0, crop_max=1800.0, sg_deriv=0, norm="vector"),
    dict(crop_min=400.0, crop_max=1800.0, sg_deriv=0, norm="snv"),
    dict(crop_min=400.0, crop_max=1800.0, sg_deriv=1, norm="vector"),
    dict(crop_min=400.0, crop_max=1800.0, sg_deriv=1, norm="snv"),
    dict(crop_min=400.0, crop_max=1800.0, sg_deriv=2, norm="vector"),
    dict(crop_min=800.0, crop_max=1800.0, sg_deriv=1, norm="vector"),
    dict(crop_min=400.0, crop_max=1800.0, sg_deriv=0, norm="vector",
         als_lambda=1e6),
]
try:
    import pybaselines  # noqa: F401
    GRID += [
        dict(crop_min=400.0, crop_max=1800.0, sg_deriv=0, norm="vector",
             baseline_method="arpls"),
        dict(crop_min=400.0, crop_max=1800.0, sg_deriv=1, norm="vector",
             baseline_method="arpls"),
        dict(crop_min=400.0, crop_max=1800.0, sg_deriv=0, norm="snv",
             baseline_method="arpls"),
    ]
except ImportError:
    pass
# B4 (2026-09-05): the project's own measured-but-never-swept axes —
# Phe-1003 alignment (the literature note says it beat every intensity
# normalization), adaptive wavelet rules (bayes/garrote+cycle-spin
# measured −23% RMSE on Lorentzians), ALS asymmetry — one axis at a
# time around the best-known config (400-1800 / deriv1 / vector).
GRID += [
    dict(crop_min=400.0, crop_max=1800.0, sg_deriv=1, norm="vector",
         wn_calibrate=True),
    dict(crop_min=400.0, crop_max=1800.0, sg_deriv=1, norm="vector",
         wavelet_threshold="bayes"),
    dict(crop_min=400.0, crop_max=1800.0, sg_deriv=1, norm="vector",
         wavelet_threshold="garrote", wavelet_cycle=1),
    dict(crop_min=400.0, crop_max=1800.0, sg_deriv=0, norm="vector",
         als_p=0.001),
]


def _models() -> list[tuple[str, object]]:
    return [
        ("PCA+SVM", Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=0)),
            ("svm", SVC(kernel="rbf", class_weight="balanced"))])),
        ("RandomForest", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1,
            random_state=0)),
        ("PCA+LogReg", Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=0)),
            ("lr", LogisticRegression(max_iter=2000,
                                      class_weight="balanced"))])),
    ]


def _cv_macro_f1(est, X, y, groups, k: int, seed: int) -> tuple[float, float]:
    if groups is not None and len(set(groups)) >= k:
        sp = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                  random_state=seed)
        splits = sp.split(X, y, groups)
    else:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = skf.split(X, y)
    f1s = []
    for tr, te in splits:
        est = clone(est)
        est.fit(X[tr], y[tr])
        f1s.append(f1_score(y[te], est.predict(X[te]), average="macro"))
    return float(np.mean(f1s)), float(np.std(f1s))


def optimize_preprocessing(X_raw: np.ndarray, wn: np.ndarray, y: list[str],
                           groups: list[str] | None = None,
                           grid: list[dict] | None = None, k: int = 5,
                           seed: int = 42,
                           exclude: list[bool] | None = None,
                           progress=print) -> dict:
    """
    Search `grid` preprocessing configs x 3 fast models under (grouped) CV.
    Returns {"results": ranked list, "best": best entry,
             "params": winning PreprocessParams}.
    """
    if exclude is not None:
        keep = [i for i, bad in enumerate(exclude) if not bad]
        X_raw = X_raw[keep]
        y = [y[i] for i in keep]
        groups = [groups[i] for i in keep] if groups is not None else None
    y_arr = np.asarray(y)
    grid = grid if grid is not None else GRID
    results = []
    t0 = time.time()
    for ci, over in enumerate(grid, start=1):
        params = replace(pp.PreprocessParams(), **over).validate()
        label = (f"crop {over.get('crop_min', 0):.0f}-{over.get('crop_max', 0):.0f}"
                 f" / deriv {params.sg_deriv} / {params.norm}"
                 + (f" / {params.baseline_method.upper()}"
                    if over.get("baseline_method") == "arpls" else "")
                 + (f" / ALS {params.als_lambda:.0e}"
                    if "als_lambda" in over else "")
                 + (" / cal-1003" if params.wn_calibrate else "")
                 + (f" / {params.wavelet_threshold}"
                    if over.get("wavelet_threshold") else "")
                 + ("+spin" if params.wavelet_cycle else "")
                 + (f" / p={params.als_p}" if "als_p" in over else ""))
        try:
            Xp = pp.preprocess_matrix(X_raw, params, wn=wn)
        except Exception as exc:
            progress(f"[{ci}/{len(grid)}] {label}: FAILED ({exc})")
            continue
        best_here = None
        for mname, est in _models():
            mean, std = _cv_macro_f1(est, Xp, y_arr, groups, k, seed)
            if best_here is None or mean > best_here["mean_f1"]:
                best_here = {"params_over": over, "params": params,
                             "model": mname, "mean_f1": mean, "std_f1": std,
                             "n_features": int(Xp.shape[1])}
        best_here["label"] = label
        results.append(best_here)
        progress(f"[{ci}/{len(grid)}] {label}: best {best_here['model']} "
                 f"macro-F1 {best_here['mean_f1']:.3f} "
                 f"± {best_here['std_f1']:.3f} "
                 f"({best_here['n_features']} pts)")
    results.sort(key=lambda r: r["mean_f1"], reverse=True)
    progress(f"Optimization finished in {time.time() - t0:.0f}s — "
             f"winner: {results[0]['label']} with {results[0]['model']} "
             f"(macro-F1 {results[0]['mean_f1']:.3f})")
    return {"results": results, "best": results[0], "params": results[0]["params"]}


def save_best_params(params: pp.PreprocessParams, score: float,
                     model: str, path: str = BEST_PARAMS_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"preprocess": asdict(params), "score_macro_f1": score,
                   "model": model}, fh, indent=2)
    return path


# ---- B2 (2026-09-05 program): paired-mode sweep --------------------------
# The deviation features the strong mode uses have NEVER been swept —
# the old GRID only ever scored absolute spectra. One axis at a time
# around the current defaults (crop 500/2000, deriv 0, vector, ALS 1e5,
# despike off), plus the measured suspects: deriv 1 (+0.05 F1 in
# standard mode), snv, arPLS, λ 1e6, despike ON, and the PQN variant.
GRID_PAIRED: list[dict] = [
    dict(),                                            # current defaults
    dict(sg_deriv=1),
    dict(sg_deriv=2),
    dict(norm="snv"),
    dict(crop_min=400.0, crop_max=1800.0),
    dict(crop_min=400.0, crop_max=1800.0, sg_deriv=1),
    dict(crop_min=800.0, crop_max=1800.0, sg_deriv=1),
    dict(baseline_method="arpls"),
    dict(baseline_method="arpls", sg_deriv=1),
    dict(als_lambda=1e6),
    dict(despike=True),
    dict(despike=True, sg_deriv=1),
    dict(despike=True, baseline_method="arpls"),
    dict(use_pqn=True),
    dict(use_pqn=True, sg_deriv=1),
    # B4: alignment / adaptive-wavelet / ALS-p axes for the paired
    # deviations too (same rationale as the standard GRID additions)
    dict(wn_calibrate=True, sg_deriv=1),
    dict(wavelet_threshold="bayes", sg_deriv=1),
    dict(wavelet_threshold="garrote", wavelet_cycle=1, sg_deriv=1),
    dict(als_p=0.001),
]


def optimize_paired(X_raw: np.ndarray, y: list[str], groups, wn: np.ndarray,
                    k: int = 5, seed: int = 42,
                    grid: list[dict] | None = None, progress=print) -> dict:
    """
    B2: sweep preprocessing configs ON the PAIRED deviation features
    (paired_features preprocesses internally, so every config rebuilds
    the deviations — exactly what a paired-mode Train run would see).
    Same return shape as optimize_preprocessing: ranked results, best
    entry and winning PreprocessParams.  `use_pqn` entries are scored
    for comparison; the winner note says to switch mode if PQN wins.
    """
    import paired as pmod
    grid = grid if grid is not None else GRID_PAIRED
    results = []
    t0 = time.time()
    for ci, over in enumerate(grid, start=1):
        over = dict(over)
        use_pqn = over.pop("use_pqn", False)
        params = replace(pp.PreprocessParams(), **over).validate()
        label = (f"crop {params.crop_min:.0f}-{params.crop_max:.0f}"
                 f" / deriv {params.sg_deriv} / {params.norm}"
                 + (f" / {params.baseline_method.upper()}"
                    if params.baseline_method == "arpls" else "")
                 + (f" / ALS {params.als_lambda:.0e}"
                    if params.als_lambda != 1e5 else "")
                 + (" / despike" if params.despike else "")
                 + (" / PQN" if use_pqn else "")
                 + (" / cal-1003" if params.wn_calibrate else "")
                 + (f" / {params.wavelet_threshold}"
                    if over.get("wavelet_threshold") else "")
                 + ("+spin" if params.wavelet_cycle else "")
                 + (f" / p={params.als_p}" if "als_p" in over else ""))
        try:
            pd_ = pmod.paired_features(X_raw, y, groups, wn, params,
                                       use_pqn=use_pqn)
        except Exception as exc:
            progress(f"[{ci}/{len(grid)}] {label}: FAILED ({exc})")
            continue
        best_here = None
        for mname, est in _models():
            mean, std = _cv_macro_f1(est, pd_.X, np.asarray(pd_.y),
                                     pd_.groups, k, seed)
            if best_here is None or mean > best_here["mean_f1"]:
                best_here = {"params": params, "model": mname,
                             "mean_f1": mean, "std_f1": std,
                             "n_features": int(pd_.X.shape[1]),
                             "use_pqn": use_pqn}
        best_here["label"] = label
        results.append(best_here)
        progress(f"[{ci}/{len(grid)}] {label}: best {best_here['model']} "
                 f"macro-F1 {best_here['mean_f1']:.3f} "
                 f"± {best_here['std_f1']:.3f}")
    results.sort(key=lambda r: r["mean_f1"], reverse=True)
    progress(f"Paired sweep finished in {time.time() - t0:.0f}s — "
             f"winner: {results[0]['label']} with {results[0]['model']} "
             f"(macro-F1 {results[0]['mean_f1']:.3f})"
             + ("  [switch mode to Margin+PQN]" if results[0]["use_pqn"]
                else ""))
    return {"results": results, "best": results[0],
            "params": results[0]["params"]}


def load_best_params(path: str = BEST_PARAMS_PATH
                     ) -> tuple[pp.PreprocessParams, dict] | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return pp.PreprocessParams(**data["preprocess"]), data
    except Exception:
        return None


def _load_any(folder: str):
    """Clinical layout if present, else a flat folder of labelled spectra."""
    if is_clinical_layout(folder):
        cd = load_clinical_dataset(folder)
        X_raw, _ = ds.to_matrix(cd.spectra, cd.grid)
        y = [s.label for s in cd.spectra]
        return X_raw, cd.grid, y, cd.groups, cd.flagged, cd.report
    spectra = [s for s in ds.load_folder(folder) if s.label]
    if len(spectra) < 10:
        raise SystemExit(f"Not enough labelled spectra in {folder}")
    grid = ds.common_grid(spectra)
    X_raw, y = ds.to_matrix(spectra, grid)
    return X_raw, grid, y, None, None, {}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Auto-tune the preprocessing pipeline (grouped CV)")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(here),
                                                   "Data"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-flagged", action="store_true",
                    help="include low-quality (spiked) spectra")
    ap.add_argument("--mode", choices=["standard", "paired"],
                    default="standard",
                    help="paired: sweep ON the margin deviation features "
                         "(clinical data with patient groups)")
    ap.add_argument("--out", default=BEST_PARAMS_PATH)
    args = ap.parse_args()

    X_raw, wn, y, groups, flagged, report = _load_any(args.data)
    exclude = None if (args.keep_flagged or flagged is None) else flagged
    if exclude:
        print(f"Excluding {sum(exclude)} low-quality (spiked) spectra "
              f"of {len(y)}")
        if args.mode == "paired":        # optimize_paired has no exclude kw
            keep = [i for i, bad in enumerate(exclude) if not bad]
            X_raw = X_raw[keep]
            y = [y[i] for i in keep]
            groups = [groups[i] for i in keep]
    print(f"Optimizing preprocessing ({args.mode} mode) on {len(y)} spectra "
          f"({len(set(groups)) if groups else '-'} patients), "
          f"{args.folds}-fold grouped CV\n")
    if args.mode == "paired":
        if not groups:
            raise SystemExit("paired mode needs patient groups "
                             "(clinical data layout)")
        out = optimize_paired(X_raw, y, groups, wn, k=args.folds,
                              seed=args.seed)
    else:
        out = optimize_preprocessing(X_raw, wn, y, groups=groups,
                                     k=args.folds, seed=args.seed,
                                     exclude=exclude)

    print("\nRanked results (macro-F1, grouped CV):")
    for i, r in enumerate(out["results"], start=1):
        print(f"  {i:2d}. {r['label']:<46} {r['model']:<13} "
              f"{r['mean_f1']:.3f} ± {r['std_f1']:.3f}")
    path = save_best_params(out["params"], out["best"]["mean_f1"],
                            out["best"]["model"], args.out)
    print(f"\nSaved best preprocessing: {path}")
    print("vit_train.py and the GUI can now use these settings.")


if __name__ == "__main__":
    main()
