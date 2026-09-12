"""
prep_3sse_check.py -- does the deep-search preprocessing winner transfer
to the FULL model registry 3SSE uses? (2026-09-12, follow-up to §52)

The deep search (prep_deep_search.py) selected/validated with 3+2
classical models.  3SSE chains EVERY registry model on the same
features.  This script measures, in paired mode (3SSE's best mode),
defaults vs the deep-search winner across a diverse registry panel:

    PCA+SVM, PCA+LDA, PCA+LogReg, RF, ExtraTrees, HGB, PLS-DA,
    XGBoost, LightGBM, CatBoost, Peak bands+RF, Spectral+band,
    Ensemble (top-3)

through modeling.evaluate_models (nested grouped CV, k=5, seed 42) --
the exact estimator path a 3SSE validate_top run uses.  1D-CNN /
TabPFN excluded for runtime (historically weak at n~300; SLOW_MODELS).

Usage:  python prep_3sse_check.py
Output: vit_outputs/prep_3sse_check.json + console table
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

os.environ.setdefault("RAMAN_DEVICE", "cpu")

import dataset as ds
import modeling
import preprocessing as pp
from clinical_data import find_data_root, load_clinical_dataset
from prep_deep_search import Scorer, describe

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vit_outputs", "prep_3sse_check.json")

MODELS = [
    "PCA + SVM (RBF)", "PCA + LDA", "PCA + Logistic Regression",
    "Random Forest", "Extra Trees", "Hist Gradient Boosting",
    "PLS-DA", "XGBoost", "LightGBM", "CatBoost",
    "Peak bands + RF", "Spectral + band features",
    "Ensemble (top-3)",
]

DEFAULTS = pp.PreprocessParams().validate()          # crop 500-2000 d0 vector
WINNER = pp.PreprocessParams(                        # §52 paired winner
    crop_min=0.0, crop_max=0.0, sg_deriv=2, norm="none",
    sg_window=11, sg_poly=4, wavelet_name="db6", wavelet_level=2,
).validate()


def main():
    root = find_data_root()
    if not root:
        raise SystemExit("no data root")
    cd = load_clinical_dataset(root)
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)
    labels = [s.label for s in cd.spectra]
    groups = cd.groups or None

    scorer = Scorer(X_raw, grid, labels, groups, cd.flagged or None,
                    mode="paired", k=5, seed=42)
    out = {"models": MODELS, "runs": {}}
    for tag, params in (("defaults", DEFAULTS), ("winner", WINNER)):
        print(f"\n=== {tag}: {describe(params)}", flush=True)
        X, y, g, wn_c = scorer.build(params)
        print(f"    features {X.shape[0]}x{X.shape[1]}", flush=True)
        t0 = time.time()
        results, winner = modeling.evaluate_models(
            X, list(y), model_names=MODELS, k_folds=5, seed=42,
            groups=g, wavenumbers=wn_c)
        rows = {}
        for r in results:
            if r.error:
                rows[r.name] = {"error": r.error}
                print(f"  {r.name:<28} ERROR {r.error}", flush=True)
                continue
            rows[r.name] = {"f1": r.macro_f1(),
                            "sens": r.macro.get("sens", (float("nan"), 0))[0],
                            "spec": r.macro.get("spec", (float("nan"), 0))[0]}
            print(f"  {r.name:<28} F1 {rows[r.name]['f1']:.3f}  "
                  f"sens {rows[r.name]['sens']:.3f}  "
                  f"spec {rows[r.name]['spec']:.3f}", flush=True)
        print(f"  -> winner: {winner.name} F1 {winner.macro_f1():.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        out["runs"][tag] = {"params": asdict(params), "rows": rows,
                            "winner": winner.name,
                            "winner_f1": winner.macro_f1()}

    # side-by-side delta
    print("\n=== defaults -> winner (macro-F1) ===", flush=True)
    out["delta"] = {}
    for m in MODELS:
        f_d = out["runs"]["defaults"]["rows"].get(m, {}).get("f1")
        f_w = out["runs"]["winner"]["rows"].get(m, {}).get("f1")
        if f_d is None or f_w is None:
            continue
        d = f_w - f_d
        out["delta"][m] = {"defaults": f_d, "winner": f_w, "delta": d}
        print(f"  {m:<28} {f_d:.3f} -> {f_w:.3f}  ({d:+.3f})",
              flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
