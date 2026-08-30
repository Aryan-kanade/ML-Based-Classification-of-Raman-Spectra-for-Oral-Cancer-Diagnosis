"""
smoke_test.py — headless end-to-end test (no GUI).

1. generates a demo dataset from the real spectrum (if present)
2. loads it and preprocesses it
3. cross-validates the model suite (sensitivity/specificity/F1)
4. saves the best model and predicts a held-out file

Run:  python smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

import dataset
import modeling
import preprocessing
from preprocessing import PreprocessParams

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(APP_DIR, "demo_data")


def main() -> int:
    print("=" * 70)
    print("SMOKE TEST — Raman classifier pipeline")
    print("=" * 70)

    # 1. demo data (isolated temp folder — DEMO_DIR may contain user data)
    source = dataset.find_default_source_spectrum()
    if not source:
        print("No real source spectrum found — FAIL")
        return 1
    print(f"[1] source spectrum: {source}")
    out = dataset.generate_demo_data(
        source, tempfile.mkdtemp(prefix="smoke_demo_"), n_per_class=15)
    spectra = dataset.load_folder(out)
    labels = [s.label for s in spectra]
    print(f"    generated {len(spectra)} spectra, "
          f"classes={sorted(set(labels))}")
    assert len(spectra) == 45, f"expected 45, got {len(spectra)}"
    assert sorted(set(labels)) == ["C1", "C5", "C8"], labels

    # 2. preprocess --------------------------------------------------------
    grid = dataset.common_grid(spectra)
    X_raw, _ = dataset.to_matrix(spectra, grid, labels)
    params = PreprocessParams()
    X = preprocessing.preprocess_matrix(X_raw, params, wn=grid)
    assert np.isfinite(X).all()
    print(f"[2] preprocessed X: {X.shape}  "
          f"(grid {grid.min():.0f}-{grid.max():.0f} cm-1, "
          f"crop {params.crop_min:.0f}-{params.crop_max:.0f})")

    # 3. cross-validate the FULL suite --------------------------------------
    print(f"[3] evaluating ALL models: {modeling.ALL_MODEL_NAMES}")
    results, winner = modeling.evaluate_models(
        X, labels, model_names=None, k_folds=5,
        progress_cb=lambda pct, msg: print(f"    {pct:3d}%  {msg}"))

    print(f"\n    {'Model':<28}{'Sens':<16}{'Spec':<16}{'F1':<16}")
    print("    " + "-" * 76)
    for r in results:
        if r.error:
            print(f"    {r.name:<28}FAILED")
            continue
        s, sp, f1 = r.summary_row()[1:]
        mark = "  <- BEST" if r is winner else ""
        print(f"    {r.name:<28}{s:<16}{sp:<16}{f1:<16}{mark}")

    w_f1 = winner.macro_f1()
    print(f"\n    winner: {winner.name}  macro-F1={w_f1:.3f}")
    for cls, per in winner.per_class.items():
        print(f"      class {cls}: sens={per['sens'][0]:.3f} "
              f"spec={per['spec'][0]:.3f} f1={per['f1'][0]:.3f}")
    assert w_f1 > 0.6, f"demo data should be easy; got F1={w_f1:.3f}"

    # 4. save + predict ----------------------------------------------------
    path = os.path.join(tempfile.gettempdir(), "smoke_model.joblib")
    modeling.save_bundle(path, winner, grid, params, "demo")
    bundle = modeling.load_bundle(path)
    test_file = spectra[0].path          # a C1 demo file
    wn, it = dataset.load_spectrum(test_file)
    out_pred = modeling.predict_with_bundle(bundle, wn, it)
    print(f"\n[4] saved {path}")
    print(f"    predict {os.path.basename(test_file)} -> "
          f"{out_pred['prediction']}  probs={out_pred['probabilities']}")
    assert out_pred["prediction"] == "C1", out_pred

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
