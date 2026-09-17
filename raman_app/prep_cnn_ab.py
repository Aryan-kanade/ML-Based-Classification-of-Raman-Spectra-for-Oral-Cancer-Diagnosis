"""
prep_cnn_ab.py -- A/B test: does the §53 CNN upgrade (attention gate /
cosine LR) beat the current 1D-CNN on the real data?

Four variants, identical folds/seeds (paired mode, §52 winner
preprocessing, grouped 5-fold x seeds {42,43,44}):

    baseline            CNN1DClassifier()                    (legacy)
    attention           CNN1DClassifier(attention=True)
    cosine              CNN1DClassifier(scheduler="cosine")
    attention+cosine    CNN1DClassifier(attention=True, scheduler="cosine")
                        (= the new "1D-CNN (attention)" registry entry)

Plus a Grad-CAM sanity check on the attention net (top cam peaks vs
raw-energy peaks).

Usage:  python prep_cnn_ab.py
Output: vit_outputs/prep_cnn_ab.json + console table
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("RAMAN_DEVICE", "cpu")   # keep fits deterministic

import numpy as np

import dataset as ds
import modeling
import optimize as opt
import preprocessing as pp
from clinical_data import find_data_root, load_clinical_dataset
from prep_deep_search import Scorer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vit_outputs", "prep_cnn_ab.json")

WINNER = pp.PreprocessParams(                        # §52 paired winner
    crop_min=0.0, crop_max=0.0, sg_deriv=2, norm="none",
    sg_window=11, sg_poly=4, wavelet_name="db6", wavelet_level=2,
).validate()

VARIANTS = [
    ("baseline", modeling.CNN1DClassifier()),
    ("attention", modeling.CNN1DClassifier(attention=True)),
    ("cosine", modeling.CNN1DClassifier(scheduler="cosine")),
    ("attention+cosine",
     modeling.CNN1DClassifier(attention=True, scheduler="cosine")),
]
SEEDS = [42, 43, 44]


def main():
    root = find_data_root()
    if not root:
        raise SystemExit("no data root")
    cd = load_clinical_dataset(root)
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)
    scorer = Scorer(X_raw, grid, [s.label for s in cd.spectra],
                    cd.groups, cd.flagged, mode="paired", k=5, seed=42)
    X, y, g, wn_c = scorer.build(WINNER)
    y = np.asarray(y)
    print(f"paired features {X.shape} | "
          f"variants {[n for n, _ in VARIANTS]}", flush=True)

    out = {"seeds": SEEDS, "rows": {}}
    for tag, est0 in VARIANTS:
        f1s = []
        t0 = time.time()
        for s in SEEDS:
            params = est0.get_params()
            params["seed"] = s
            est = modeling.CNN1DClassifier(**params)
            mean, std = opt._cv_macro_f1(est, X, y, g, 5, s)
            f1s.append(mean)
            print(f"  {tag:<18} seed {s}: F1 {mean:.3f} ± {std:.3f}",
                  flush=True)
        row = {"f1s": [round(f, 4) for f in f1s],
               "mean": float(np.mean(f1s)),
               "std": float(np.std(f1s)),
               "seconds": round(time.time() - t0)}
        out["rows"][tag] = row
        print(f"== {tag}: {row['mean']:.3f} ± {row['std']:.3f} "
              f"({row['seconds']}s)", flush=True)

    # Grad-CAM sanity: fit the attention+cosine net on ALL rows — do
    # the cam peaks differ from raw |intensity| energy peaks?
    params = VARIANTS[3][1].get_params()
    params["seed"] = 42
    est = modeling.CNN1DClassifier(**params).fit(X, list(y))
    cam = est.grad_cam(X[:40]).mean(axis=0)
    top_cam = wn_c[np.argsort(cam)[-3:][::-1]]
    energy = np.abs(X[:40]).mean(axis=0)
    top_energy = wn_c[np.argsort(energy)[-3:][::-1]]
    out["grad_cam"] = {"top_cam_cm": [round(float(v), 1) for v in top_cam],
                       "top_energy_cm": [round(float(v), 1)
                                         for v in top_energy]}
    print(f"grad-cam top bands: {out['grad_cam']['top_cam_cm']} | "
          f"raw-energy top: {out['grad_cam']['top_energy_cm']}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
