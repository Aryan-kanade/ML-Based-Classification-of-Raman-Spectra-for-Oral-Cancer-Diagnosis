"""
eval_multiview.py -- L16: multi-view preprocessing concat.  hstack
complementary preprocessing views of the SAME paired deviations
(deriv-2 baseline | deriv-1 | vector-norm) so tree layers see both
shape and amplitude structure.  Rows align because row selection is
independent of deriv/norm settings (verified by assertion).
SAFETY: pure experiment script -> experiments/ only.
"""

from __future__ import annotations

import json
import os

import numpy as np

import dataset as ds
import paired as pmod
import sequential
from clinical_data import load_clinical_dataset
from exp_common import OUT_DIR, oof_f1, record, stamp
from run_3sse_d2 import WINNER

import preprocessing as pp


def _paired_view(params) -> tuple[np.ndarray, list, list, np.ndarray]:
    cd = load_clinical_dataset(sequential.DEFAULT_DATA)
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)
    labels = [s.label for s in cd.spectra]
    flagged = cd.flagged or [False] * len(labels)
    keep = [i for i, lab in enumerate(labels)
            if lab.strip() and (params.despike or not flagged[i])]
    y = [labels[i].strip() for i in keep]
    g = [cd.groups[i] for i in keep]
    pd_ = pmod.paired_features(X_raw[keep], y, g, grid, params)
    return np.asarray(pd_.X, dtype=np.float32), list(pd_.y), \
        list(pd_.groups), np.asarray(pd_.wn)


def _view(**over) -> pp.PreprocessParams:
    d = dict(crop_min=WINNER.crop_min, crop_max=WINNER.crop_max,
             sg_deriv=WINNER.sg_deriv, norm=WINNER.norm,
             sg_window=WINNER.sg_window, sg_poly=WINNER.sg_poly,
             wavelet_name=WINNER.wavelet_name,
             wavelet_level=WINNER.wavelet_level)
    d.update(over)
    return pp.PreprocessParams(**d).validate()


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE_DIR, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]
    print(f"[view] {stamp()} arch {' -> '.join(arch)}", flush=True)

    X0, y, groups, wn = _paired_view(_view())
    X1, y1, g1, wn1 = _paired_view(_view(sg_deriv=1))
    X2, y2, g2, wn2 = _paired_view(_view(norm="vector"))
    assert (X0.shape[0], len(y), len(groups)) == \
        (X1.shape[0], len(y1), len(g1)) == \
        (X2.shape[0], len(y2), len(g2)), "view rows misaligned"
    # normalize scales per view so no block dominates
    def _std(a):
        return (a - a.mean(axis=1, keepdims=True)) / \
            (a.std(axis=1, keepdims=True) + 1e-9)
    Xv = np.hstack([_std(X0), _std(X1), _std(X2)])
    print(f"[view] multiview matrix: {Xv.shape[1]} features "
          f"(3 x {X0.shape[1]})", flush=True)

    factories = sequential._factories(wn)
    m = sequential.validate_arch(arch, factories, Xv, y, groups,
                                 seed=42, k_outer=5, wn=None)
    f1 = oof_f1(m)
    print(f"[view] multiview F1 {f1:.3f} (baseline 0.753)", flush=True)
    record("L16 multiview", "deriv2|deriv1|vector concat (std-normed)",
           f1)
    return 0


HERE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    raise SystemExit(main())
