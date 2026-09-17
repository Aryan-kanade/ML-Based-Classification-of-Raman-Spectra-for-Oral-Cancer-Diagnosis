"""
eval_windows.py -- L34 informative-window systematics (SAFETY: pure
experiment script -> experiments/ only).

The current no-crop config feeds 2000 columns of which the 1800-2800
cm-1 SILENT region (no tissue Raman peaks) and the >3200 water band
are pure noise.  Arms (each scored by the winning chain, identical
folds):
  fp_d{0,1,2}      fingerprint 400-1800 only
  fphigh_d{0,1,2}  fingerprint + high-wavenumber 2800-3050 (concat)
  drop_d{0,1,2}    full range minus silent minus >3200 (post-mask)
"""

from __future__ import annotations

import json
import os

import numpy as np

import paired as pmod
import preprocessing as pp
import sequential
from clinical_data import load_clinical_dataset
from exp_common import HERE, OUT_DIR, record, stamp
from run_3sse_d2 import WINNER

import dataset as ds


def _view(crop_min, crop_max, deriv):
    p = pp.PreprocessParams(
        crop_min=crop_min, crop_max=crop_max, sg_deriv=deriv,
        norm=WINNER.norm, sg_window=WINNER.sg_window,
        sg_poly=WINNER.sg_poly, wavelet_name=WINNER.wavelet_name,
        wavelet_level=WINNER.wavelet_level).validate()
    cd = load_clinical_dataset(sequential.DEFAULT_DATA)
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)
    labels = [s.label for s in cd.spectra]
    flagged = cd.flagged or [False] * len(labels)
    keep = [i for i, lab in enumerate(labels)
            if lab.strip() and (p.despike or not flagged[i])]
    y = [labels[i].strip() for i in keep]
    g = [cd.groups[i] for i in keep]
    pd_ = pmod.paired_features(X_raw[keep], y, g, grid, p)
    return np.asarray(pd_.X, dtype=np.float32), list(pd_.y), \
        list(pd_.groups), np.asarray(pd_.wn)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]
    print(f"[win] {stamp()} arch {' -> '.join(arch)}", flush=True)

    # full-range view once, for the drop-mask arm
    Xfull, y, groups, wn = _view(0.0, 0.0, WINNER.sg_deriv)
    mask = ~(((wn >= 1800) & (wn <= 2800)) | (wn > 3200))

    arms = {}
    for d in (0, 1, 2):
        Xf, yf, gf, wnf = _view(400.0, 1800.0, d)
        Xh, yh, gh, wnh = _view(2800.0, 3050.0, d)
        Xd, yd, gd, wnd = _view(0.0, 0.0, d)
        if not (Xf.shape[0] == Xh.shape[0] == Xd.shape[0]
                and yf == y == yh == yd and gf == groups == gh == gd):
            print(f"[win] d={d}: view rows misaligned — skipped",
                  flush=True)
            continue
        arms[f"fp_d{d}"] = (Xf, yf, gf, wnf)
        arms[f"fphigh_d{d}"] = (np.hstack([Xf, Xh]), yf, gf,
                                np.concatenate([wnf, wnh]))
        arms[f"drop_d{d}"] = (Xd[:, mask], yd, gd, wnd[mask])

    results = {}
    for name, (Xa, ya, ga, wna) in arms.items():
        facs = sequential._factories(wna)
        m = sequential.validate_arch(arch, facs, Xa, ya, ga, seed=42,
                                     k_outer=5, wn=wna)
        f1 = sequential_oof_f1(m)
        results[name] = f1
        print(f"[win] {name:<12}: F1 {f1:.3f}", flush=True)
        record("L34 windows", name, f1)
    best = max(results, key=results.get)
    print(f"[win] BEST: {best} F1 {results[best]:.3f}", flush=True)
    return 0


def sequential_oof_f1(m: dict) -> float:
    return float(m.get("f1_mean", m.get("f1", float("nan"))))


if __name__ == "__main__":
    raise SystemExit(main())
