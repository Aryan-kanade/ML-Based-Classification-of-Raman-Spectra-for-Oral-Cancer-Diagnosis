"""
convert_rsclass.py -- rebuild of the lost §16d converter (2026-09-16).
Raw source: RSClassification release dataV1.0.0 oral_cancer
(ISCLab-Bistu/RSClassification) — 3,535 spectra x 1,038 wavenumbers
(1.7-4075 cm^-1), labels 0=Health 1=Benign 2/3/4=TNM-staged malignant
(raman_type strings carry the stage).

Binary task: Health -> Normal (0), malignant (2,3,4) -> Tumor (1).
Benign EXCLUDED from the binary task (ambiguous adjacency; including
it as Normal would import label noise).  Stage strings are preserved
for the future TNM multi-task arm.

Usage:  python convert_rsclass.py [--src D:/BARC/_external/rsclass/...]
Output: _external/processed/oral_binary.npz  (X, y, wn, tnm, raw_label)
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(
    os.path.dirname(HERE), "_external", "rsclass", "data",
    "oral_cancer", "results", "oral_cancer.csv")
OUT = os.path.join(os.path.dirname(HERE), "_external", "processed",
                   "oral_binary.npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    args = ap.parse_args()
    with open(args.src, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    wn = np.asarray([float(v) for v in rows[1][2:]])
    X, y, tnm, raw = [], [], [], []
    for r in rows[2:]:
        lab = int(r[1])
        if lab == 1:            # Benign: excluded from binary task
            continue
        X.append([float(v) for v in r[2:]])
        y.append(1 if lab >= 2 else 0)
        tnm.append(r[0] if lab >= 2 else "Health")
        raw.append(lab)
    X = np.asarray(X, dtype=np.float32)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, X=X, y=np.asarray(y), wn=wn,
                        tnm=np.asarray(tnm), raw=np.asarray(raw))
    n_t = int((np.asarray(y) == 1).sum())
    print(f"[rsclass] {X.shape[0]} spectra ({n_t} tumor / "
          f"{X.shape[0] - n_t} normal) x {X.shape[1]} pts -> {OUT}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
