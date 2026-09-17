"""
run_3sse_d2.py -- 3SSE architecture search on the §52 WINNER
preprocessing (paired mode): no-crop, deriv 2, norm none, db6 L2,
SG 11/4.  Delegates to sequential.main() with prepare_dataset
monkey-patched, so every artifact (screening.jsonl, validated.json,
report.txt, winner.joblib) is produced by the standard machinery.

Model subset: PCA + LDA and PLS-DA are EXCLUDED (measured -0.118 /
-0.055 on deriv-2 features, prep_3sse_check), the torch CNNs and
TabPFN are excluded (slow, not part of the transferability check).
Output: study_run_3sse_d2/ (the old study_run_3sse stays untouched).

Usage:  python run_3sse_d2.py [--k 5] [--top 20] [--resume]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("RAMAN_DEVICE", "cpu")   # 3SSE loky children: CPU

import dataset as ds
import paired as pmod
import preprocessing as pp
import sequential
from clinical_data import load_clinical_dataset

WINNER = pp.PreprocessParams(                        # §52 paired winner
    crop_min=0.0, crop_max=0.0, sg_deriv=2, norm="none",
    sg_window=11, sg_poly=4, wavelet_name="db6", wavelet_level=2,
).validate()

EXCLUDE = {
    "PCA + LDA", "PLS-DA", "Sparse PLS-DA",           # deriv-2 losers
    "1D-CNN", "1D-CNN (attention)",
    "1D-CNN ensemble (5 seeds)", "TabPFN (foundation model)",
}


def _prepare_winner(root: str, mode: str = "standard"):
    """sequential.prepare_dataset replacement built on the §52 winner
    params (paired deviations, spike-flagged excluded — GUI parity)."""
    cd = load_clinical_dataset(root)
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)
    labels = [s.label for s in cd.spectra]
    flagged = cd.flagged or [False] * len(labels)
    keep = [i for i, lab in enumerate(labels)
            if lab.strip() and (WINNER.despike or not flagged[i])]
    y = [labels[i].strip() for i in keep]
    g = [cd.groups[i] for i in keep]
    pd_ = pmod.paired_features(X_raw[keep], y, g, grid, WINNER)
    meta = {"n_spectra": len(pd_.y),
            "n_patients": len(set(pd_.groups)), "mode": "paired",
            "preprocessing": "d2-noCrop-none-db6L2-SG11-4 (§52 winner)"}
    print(f"[3sse-d2] features {pd_.X.shape[0]}x{pd_.X.shape[1]} | "
          f"{meta['n_patients']} patients | {meta['preprocessing']}",
          flush=True)
    return pd_.X, pd_.y, pd_.groups, pd_.wn, grid, WINNER, meta


def main(argv=None) -> int:
    import modeling
    names = [s["name"] for s in modeling.model_specs()
             if s["name"] not in EXCLUDE]
    print(f"[3sse-d2] models ({len(names)}): {', '.join(names)}",
          flush=True)
    sequential.prepare_dataset = _prepare_winner
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "study_run_3sse_d2")
    args = ["--mode", "paired", "--k", "5", "--top",
            "20", "--prune-pairs", "50", "--out", out_dir,
            "--models", ",".join(names)]
    args += [a for a in (argv or []) if a.startswith("--resume")]
    return sequential.main(args)


if __name__ == "__main__":
    sys.exit(main())
