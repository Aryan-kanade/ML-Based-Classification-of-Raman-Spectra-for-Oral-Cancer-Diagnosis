"""
paired.py — paired-reference classification (tumor-margin scenario).

Pilot finding on the clinical dataset (patient-grouped CV):
  * ~61% of the spectral variance is BETWEEN patients (one-way ANOVA);
  * defining each spectrum as its DEVIATION FROM THE PATIENT'S OWN MEAN
    NORMAL reference removes that offset and separates the classes far
    better: macro-F1 0.571 -> 0.709 (PCA+SVC), positive in 5/5 folds,
    while label-free patient centering alone gains nothing.
This module builds those paired-reference features.  It models the
clinical scenario where a normal reference site from the SAME patient is
measured alongside the suspect tissue (e.g. margin assessment).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

import preprocessing as pp


@dataclass
class PairedData:
    X: np.ndarray                    # deviation features
    y: list[str]                     # class per row
    groups: list[str]                # patient key per row
    wn: np.ndarray                   # (cropped) wavenumber axis
    n_patients: int = 0
    n_unpaired_excluded: int = 0
    n_unlabeled_dropped: int = 0     # rows with an empty label (2026-09-12)
    n_self_ref_rows: int = 0         # k=1-normal rows (deviation ≡ 0)


def paired_features(spectra_X: np.ndarray, labels: list[str],
                    groups: list[str], wn_full: np.ndarray,
                    params: pp.PreprocessParams,
                    exclude: list[bool] | None = None,
                    use_pqn: bool = False
                    ) -> "PairedData":
    """
    Build deviation-from-own-normal features.

    spectra_X: raw intensity matrix on the common grid wn_full.
    labels/groups: one entry per row; exclude: flagged rows to skip.
    For every patient with at least one Normal AND one Tumor spectrum,
    TUMOR rows use the mean of the patient's (preprocessed) Normal
    spectra as reference; NORMAL rows use the LEAVE-ONE-OUT mean of the
    patient's OTHER normals (2026-09-12 audit: including the row itself
    shrank Normal deviations to exactly (k-1)/k — 0.0 for k=1 — and
    mismatched deploy time, where references never contain the spectrum
    being classified).  k=1 normals still yield a zero deviation; they
    are kept but counted in `n_self_ref_rows`.  Patients without a
    normal reference are excluded (counted).

    With use_pqn=True each spectrum is first PQN-normalized (Dieterle
    2006) against its own reference — leakage-free by construction —
    which corrects coupling/dilution scale differences before the
    deviation is taken.
    """
    keep = [i for i in range(len(labels))
            if labels[i] and (exclude is None or not exclude[i])]
    n_unlabeled = len(labels) - len(keep) - (
        0 if exclude is None else sum(1 for e in exclude if e))
    X = pp.preprocess_matrix(spectra_X[keep], params, wn=wn_full)
    wn = np.asarray(wn_full)[pp.crop_mask(wn_full, params)]
    labels = [labels[i] for i in keep]
    groups = [groups[i] for i in keep]

    def _canon(s: str) -> str:
        return (s or "").strip().casefold()

    by_patient: dict[str, dict[str, list[int]]] = {}
    for i, (g, lab) in enumerate(zip(groups, labels, strict=True)):
        by_patient.setdefault(g, {}).setdefault(_canon(lab), []).append(i)

    rows_X, rows_y, rows_g = [], [], []
    n_unpaired = 0
    n_self_ref = 0
    for g, cls_idx in by_patient.items():
        normals = cls_idx.get("normal")
        tumors = cls_idx.get("tumor")
        if not normals or not tumors:
            n_unpaired += 1
            continue
        ref_tumor = X[normals].mean(axis=0)     # what a tumor is compared to
        for lab, idxs in (("Normal", normals), ("Tumor", tumors)):
            for i in idxs:
                row = X[i]
                if lab == "Normal":
                    others = [j for j in normals if j != i]
                    if others:                   # leave-one-out reference
                        ref = X[others].mean(axis=0)
                    else:                        # k=1: degenerate zero row
                        ref = X[i]
                        n_self_ref += 1
                else:
                    ref = ref_tumor
                if use_pqn:
                    row = pp.pqn_normalize(row, ref)
                rows_X.append(row - ref)
                rows_y.append(lab)
                rows_g.append(g)
    return PairedData(
        X=np.vstack(rows_X) if rows_X else np.zeros((0, len(wn))),
        y=rows_y, groups=rows_g, wn=wn,
        n_patients=len(by_patient) - n_unpaired,
        n_unpaired_excluded=n_unpaired,
        n_unlabeled_dropped=max(0, n_unlabeled),
        n_self_ref_rows=n_self_ref)


def reference_vector(bundle: dict, paths: list[str]) -> np.ndarray:
    """
    Build the paired normal reference for prediction: the mean of the
    (preprocessed, cropped) spectra from the reference files, using the
    bundle's stored wavenumber grid and preprocessing settings.
    """
    import dataset as ds

    grid = np.asarray(bundle["wavenumbers"])
    raw_params = bundle["prep_params"]
    # tolerate plain dicts at this trust boundary
    params = (raw_params if hasattr(raw_params, "validate")
              else pp.PreprocessParams(**raw_params))
    m = pp.crop_mask(grid, params)
    accs = []
    skipped: list[str] = []
    for p in paths:
        try:
            wn, it = ds.load_spectrum(p)
        except (ValueError, OSError) as exc:
            # junk files (split_log.txt etc.) must not kill the whole
            # prediction — skip them, report when nothing loads at all
            # (2026-09-06: one stray .txt in the reference folder
            # failed the entire run)
            skipped.append(f"{os.path.basename(p)} ({exc})")
            continue
        if not np.all(np.isfinite(it)):
            # 2026-09-08 audit BUG-1: a NaN reference spectrum would be
            # silently zero-filled by nan_to_num and poison EVERY
            # prediction built from it — skip with a reason instead
            skipped.append(f"{os.path.basename(p)} (NaN/Inf values)")
            continue
        if float(np.max(np.abs(it), initial=0.0)) < 1e-9:
            # round 3: an all-zero (or subnormal) reference file WARPS
            # the reference mean instead of contributing signal
            skipped.append(f"{os.path.basename(p)} (no signal — "
                           "all-zero/near-zero)")
            continue
        # align_to_grid (2026-09-06): apply the Phe-1003 calibration the
        # model was trained with — training averages CALIBRATED normals
        # (preprocess_matrix), so the deploy reference must match or every
        # paired deviation goes out-of-distribution
        y = pp.align_to_grid(wn, it, grid, params)
        accs.append(pp.preprocess_spectrum(y[m], params))
    if not accs:
        detail = ("; skipped: " + "; ".join(skipped)) if skipped else ""
        raise ValueError(
            "No reference spectra could be loaded from the folder."
            f"{detail}")
    return np.mean(accs, axis=0)


def variance_analysis(X: np.ndarray, labels: list[str],
                      groups: list[str]) -> dict:
    """
    Decompose the spectral variance (one-way ANOVA, sum of squares) into
    the BETWEEN-PATIENT share and compute class separability (d') at the
    spectrum level and on patient-class means — the numbers that explain
    why paired referencing helps (or does not).
    """
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    grand = X.mean(axis=0)
    ss_between = ss_within = 0.0
    for g in uniq:
        rows = X[groups == g]
        ss_between += len(rows) * ((rows.mean(axis=0) - grand) ** 2).sum()
        ss_within += ((rows - rows.mean(axis=0)) ** 2).sum()
    total = ss_between + ss_within

    def d_prime(mat, labs):
        a = mat[[i for i, l in enumerate(labs) if l == "Tumor"]]
        b = mat[[i for i, l in enumerate(labs) if l == "Normal"]]
        if len(a) == 0 or len(b) == 0:
            return float("nan")
        pooled = np.sqrt((a.var(axis=0, ddof=1)
                          + b.var(axis=0, ddof=1)) / 2.0)
        pooled[pooled == 0] = np.inf
        return float(np.mean(np.abs(a.mean(axis=0) - b.mean(axis=0))
                             / pooled))

    # patient-class means: average each (patient, class) block
    keys = sorted({(g, l) for g, l in zip(groups, labels, strict=True)})
    pm, pl = [], []
    for g, l in keys:
        m = X[(groups == g) & (np.asarray(labels) == l)]
        if len(m):
            pm.append(m.mean(axis=0))
            pl.append(l)
    return {
        "between_patient_var_share":
            float(ss_between / total) if total > 0 else 0.0,
        "d_prime_spectrum": d_prime(X, labels),
        "d_prime_patient_means": (d_prime(np.vstack(pm), pl)
                                  if pm else float("nan")),
        "n_rows": int(X.shape[0]), "n_patients": int(len(uniq)),
    }
