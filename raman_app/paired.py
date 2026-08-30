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
    the reference is the mean of their (preprocessed) Normal spectra;
    every kept spectrum becomes (preprocessed spectrum - reference).
    Patients without a normal reference are excluded (counted).

    With use_pqn=True each spectrum is first PQN-normalized (Dieterle
    2006) against the patient's own normal reference — leakage-free by
    construction — which corrects coupling/dilution scale differences
    before the deviation is taken.
    """
    keep = [i for i in range(len(labels))
            if labels[i] and (exclude is None or not exclude[i])]
    X = pp.preprocess_matrix(spectra_X[keep], params, wn=wn_full)
    wn = np.asarray(wn_full)[pp.crop_mask(wn_full, params)]
    labels = [labels[i] for i in keep]
    groups = [groups[i] for i in keep]

    by_patient: dict[str, dict[str, list[int]]] = {}
    for i, (g, lab) in enumerate(zip(groups, labels, strict=True)):
        by_patient.setdefault(g, {}).setdefault(lab, []).append(i)

    rows_X, rows_y, rows_g = [], [], []
    n_unpaired = 0
    for g, cls_idx in by_patient.items():
        normals = cls_idx.get("Normal") or cls_idx.get("normal")
        tumors = cls_idx.get("Tumor") or cls_idx.get("tumor")
        if not normals or not tumors:
            n_unpaired += 1
            continue
        ref = X[normals].mean(axis=0)
        for lab, idxs in (("Normal", normals), ("Tumor", tumors)):
            for i in idxs:
                row = X[i]
                if use_pqn:
                    row = pp.pqn_normalize(row, ref)
                rows_X.append(row - ref)
                rows_y.append(lab)
                rows_g.append(g)
    return PairedData(
        X=np.vstack(rows_X) if rows_X else np.zeros((0, len(wn))),
        y=rows_y, groups=rows_g, wn=wn,
        n_patients=len(by_patient) - n_unpaired,
        n_unpaired_excluded=n_unpaired)


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
    for p in paths:
        wn, it = ds.load_spectrum(p)
        order = np.argsort(wn)
        y = np.interp(grid, np.asarray(wn)[order], np.asarray(it)[order])
        accs.append(pp.preprocess_spectrum(y[m], params))
    if not accs:
        raise ValueError("No reference spectra could be loaded.")
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
