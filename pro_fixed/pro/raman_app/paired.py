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
    n_normals_dropped: int = 0       # single-normal patients: no honest ref


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
    every kept spectrum becomes (preprocessed spectrum - reference).

    The reference is the mean of that patient's (preprocessed) Normal
    spectra, EXCLUDING the spectrum being encoded: a Tumor row uses all
    of the patient's normals, a Normal row uses the patient's OTHER
    normals (leave-one-out).  A Normal spectrum that is its patient's
    only normal has no honest reference and is dropped
    (`n_normals_dropped`); the patient still contributes Tumor rows.
    Patients without any normal reference are excluded entirely
    (`n_unpaired_excluded`).

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
    n_dropped = 0
    for g, cls_idx in by_patient.items():
        normals = cls_idx.get("Normal") or cls_idx.get("normal")
        tumors = cls_idx.get("Tumor") or cls_idx.get("tumor")
        if not normals or not tumors:
            n_unpaired += 1
            continue
        ref_all = X[normals].mean(axis=0)
        for lab, idxs in (("Normal", normals), ("Tumor", tumors)):
            for i in idxs:
                if lab == "Normal":
                    # LEAVE-ONE-OUT reference (2026-09-07 fix): a Normal
                    # spectrum must NEVER be part of the mean it is
                    # subtracted from.  Self-inclusion shrinks every
                    # Normal row toward zero (a patient with a single
                    # normal produced the EXACT zero vector) while Tumor
                    # rows keep full magnitude, so |row| alone separated
                    # the classes (AUC 0.835 on the clinical cohort) and
                    # models learned "small deviation = Normal".  At
                    # predict time reference_vector() builds the
                    # reference from OTHER files, so deployed normals
                    # arrived at full magnitude and ~80% were called
                    # Tumor.  With an honest reference the two sides are
                    # constructed identically, in CV and in deployment.
                    others = [j for j in normals if j != i]
                    if not others:
                        n_dropped += 1     # single normal: no honest ref
                        continue
                    ref_i = X[others].mean(axis=0)
                else:
                    ref_i = ref_all
                row = X[i]
                if use_pqn:
                    row = pp.pqn_normalize(row, ref_i)
                rows_X.append(row - ref_i)
                rows_y.append(lab)
                rows_g.append(g)
    if rows_X and len(set(rows_y)) < 2:
        # Clear, actionable failure: without this the degenerate matrix
        # reached evaluate_models and surfaced as "All models failed"
        # with a wall of sklearn tracebacks (2026-09-07).
        raise ValueError(
            f"Paired mode produced only one class ({sorted(set(rows_y))}). "
            f"{n_dropped} Normal spectra were dropped because they were "
            "their patient's ONLY normal, so no leave-one-out reference "
            "exists for them. Paired mode needs patients with at least "
            "TWO normal spectra plus a tumor spectrum — use Standard "
            "mode for this cohort.")
    return PairedData(
        X=np.vstack(rows_X) if rows_X else np.zeros((0, len(wn))),
        y=rows_y, groups=rows_g, wn=wn,
        n_patients=len(set(rows_g)),
        n_unpaired_excluded=n_unpaired,
        n_normals_dropped=n_dropped)


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
