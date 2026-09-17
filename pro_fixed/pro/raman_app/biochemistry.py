"""
biochemistry.py — what the signal is MADE of, medically read.

A tissue Raman spectrum is a sum of molecular vibrational signatures:
proteins, amino acids, nucleic acids, lipids, collagen, keratin.  Cancer
changes that mixture in known directions (uncontrolled proliferation
raises DNA/RNA and aromatic-amino-acid bands; invasion degrades the
collagen stroma; keratinisation varies by anatomical site and confounds
naive comparisons).  This module makes those contributors explicit:

  * BANDS — literature assignments with the expected tumor direction
  * band_area / band_ratios — classical biomedical intensity markers
  * plausibility — do the model's discriminative bands agree with the
    literature? (scientific validation of the explainability)
  * nmf_components — non-negative unmixing into biochemical components,
    auto-labelled by their dominant bands
  * keratin_index — a site-effect / confounder guard

References: Sharma 2021 (OSCC: ↑ nucleic acid + protein content); Faur
2022 (↑ proteins, nucleic acids, water); Matthies 2021 and Hanna 2024
(keratin as marker AND confounder; MCR-based keratin analysis); standard
band tables as reviewed in Zhang 2022.
"""

from __future__ import annotations

import numpy as np

# (center cm-1, halfwidth, molecule, assignment, direction)
# direction: +1 band rises in tumor, -1 falls, 0 = unspecific
BANDS: tuple[tuple[float, float, str, str, int], ...] = (
    (785.0, 8.0, "nucleic acids", "DNA/RNA phosphate backbone (O-P-O)", +1),
    (815.0, 8.0, "collagen", "C-C stretch, proline-rich collagen", -1),
    (854.0, 8.0, "collagen", "proline / hydroxyproline ring", -1),
    (938.0, 10.0, "keratin", "C-C backbone (keratin marker)", -1),
    (1003.0, 8.0, "protein", "phenylalanine ring breathing", +1),
    (1090.0, 10.0, "nucleic acids", "PO2- stretch", +1),
    (1155.0, 8.0, "carotenoids", "C-C stretch (falls in cancer)", -1),
    (1520.0, 10.0, "carotenoids", "C=C stretch (falls in cancer)", -1),
    (1240.0, 12.0, "collagen/protein", "amide III", -1),
    (1335.0, 10.0, "nucleic acids", "purine bases (A, G)", +1),
    (1445.0, 10.0, "lipids/protein", "CH2/CH3 deformation", 0),
    (1554.0, 10.0, "protein", "tryptophan ring", +1),
    (1578.0, 8.0, "nucleic acids", "purine bases (A, G)", +1),
    (1655.0, 12.0, "protein/lipid", "amide I (alpha-helix) / C=C", +1),
)

# classical intensity-ratio markers: (key, numerator, denominator, meaning)
RATIOS: tuple[tuple[str, float, float, str], ...] = (
    ("nucleic/protein", 785.0, 1003.0,
     "proliferation: DNA content vs protein matrix"),
    ("collagen/protein", 854.0, 1003.0,
     "stroma integrity: collagen vs protein"),
    ("lipid/protein", 1445.0, 1003.0,
     "membrane turnover: lipids vs protein"),
    ("amide I/III", 1655.0, 1240.0,
     "protein secondary structure / collagen content"),
)

MATCH_TOL_CM1 = 25.0     # band-matching tolerance for the plausibility check

# Top discriminative wavenumbers from Bidipta Rana's internship report
# (BARC/IIT-M, July 2026) — Welch t-test + Cohen's d on the OSCC dataset
# (Table 4.1, p<0.05, |d|>0.1), with literature assignments.  Used by the
# band-agreement diagnostics to check whether the trained models
# independently rediscover the same biochemistry.
REPORT_BANDS: tuple[tuple[float, float, str, str], ...] = (
    # (shift cm-1, |Cohen's d|, molecule, assignment)
    (743.6, 0.526, "nucleic acids/phospholipid", "adenine, C-S stretch"),
    (739.3, 0.504, "nucleic acids/phospholipid", "adenine, C-S stretch"),
    (737.2, 0.482, "nucleic acids/phospholipid", "adenine, C-S stretch"),
    (1222.4, 0.490, "collagen/protein", "amide III"),
    (1242.3, 0.485, "collagen/protein", "amide III"),
    (1519.0, 0.475, "carotenoids", "C=C stretch (falls in cancer)"),
    (1752.0, 0.459, "lipids", "C=O ester stretch"),
    (1608.9, 0.446, "protein/lipid", "phenylalanine / C=C"),
    (1786.3, 0.445, "lipids", "C=O ester stretch"),
    (1569.8, 0.438, "protein/nucleic acids", "tryptophan / guanine-adenine"),
    (1317.1, 0.433, "collagen/nucleic acids", "adenine / protein CH-twist"),
    (1391.0, 0.431, "protein/lipids", "CH3 symmetric deformation"),
    (1614.5, 0.428, "protein/lipid", "phenylalanine / C=C"),
    (865.4, 0.427, "collagen", "proline / hydroxyproline ring"),
    (1303.4, 0.413, "collagen/protein", "amide III / CH wag"),
    (1416.1, 0.411, "lipids/protein", "CH2/CH3 deformation"),
    (1797.1, 0.412, "lipids", "C=O ester stretch"),
    (1299.5, 0.407, "collagen/protein", "amide III / CH wag"),
    (1798.9, 0.405, "lipids", "C=O ester stretch"),
)


def _window(wn: np.ndarray, center: float, halfwidth: float) -> np.ndarray:
    wn = np.asarray(wn, dtype=float)
    return (wn >= center - halfwidth) & (wn <= center + halfwidth)


def band_area(wn, y, center: float, halfwidth: float) -> float:
    """Trapezoid area of the band window (robust vs point intensities)."""
    wn = np.asarray(wn, dtype=float)
    y = np.asarray(y, dtype=float)
    m = _window(wn, center, halfwidth)
    if m.sum() < 2:
        return float("nan")
    return float(np.trapezoid(y[m], wn[m]))


def band_ratios(wn, y) -> dict[str, float]:
    """The classical biomedical markers for one (cropped) spectrum."""
    areas = {}
    for center, halfwidth, _mol, _assign, _dir in BANDS:
        areas[center] = band_area(wn, y, center, halfwidth)
    out: dict[str, float] = {}
    for key, num, den, _meaning in RATIOS:
        d = areas.get(den, float("nan"))
        out[key] = (areas.get(num, float("nan")) / d
                    if d and np.isfinite(d) and d != 0 else float("nan"))
    return out


def ratio_table(wn, X, y, groups) -> tuple[list[tuple], list[tuple]]:
    """
    Class means of every ratio + PAIRED deltas (tumor minus the SAME
    patient's normal — the paired design's unique strength: each patient
    is their own control).
    Returns (class_rows, paired_rows):
      class_rows  = (ratio, meaning, {class: mean})
      paired_rows = (ratio, n_patients, mean_delta_tumor_minus_normal)
    """
    wn = np.asarray(wn, dtype=float)
    X = np.asarray(X, dtype=float)
    classes = sorted(set(y))
    per_row = [band_ratios(wn, row) for row in X]
    class_rows = []
    for key, _num, _den, meaning in RATIOS:
        means = {c: float(np.nanmean([per_row[i][key]
                                      for i in range(len(y)) if y[i] == c]))
                 for c in classes}
        class_rows.append((key, meaning, means))
    # paired deltas: tumor minus the same patient's normal, per ratio
    # (needs patient groups; flat data simply yields no pairing)
    paired: dict[str, list[float]] = {}
    for g in dict.fromkeys(groups or []):
        rows_n = [per_row[i] for i in range(len(y))
                  if groups[i] == g and y[i] == classes[0]]
        rows_t = [per_row[i] for i in range(len(y))
                  if groups[i] == g and y[i] == classes[-1]]
        if not rows_n or not rows_t:
            continue
        for key, _num, _den, _meaning in RATIOS:
            dn = np.nanmean([r[key] for r in rows_n])
            dt = np.nanmean([r[key] for r in rows_t])
            if np.isfinite(dn) and np.isfinite(dt):
                paired.setdefault(key, []).append(float(dt - dn))
    paired_rows = [(key, len(vals), float(np.mean(vals)))
                   for key, vals in paired.items() if vals]
    return class_rows, paired_rows


def plausibility(bands, tol: float = MATCH_TOL_CM1) -> tuple[list[tuple],
                                                             tuple[int, int]]:
    """
    Check the model's discriminative bands against the literature.

    bands: (center, share, name, sign) tuples as stored from the Train
    page (sign: +1 pushes toward the positive class, -1 away, 0 unknown
    — Gini importances have no sign).  Returns (rows, (agree, known))
    with rows = (center, molecule, assignment, expected, model_sign,
    verdict) and verdict in agree / opposite / unknown-sign / unassigned.
    """
    rows = []
    agree = known = 0
    for b in bands:
        center = float(b[0])
        sign = int(b[3]) if len(b) > 3 else 0
        best = None
        for bc, _hw, mol, assign, direction in BANDS:
            if abs(bc - center) <= tol and (best is None
                                            or abs(bc - center) < best[0]):
                best = (abs(bc - center), bc, mol, assign, direction)
        if best is None:
            rows.append((center, "-", "-", "–", sign, "unassigned"))
            continue
        _, bc, mol, assign, direction = best
        known += 1
        if sign == 0 or direction == 0:
            verdict = "unknown-sign"
        elif sign == direction:
            verdict = "agree"
            agree += 1
        else:
            verdict = "opposite"
        rows.append((center, mol, assign, direction, sign, verdict))
    return rows, (agree, known)


def agreement_report(wn, importance, top_k: int = 20,
                     tol: float = 30.0) -> dict:
    """
    Cross-model / cross-source band agreement: take the `top_k` most
    important wavenumbers of ANY model's importance profile (SHAP, VIP,
    Grad-CAM, …) and check them against (a) the literature BANDS table
    and (b) the internship report's significant bands (REPORT_BANDS).

    Returns {'matches': [(shift, source, molecule, assignment), ...],
             'lit_frac', 'report_frac', 'spearman'} where spearman is
    the rank correlation between the importance profile and a synthetic
    literature profile (Gaussian bumps at every literature band).
    """
    from scipy.stats import spearmanr
    wn = np.asarray(wn, dtype=float)
    imp = np.asarray(importance, dtype=float)
    if len(wn) != len(imp) or len(imp) < 10:
        return {"matches": [], "lit_frac": 0.0, "report_frac": 0.0,
                "spearman": float("nan")}
    top = wn[np.argsort(-imp)[:top_k]]
    lit_centers = [b[0] for b in BANDS]
    rep_centers = [b[0] for b in REPORT_BANDS]

    def _near(x, centers):
        return any(abs(x - c) <= tol for c in centers)

    lit_hits = [s for s in top if _near(s, lit_centers)]
    rep_hits = [s for s in top if _near(s, rep_centers)]
    matches = []
    for s in lit_hits:
        bc, _hw, mol, assign, _d = min(
            BANDS, key=lambda b: abs(b[0] - s))
        matches.append((float(s), "literature", mol, assign))
    for s in rep_hits:
        sh, _d, mol, assign = min(
            REPORT_BANDS, key=lambda b: abs(b[0] - s))
        matches.append((float(s), "internship report", mol, assign))
    # synthetic literature profile: unit Gaussians at every band center
    sig = 20.0
    ref = np.zeros_like(wn)
    for c in lit_centers:
        ref += np.exp(-((wn - c) / sig) ** 2)
    rho = float(spearmanr(ref, imp).statistic) \
        if np.isfinite(imp).all() and imp.std() > 0 else float("nan")
    return {"matches": matches,
            "lit_frac": len(lit_hits) / top_k,
            "report_frac": len(rep_hits) / top_k,
            "spearman": rho}


# saliva SERS: the review-endorsed leading biofluid (Hanna 2024 — saliva
# 93.6/94.0 vs serum 85/90); its thiocyanate band doubles as QC AND
# discriminative feature (about 2x lower area in cancer: Fălămaș 2020,
# Faur 2023)
SALIVA_BANDS: tuple[tuple[float, float, str, str, int], ...] = (
    (875.0, 10.0, "protein", "tryptophan ring (Trp-caged)", +1),
    (1003.0, 8.0, "protein", "phenylalanine ring breathing", +1),
    (1078.0, 10.0, "protein", "C-C stretch", +1),
    (1445.0, 10.0, "lipids/protein", "CH2/CH3 deformation", 0),
    (1590.0, 10.0, "protein/nucleic acids", "C=C / purine", +1),
    (2120.0, 16.0, "thiocyanate", "SCN- C≡N stretch", -1),
)


def thiocyanate_index(wn, X) -> np.ndarray:
    """
    Normalized 2100–2136 cm-1 thiocyanate band area per spectrum — the
    saliva-specific sample-quality and discrimination marker.  NaN when
    the wavenumber range does not cover it (tissue datasets).
    """
    wn = np.asarray(wn, dtype=float)
    X = np.asarray(X, dtype=float)
    m = _window(wn, 2118.0, 18.0)
    if m.sum() < 3:
        return np.full(len(X), np.nan)
    area = np.array([np.trapezoid(row[m], wn[m]) for row in X])
    total = np.array([np.abs(np.trapezoid(row, wn)) for row in X])
    out = np.where(total > 1e-12, area / np.maximum(total, 1e-12),
                   np.nan)
    return out


def biochemical_shift(X, wn, labels) -> list[tuple]:
    """
    The one-glance cancer narrative: class-mean band-area deltas over
    the literature BANDS with Welch p-values — normal tissue is
    lipid/carotenoid-dominated; cancer shifts toward protein + nucleic
    acids (Baraga/Feld fingerprint; Li 2023).  Positive class = the
    alphabetically-later class (Tumor > Normal).
    Returns rows (center, molecule, expected, delta_norm, p).
    """
    from scipy.stats import ttest_ind
    X = np.asarray(X, dtype=float)
    wn = np.asarray(wn, dtype=float)
    labels = np.asarray(labels)
    classes = sorted(set(labels))
    if len(classes) != 2:
        return []
    a = X[labels == classes[0]]
    b = X[labels == classes[1]]
    rows = []
    for center, hw, mol, _assign, direction in BANDS:
        m = _window(wn, center, hw)
        if m.sum() < 3:
            continue
        aa = np.array([np.trapezoid(r[m], wn[m]) for r in a])
        bb = np.array([np.trapezoid(r[m], wn[m]) for r in b])
        denom = max(abs(np.mean(aa)) + abs(np.mean(bb)), 1e-12)
        delta = (np.mean(bb) - np.mean(aa)) / denom
        try:
            p = float(ttest_ind(bb, aa, equal_var=False).pvalue)
        except Exception:
            p = float("nan")
        rows.append((center, mol, direction, float(delta), p))
    return rows


def dataset_qc(X, wn) -> dict:
    """
    Cohort quality snapshot: per-spectrum SNR (range vs robust HF-noise
    estimate from first differences), spike proxy, and whether the
    saliva thiocyanate QC band is inside the measured range.
    """
    X = np.asarray(X, dtype=float)
    noise = np.median(np.abs(np.diff(X, axis=1)), axis=1) * 1.4826
    snr = (X.max(axis=1) - X.min(axis=1)) / np.maximum(noise, 1e-12)
    jumps = np.abs(np.diff(X, axis=1)).max(axis=1) / np.maximum(
        np.median(np.abs(np.diff(X, axis=1)), axis=1), 1e-12)
    return {
        "snr_median": float(np.median(snr)),
        "snr_p10": float(np.percentile(snr, 10)),
        "spike_proxy_max": float(jumps.max()),
        "thiocyanate_band": bool(_window(np.asarray(wn, dtype=float),
                                         2118.0, 18.0).sum() >= 3),
    }


def nmf_components(X, wn, k: int = 5, seed: int = 0):
    """
    Non-negative unmixing of the spectra into biochemical components.

    Returns (component_spectra (k, len(wn)), labels list[str],
    weights (n, k)).  Each component is auto-labelled by its dominant
    band (highest z-scored band area across components).
    """
    from sklearn.decomposition import NMF

    X = np.ascontiguousarray(X, dtype=float)  # NMF's Cython kernel needs
    X = np.clip(X, 0.0, None)                 # C-order, non-view arrays
    model = NMF(n_components=k, init="nndsvda", max_iter=600,
                random_state=seed)
    W = model.fit_transform(X)          # (n, k) weights
    H = model.components_               # (k, features) spectra
    wn = np.asarray(wn, dtype=float)
    # label each component by its dominant literature band
    mols = sorted({mol for _c, _hw, mol, _a, _d in BANDS})
    areas = np.zeros((k, len(BANDS)))
    for b_i, (center, halfwidth, _mol, _assign, _dir) in enumerate(BANDS):
        for c_i in range(k):
            areas[c_i, b_i] = band_area(wn, H[c_i], center, halfwidth)
    z = (areas - areas.mean(axis=0)) / (areas.std(axis=0) + 1e-12)
    labels = []
    for c_i in range(k):
        b_i = int(np.argmax(z[c_i]))
        center, _hw, mol, _assign, _dir = BANDS[b_i]
        labels.append(f"{mol} ({center:.0f})" if mol in mols else mol)
    return H, labels, W


def component_deltas(W, y, groups) -> list[tuple]:
    """
    Paired component-weight changes (last class minus first class,
    same patient) — which biochemical contributors shift in tumor.
    Returns (label-less) rows of (n_patients, mean_delta) per component.
    """
    W = np.asarray(W, dtype=float)
    classes = sorted(set(y))
    k = W.shape[1]
    out = []
    for c_i in range(k):
        deltas = []
        for g in dict.fromkeys(groups or []):
            wn_ = [W[i, c_i] for i in range(len(y))
                   if groups[i] == g and y[i] == classes[0]]
            wt = [W[i, c_i] for i in range(len(y))
                  if groups[i] == g and y[i] == classes[-1]]
            if wn_ and wt:
                deltas.append(float(np.mean(wt) - np.mean(wn_)))
        out.append((c_i, len(deltas),
                    float(np.mean(deltas)) if deltas else float("nan")))
    return out


def keratin_index(wn, y) -> float:
    """
    Site-effect guard: keratin-like content of one spectrum, as the
    938 cm-1 keratin band normalized to the 1003 cm-1 protein band
    (keratinisation varies by anatomical site and can masquerade as
    class structure — flag it instead of silently absorbing it).
    """
    den = band_area(wn, y, 1003.0, 8.0)
    if not np.isfinite(den) or den == 0:
        return float("nan")
    return band_area(wn, y, 938.0, 10.0) / den


def keratin_flags(wn, spectra, z_thresh: float = 2.5) -> tuple[float, float,
                                                               int, int]:
    """
    Batch-relative keratin outliers: (median, mad, n_flagged, n_total).
    A spectrum is flagged when its index is more than z_thresh
    median-absolute-deviations above the batch median.
    """
    idx = [keratin_index(wn, s) for s in spectra]
    idx = np.asarray([v for v in idx if np.isfinite(v)], dtype=float)
    if len(idx) == 0:
        return float("nan"), float("nan"), 0, 0
    med = float(np.median(idx))
    mad = float(np.median(np.abs(idx - med))) * 1.4826
    if mad <= 0:
        mad = float(np.std(idx)) or 1.0
    flagged = int(np.sum(idx > med + z_thresh * mad))
    return med, mad, flagged, int(len(idx))
