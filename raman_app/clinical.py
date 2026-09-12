"""
clinical.py — clinical-grade evaluation on top of the ML pipeline.

Medical diagnostics does not read a macro-F1; it reads operating points,
triage tiers, predictive values at real prevalence, calibration and net
benefit.  Everything here is computed from the pooled out-of-fold
probabilities the training loop already produces (patient-grouped, so no
leakage) and follows the standards used by the reporting guidelines:

  * TRIPOD+AI (Collins et al., BMJ 2024) — report discrimination,
    calibration AND clinical utility, not one score.
  * PROBAST+AI (Moons et al., BMJ 2025) — data leakage is the top
    ML-specific bias; every input here comes from patient-grouped CV.

All functions are plain numpy/sklearn and unit-tested in test_all.py.
"""

from __future__ import annotations

import numpy as np

TRIAGE_NEGATIVE = "NEGATIVE"
TRIAGE_INDETERMINATE = "INDETERMINATE"
TRIAGE_POSITIVE = "POSITIVE"

# what each tier recommends (shown verbatim on the Result page)
TRIAGE_ACTIONS = {
    TRIAGE_NEGATIVE: "routine follow-up",
    TRIAGE_INDETERMINATE: "re-sample or biopsy",
    TRIAGE_POSITIVE: "biopsy / refer",
}

# plausible disease prevalence by setting — PPV/NPV must be read against
# one of these (a test with identical sens/spec has very different PPV
# in a dental screening vs a biopsy queue; Patton, JADA)
PREVALENCE_SCENARIOS = (
    ("dental screening", 0.05),
    ("lesion clinic", 0.30),
    ("biopsy queue", 0.60),
)

# published comparison values (sens, spec) — ONE home so the txt and
# HTML reports can never drift apart; "~" marks typical/approximate
LITERATURE = (
    ("Han 2022 meta (13 studies)", 0.89, 0.84),
    ("2025 meta-analysis (OSCC subgroup)", 0.89, 0.91),
    ("Purohit 2026 review (pooled)", 0.90, 0.89),
    ("VELscope autofluorescence (typical)", 0.84, 0.45),
    ("Toluidine blue (typical)", 0.63, 0.83),
)


def operating_points(y_true, p_pos: np.ndarray, sens_target: float = 0.90,
                     spec_target: float = 0.90):
    """
    The two clinically useful thresholds on the ROC curve:

      rule-out  — the HIGHEST probability threshold that still catches
                  >= sens_target of the positives (miss as little cancer
                  as possible; below it a case is cleared);
      rule-in   — the LOWEST threshold that keeps >= spec_target of the
                  negatives clear (above it, act on the result).

    Returns (thr_ruleout, thr_rulein, sens_at_ruleout, spec_at_rulein);
    entries are None when the target is unreachable on this ROC.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true)
    p_pos = np.asarray(p_pos, dtype=float)
    ok = ~np.isnan(p_pos)
    fpr, tpr, thr = roc_curve(y_true[ok], p_pos[ok])
    # roc_curve puts inf at thr[0] (the never-positive point) — replace
    # with the largest real probability so callers get usable thresholds
    thr = np.asarray(thr, dtype=float)
    thr[~np.isfinite(thr)] = float(np.nanmax(p_pos)) if ok.any() else 1.0
    # roc_curve: thresholds descending, tpr non-decreasing, fpr
    # non-decreasing with the index
    idx = np.flatnonzero(tpr >= sens_target)
    if len(idx) == 0:
        return None, None, None, None
    i = int(idx[0])                  # highest threshold with sens kept
    spec_ok = fpr <= 1.0 - spec_target
    if not spec_ok.any():
        return float(thr[i]), None, float(tpr[i]), None
    j = int(np.max(np.flatnonzero(spec_ok)))   # lowest thr with spec kept
    return (float(thr[i]), float(thr[j]), float(tpr[i]),
            float(1.0 - fpr[j]))


def triage(p: float, thr_ruleout: float | None,
           thr_rulein: float | None, fallback: float | None = 0.5):
    """
    Three-tier clinical verdict from P(positive):

      p >= rule-in  -> POSITIVE
      p <  rule-out -> NEGATIVE    (STRICT: operating_points guarantees
      in between    -> INDETERMINATE  sens>=target under sklearn's
                                    `p >= thr` positive rule, so a
                                    spectrum exactly AT the rule-out
                                    threshold is a positive call and
                                    must NOT be cleared — the old
                                    inclusive `p <= rule-out` realized
                                    sens below target at the boundary;
                                    2026-09-12 audit)

    Falls back to a single-threshold split when the ROC cannot support
    two targets (rule-out >= rule-in: the model is too weak to tier).
    """
    if np.isnan(p):
        return TRIAGE_INDETERMINATE
    if (thr_ruleout is None or thr_rulein is None
            or thr_ruleout >= thr_rulein):
        t = fallback if fallback is not None else 0.5
        return TRIAGE_POSITIVE if p >= t else TRIAGE_NEGATIVE
    if p >= thr_rulein:
        return TRIAGE_POSITIVE
    if p < thr_ruleout:
        return TRIAGE_NEGATIVE
    return TRIAGE_INDETERMINATE


def ppv_npv(sens: float, spec: float, prev: float):
    """
    Predictive values at a given disease prevalence (Bayes).

    Sensitivity/specificity are properties of the test; PPV/NPV are what
    a clinician experiences and they move strongly with prevalence.
    Returns (ppv, npv); NaN-safe (no division by zero).
    """
    num = sens * prev
    den = num + (1.0 - spec) * (1.0 - prev)
    ppv = num / den if den > 0 else float("nan")
    num2 = spec * (1.0 - prev)
    den2 = (1.0 - sens) * prev + num2
    npv = num2 / den2 if den2 > 0 else float("nan")
    return ppv, npv


def calibration_bins(y_true, p_pos: np.ndarray, n_bins: int = 8):
    """
    Reliability diagram data (TRIPOD+AI calibration item): equal-count
    bins of predicted probability vs observed event rate.
    Returns list of (mean_predicted, observed_rate, n) per bin.
    """
    y_true = np.asarray(y_true, dtype=float)
    p_pos = np.asarray(p_pos, dtype=float)
    ok = ~np.isnan(p_pos)
    y, p = y_true[ok], p_pos[ok]
    order = np.argsort(p, kind="stable")
    # CONTIGUOUS equal-count bins: block b of the sorted probabilities.
    # (Interleaving — order[b::n_bins] — puts every bin across the whole
    # [0, 1] range, flattening each bin to the global mean and destroying
    # the reliability diagram; fixed 2026-09-05.)
    out = []
    for sel in np.array_split(order, n_bins):
        if len(sel) == 0:
            continue
        out.append((float(np.mean(p[sel])), float(np.mean(y[sel])),
                    int(len(sel))))
    return out


def decision_curve(y_true, p_pos: np.ndarray,
                   thresholds=None) -> tuple[np.ndarray, ...]:
    """
    Net benefit of treating by the model vs treat-all vs treat-none
    (decision curve analysis — the standard 'clinical utility' plot).

    NB(t) = TP/n - FP/n * t/(1-t)

    Returns (thresholds, nb_model, nb_treat_all); treat-none is the
    zero line by definition.
    """
    y_true = np.asarray(y_true, dtype=float)
    p_pos = np.asarray(p_pos, dtype=float)
    ok = ~np.isnan(p_pos)
    y, p = y_true[ok], p_pos[ok]
    n = max(len(y), 1)
    prev = float(np.mean(y)) if n else 0.0
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    thr = np.asarray(thresholds, dtype=float)
    nb_model = np.zeros_like(thr)
    for i, t in enumerate(thr):
        if t >= 1.0:
            continue
        call = p >= t
        tp = float(np.sum(y[call] == 1))
        fp = float(np.sum(y[call] == 0))
        nb_model[i] = tp / n - fp / n * (t / (1.0 - t))
    nb_all = np.where(thr < 1.0,
                      prev - (1.0 - prev) * thr / (1.0 - thr), 0.0)
    return thr, nb_model, nb_all


# --------------------------------------------------------------------------
# Probability calibration (Platt scaling)
# --------------------------------------------------------------------------
def fit_platt(y_true, p_pos: np.ndarray) -> tuple[float, float] | None:
    """
    Platt scaling: fit (a, b) so that sigmoid(a * logit(p) + b) maps the
    model's raw probabilities onto observed event rates.  Returns None
    when the data cannot support a fit (< 10 rows or a single class).
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pos, dtype=float), 1e-6, 1.0 - 1e-6)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    if len(y) < 10 or len(np.unique(y)) < 2:
        return None
    z = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    lr.fit(z, y)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    return (a, b) if np.isfinite(a) and np.isfinite(b) else None


def apply_platt(p, ab: tuple[float, float]) -> np.ndarray | float:
    """Map raw probabilities through the fitted Platt sigmoid."""
    a, b = ab
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    z = a * np.log(p / (1.0 - p)) + b
    out = 1.0 / (1.0 + np.exp(-z))
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------
# Conformal prediction sets (split-conformal LAC; hand-rolled, no MAPIE)
# --------------------------------------------------------------------------
def conformal_q(oof_proba: np.ndarray, y_codes: np.ndarray,
                alpha: float = 0.10) -> float:
    """
    Split-conformal threshold for Least-Ambiguous set-valued Classifier
    scores s_i = 1 - p_i(y_i):  q̂ = the ceil((n+1)(1-alpha))/n empirical
    quantile.  Fitted on pooled OUT-OF-FOLD probabilities of the winner
    (the only honest calibration source), it guarantees ~1-alpha marginal
    coverage of the true class in the prediction sets.
    """
    p = np.asarray(oof_proba, dtype=float)
    s = 1.0 - p[np.arange(len(y_codes)), np.asarray(y_codes)]
    n = len(s)
    if n < 10:
        return 1.0                      # too small to promise anything
    rank = min(n, int(np.ceil((n + 1) * (1.0 - alpha))))
    return float(np.sort(s)[rank - 1])


def conformal_sets(proba: np.ndarray, q: float) -> list[list[int]]:
    """Per-sample prediction sets: classes with p >= 1 - q̂."""
    p = np.asarray(proba, dtype=float)
    return [list(np.where(row >= 1.0 - q)[0]) for row in p]


def conformal_metrics(sets: list[list[int]], y_codes: np.ndarray) -> dict:
    """Coverage (true class inside the set), mean set size, abstain rate."""
    sets = [list(s) for s in sets]
    cover = sum(1 for s, y in zip(sets, y_codes, strict=True) if y in s)
    return {
        "coverage": cover / max(len(sets), 1),
        "mean_size": float(np.mean([len(s) for s in sets])) if sets else 0.0,
        "abstain_rate": float(np.mean([len(s) == 0 for s in sets]))
        if sets else 0.0,
    }


# --------------------------------------------------------------------------
# Calibration quality (ECE) + isotonic comparison
# --------------------------------------------------------------------------
def ece(y_true, p_pos: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: |predicted - observed| mass-weighted
    over equal-count bins (0 = perfectly calibrated)."""
    bins = calibration_bins(y_true, p_pos, n_bins)
    n = sum(b[2] for b in bins)
    if n == 0:
        return float("nan")
    return float(sum(b[2] * abs(b[0] - b[1]) for b in bins) / n)


def isotonic_compare(y_true, p_pos: np.ndarray, k: int = 5,
                     seed: int = 0) -> dict:
    """
    Fit isotonic mapping alongside Platt and report both ECEs — the
    honest comparison at n<500 (isotonic usually overfits small samples;
    Platt stays applied, isotonic is reported for transparency).

    Both mappings are CROSS-FITTED (k-fold: fit on k-1 folds, transform
    the held-out fold, pool) before the ECE — the old in-sample ECEs
    were near-zero by construction, especially for isotonic, which
    interpolates its own training points (2026-09-12 audit).
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import KFold
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pos, dtype=float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {}
    folds = list(KFold(n_splits=max(2, min(k, len(y) // 2)),
                       shuffle=True, random_state=seed).split(p))
    iso_p = np.empty_like(p)
    platt_p = np.empty_like(p)
    for tr, te in folds:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_p[te] = iso.fit(p[tr], y[tr]).predict(p[te])
        ab = fit_platt(y[tr], p[tr])
        platt_p[te] = apply_platt(p[te], ab) if ab else p[te]
    return {"ece_raw": ece(y, p),
            "ece_platt": ece(y, platt_p),
            "ece_isotonic": ece(y, iso_p)}


# --------------------------------------------------------------------------
# AUC power / sample size (Hanley & McNeil 1982 variance)
# --------------------------------------------------------------------------
def auc_power(n_pos: int, n_neg: int, auc: float = 0.85,
              alpha: float = 0.05) -> dict:
    """
    Detectable AUC at 80% power for this case/control split (and the
    sample needed for a target AUC) using the Hanley-McNeil variance —
    answers 'is this study big enough to see the effect it claims?'.
    """
    import math
    from scipy.stats import norm
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc ** 2 / (1.0 + auc)
    v = (auc * (1 - auc)
         + (n_pos - 1) * (q1 - auc ** 2)
         + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    se = math.sqrt(max(v, 1e-12))
    z_a = norm.ppf(1.0 - alpha / 2.0)
    z_b = 0.8416                              # 80% power
    # Detectable AUC vs the 0.5 null: delta = (z_a + z_b) * SE(AUC).
    # (A previous version divided by sqrt(2) — that factor belongs to a
    # DIFFERENCE of two independent AUCs, not a one-sample test vs 0.5;
    # dividing shrank the detectable effect ~29% and halved n_needed,
    # making every power claim optimistic. Fixed 2026-09-05.)
    detectable = 0.5 + (z_a + z_b) * se
    n_needed = None
    for n in range(10, 100000, 5):            # equal groups search
        se_n = se * math.sqrt(n_pos * n_neg / (n * n))
        if 0.5 + (z_a + z_b) * se_n <= auc:
            n_needed = n
            break
    return {"detectable_auc_80pct": float(min(detectable, 1.0)),
            "se_auc": float(se),
            "n_per_group_for_target": n_needed}


# --------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n (never degenerate)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _midrank(v: np.ndarray) -> np.ndarray:
    """Midranks with ties averaged (DeLong's tie-corrected ranks)."""
    order = np.argsort(v, kind="mergesort")
    sv = np.asarray(v)[order]
    out = np.empty(len(v), dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        out[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return out


def delong_auc_ci(y_true, p_pos: np.ndarray, z: float = 1.96):
    """
    DeLong ROC-AUC with analytical standard error and 95% CI
    (structural-components method, tie-corrected).  Returns
    (auc, se, lo, hi); NaNs when a class is empty.
    """
    y = np.asarray(y_true)
    p = np.asarray(p_pos, dtype=float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return (float("nan"),) * 4
    ranks = _midrank(np.concatenate([pos, neg]))
    v10 = (ranks[:m] - _midrank(pos)) / n
    v01 = 1.0 - (ranks[m:] - _midrank(neg)) / m
    auc = float(np.mean(v10))
    se = float(np.sqrt(np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n))
    return (auc, se, auc - z * se, min(1.0, auc + z * se))
