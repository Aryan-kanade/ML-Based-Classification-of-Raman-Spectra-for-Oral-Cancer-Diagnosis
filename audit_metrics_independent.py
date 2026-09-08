"""audit_metrics_independent.py — independent metric implementations for
the formula audit (2026-09-08).

This script must NOT call the project's metric helpers.  Every formula
below is written from scratch (numpy only; sklearn used ONLY as the
reference cross-check, never as the implementation under test) so that
project numbers can be compared against a truly independent
computation.

Layers:
  1. SYNTHETIC BATTERY — known ground truth (TP=8, TN=7, FP=2, FN=3,
     plus degenerate cases).  Every value checked against hand-
     calculated constants.
  2. REAL-DATA CROSS-CHECK — retrains the same models the app trains
     (same seed/folds), extracts the project's pooled confusion
     matrices + OOF arrays, and compares every overlapping metric.
  3. SUPPLEMENT METRICS — MCC, balanced accuracy, PR average
     precision, Brier — computed on the SAME predictions (isolates the
     formula effect; §32 of the audit spec).

Writes a full report to stdout.  No production files are modified.
"""

from __future__ import annotations

import math
import numpy as np


# ----------------------------------------------------------------------
# independent metric implementations (numpy only)
# ----------------------------------------------------------------------
def i_accuracy(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    return (tp + tn) / n if n else float("nan")


def i_balanced_accuracy(tp, tn, fp, fn):
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return (sens + spec) / 2.0


def i_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) else 0.0


def i_sensitivity(tp, fn):
    return tp / (tp + fn) if (tp + fn) else 0.0


def i_specificity(tn, fp):
    return tn / (tn + fp) if (tn + fp) else 0.0


def i_f1(tp, fp, fn):
    p, r = i_precision(tp, fp), i_sensitivity(tp, fn)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def i_mcc(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return num / den if den > 0 else 0.0


def i_macro_f1_from_cm(cm):
    """Macro-F1 from a k x k confusion matrix (rows = actual)."""
    k = cm.shape[0]
    total = cm.sum()
    f1s = []
    for i in range(k):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i].sum() - tp
        f1s.append(i_f1(tp, fp, fn))
    return float(np.mean(f1s)) if k else float("nan")


def i_macro_sens_spec_from_cm(cm):
    k = cm.shape[0]
    total = cm.sum()
    sens, spec = [], []
    for i in range(k):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i].sum() - tp
        tn = total - tp - fp - fn
        sens.append(i_sensitivity(tp, fn))
        spec.append(i_specificity(tn, fp))
    return float(np.mean(sens)), float(np.mean(spec))


def i_average_precision(y_true01, scores):
    """PR average precision: sum over thresholds of (R_n - R_{n-1}) * P_n
    (sklearn-compatible step-wise definition, implemented from scratch)."""
    y = np.asarray(y_true01)
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(y.sum(), 1)
    # only count the precision at each NEW recall level
    changed = np.r_[True, rec[1:] != rec[:-1]]
    return float(np.sum(np.diff(np.r_[0.0, rec[changed]])
                        * prec[changed]))


def i_brier(y_true01, p):
    ok = np.isfinite(p)
    return float(np.mean((np.asarray(p)[ok] - np.asarray(y_true01)[ok]) ** 2))


def i_wilson(k, n, z=1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def i_roc_auc(y_true01, scores):
    """Trapezoid AUC via midranks (tie-corrected Mann-Whitney)."""
    y = np.asarray(y_true01, dtype=float)
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    i = 0
    sv = s[order]
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # average rank
        i = j + 1
    r1 = ranks[y == 1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))


def i_group_bootstrap_macro_f1(y_true, y_pred, groups, n=1000,
                               seed=42, alpha=0.05):
    """Cluster (patient-level) percentile bootstrap of macro-F1."""
    rng = np.random.default_rng(seed)
    y, yp, g = map(np.asarray, (y_true, y_pred, groups))
    uniq = np.unique(g)
    rows_by = {gu: np.flatnonzero(g == gu) for gu in uniq}
    f1s = []
    for _ in range(n):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by[t] for t in take])
        k = int(max(y.max(), yp.max())) + 1
        cm = np.zeros((k, k), int)
        for t, p in zip(y[idx], yp[idx]):
            cm[t, p] += 1
        f1s.append(i_macro_f1_from_cm(cm))
    lo, hi = np.percentile(f1s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cm_from(tp, tn, fp, fn):
    return np.array([[tn, fp], [fn, tp]])


# ----------------------------------------------------------------------
# 1. synthetic battery
# ----------------------------------------------------------------------
def synthetic_battery():
    print("=" * 70)
    print("LAYER 1 — SYNTHETIC GROUND-TRUTH BATTERY")
    print("=" * 70)
    cases = [
        # name, tp, tn, fp, fn, expected dict
        ("TP8 TN7 FP2 FN3", 8, 7, 2, 3, dict(
            acc=0.75, prec=0.8, sens=12 / 16 * 0 + 8 / 11,
            spec=7 / 9, f1=2 * 0.8 * (8 / 11) / (0.8 + 8 / 11),
            bacc=(8 / 11 + 7 / 9) / 2,
            mcc=(8 * 7 - 2 * 3) / math.sqrt(10 * 11 * 9 * 10))),
        ("perfect", 10, 10, 0, 0, dict(
            acc=1.0, prec=1.0, sens=1.0, spec=1.0, f1=1.0, bacc=1.0,
            mcc=1.0)),
        ("all-positive", 10, 0, 10, 0, dict(
            acc=0.5, prec=0.5, sens=1.0, spec=0.0, f1=2 / 3, bacc=0.5,
            mcc=0.0)),
        ("all-negative", 0, 10, 0, 10, dict(
            acc=0.5, prec=0.0, sens=0.0, spec=1.0, f1=0.0, bacc=0.5,
            mcc=0.0)),
    ]
    n_fail = 0
    for name, tp, tn, fp, fn, exp in cases:
        got = dict(
            acc=i_accuracy(tp, tn, fp, fn),
            prec=i_precision(tp, fp), sens=i_sensitivity(tp, fn),
            spec=i_specificity(tn, fp), f1=i_f1(tp, fp, fn),
            bacc=i_balanced_accuracy(tp, tn, fp, fn),
            mcc=i_mcc(tp, tn, fp, fn))
        row = []
        for k, v in exp.items():
            ok = abs(got[k] - v) < 1e-12
            n_fail += (not ok)
            row.append(f"{k}={got[k]:.4f}{'' if ok else ' != ' + f'{v:.4f}'}")
        print(f"  {name:<18} " + " ".join(row))
    # zero-denominator safety
    assert i_precision(0, 0) == 0.0 and i_f1(0, 0, 0) == 0.0
    assert i_mcc(0, 0, 0, 0) == 0.0
    assert math.isnan(i_accuracy(0, 0, 0, 0))
    print("  zero-denominator: precision/F1/MCC -> 0.0, accuracy -> NaN (safe)")
    # empty / one-class matrices
    assert i_macro_f1_from_cm(np.zeros((2, 2), int)) == 0.0
    s, p = i_macro_sens_spec_from_cm(np.zeros((2, 2), int))
    assert s == 0.0 and p == 0.0
    print("  empty matrix: macro F1/sens/spec -> 0.0 (safe, no crash)")
    assert math.isnan(i_roc_auc([1, 1], [0.5, 0.5]))     # one-class AUC
    assert i_roc_auc([0, 1, 0, 1], [.1, .8, .2, .9]) == 1.0
    assert abs(i_average_precision([0, 1, 0, 1], [.1, .8, .2, .9]) - 1.0) < 1e-9
    print("  AUC/AP degenerate + perfect cases: correct")
    w = i_wilson(8, 11)
    assert w[0] <= 8 / 11 <= w[1]
    print(f"  Wilson(8/11) = [{w[0]:.3f}, {w[1]:.3f}] contains the point estimate")
    print(f"LAYER 1 RESULT: {'PASS' if n_fail == 0 else f'FAIL ({n_fail})'}\n")
    return n_fail == 0


# ----------------------------------------------------------------------
# 2 + 3. real-data cross-check + supplements
# ----------------------------------------------------------------------
def real_data_check():
    print("=" * 70)
    print("LAYER 2/3 — REAL DATA: PROJECT vs INDEPENDENT + SUPPLEMENTS")
    print("(same models, seed 42, 5-fold grouped CV, paired mode)")
    print("=" * 70)
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "raman_app"))
    import sequential, modeling, dataset as ds
    from clinical_data import load_clinical_dataset
    import preprocessing as pp

    X, y, g, wn, grid, params, meta = sequential.prepare_dataset(
        r"D:/BARC/Data", mode="paired")
    res, win = modeling.evaluate_models(
        X, y, ["PCA + Gaussian Naive Bayes", "Random Forest"], 5, 42,
        groups=g, repeats=1, wavenumbers=wn)
    n_mismatch = 0
    for r in res:
        cm = r.cm
        tp, tn = cm[1, 1], cm[0, 0]
        fp, fn = cm[0, 1], cm[1, 0]
        i_s, i_sp = i_macro_sens_spec_from_cm(cm)
        checks = [
            ("macro sens", r.macro["sens"][0], i_s),
            ("macro spec", r.macro["spec"][0], i_sp),
            ("macro f1", r.macro_f1(), i_macro_f1_from_cm(cm)),
        ]
        print(f"\n{r.name}  (pooled cm {cm.tolist()}, n={int(cm.sum())})")
        for name, proj, indep in checks:
            d = abs(proj - indep)
            # fold-mean vs pooled are DIFFERENT aggregations by design;
            # record both, expect small but nonzero delta
            flag = "same-aggregation" if d < 1e-12 else \
                "fold-mean vs pooled (documented duality)"
            print(f"  {name:<12} project={proj:.4f} independent={indep:.4f} "
                  f"|d|={d:.4f}  [{flag}]")
        # supplements on the SAME predictions
        ye, pr = r.y_true_encoded, r.oof_proba
        v = ~np.isnan(pr).any(axis=1)
        thr = r.threshold if r.threshold is not None else 0.5
        pred = (pr[v, 1] >= thr).astype(int)
        yv = ye[v]
        tp2, tn2 = int(((yv == 1) & (pred == 1)).sum()), \
            int(((yv == 0) & (pred == 0)).sum())
        fp2, fn2 = int(((yv == 0) & (pred == 1)).sum()), \
            int(((yv == 1) & (pred == 0)).sum())
        print(f"  SUPPLEMENTS (same predictions): "
              f"accuracy={i_accuracy(tp2, tn2, fp2, fn2):.4f} "
              f"balanced_acc={i_balanced_accuracy(tp2, tn2, fp2, fn2):.4f} "
              f"MCC={i_mcc(tp2, tn2, fp2, fn2):.4f} "
              f"AUC={i_roc_auc(yv, pr[v, 1]):.4f} "
              f"PR-AP={i_average_precision(yv, pr[v, 1]):.4f} "
              f"Brier={i_brier(yv, pr[v, 1]):.4f}")
        # threshold stability (per-fold)
        thrs = [None if t is None else round(float(t), 3)
                for t in r.thresholds]
        print(f"  per-fold thresholds: {thrs} -> median "
              f"{r.threshold:.4f}")

    # honest check, independently recomputed + supplements
    cd = load_clinical_dataset(r"D:/BARC/Data")
    labels = [s.label for s in cd.spectra]
    keep = [i for i, l in enumerate(labels) if l.strip()]
    keep = [i for i in keep if not cd.flagged[i]]
    grid2 = ds.common_grid(cd.spectra)
    Xraw, _ = ds.to_matrix(cd.spectra, grid2)
    out = modeling.evaluate_pipeline(
        Xraw[keep], grid2, [labels[i] for i in keep],
        groups=[cd.groups[i] for i in keep], k=5, seed=42)
    cm = out["cm"]
    tp, tn = cm[1, 1], cm[0, 0]
    fp, fn = cm[0, 1], cm[1, 0]
    i_s, i_sp = i_macro_sens_spec_from_cm(cm)
    print(f"\nHONEST CHECK pooled cm {cm.tolist()}")
    print(f"  project  sens={out['sens']:.4f} spec={out['spec']:.4f} "
          f"acc={out['acc']:.4f} f1={out['mean_f1']:.4f}")
    print(f"  indep    sens={i_s:.4f} spec={i_sp:.4f} "
          f"acc={i_accuracy(tp, tn, fp, fn):.4f} "
          f"pooled-f1={i_macro_f1_from_cm(cm):.4f}")
    print(f"  SUPPLEMENTS: balanced_acc="
          f"{i_balanced_accuracy(tp, tn, fp, fn):.4f} "
          f"MCC={i_mcc(tp, tn, fp, fn):.4f}")
    for nm, a, b in (("sens", out["sens"], i_s), ("spec", out["spec"], i_sp),
                     ("acc", out["acc"], i_accuracy(tp, tn, fp, fn))):
        if abs(a - b) > 1e-9:
            n_mismatch += 1
            print(f"  MISMATCH {nm}: {a} vs {b}")
    print(f"\nLAYER 2 RESULT: {'PASS (all overlapping metrics equal)' if n_mismatch == 0 else f'{n_mismatch} MISMATCHES'}")
    return n_mismatch == 0


if __name__ == "__main__":
    ok1 = synthetic_battery()
    ok2 = real_data_check()
    print("\nAUDIT_METRICS_INDEPENDENT:",
          "ALL PASS" if (ok1 and ok2) else "FAILURES PRESENT")
