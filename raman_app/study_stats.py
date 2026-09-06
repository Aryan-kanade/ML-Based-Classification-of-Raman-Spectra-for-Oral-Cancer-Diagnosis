"""
study_stats.py — deeper study statistics on top of the trained models.

Everything here answers a question a reviewer or clinician will ask:

  * lopo_evaluate        — leave-one-PATIENT-out: the strictest check
                           for small paired datasets
  * friedman_nemenyi     — are the model differences real, or noise?
                           (Friedman chi-square + Nemenyi critical
                           difference, the standard benchmark test)
  * seed_stability       — how much do the numbers move with the seed?
  * noise_robustness     — calibrated noise at inference: does the
                           verdict survive a rougher measurement?
  * per_patient_rollup   — which PATIENTS does the model fail on?
  * band_stats_paired    — patient-paired band statistics with
                           Benjamini-Hochberg FDR (the classic
                           spectroscopy table)
  * triage_arithmetic    — per-1000-patients consequences of the
                           three-tier triage at a given prevalence

Pure numpy/scipy/sklearn; no new dependencies.
"""

from __future__ import annotations

import numpy as np

# Nemenyi q_alpha (alpha=0.05) for k=2..10 classifiers, from the
# studentized-range table (Demšar 2006, JMLR) — used by hand, no
# statsmodels dependency
NEMENYI_Q = {2: 1.960, 3: 2.349, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.936,
             8: 2.998, 9: 3.045, 10: 3.081}


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (monotone, sorted back)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]      # enforce monotone
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n)
    out[order] = adj
    return out.tolist()


# --------------------------------------------------------------------------
# Leave-one-patient-out
# --------------------------------------------------------------------------
def _lopo_fit_predict(X, ye, tr, te, estimator, params):
    """Worker: one LOPO fold (fit on all-but-one patient, predict the
    left-out one).  Runs inside a loky child — threads pinned to 1 so
    N workers don't oversubscribe the CPU (same pattern as the 3SSE
    search workers)."""
    import os
    from sklearn.base import clone
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    est = clone(estimator)
    if params:
        est = est.set_params(**params)
    est.fit(X[tr], ye[tr])
    return te, est.predict_proba(X[te])


def lopo_evaluate(X, y, groups, estimator, params: dict,
                  classes: list[str], seed: int = 0,
                  jobs: int = 1) -> dict:
    """
    Refit `estimator` (with fixed `params`, no re-tuning) once per
    left-out PATIENT and evaluate every patient unseen.  Returns pooled
    metrics + per-patient accuracy rows.  This is the strictest
    generalization check available on a small paired dataset.

    `jobs` > 1 validates folds in a joblib process pool — DEFAULT 1
    (serial): the measured rejection stands (tree winners refit with
    n_jobs=-1 internally; process × thread oversubscription lost more
    than the parallelism won, §16b-A4).
    """
    from sklearn.metrics import (confusion_matrix, f1_score,
                                 roc_auc_score)

    X = np.asarray(X)
    groups = np.asarray(groups)
    y = list(y)
    lut = {c: i for i, c in enumerate(classes)}
    ye = np.array([lut[v] for v in y], dtype=int)
    tasks = []
    for g in sorted(set(groups.tolist())):
        te = np.flatnonzero(groups == g)
        tr = np.flatnonzero(groups != g)
        if len(tr) == 0 or len(set(ye[tr].tolist())) < 2:
            continue
        tasks.append((X, ye, tr, te, estimator, params))
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=jobs)(
        delayed(_lopo_fit_predict)(*t) for t in tasks)
    per_patient = []
    y_true, y_pred, y_score = [], [], []
    for (te, proba) in results:
        pred = np.argmax(proba, axis=1)
        acc = float(np.mean(pred == ye[te]))
        pos_rate = (float(np.mean(proba[:, 1]))
                    if len(classes) == 2 else float("nan"))
        # the patient's dominant TRUE class (paired datasets have both)
        vals, counts = np.unique(ye[te], return_counts=True)
        true_cls = classes[int(vals[np.argmax(counts)])]
        per_patient.append((str(groups[te][0]), int(len(te)), acc,
                            pos_rate, true_cls))
        y_true.extend(ye[te].tolist())
        y_pred.extend(pred.tolist())
        if len(classes) == 2:
            y_score.extend(proba[:, 1].tolist())
    out = {"per_patient": per_patient,
           "n_patients": len(per_patient),
           "f1": float(f1_score(y_true, y_pred, average="macro"))
           if y_true else float("nan"),
           "cm": confusion_matrix(y_true, y_pred,
                                  labels=range(len(classes)))
           if y_true else None,
           "auc": float("nan"),
           "accs": [r[2] for r in per_patient]}
    if y_true and len(classes) == 2 and len(set(y_true)) == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    if out["cm"] is not None and len(classes) == 2:
        cm = out["cm"]
        tp = int(cm[1, 1]); fn_ = int(cm[1].sum()) - tp
        tn = int(cm[0, 0]); fp_ = int(cm[0].sum()) - tn
        out["sens"] = tp / (tp + fn_) if tp + fn_ else float("nan")
        out["spec"] = tn / (tn + fp_) if tn + fp_ else float("nan")
    return out


# --------------------------------------------------------------------------
# Friedman + Nemenyi across models
# --------------------------------------------------------------------------
def friedman_nemenyi(scores: dict[str, list[float]],
                     alpha_q: float | None = None) -> dict:
    """
    scores: model name -> list of per-fold macro-F1 (equal lengths).
    Friedman chi-square over models; when significant, Nemenyi critical
    difference on average ranks — the standard 'are these models really
    different' test (Demšar 2006).  Returns ranks, CD and the pairwise
    verdicts against the best model.
    """
    from scipy.stats import friedmanchisquare

    names = [n for n, v in scores.items() if len(v) > 1]
    cols = [list(scores[n]) for n in names]
    n_blocks = min(len(c) for c in cols)
    cols = [c[:n_blocks] for c in cols]
    if len(names) < 2 or n_blocks < 3:
        return {"ok": False, "reason": "need >= 2 models and >= 3 folds"}
    if len(names) >= 3:
        chi2, p = friedmanchisquare(*cols)
        if not np.isfinite(p):
            chi2, p = float("nan"), 1.0    # fully tied -> no difference
    else:
        # 2 models: Friedman degenerates — use the paired Wilcoxon
        from scipy.stats import wilcoxon
        diff_ok = not np.allclose(cols[0], cols[1])
        if diff_ok:
            try:
                w_p = float(wilcoxon(cols[0], cols[1]).pvalue)
            except Exception:
                w_p = 1.0
            if not np.isfinite(w_p):
                w_p = 1.0
        else:
            w_p = 1.0
        chi2 = float("nan")
        p = w_p
    mat = np.asarray(cols, dtype=float)              # models x blocks
    # average ranks per block (1 = best, ties averaged)
    ranks = np.zeros_like(mat)
    for j in range(n_blocks):
        order = np.argsort(-mat[:, j], kind="mergesort")
        sorted_vals = mat[order, j]
        r = np.empty(len(names))
        i = 0
        while i < len(sorted_vals):
            k = i
            while k + 1 < len(sorted_vals) and sorted_vals[k + 1] == \
                    sorted_vals[i]:
                k += 1
            avg = 0.5 * (i + k) + 1.0
            r[order[i:k + 1]] = avg
            i = k + 1
        ranks[:, j] = r
    avg_ranks = ranks.mean(axis=1)
    k = len(names)
    q = alpha_q if alpha_q is not None else NEMENYI_Q.get(
        k, 3.081 + 0.02 * (k - 10))
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    best = int(np.argmin(avg_ranks))
    vs_best = {names[i]: (float(avg_ranks[i] - avg_ranks[best]),
                          bool(avg_ranks[i] - avg_ranks[best] > cd))
               for i in range(k) if i != best}
    return {"ok": True, "chi2": float(chi2), "p": float(p),
            "significant": bool(p < 0.05), "cd": float(cd),
            "n_blocks": int(n_blocks),
            "avg_ranks": {names[i]: float(avg_ranks[i])
                          for i in range(k)},
            "best": names[best], "vs_best": vs_best}


# --------------------------------------------------------------------------
# Seed stability & noise robustness (winner template, fixed params)
# --------------------------------------------------------------------------
def _grouped_f1(X, ye, groups, estimator, params, k: int, seed: int,
                noise_level: float = 0.0,
                rng=None) -> float:
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    X = np.asarray(X, dtype=float)
    n_groups = len(set(np.asarray(groups).tolist()))
    k = int(min(k, n_groups))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                random_state=seed)
    y_true, y_pred = [], []
    for tr, te in sgkf.split(X, ye, groups):
        est = clone(estimator)
        if params:
            est = est.set_params(**params)
        est.fit(X[tr], ye[tr])
        Xte = X[te]
        if noise_level > 0 and rng is not None:
            scale = np.std(Xte, axis=1, keepdims=True)
            Xte = Xte + rng.normal(0, 1, Xte.shape) * scale * noise_level
        pred = est.predict(Xte)
        y_true.extend(ye[te].tolist())
        y_pred.extend(np.asarray(pred).tolist())
    return float(f1_score(y_true, y_pred, average="macro"))


def seed_stability(X, y, groups, estimator, params, classes: list[str],
                   seeds=(0, 1, 2, 3, 4), k: int = 5) -> list[float]:
    """Macro-F1 of the fixed winner across several CV seeds."""
    lut = {c: i for i, c in enumerate(classes)}
    ye = np.array([lut[v] for v in y], dtype=int)
    return [_grouped_f1(X, ye, groups, estimator, params, k, s)
            for s in seeds]


def noise_robustness(X, y, groups, estimator, params, classes: list[str],
                     levels=(0.0, 0.01, 0.02, 0.05), k: int = 5,
                     seed: int = 0) -> list[tuple[float, float]]:
    """
    Fit on clean training folds, predict test folds with calibrated
    Gaussian noise added at inference (per-spectrum scale).  Returns
    [(noise_fraction, macro_f1)] — the degradation curve.

    Every level shares the SAME folds and the SAME clean fits — each
    fold is fit ONCE and only re-predicted per noise level (the naive
    level-by-level loop refit 4x for nothing).  Noise is pre-drawn in
    the identical order the sequential version consumed it, so the
    curve is bit-identical to the original implementation.
    """
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    lut = {c: i for i, c in enumerate(classes)}
    ye = np.array([lut[v] for v in y], dtype=int)
    X = np.asarray(X, dtype=float)
    n_groups = len(set(np.asarray(groups).tolist()))
    k = int(min(k, n_groups))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                random_state=seed)
    folds = list(sgkf.split(X, ye, groups))
    rng = np.random.default_rng(seed)
    noise = {lv: ([rng.normal(0, 1, X[te].shape) for _tr, te in folds]
                  if lv > 0 else [None] * len(folds))
             for lv in levels}
    preds = {lv: [] for lv in levels}
    y_true: list = []
    for i, (tr, te) in enumerate(folds):
        est = clone(estimator)
        if params:
            est = est.set_params(**params)
        est.fit(X[tr], ye[tr])
        scale = np.std(X[te], axis=1, keepdims=True)
        for lv in levels:
            Xte = X[te]
            if noise[lv][i] is not None:
                Xte = Xte + noise[lv][i] * scale * lv
            preds[lv].extend(np.asarray(est.predict(Xte)).tolist())
        y_true.extend(ye[te].tolist())
    return [(lv, float(f1_score(y_true, preds[lv], average="macro")))
            for lv in levels]


# --------------------------------------------------------------------------
# Per-patient rollup of out-of-fold predictions
# --------------------------------------------------------------------------
def patient_level_metrics(y, groups, oof_proba) -> dict | None:
    """
    B6: pool the out-of-fold probabilities per PATIENT (mean), call
    each patient by its dominant true class, and report macro-F1 (+AUC
    for 2-class) at the PATIENT level — the operating level the field
    reports (Jeng 2019 ~+6pp; Farnesi 2025 majority-vote standard).
    Works with string labels or pre-encoded ints; returns None when
    there are no groups."""
    from sklearn.metrics import f1_score, roc_auc_score

    oof = np.asarray(oof_proba, dtype=float)
    if groups is None:
        return None
    y = list(y)
    groups = np.asarray(groups)
    keep = ~np.isnan(oof).any(axis=1)
    oof, y, groups = oof[keep], [v for v, k in zip(y, keep, strict=True)
                                 if k], groups[keep]
    if not len(y) or len(set(groups.tolist())) < 2:
        return None
    classes = sorted(set(y))
    lut = {c: i for i, c in enumerate(classes)}
    ye = np.array([lut[v] for v in y], dtype=int)
    y_pat, p_pat = [], []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        p_pat.append(oof[m].mean(axis=0))
        vals, counts = np.unique(ye[m], return_counts=True)
        y_pat.append(int(vals[np.argmax(counts)]))
    P = np.vstack(p_pat)
    Y = np.array(y_pat)
    out = {"n_patients": int(len(Y)),
           "f1": float(f1_score(Y, P.argmax(axis=1), average="macro"))}
    if len(classes) == 2 and len(set(Y.tolist())) == 2:
        try:
            out["auc"] = float(roc_auc_score(Y, P[:, 1]))
        except Exception:
            out["auc"] = float("nan")
    return out


def per_patient_rollup(y, groups, oof_proba, classes: list[str],
                       pos_idx: int = 1) -> list[tuple]:
    """
    Per-patient out-of-fold accuracy + mean P(positive).  Rows:
    (patient, n, accuracy, mean_p_pos, dominant_true_class, hard_flag)
    sorted worst-first — the 'hard patients' failure analysis.
    """
    ye_pred = np.argmax(np.asarray(oof_proba), axis=1)
    lut = {c: i for i, c in enumerate(classes)}
    ye = np.array([lut[v] for v in y], dtype=int)
    groups = np.asarray(groups)
    rows = []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        acc = float(np.mean(ye_pred[m] == ye[m]))
        p_pos = (float(np.nanmean(np.asarray(oof_proba)[m, pos_idx]))
                 if len(classes) == 2 else float("nan"))
        vals, counts = np.unique(ye[m], return_counts=True)
        true_cls = classes[int(vals[np.argmax(counts)])]
        rows.append((str(g), int(m.sum()), acc, p_pos, true_cls,
                     acc < 0.5))
    rows.sort(key=lambda r: r[2])
    return rows


# --------------------------------------------------------------------------
# Patient-paired band statistics with FDR
# --------------------------------------------------------------------------
def band_stats_paired(X, y, groups, wn) -> list[tuple]:
    """
    For every literature band: patient-paired Wilcoxon signed-rank on
    (mean tumor band area − mean own-normal band area), with
    Benjamini-Hochberg FDR across bands.  Rows:
    (center, molecule, direction, median_delta, p_raw, p_fdr, significant)
    """
    from scipy.stats import wilcoxon
    import biochemistry as bio

    X = np.asarray(X, dtype=float)
    wn = np.asarray(wn, dtype=float)
    y = list(y)
    groups = list(groups)
    classes = sorted(set(y))
    if len(classes) != 2:
        return []
    # patient -> class -> mean band area
    per: dict[tuple, dict[float, float]] = {}
    for i in range(len(y)):
        key = (groups[i], y[i])
        d = per.setdefault(key, {})
        for center, hw, _mol, _assign, _dir in bio.BANDS:
            a = bio.band_area(wn, X[i], center, hw)
            d.setdefault(center, []).append(a)
    rows = []
    for center, _hw, mol, assign, direction in bio.BANDS:
        deltas = []
        for g in dict.fromkeys(groups):
            n_ = per.get((g, classes[0]), {}).get(center)
            t_ = per.get((g, classes[-1]), {}).get(center)
            if n_ and t_:
                dn = float(np.nanmean(n_))
                dt = float(np.nanmean(t_))
                if np.isfinite(dn) and np.isfinite(dt):
                    deltas.append(dt - dn)
        p = float("nan")
        if len(deltas) >= 5:
            try:
                if np.allclose(deltas, 0):
                    p = 1.0
                else:
                    p = float(wilcoxon(deltas).pvalue)
            except Exception:
                p = float("nan")
        rows.append([float(center), mol, assign, direction,
                     float(np.median(deltas)) if deltas else float("nan"),
                     p, False])
    finite = [i for i, r in enumerate(rows) if np.isfinite(r[5])]
    if finite:
        adj = bh_fdr([rows[i][5] for i in finite])
        for i, a in zip(finite, adj, strict=True):
            rows[i][6] = bool(a < 0.05)
            rows[i][5] = a           # report FDR-adjusted p
    return [tuple(r) for r in rows]


# --------------------------------------------------------------------------
# Triage arithmetic (per-1000 consequences)
# --------------------------------------------------------------------------
def triage_arithmetic(y_true, tiers: list[str], prevalence: float,
                      positive_tier: str = "POSITIVE",
                      indeterminate_tier: str = "INDETERMINATE",
                      negative_tier: str = "NEGATIVE",
                      per_1000: int = 1000) -> dict:
    """
    Translate the empirical triage-tier rates (from out-of-fold
    predictions) into consequences per 1,000 patients at a given
    prevalence: how many biopsies the triage saves and how many
    cancers it misses at the NEGATIVE tier.
    """
    y_true = np.asarray(y_true)
    tiers = np.asarray(tiers)
    dis = y_true == 1
    n_dis, n_no = int(dis.sum()), int((~dis).sum())
    if n_dis == 0 or n_no == 0:
        return {}
    p_tier = {}
    for tier in (positive_tier, indeterminate_tier, negative_tier):
        p_tier[tier] = (float(np.mean(tiers[dis] == tier)),
                        float(np.mean(tiers[~dis] == tier)))
    k = per_1000
    d = k * prevalence
    nd = k * (1.0 - prevalence)
    pos_d = d * p_tier[positive_tier][0]
    ind_d = d * p_tier[indeterminate_tier][0]
    neg_d = d * p_tier[negative_tier][0]
    pos_n = nd * p_tier[positive_tier][1]
    ind_n = nd * p_tier[indeterminate_tier][1]
    return {"biopsies_all": k,
            "biopsies_triage": round(pos_d + ind_d + pos_n + ind_n),
            "cancers_caught": round(pos_d),
            "cancers_in_indeterminate": round(ind_d),
            "cancers_missed_negative": round(neg_d),
            "reduction_pct": 100.0 * (1.0 - (pos_d + ind_d + pos_n
                                             + ind_n) / k),
            "caught_or_flagged_pct": 100.0 * (pos_d + ind_d) / d
            if d else float("nan")}
