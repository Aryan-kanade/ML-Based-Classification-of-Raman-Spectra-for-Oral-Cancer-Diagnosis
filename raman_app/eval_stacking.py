"""
eval_stacking.py -- L38 signed-log, L39 multi-view ensemble, L40
logistic stacking meta-learner, L41 inner-CV bagging, L42 PCA-denoise
(SAFETY: pure experiment script -> experiments/ only).

All arms run the winner arch through a shared custom outer loop with
per-arm feature or feature-construction variants, scored identically
(patient-grouped 5-fold, seed 42).
"""

from __future__ import annotations

import os

import numpy as np

import sequential
from exp_common import OUT_DIR, load_data, record, stamp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def build_oof(arch, factories, X, y_arr, groups, classes, seed, wn,
              inner_seeds=(42,), pca_denoise=False):
    outer = sequential._splitter(5, seed, y_arr, groups)
    oof = np.full((len(y_arr), len(classes)), np.nan)
    for tr, te in outer.split(X, y_arr, groups):
        F_tr, F_te = X[tr], X[te]
        if pca_denoise:
            pca = PCA(n_components=0.999, svd_solver="full")
            Z = pca.fit_transform(F_tr)
            F_tr = pca.inverse_transform(Z)
            F_te = pca.inverse_transform(pca.transform(F_te))
        g_tr = groups[tr]
        y_tr = y_arr[tr]
        k_in = max(2, min(3, int(min(np.bincount(y_tr)))))
        k_in = min(k_in, len(set(g_tr.tolist())))
        for pos, name in enumerate(arch[:-1]):
            Ps = [sequential._oof_proba(
                sequential._layer_est(name, factories, pos, wn),
                F_tr, list(y_tr), g_tr, k_in, s) for s in inner_seeds]
            P_tr = np.mean(Ps, axis=0)
            est = sequential.fit_maybe_grouped(
                sequential._layer_est(name, factories, pos, wn),
                F_tr, y_tr, g_tr)
            F_tr = np.hstack([F_tr, P_tr])
            F_te = np.hstack([F_te, est.predict_proba(F_te)])
        last = sequential.fit_maybe_grouped(
            sequential._layer_est(arch[-1], factories, len(arch) - 1,
                                  wn), F_tr, y_tr, g_tr)
        oof[te] = last.predict_proba(F_te)
    return oof


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    arch, X, y, groups, wn, _meta = load_data()
    X = np.asarray(X, dtype=np.float32)
    y_arr = np.searchsorted(sorted(set(y)), np.asarray(y))
    groups = np.asarray(groups)
    classes = sorted(set(y))
    factories = sequential._factories(wn)
    print(f"[stk] {stamp()} arch {' -> '.join(arch)}", flush=True)

    def score(oof):
        v = ~np.isnan(oof).any(axis=1)
        from sklearn.metrics import f1_score
        return float(f1_score(y_arr[v], oof[v].argmax(axis=1),
                              average="macro"))

    # L38 signed-log compression of the deviation matrix
    Xs = np.sign(X) * np.log1p(np.abs(X))
    f = score(build_oof(arch, factories, Xs, y_arr, groups, classes,
                        42, wn))
    print(f"[stk] L38 signed-log F1 {f:.3f}", flush=True)
    record("L38 signed-log", "sign(x)*log1p(|x|) before chain", f)

    # L41 inner-CV bagging (2 inner seeds averaged for layer features)
    f = score(build_oof(arch, factories, X, y_arr, groups, classes, 42,
                        wn, inner_seeds=(42, 43)))
    print(f"[stk] L41 inner-bagged F1 {f:.3f}", flush=True)
    record("L41 inner-bagging", "layer OOF averaged over 2 seeds", f)

    # L42 PCA-denoise prefix
    f = score(build_oof(arch, factories, X, y_arr, groups, classes, 42,
                        wn, pca_denoise=True))
    print(f"[stk] L42 PCA-denoise F1 {f:.3f}", flush=True)
    record("L42 PCA-denoise", "PCA 99.9% reconstruct in-fold", f)

    # L39 multi-view ensemble: deriv2 | deriv1 | vector-norm | signed-log
    import eval_multiview as mv
    X1v, y1, g1, wn1 = mv._paired_view(mv._view(sg_deriv=1))
    X2v, y2, g2, wn2 = mv._paired_view(mv._view(norm="vector"))
    views = [X, X1v, X2v, Xs]
    oofs = [build_oof(arch, factories, V, y_arr, groups, classes, 42,
                      wn) for V in views]
    f = score(np.mean(oofs, axis=0))
    print(f"[stk] L39 view-ensemble F1 {f:.3f}", flush=True)
    record("L39 view-ensemble", "4 views, probs averaged", f)

    # L40 logistic stacking over member chains (winner + sibling +
    # peak-bands chain + each view chain)
    POOL = [
        ["PLS + XGBoost", "Random Forest", "Extra Trees"],
        ["PLS + XGBoost", "Extra Trees", "Random Forest"],
        ["Peak bands + RF", "Random Forest", "Extra Trees"],
    ]
    member_oofs = [build_oof(a, factories, X, y_arr, groups, classes,
                             42, wn) for a in POOL]
    member_oofs += oofs[:3]                    # view chains (d2/d1/vec)
    M = np.stack(member_oofs)                  # (m, n, 2)
    outer = sequential._splitter(5, 42, y_arr, groups)
    oof_meta = np.full((len(y_arr), len(classes)), np.nan)
    for tr, te in outer.split(X, y_arr, groups):
        feats_tr = np.hstack([M[i, tr] for i in range(len(M))])
        feats_te = np.hstack([M[i, te] for i in range(len(M))])
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(feats_tr, y_arr[tr])
        oof_meta[te] = lr.predict_proba(feats_te)
    f = score(oof_meta)
    print(f"[stk] L40 log-stack F1 {f:.3f} "
          f"({len(M)} members)", flush=True)
    record("L40 logistic stacking",
           f"{len(M)} chains+views, LogReg meta in-fold", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
