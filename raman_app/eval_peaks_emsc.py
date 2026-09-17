"""
eval_peaks_emsc.py -- L35 peak-parameter features + L36 EMSC scatter
correction (SAFETY: pure experiment script -> experiments/ only).

L35: scipy find_peaks per deviation spectrum -> top-30 peaks by
prominence -> (position, height, width) block appended to the matrix.
L36: EMSC in-fold — per outer fold, reference = train-fold mean
spectrum; each spectrum fit x ~ a*ref + b*wl + d, corrected
(x - b*wl - d)/a.  Multiplicative+additive, where ALS is additive
only.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.signal import find_peaks, peak_widths

import sequential
from exp_common import OUT_DIR, load_data, record, stamp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = OUT_DIR


def peak_block(X: np.ndarray, wn: np.ndarray, k: int = 30) -> np.ndarray:
    feats = np.zeros((X.shape[0], k * 3), dtype=np.float32)
    scale = (wn.max() - wn.min()) or 1.0
    for i in range(X.shape[0]):
        x = X[i]
        pk, props = find_peaks(x, prominence=x.std() * 1.5)
        if len(pk) == 0:
            continue
        order = np.argsort(props["prominences"])[::-1][:k]
        pk = pk[order]
        prom = props["prominences"][order]
        widths = peak_widths(x, pk, rel_height=0.5)[0]
        for j, (p, pr, w) in enumerate(zip(pk, prom, widths)):
            feats[i, j * 3] = (wn[p] - wn.min()) / scale
            feats[i, j * 3 + 1] = pr / (x.std() + 1e-9)
            feats[i, j * 3 + 2] = w / len(wn)
    return feats


def emsc_fold_correct(F_tr: np.ndarray, F_te: np.ndarray,
                      wn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = F_tr.mean(axis=0)
    wl = (wn - wn.mean()) / (wn.std() + 1e-9)

    def correct(M):
        A = np.column_stack([ref, wl, np.ones(len(wn))])
        coef, *_ = np.linalg.lstsq(A, M.T, rcond=None)
        a, b, d = coef                       # each (n_spec,)
        num = M.T - wl[:, None] * b[None, :] - d[None, :]
        # sign/near-zero guard: deviation-spectra means can be flat,
        # which would flip or blow up the division (audit finding 18)
        safe_a = np.where(np.abs(a) > 1e-6, a, 1.0)
        return (num / safe_a[None, :]).T

    return correct(F_tr), correct(F_te)


def chain_oof_custom(arch, factories, X, y_arr, groups, classes, seed,
                     wn, emsc=False, extra_feats=None):
    """validate_arch loop with optional in-fold EMSC + feature append
    (extra computed once on ALL rows — peak positions are per-row
    physical quantities, no label leakage)."""
    if extra_feats is not None:
        X = np.hstack([X, extra_feats])
    outer = sequential._splitter(5, seed, y_arr, groups)
    oof = np.full((len(y_arr), len(classes)), np.nan)
    for tr, te in outer.split(X, y_arr, groups):
        F_tr, F_te = X[tr].copy(), X[te].copy()
        if emsc:
            F_tr, F_te = emsc_fold_correct(F_tr, F_te, wn)
        g_tr = groups[tr]
        y_tr = y_arr[tr]
        k_in = max(2, min(3, int(min(np.bincount(y_tr)))))
        k_in = min(k_in, len(set(g_tr.tolist())))
        for pos, name in enumerate(arch[:-1]):
            P_tr = sequential._oof_proba(
                sequential._layer_est(name, factories, pos, wn),
                F_tr, list(y_tr), g_tr, k_in, seed)
            est = sequential.fit_maybe_grouped(
                sequential._layer_est(name, factories, pos, wn),
                F_tr, y_tr, g_tr)
            F_tr = np.hstack([F_tr, P_tr])
            F_te = np.hstack([F_te, est.predict_proba(F_te)])
        last = sequential.fit_maybe_grouped(
            sequential._layer_est(arch[-1], factories, len(arch) - 1,
                                  wn), F_tr, y_tr, g_tr)
        oof[te] = last.predict_proba(F_te)
    valid = ~np.isnan(oof).any(axis=1)
    return sequential._metrics_from_oof(y_arr[valid], oof[valid],
                                        classes, groups[valid])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    arch, X, y, groups, wn, _meta = load_data()
    y_arr = np.searchsorted(sorted(set(y)), np.asarray(y))
    groups = np.asarray(groups)
    classes = sorted(set(y))
    factories = sequential._factories(wn)
    print(f"[pke] {stamp()} arch {' -> '.join(arch)}", flush=True)

    peaks = peak_block(X, wn)
    print(f"[pke] peak block: {peaks.shape[1]} features", flush=True)

    m0 = chain_oof_custom(arch, factories, X, y_arr, groups, classes,
                          42, wn)
    f0 = float(m0.get("f1_mean", m0["f1"]))
    print(f"[pke] reference (custom loop) F1 {f0:.3f}", flush=True)

    m1 = chain_oof_custom(arch, factories, X, y_arr, groups, classes,
                          42, wn, extra_feats=peaks)
    f1 = float(m1.get("f1_mean", m1["f1"]))
    print(f"[pke] +peak features F1 {f1:.3f}", flush=True)
    record("L35 peak-features", "top-30 pos/int/width appended", f1)

    m2 = chain_oof_custom(arch, factories, X, y_arr, groups, classes,
                          42, wn, emsc=True)
    f2 = float(m2.get("f1_mean", m2["f1"]))
    print(f"[pke] EMSC in-fold F1 {f2:.3f}", flush=True)
    record("L36 EMSC", "per-fold multiplicative+additive correction",
           f2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
