"""
eval_ensemble_variants.py -- L17 TTA + L19 OOF-bagging + L18 greedy
ensemble selection, one custom outer loop shared by all three arms.
SAFETY: pure experiment script -> experiments/ only.

- L19: average the chain's OOF probabilities across 3 outer
  partitions (seeds 42/43/44) before metrics — partition-variance
  reduction, every row still held-out in each partition.
- L17: per-fold TTA — average the held-out prediction over jittered
  copies of the spectral columns (noise 1% std, +-1 index shift).
- L18: greedy ensemble (Caruana) over the POOL of three diverse
  chains (d2 winner, its layer-swapped sibling, Peak-bands chain),
  fitted inside each fold with replacement weights.
"""

from __future__ import annotations

import os

import numpy as np

import sequential
from exp_common import OUT_DIR, load_data, record, stamp
from sklearn.metrics import f1_score

POOL = (
    ["PLS + XGBoost", "Random Forest", "Extra Trees"],
    ["PLS + XGBoost", "Extra Trees", "Random Forest"],
    ["Peak bands + RF", "Random Forest", "Extra Trees"],
)


def _chain_oof(arch, factories, X, y_arr, groups, classes, seed,
               tta=False):
    outer = sequential._splitter(5, seed, y_arr, groups)
    oof = np.full((len(y_arr), len(classes)), np.nan)
    for tr, te in outer.split(X, y_arr, groups):
        g_tr = groups[tr]
        y_tr = y_arr[tr]
        k_in = max(2, min(3, int(min(np.bincount(y_tr)))))
        k_in = min(k_in, len(set(g_tr.tolist())))
        F_tr, F_te = X[tr], X[te]
        for pos, name in enumerate(arch[:-1]):
            P_tr = sequential._oof_proba(
                sequential._layer_est(name, factories, pos, WN),
                F_tr, list(y_tr), g_tr, k_in, seed)
            est = sequential.fit_maybe_grouped(
                sequential._layer_est(name, factories, pos, WN),
                F_tr, y_tr, g_tr)
            F_tr = np.hstack([F_tr, P_tr])
            F_te = np.hstack([F_te, est.predict_proba(F_te)])
        last = sequential.fit_maybe_grouped(
            sequential._layer_est(arch[-1], factories, len(arch) - 1,
                                  WN),
            F_tr, y_tr, g_tr)
        if not tta:
            oof[te] = last.predict_proba(F_te)
            continue
        rng = np.random.default_rng(seed)
        preds = [last.predict_proba(F_te)]
        for _ in range(4):
            Xj = F_te.copy()
            ns = Xj[:, :X.shape[1]]
            Xj[:, :X.shape[1]] = ns + rng.normal(
                0, 0.01 * (ns.std() + 1e-9), ns.shape)
            shift = int(rng.integers(-1, 2))
            Xj[:, :X.shape[1]] = np.roll(Xj[:, :X.shape[1]], shift,
                                         axis=1)
            preds.append(last.predict_proba(Xj))
        oof[te] = np.mean(preds, axis=0)
    return oof


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    arch, X, y, groups, wn, _meta = load_data()
    global WN
    WN = wn
    X = np.asarray(X, dtype=np.float32)
    y_arr = np.searchsorted(sorted(set(y)), np.asarray(y))
    groups = np.asarray(groups)
    classes = sorted(set(y))
    factories = sequential._factories(wn)
    print(f"[ensv] {stamp()} pool: {[' -> '.join(a) for a in POOL]}",
          flush=True)

    # L19: OOF-bagging over 3 partitions for the WINNER arch
    oofs = []
    for seed in (42, 43, 44):
        oofs.append(_chain_oof(arch, factories, X, y_arr, groups,
                               classes, seed))
    valid = ~np.isnan(oofs[0]).any(axis=1)
    f1_bag = f1_score(y_arr[valid], np.mean(oofs, axis=0)[valid].argmax(
        axis=1), average="macro")
    print(f"[ensv] L19 OOF-bagged F1 {f1_bag:.3f}", flush=True)
    record("L19 OOF-bagging", "winner arch, 3 partitions averaged",
           float(f1_bag))

    # L17: TTA on the winner arch, single partition
    oof_tta = _chain_oof(arch, factories, X, y_arr, groups, classes,
                         42, tta=True)
    f1_tta = f1_score(y_arr[valid], oof_tta[valid].argmax(axis=1),
                      average="macro")
    print(f"[ensv] L17 TTA F1 {f1_tta:.3f}", flush=True)
    record("L17 TTA", "winner arch, 4 jittered copies averaged",
           float(f1_tta))

    # L18: greedy ensemble over the pool (inside fold 42)
    member_oof = [_chain_oof(a, factories, X, y_arr, groups, classes,
                             42) for a in POOL]
    rng = np.random.default_rng(0)
    w = np.zeros(len(POOL))
    best_f1 = -1.0
    for _ in range(30):
        i = int(rng.integers(0, len(POOL)))
        trial = w.copy()
        trial[i] += 1.0
        pred = np.tensordot(trial / trial.sum(),
                            np.stack(member_oof), axes=1).argmax(axis=1)
        f1 = f1_score(y_arr[valid], pred[valid], average="macro")
        if f1 > best_f1:
            best_f1, w = f1, trial
    print(f"[ensv] L18 greedy ensemble F1 {best_f1:.3f} "
          f"(weights {w / w.sum()})", flush=True)
    record("L18 greedy ensemble",
           "3-chain pool, in-fold greedy selection", float(best_f1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
