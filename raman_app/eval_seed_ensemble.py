"""
eval_seed_ensemble.py -- L2 of the push-to-0.8 program (SAFETY: pure
experiment script; reads d2 artifacts, writes experiments/ only).

Question: does seed-ensembling the d2 winner chain (PLS + XGBoost ->
Random Forest -> Extra Trees) raise the HONEST nested-CV number?
The deployed bundle already wraps the chain in AveragedChain(n_seeds=3)
at inference, but validate_arch scores a SINGLE seed -- the ensemble's
own honest score was never measured.

Protocol: identical outer folds (StratifiedGroupKFold k=5 seed 42) for
every variant.  Single-seed = validate_arch as-is.  Ensemble variants
rebuild the per-fold features n_seeds times (inner split seed + s) and
average the members' predict_proba -- exactly AveragedChain semantics,
inside the nested loop (no leakage: every member sees only the fold's
training rows).

Usage:  python eval_seed_ensemble.py [--seeds 3 5]
Output: experiments/seed_ensemble.json + console table.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import sequential
from run_3sse_d2 import _prepare_winner

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "experiments")


def _ensemble_validate_arch(arch, factories, X, y, groups, seed,
                            k_outer, wn, n_seeds):
    """validate_arch with an n_seeds member average per outer fold."""
    X = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y)
    classes = sorted(set(y_arr.tolist()))
    ye = np.searchsorted(np.asarray(classes), y_arr)
    outer = sequential._splitter(k_outer, seed, ye, groups)
    oof = np.full((len(ye), len(classes)), np.nan)
    for tr, te in outer.split(X, ye, groups):
        g_tr = np.asarray(groups)[tr] if groups is not None else None
        y_tr = ye[tr]
        min_class = min(np.bincount(y_tr))
        k_in = max(2, min(3, min_class))
        if g_tr is not None:
            k_in = min(k_in, len(set(g_tr.tolist())))
        members = []
        for s in range(n_seeds):
            F_tr, F_te = X[tr], X[te]
            for pos, name in enumerate(arch[:-1]):
                P_tr = sequential._oof_proba(
                    sequential._layer_est(name, factories, pos, wn),
                    F_tr, list(y_tr), g_tr, k_in, seed + s)
                est = sequential.fit_maybe_grouped(
                    sequential._layer_est(name, factories, pos, wn),
                    F_tr, y_tr, g_tr)
                F_tr = np.hstack([F_tr, P_tr])
                F_te = np.hstack([F_te, est.predict_proba(F_te)])
            last = sequential.fit_maybe_grouped(
                sequential._layer_est(arch[-1], factories,
                                      len(arch) - 1, wn),
                F_tr, y_tr, g_tr)
            members.append(last.predict_proba(F_te))
        oof[te] = np.mean(members, axis=0)
    valid = ~np.isnan(oof).any(axis=1)
    sub_g = (np.asarray(groups)[valid] if groups is not None else None)
    return sequential._metrics_from_oof(ye[valid], oof[valid], classes,
                                        sub_g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 5])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]
    print(f"[ens] arch: {' -> '.join(arch)}", flush=True)

    X, y, groups, wn, _grid, _params, meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    print(f"[ens] data: {X.shape[0]} rows · {meta['n_patients']} "
          "patients", flush=True)
    factories = sequential._factories(wn)

    results = {}
    single = sequential.validate_arch(arch, factories, X, list(y),
                                      list(groups), seed=42, k_outer=5,
                                      wn=wn)
    results["single (deployed protocol)"] = _pick(single)
    print(f"[ens] single          : {_fmt(results['single (deployed protocol)'])}",
          flush=True)
    for n in args.seeds:
        m = _ensemble_validate_arch(arch, factories, X, list(y),
                                    list(groups), seed=42, k_outer=5,
                                    wn=wn, n_seeds=n)
        results[f"ensemble n_seeds={n}"] = _pick(m)
        print(f"[ens] ensemble n={n:<3}  : {_fmt(results[f'ensemble n_seeds={n}'])}",
              flush=True)

    base = results["single (deployed protocol)"]["f1"]
    verdict = max((v["f1"], k) for k, v in results.items()
                  if k != "single (deployed protocol)")
    results["_verdict"] = (
        f"best ensemble {verdict[1]} F1 {verdict[0]:.3f} vs single "
        f"{base:.3f} ({verdict[0] - base:+.3f})")
    print(f"[ens] {results['_verdict']}", flush=True)
    out = os.path.join(OUT_DIR, "seed_ensemble.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[ens] saved {out}", flush=True)
    return 0


def _pick(m: dict) -> dict:
    keys = ("f1", "sens", "spec", "auc", "pat_f1")
    return {k: (float(m[k]) if k in m
                and isinstance(m[k], (int, float)) else None)
            for k in keys}


def _fmt(d: dict) -> str:
    def f(v):
        return f"{v:.3f}" if isinstance(v, float) else "  —  "
    return (f"F1 {f(d['f1'])} · sens {f(d['sens'])} · "
            f"spec {f(d['spec'])} · patF1 {f(d['pat_f1'])}")


if __name__ == "__main__":
    raise SystemExit(main())
