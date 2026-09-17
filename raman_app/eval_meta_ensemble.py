"""
eval_meta_ensemble.py -- L7 of the push-to-0.8 program (SAFETY: pure
experiment script; writes experiments/ only).

L2 showed averaging SEEDS of one chain does not help.  This lever
averages DIFFERENT ARCHITECTURES: the OOF probabilities of the top-3
most stable chains (from experiments/multiseed_top.json, L4) are
averaged per row under identical outer folds.  Architecture diversity
is a different variance axis than seed diversity.

Usage:  python eval_meta_ensemble.py [--top 3]
Output: experiments/meta_ensemble.json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    ranked_path = os.path.join(OUT_DIR, "multiseed_top.json")
    if not os.path.isfile(ranked_path):
        print("[meta] experiments/multiseed_top.json missing — run "
              "eval_multiseed_top.py first")
        return 1
    with open(ranked_path, encoding="utf-8") as fh:
        ranked = json.load(fh)[:args.top]
    archs = [c["arch"] for c in ranked]
    for a in archs:
        print(f"[meta] member: {' -> '.join(a)}", flush=True)

    X, y, groups, wn, _grid, _params, meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    X = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y)
    classes = sorted(set(y_arr.tolist()))
    ye = np.searchsorted(np.asarray(classes), y_arr)
    groups = np.asarray(groups)
    factories = sequential._factories(wn)

    seed, k_outer = 42, 5
    outer = sequential._splitter(k_outer, seed, ye, groups)
    singles = [np.full((len(ye), len(classes)), np.nan) for _ in archs]
    for tr, te in outer.split(X, ye, groups):
        g_tr = groups[tr]
        y_tr = ye[tr]
        k_in = max(2, min(3, int(min(np.bincount(y_tr)))))
        k_in = min(k_in, len(set(g_tr.tolist())))
        for i, arch in enumerate(archs):
            F_tr, F_te = X[tr], X[te]
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
                sequential._layer_est(arch[-1], factories,
                                      len(arch) - 1, wn),
                F_tr, y_tr, g_tr)
            singles[i][te] = last.predict_proba(F_te)

    out = {"members": [" -> ".join(a) for a in archs], "single": [],
           "f1_by_member": []}
    valid = ~np.isnan(singles[0]).any(axis=1)
    for i in range(len(archs)):
        m = sequential._metrics_from_oof(ye[valid], singles[i][valid],
                                         classes, groups[valid])
        out["single"].append(float(m["f1"]))
        print(f"[meta] member {i} F1 {m['f1']:.3f}", flush=True)
    avg = np.mean(singles, axis=0)
    mavg = sequential._metrics_from_oof(ye[valid], avg[valid], classes,
                                        groups[valid])
    out["meta_f1"] = float(mavg["f1"])
    out["meta_sens"] = float(mavg["sens"])
    out["meta_spec"] = float(mavg["spec"])
    out["verdict"] = (
        f"meta-ensemble F1 {mavg['f1']:.3f} vs best member "
        f"{max(out['single']):.3f} "
        f"({mavg['f1'] - max(out['single']):+.3f})")
    print(f"[meta] {out['verdict']}", flush=True)
    path = os.path.join(OUT_DIR, "meta_ensemble.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[meta] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
