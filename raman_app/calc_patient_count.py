"""
calc_patient_count.py -- patient-count-to-target calculation (SAFETY:
pure experiment script; writes experiments/ only).

Learning curve of the d2 winner architecture over PATIENT subsets
(fractions 0.3/0.5/0.7/1.0, stratified by class, 3 outer seeds each,
patient-grouped 5-fold), then a saturating fit F1(n) = a - b/n and
extrapolation: how many paired patients for honest F1 0.80 / 0.85?

Extrapolation caveat: assumes the measured trend continues; at 2x
beyond the data it is an estimate, not a promise.

Usage:  python calc_patient_count.py
Output: experiments/patient_count.json
"""

from __future__ import annotations

import json
import os

import numpy as np

import sequential
from run_3sse_d2 import _prepare_winner

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "experiments")
FRACTIONS = (0.3, 0.5, 0.7, 1.0)
SEEDS = (42, 43, 44)
TARGETS = (0.80, 0.85)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]
    X, y, groups, wn, _grid, _params, meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    patients = sorted(set(groups))
    y_by_pat = {p: y[i] for i, p in enumerate(groups)}
    n_pat = len(patients)
    print(f"[count] arch {' -> '.join(arch)} · {n_pat} patients",
          flush=True)
    factories = sequential._factories(wn)
    rng = np.random.default_rng(0)

    curve = []
    for frac in FRACTIONS:
        k = max(4, int(round(n_pat * frac)))
        f1s = []
        for seed in SEEDS:
            # stratified patient subsample: keep the class balance
            picks = {c: [] for c in set(y_by_pat.values())}
            order = rng.permutation(n_pat)
            for idx in order:
                p = patients[idx]
                if len(picks[y_by_pat[p]]) < max(
                        1, round(k * sum(1 for q in patients
                                         if y_by_pat[q] == y_by_pat[p])
                                / n_pat)):
                    picks[y_by_pat[p]].append(p)
            keep_pats = set(picks["Normal"]) | set(picks["Tumor"]) \
                if "Normal" in picks else {p for lst in picks.values()
                                           for p in lst}
            keep = [i for i, p in enumerate(groups) if p in keep_pats]
            Xs = np.asarray(X)[keep]
            ys = [y[i] for i in keep]
            gs = [groups[i] for i in keep]
            m = sequential.validate_arch(arch, factories, Xs, ys, gs,
                                         seed=seed, k_outer=5, wn=wn)
            f1s.append(float(m["f1"]))
        curve.append({"n_patients": len(keep_pats),
                      "f1_mean": float(np.mean(f1s)),
                      "f1_std": float(np.std(f1s))})
        print(f"[count] n={len(keep_pats):>3}: F1 {np.mean(f1s):.3f} "
              f"+/- {np.std(f1s):.3f}", flush=True)

    # saturating fit F1(n) = a - b/n  (least squares on the curve)
    ns = np.array([c["n_patients"] for c in curve], dtype=float)
    fs = np.array([c["f1_mean"] for c in curve])
    A = np.vstack([np.ones_like(ns), -1.0 / ns]).T
    coef, *_ = np.linalg.lstsq(A, fs, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    out = {"arch": arch, "curve": curve, "fit_a": a, "fit_b": b,
           "targets": {}}
    print(f"[count] fit: F1(n) = {a:.3f} - {b:.3f}/n "
          f"(ceiling a = {a:.3f})", flush=True)
    for t in TARGETS:
        if a <= t:
            out["targets"][str(t)] = None
            print(f"[count] F1 {t}: NOT REACHABLE — fitted ceiling "
                  f"{a:.3f} is below the target", flush=True)
        else:
            n_need = b / (a - t)
            out["targets"][str(t)] = float(n_need)
            print(f"[count] F1 {t}: ~{n_need:.0f} paired patients "
                  f"(have {n_pat})", flush=True)
    path = os.path.join(OUT_DIR, "patient_count.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[count] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
