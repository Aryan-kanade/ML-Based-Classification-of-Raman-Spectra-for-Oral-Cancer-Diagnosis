"""
eval_multiseed_top.py -- L4 of the push-to-0.8 program (SAFETY: pure
experiment script; writes experiments/ only).

Anti-fold-luck: the leaderboard ranks chains by ONE nested validation
(seed 42).  Re-validate the top level-3 chains from BOTH searches at
outer seeds 42/43/44 and rank by mean F1 with a variance penalty --
a chain that got 0.76 on a lucky split but collapses elsewhere loses
to a stable 0.74.

Usage:  python eval_multiseed_top.py [--top 4]
Output: experiments/multiseed_top.json + console ranking.
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
RUNS = ("study_run_3sse_d2", "study_run_3sse_proven")
SEEDS = (42, 43, 44)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # candidate chains: level-3 entries of every available search
    cands: list[dict] = []
    for run in RUNS:
        path = os.path.join(HERE, run, "validated.json")
        if not os.path.isfile(path):
            print(f"[multi] {run}: not finished yet — skipped")
            continue
        with open(path, encoding="utf-8") as fh:
            v = json.load(fh)
        entries = v.get("3") or v.get(3) or []
        for e in entries[:args.top]:
            cands.append({"run": run, "arch": e["arch"],
                          "screen_f1": e["metrics"]["f1"]})
    if not cands:
        print("[multi] no candidates available yet")
        return 1
    print(f"[multi] {len(cands)} candidate chains", flush=True)

    X, y, groups, wn, _grid, _params, _meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    factories = sequential._factories(wn)

    for c in cands:
        f1s = []
        for seed in SEEDS:
            m = sequential.validate_arch(c["arch"], factories, X,
                                         list(y), list(groups),
                                         seed=seed, k_outer=5, wn=wn)
            f1s.append(float(m["f1"]))
        arr = np.asarray(f1s)
        c["f1_by_seed"] = {str(s): f for s, f in zip(SEEDS, f1s,
                                                      strict=True)}
        c["mean_f1"] = float(arr.mean())
        c["std_f1"] = float(arr.std())
        c["score"] = c["mean_f1"] - 0.5 * c["std_f1"]
        print(f"[multi] {' -> '.join(c['arch'])}\n"
              f"        seeds {[f'{f:.3f}' for f in f1s]} · "
              f"mean {c['mean_f1']:.3f} ± {c['std_f1']:.3f} · "
              f"score {c['score']:.3f}", flush=True)

    cands.sort(key=lambda c: -c["score"])
    out = os.path.join(OUT_DIR, "multiseed_top.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(cands, fh, indent=2)
    best = cands[0]
    print(f"[multi] WINNER: {' -> '.join(best['arch'])} "
          f"(mean {best['mean_f1']:.3f} ± {best['std_f1']:.3f})\n"
          f"[multi] saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
