"""
run_program_queue.py -- L23 orchestrator: runs the remaining push-to-
0.8 lever scripts SEQUENTIALLY as CPU frees (never two loky pools at
once — gotcha #16).  Each child writes its own experiments/*.json and
appends to SUMMARY.md; a failing child is logged and skipped, never
fatal.  Re-run safe: pass --only name1,name2 to restrict.

Usage:  python run_program_queue.py [--only l3,l5,...]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = [
    ("L4-re-rank", "eval_multiseed_top.py", ["--top", "5"]),
    ("L3-prep-arms", "prep_chain_prep_search.py", []),
    ("L34-windows", "eval_windows.py", []),
    ("L35-36-peaks-emsc", "eval_peaks_emsc.py", []),
    ("L5-band-feats", "eval_feature_expand.py", []),
    ("L38-42-stacking", "eval_stacking.py", []),
    ("L7-meta-ensemble", "eval_meta_ensemble.py", []),
    ("L8-standard-arm", "eval_standard_arm.py", []),
    ("L12-wide-tuning", "eval_wide_tuning.py", []),
    ("L13-augment", "eval_augment.py", []),
    ("L16-multiview", "eval_multiview.py", []),
    ("L17-19-ensembles", "eval_ensemble_variants.py", []),
    ("L28-31-33-closers", "eval_closers.py", []),
    ("L45-50-signals", "eval_signals.py", []),
    ("L54-58-deep", "eval_deep_arms.py", []),
    ("L51-pretrain", "eval_pretrain.py", []),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated queue names to run")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",")} \
        if args.only else None
    t0 = time.time()
    for name, script, extra in QUEUE:
        if only and name not in only:
            continue
        print(f"\n=== [{time.strftime('%H:%M:%S')}] {name} "
              f"({script}) ===", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, script)]
                           + extra, cwd=HERE)
        status = "ok" if r.returncode == 0 else \
            f"FAILED rc={r.returncode} (skipped)"
        print(f"=== {name}: {status} · elapsed "
              f"{(time.time() - t0) / 60:.0f} min total ===", flush=True)
    print(f"\nQUEUE COMPLETE in {(time.time() - t0) / 60:.0f} min — "
          "see experiments/SUMMARY.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
