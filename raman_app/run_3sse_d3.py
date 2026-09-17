"""
run_3sse_d3.py -- FINAL full 3SSE search on the RECOVERED tree
(2026-09-17): identical protocol to run_3sse_d2 (proven 18-model
subset, §52 winner paired preprocessing, k=5, grouped) but on the
restored 317-spectrum / 72-subject Data, with a wider nested
validation (--top 30) and its own output folder so the precious d2
artifacts are never overwritten.

This is the last exhaustive honest lever: if a better chain than
0.757 exists in this search space, this finds it; its winner then
gets 3-seed + McNemar adjudication before any claim.

Usage:  python run_3sse_d3.py [--resume]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("RAMAN_DEVICE", "cpu")   # 3SSE loky children: CPU

import modeling
import sequential
from run_3sse_d2 import EXCLUDE, _prepare_winner


def main(argv=None) -> int:
    names = [s["name"] for s in modeling.model_specs()
             if s["name"] not in EXCLUDE]
    print(f"[3sse-d3] models ({len(names)}): {', '.join(names)}",
          flush=True)
    sequential.prepare_dataset = _prepare_winner
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "study_run_3sse_d3")
    args = ["--mode", "paired", "--k", "5", "--top",
            "30", "--prune-pairs", "50", "--out", out_dir,
            "--models", ",".join(names)]
    args += [a for a in (argv or []) if a.startswith("--resume")]
    return sequential.main(args)


if __name__ == "__main__":
    sys.exit(main())
