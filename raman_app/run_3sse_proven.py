"""
run_3sse_proven.py -- second-draw 3SSE search on the §52 winner
preprocessing (paired): the ⭐ Proven subset (= run_3sse_d2's EXCLUDE
set), but a WIDER exploration than d2 -- seed 43 (independent fold
draw), top 25 validated per level (d2: 20), prune-pairs 100 (d2: 50).
Output: study_run_3sse_proven/ (d2 and everything else untouched).

Usage:  python run_3sse_proven.py [--resume]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("RAMAN_DEVICE", "cpu")   # 3SSE loky children: CPU

import sequential
from run_3sse_d2 import EXCLUDE, _prepare_winner


def main(argv=None) -> int:
    import modeling
    names = [s["name"] for s in modeling.model_specs()
             if s["name"] not in EXCLUDE]
    print(f"[3sse-proven] models ({len(names)}): {', '.join(names)}",
          flush=True)
    sequential.prepare_dataset = _prepare_winner
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "study_run_3sse_proven")
    args = ["--mode", "paired", "--k", "5", "--seed", "43",
            "--top", "25", "--prune-pairs", "100",
            "--out", out_dir,
            "--models", ",".join(names)]
    args += [a for a in (argv or []) if a.startswith("--resume")]
    return sequential.main(args)


if __name__ == "__main__":
    sys.exit(main())
