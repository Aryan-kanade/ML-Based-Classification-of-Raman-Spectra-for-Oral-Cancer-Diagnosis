"""
eval_label_errors.py -- L11 (cheap, runs first; SAFETY: reads d2
artifacts, writes experiments/label_errors.txt only).

modeling.label_error_report on the best chain's OOF probabilities —
flags spectra whose GIVEN class got a near-zero probability while
another class dominates.  Historically the largest HONEST lever on
small clinical datasets: removing confirmed mislabels raises every
downstream number legitimately.

Rows are mapped back to the original spectrum name via the keep-order
of _prepare_winner (load_clinical_dataset order minus excluded).
"""

from __future__ import annotations

import json
import os

import numpy as np

import modeling
import sequential
from clinical_data import load_clinical_dataset
from exp_common import HERE, OUT_DIR, record, stamp
from run_3sse_d2 import WINNER


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, "study_run_3sse_d2", "validated.json"),
              encoding="utf-8") as fh:
        v = json.load(fh)
    entries = v.get("3") or v.get(3) or []
    if not entries:
        print("[labels] no level-3 entries", flush=True)
        return 1
    # best validated triple by f1
    best = max(entries, key=lambda e: e["metrics"]["f1"])
    oof = np.asarray(best["metrics"]["oof_proba"])
    y_enc = np.asarray(best["metrics"]["y_true"])

    # rebuild the keep order to name rows (same recipe as d2's
    # _prepare_winner: load order minus unlabeled/flagged)
    cd = load_clinical_dataset(sequential.DEFAULT_DATA)
    labels = [s.label for s in cd.spectra]
    flagged = cd.flagged or [False] * len(labels)
    keep = [i for i, lab in enumerate(labels)
            if lab.strip() and (WINNER.despike or not flagged[i])]
    names = [cd.spectra[i].name for i in keep]
    classes = sorted(set(labels[i].strip() for i in keep))
    if len(names) != len(y_enc):
        names = names[-len(y_enc):]     # paired_features may drop rows

    rows = modeling.label_error_report(oof, y_enc)
    lines = [f"label-error review — {stamp()}",
             f"arch: {' -> '.join(best['arch'])}",
             f"rows: {len(y_enc)} · flagged: {len(rows)}", ""]
    for r in rows[:40]:
        nm = names[r["i"]] if r["i"] < len(names) else f"row{r['i']}"
        lines.append(
            f"{r['tier'].upper():>6}  {nm:<28} given={r['given']} "
            f"p_self={r['p_self']:.3f}  looks like "
            f"{r['other']} p={r['p_other']:.3f}")
    out = os.path.join(OUT_DIR, "label_errors.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    n_likely = sum(1 for r in rows if r["tier"] == "likely")
    print(f"[labels] {len(rows)} flagged ({n_likely} likely-mislabeled) "
          f"-> {out}", flush=True)
    # OOF F1 of this chain for the dashboard context
    m = sequential._metrics_from_oof(
        y_enc[~np.isnan(oof).any(axis=1)],
        oof[~np.isnan(oof).any(axis=1)], classes, None)
    record("L11 label-errors", "review-list", float(m.get("f1",
                                                          float("nan"))),
           extra=f"{len(rows)} flagged ({n_likely} likely) — human "
                 "review next; cleaning confirmed mislabels is a "
                 "legitimate lift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
