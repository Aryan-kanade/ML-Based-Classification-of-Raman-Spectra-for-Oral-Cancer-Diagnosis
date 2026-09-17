"""
eval_3sse_gui_adj.py -- adjudicate the GUI's fresh 3SSE winner
(XGBoost -> Extra Trees -> Random Forest, F1 0.785, study_run_3sse/):
3 outer seeds + exact McNemar vs the d2-baseline OOF.  Same protocol
as eval_pretrain_adj.py.

Usage:  python eval_3sse_gui_adj.py
Output: experiments/gui3sse_adjudicated.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import binomtest

import exp_common
import pat_common
import sequential


def main() -> int:
    _arch, X, y, g, wn, _meta = exp_common.load_data()
    gui_arch = ["XGBoost", "Extra Trees", "Random Forest"]
    factories = sequential._factories(wn)
    f1s = []
    oof42 = None
    for seed in (42, 43, 44):
        m = sequential.validate_arch(gui_arch, factories, X, y, g,
                                     seed=seed, k_outer=5, wn=wn)
        f1s.append(exp_common.oof_f1(m))
        if seed == 42:
            oof42 = (np.asarray(m["y_true"]),
                     np.asarray(m["oof_proba"], dtype=np.float32))
        print(f"[adj] seed {seed}: F1 {f1s[-1]:.3f}", flush=True)

    ye_b, proba_b, _g = pat_common.winner_oof()
    yt, oof = oof42
    keep = ~np.isnan(oof).any(axis=1)
    pred_n = oof[keep].argmax(axis=1)
    pred_b = proba_b[keep].argmax(axis=1)
    yv = yt[keep].astype(int)
    b_ = int(((pred_n == yv) & (pred_b != yv)).sum())
    c_ = int(((pred_n != yv) & (pred_b == yv)).sum())
    p = float(binomtest(b_, b_ + c_, 0.5).pvalue) if b_ + c_ else 1.0

    out = {"arch": gui_arch, "f1s": f1s,
           "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
           "mcnemar": {"b": b_, "c": c_, "p": p},
           "verdict": (f"GUI 3SSE chain {np.mean(f1s):.3f}"
                       f"±{np.std(f1s):.3f} · McNemar b={b_} c={c_} "
                       f"p={p:.4f} vs 0.757 d2 winner")}
    print(f"[adj] {out['verdict']}", flush=True)
    path = os.path.join(exp_common.OUT_DIR, "gui3sse_adjudicated.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    import method_bank
    method_bank.add({"id": "MG20",
                     "mechanism": "GUI 3SSE XGBoost->ET->RF (adjudicated)",
                     "family": "chain",
                     "f1_mean": float(np.mean(f1s)),
                     "f1_std": float(np.std(f1s)),
                     "pat_f1": None, "seeds": 3,
                     "note": out["verdict"]})
    print(f"[adj] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
