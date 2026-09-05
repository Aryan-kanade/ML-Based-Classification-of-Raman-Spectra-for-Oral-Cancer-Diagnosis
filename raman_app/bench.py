"""
bench.py — fixed speed & accuracy scorecard for the Speed&Accuracy
program (2026-09-05).  Same seed, same data, same models every run so
before/after deltas are apples-to-apples.  Every change in the program
records its measured delta here (see Brain.md §13).

Usage:
    python bench.py             # quick: ~2-4 min — baselines + screening
    python bench.py --full      # + nested validation + LOPO + winner
Writes bench/latest.json (and prints the table).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(APP_DIR, "bench")

# fixed benchmark configuration — DO NOT TUNE HERE; changes go through
# the program's A/B gates so old scores stay comparable
BENCH_MODELS = ["Extra Trees", "CatBoost", "PCA + LDA", "Random Forest",
                "PCA + Logistic Regression"]
SEARCH_MODELS = BENCH_MODELS + ["XGBoost", "LightGBM",
                                "Hist Gradient Boosting",
                                "PCA + SVM (RBF)", "PLS-DA"]


def _stage(out: dict, name: str, fn):
    t0 = time.time()
    res = fn()
    out["stages"][name] = {"seconds": round(time.time() - t0, 1)}
    print(f"  {name:28s} {out['stages'][name]['seconds']:7.1f}s")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="fixed speed/accuracy bench")
    ap.add_argument("--full", action="store_true",
                    help="also run nested validation + LOPO + winner")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    import clinical_data as cdata
    import modeling
    import sequential as seq
    from sklearn.metrics import roc_auc_score

    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "seed": args.seed, "full": args.full, "stages": {},
           "accuracy": {}}
    root = cdata.find_data_root()
    if root is None:
        print("no dataset found (RAMAN_DATA_DIR / known spots)")
        return 2
    print(f"bench on {root} · seed {args.seed}"
          + (" · FULL" if args.full else " · quick"))

    X, y, g, wn, grid, params, meta = _stage(
        out, "prepare (load+preprocess+paired)",
        lambda: seq.prepare_dataset(root, "paired"))
    out["data"] = {"n": int(meta["n_spectra"]),
                   "patients": int(meta["n_patients"]),
                   "features": int(X.shape[1])}

    # ---- baseline models (5-fold grouped, the v2-winner set) ----------
    def _baselines():
        return modeling.evaluate_models(
            X, list(y), model_names=BENCH_MODELS, k_folds=5,
            seed=args.seed, groups=g, wavenumbers=wn)
    results, winner = _stage(out, "baselines (5 models, 5-fold)", _baselines)
    rows = sorted(((r.name, r.macro_f1()) for r in results
                   if r.error is None), key=lambda t: -t[1])
    out["accuracy"]["baselines"] = {n: round(f, 4) for n, f in rows}
    out["accuracy"]["winner_single"] = rows[0][0]
    auc = None
    if winner.oof_proba is not None and len(winner.classes) == 2:
        valid = ~np.isnan(winner.oof_proba[:, 1])
        auc = float(roc_auc_score(winner.y_true_encoded[valid],
                                  winner.oof_proba[valid, 1]))
    out["accuracy"]["winner_single_f1"] = round(rows[0][1], 4)
    out["accuracy"]["winner_single_auc"] = round(auc, 4) if auc else None
    # B4b: PATIENT-LEVEL headline — per-patient mean P(positive) verdict
    # (the clinically meaningful unit; literature: Farnesi 2025 13/13)
    if winner.oof_proba is not None and g is not None:
        from sklearn.metrics import f1_score as _f1
        wclasses = list(winner.classes)
        oof = winner.oof_proba
        preds, trues = [], []
        for pat in sorted(set(g)):
            idx = [i for i, gg in enumerate(g) if gg == pat]
            p_pos = float(np.nanmean(oof[idx, 1]))
            preds.append(wclasses[1] if p_pos >= 0.5 else wclasses[0])
            trues.append(y[idx[0]])
        out["accuracy"]["patient_level_f1"] = round(
            float(_f1(trues, preds, average="macro")), 4)

    # ---- 3SSE screening with the speed engine (beam, 2-fold ladder) ---
    def _screen():
        return seq.search(X, list(y), groups=g, wavenumbers=wn,
                          k=5, seed=args.seed, top=10,
                          model_names=SEARCH_MODELS)
    board = _stage(out, f"3SSE screening ({len(SEARCH_MODELS)} models, "
                        f"beam 50)", _screen)
    top5 = sorted((e for e in board["singles"] + board["pairs"]
                   + board["triples"] if "metrics" in e),
                  key=lambda e: -e["metrics"]["f1"])[:5]
    out["accuracy"]["3sse_top5"] = [
        {"arch": " → ".join(e["arch"]),
         "f1": round(e["metrics"]["f1"], 4)} for e in top5]
    out["3sse"] = {"screen_folds": board["screen_folds"],
                   "evaluated": {"singles": len(board["singles"]),
                                 "pairs": len(board["pairs"]),
                                 "triples": len(board["triples"])},
                   "pruned_early": board["pruned"]}

    if args.full:
        # ---- nested validation of the screening top -------------------
        def _validate():
            return seq.validate_top(board, X, list(y), g, wn, top=10,
                                    seed=args.seed,
                                    model_names=SEARCH_MODELS)
        validated = _stage(out, "3SSE nested validation (top 10/level)",
                           _validate)
        win = seq.pick_overall(validated)
        if win:
            out["accuracy"]["3sse_winner"] = {
                "arch": " → ".join(win["arch"]),
                "f1": round(win["metrics"]["f1"], 4),
                "sens": round(win["metrics"]["sens"], 4),
                "spec": round(win["metrics"]["spec"], 4)}

        # ---- LOPO of the best single (patient-level headline) ----------
        import study_stats as sstats
        from sklearn.base import clone
        def _lopo():
            return sstats.lopo_evaluate(
                X, list(y), g, clone(winner.pipeline), {},
                list(winner.classes))
        lo = _stage(out, "LOPO (best single)", _lopo)
        out["accuracy"]["lopo_f1"] = round(lo["f1"], 4)
        out["accuracy"]["lopo_auc"] = round(lo["auc"], 4)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "latest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n=== bench summary ===")
    print(json.dumps(out["accuracy"], indent=2))
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
