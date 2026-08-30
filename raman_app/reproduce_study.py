"""
reproduce_study.py — headless one-shot reproduction of the whole study.

Runs the complete pipeline on the clinical dataset (or synthetic demo
data) with a fixed seed and writes every artifact into an output folder:

    * model bundle (.joblib, Platt-calibrated when binary)
    * model comparison + clinical summary (summary.txt)
    * pooled confusion matrix / ROC / calibration figures (png)
    * run metadata (run_meta.json)

Usage:
    python reproduce_study.py                 # D:\\BARC\\Data (or --demo)
    python reproduce_study.py --demo          # synthetic demo data
    python reproduce_study.py --mini          # fast subset (3 models)
    python reproduce_study.py --data PATH --seed 7 --out my_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clinical as clin                      # noqa: E402
import modeling                              # noqa: E402
import plotting                              # noqa: E402
import preprocessing as pp                   # noqa: E402
from clinical_data import load_clinical_dataset  # noqa: E402

DEFAULT_DATA = r"D:\BARC\Data"
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _matrix_from_spectra(spectra):
    """Common-grid matrix exactly like the GUI: the first spectrum's
    axis is the grid; every other spectrum is interpolated onto it."""
    grid = None
    rows = []
    for s in spectra:
        wn = np.asarray(s.wavenumbers, dtype=float)
        it = np.asarray(s.intensities, dtype=float)
        if grid is None:
            grid = wn
            rows.append(it)
        else:
            order = np.argsort(wn)
            rows.append(np.interp(grid, wn[order], it[order]))
    return np.vstack(rows), grid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="One-shot headless reproduction of the study.")
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="clinical dataset root (default: %(default)s)")
    ap.add_argument("--demo", action="store_true",
                    help="use synthetic demo data instead of --data")
    ap.add_argument("--mini", action="store_true",
                    help="fast subset: PCA+LDA / Logistic / Ensemble")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mode", choices=["standard", "paired", "paired-pqn"],
                    default="standard")
    ap.add_argument("--out", default=None, help="output folder")
    args = ap.parse_args(argv)

    out_dir = args.out or os.path.join(APP_DIR, "study_run")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[reproduce] output folder: {out_dir}")

    # ---- data ------------------------------------------------------------
    groups: list[str] | None
    if args.demo:
        import dataset
        import tempfile
        src = dataset.find_default_source_spectrum()
        root = dataset.generate_demo_data(
            src, os.path.join(tempfile.mkdtemp(), "demo"))
        spectra = dataset.load_folder(root)
        labels = [s.label for s in spectra]
        groups = None
        print(f"[reproduce] demo data: {len(spectra)} spectra")
    else:
        if not os.path.isdir(args.data):
            print(f"[reproduce] data folder not found: {args.data}"
                  " (use --demo for synthetic data)")
            return 2
        _report, spectra, labels, groups, _flags = load_clinical_dataset(
            args.data)
        print(f"[reproduce] {len(spectra)} spectra, "
              f"{len(set(labels))} classes, "
              f"{len(set(groups)) if groups else 0} patients")
    if len(set(labels)) < 2:
        print("[reproduce] need at least two classes")
        return 2

    # ---- preprocessing (validated defaults, fixed seed) -------------------
    params = pp.PreprocessParams().validate()
    X_raw, grid = _matrix_from_spectra(spectra)
    X = pp.preprocess_matrix(X_raw, params, wn=grid)
    wn_cropped = np.asarray(grid)[pp.crop_mask(grid, params)]
    keep = [i for i, lab in enumerate(labels) if lab.strip()]
    X, y = X[keep], [labels[i] for i in keep]
    g = [groups[i] for i in keep] if groups else None

    # paired (margin) mode
    if args.mode in ("paired", "paired-pqn"):
        import paired as pmod
        pd_ = pmod.paired_features(X, y, g, wn_cropped, params,
                                   use_pqn=(args.mode == "paired-pqn"))
        X, y, g = pd_.X, pd_.y, pd_.groups
        wn_cropped = pd_.wn
        print(f"[reproduce] {args.mode}: {pd_.X.shape[0]} deviation "
              f"spectra from {pd_.n_patients} patients")

    # ---- model comparison -------------------------------------------------
    model_names = (["PCA + LDA", "PCA + Logistic Regression",
                    "Ensemble (top-3)"] if args.mini else None)
    print("[reproduce] training…")
    results, winner = modeling.evaluate_models(
        X, y, model_names=model_names, k_folds=args.folds,
        seed=args.seed, groups=g, repeats=args.repeats,
        wavenumbers=wn_cropped)

    lines = ["Raman Classifier — reproduced study",
             f"seed {args.seed} · mode {args.mode} · "
             f"{args.folds}-fold patient-grouped CV"
             + (f" x{args.repeats}" if args.repeats > 1 else ""),
             "-" * 60, "Model comparison (macro-F1):"]
    for r in sorted((x for x in results if x.error is None),
                    key=lambda x: -x.macro_f1()):
        lines.append(f"  {r.name:<34} {r.macro_f1():.3f}")
    lines += [f"\nWinner: {winner.name}",
              f"  sensitivity {winner.macro.get('sens', (0,))[0]:.3f}"
              f" · specificity "
              f"{winner.macro.get('spec', (0,))[0]:.3f}"
              f" · macro-F1 {winner.macro_f1():.3f}"]

    # ---- clinical layer (binary winners) ----------------------------------
    extras: dict = {}
    have_oof = (len(winner.classes) == 2
                and winner.oof_proba is not None
                and winner.y_true_encoded is not None)
    if have_oof:
        valid = ~np.isnan(winner.oof_proba[:, 1])
        yv = np.asarray(winner.y_true_encoded[valid])
        pv = winner.oof_proba[valid, 1]
        ab = clin.fit_platt(yv, pv)
        p_cal = clin.apply_platt(pv, ab) if ab else pv
        brier_raw = float(np.mean((pv - yv) ** 2))
        brier_cal = float(np.mean((p_cal - yv) ** 2))
        auc, _se, lo, hi = clin.delong_auc_ci(yv, pv)
        lines.append(f"  AUC {auc:.3f} (DeLong 95% CI {lo:.2f}-{hi:.2f})")
        lines.append(f"  Brier {brier_raw:.3f}"
                     + (f" -> {brier_cal:.3f} after Platt calibration"
                        if ab else ""))
        lo_o, hi_o, s_o, sp_o = clin.operating_points(yv, p_cal)
        if lo_o is not None and hi_o is not None:
            lines.append(f"  rule-out p<={lo_o:.2f} (sens {s_o:.0%}) · "
                         f"rule-in p>={hi_o:.2f} (spec {sp_o:.0%})")
            extras["op_points"] = (lo_o, hi_o)
        if ab:
            extras["calibrator"] = ab
    lines += ["", "Literature benchmark (pooled sens/spec):",
              "  Han 2022 meta              0.89 / 0.84",
              "  2025 OSCC meta             0.89 / 0.91",
              "  Purohit 2026 review        0.90 / 0.89",
              "  (spectrum-level splits — this study's patient-grouped "
              "numbers are the stricter standard)"]
    # Friedman + Nemenyi across models
    try:
        import study_stats as sstats
        scores = {r.name: r.fold_f1 for r in results
                  if r.error is None and len(r.fold_f1) > 2}
        if len(scores) >= 2:
            fr = sstats.friedman_nemenyi(scores)
            if fr.get("ok"):
                lines += ["", f"Friedman: p={fr['p']:.3f} — differences "
                          + ("significant" if fr["significant"]
                             else "NOT significant (models tie)")
                          + f"; best by rank: {fr['best']} "
                            f"(CD {fr['cd']:.2f})"]
    except Exception:
        pass
    # leave-one-patient-out of the winner (grouped data only)
    if g:
        try:
            from sklearn.base import clone
            lo = sstats.lopo_evaluate(X, y, g, clone(winner.pipeline),
                                      {}, list(winner.classes),
                                      seed=args.seed)
            lines.append(f"Leave-one-patient-out ({lo['n_patients']} "
                         f"patients): macro-F1 {lo['f1']:.3f}"
                         + (f", AUC {lo['auc']:.3f}"
                            if np.isfinite(lo["auc"]) else "")
                         + f", mean per-patient accuracy "
                           f"{np.mean(lo['accs']):.3f}")
        except Exception:
            pass
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(out_dir, "summary.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(summary + "\n")

    # ---- bundle + figures + meta ------------------------------------------
    bundle_path = os.path.join(out_dir, "winner.joblib")
    modeling.save_bundle(bundle_path, winner, grid, params,
                         dataset_name=args.data,
                         paired=(args.mode in ("paired", "paired-pqn")),
                         **extras)
    print(f"[reproduce] bundle: {bundle_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if winner.cm is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        plotting.plot_confusion_matrix(ax, winner.cm, winner.classes)
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
    if have_oof:
        valid = ~np.isnan(winner.oof_proba[:, 1])
        yv_i = np.asarray(winner.y_true_encoded[valid])
        pv = winner.oof_proba[valid, 1]
        fpr, tpr, auc = modeling.roc_points(yv_i, pv)
        fig, ax = plt.subplots(figsize=(6, 5))
        plotting.plot_roc(ax, fpr, tpr, auc)
        fig.savefig(os.path.join(out_dir, "roc.png"), dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5))
        cal_bins = clin.calibration_bins(yv_i == 1, pv)
        cal_extra = None
        if extras.get("calibrator"):
            cal_extra = clin.calibration_bins(
                yv_i == 1, clin.apply_platt(pv, extras["calibrator"]))
        plotting.plot_calibration(ax, cal_bins, cal_bins=cal_extra)
        fig.savefig(os.path.join(out_dir, "calibration.png"), dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print("[reproduce] figures: confusion_matrix / roc / calibration")

    meta = {"seed": args.seed, "mode": args.mode, "folds": args.folds,
            "repeats": args.repeats, "winner": winner.name,
            "macro_f1": float(winner.macro_f1()),
            "n_spectra": int(len(y)),
            "n_patients": int(len(set(g))) if g else 0}
    with open(os.path.join(out_dir, "run_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("[reproduce] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
