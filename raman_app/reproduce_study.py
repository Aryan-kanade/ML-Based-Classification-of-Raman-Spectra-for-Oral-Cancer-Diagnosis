"""
reproduce_study.py — headless one-shot reproduction of the whole study.

Runs the complete pipeline on the clinical dataset with a fixed seed
and writes every artifact into an output folder:

    * model bundle (.joblib, Platt-calibrated when binary)
    * model comparison + clinical summary (summary.txt)
    * pooled confusion matrix / ROC / calibration figures (png)
    * run metadata (run_meta.json)

Usage:
    python reproduce_study.py                 # auto-detected dataset root
    python reproduce_study.py --mini          # fast subset (3 models)
    python reproduce_study.py --data PATH --seed 7 --out my_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clinical as clin                      # noqa: E402
import dataset as ds                         # noqa: E402
import modeling                              # noqa: E402
import plotting                              # noqa: E402
import preprocessing as pp                   # noqa: E402
import study_stats as sstats                 # noqa: E402
from clinical_data import (find_data_root, load_clinical_dataset,  # noqa: E402
                           write_report)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# auto-detected per device — no hardcoded machine path to edit when the
# project moves to another computer (see clinical_data.find_data_root)
DEFAULT_DATA = find_data_root() or os.path.join(APP_DIR, "..", "Data")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="One-shot headless reproduction of the study.")
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="clinical dataset root (default: %(default)s)")
    ap.add_argument("--mini", action="store_true",
                    help="fast subset: PCA+LDA / Logistic / Ensemble")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mode", choices=["standard", "paired", "paired-pqn"],
                    default="standard")
    ap.add_argument("--crop-min", type=float, default=None,
                    help="override crop_min (default: pipeline default)")
    ap.add_argument("--crop-max", type=float, default=None,
                    help="override crop_max (default: pipeline default)")
    ap.add_argument("--despike", action="store_true",
                    help="despike (also keeps spike-flagged spectra)")
    ap.add_argument("--baseline", choices=["als", "arpls"], default=None)
    ap.add_argument("--no-locked", action="store_true",
                    help="skip the one-shot locked final exam")
    ap.add_argument("--out", default=None, help="output folder")
    args = ap.parse_args(argv)

    out_dir = args.out or os.path.join(APP_DIR, "study_run")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[reproduce] output folder: {out_dir}")

    # ---- data ------------------------------------------------------------
    groups: list[str] | None
    flagged: list[bool] | None = None
    if not os.path.isdir(args.data):
        print(f"[reproduce] data folder not found: {args.data} "
              "(pass --data PATH or set RAMAN_DATA_DIR)")
        return 2
    cd = load_clinical_dataset(args.data)
    spectra = cd.spectra
    labels = [s.label for s in spectra]
    groups = cd.groups or None
    flagged = cd.flagged or None
    write_report(os.path.join(out_dir, "data_report.txt"), cd.report)
    print(f"[reproduce] {len(spectra)} spectra, "
          f"{len(set(labels))} classes, "
          f"{len(set(groups)) if groups else 0} patients")
    if len(set(labels)) < 2:
        print("[reproduce] need at least two classes")
        return 2

    # ---- preprocessing (validated defaults, fixed seed) -------------------
    # Mirrors the GUI path exactly: common_grid/to_matrix, then either
    # paired_features (which preprocesses internally) or preprocess_matrix.
    params = pp.PreprocessParams().validate()
    overrides = {"crop_min": args.crop_min, "crop_max": args.crop_max,
                 "despike": True if args.despike else None,
                 "baseline_method": args.baseline}
    params = replace(params, **{k: v for k, v in overrides.items()
                                if v is not None}).validate()
    grid = ds.common_grid(spectra)
    X_raw, _ = ds.to_matrix(spectra, grid)
    keep = [i for i, lab in enumerate(labels) if lab.strip()]
    if flagged is not None and not params.despike:
        # GUI parity: spiked spectra are excluded unless despiking (which
        # recovers them) is on
        n_flagged = sum(1 for i in keep if flagged[i])
        if n_flagged:
            print(f"[reproduce] excluding {n_flagged} quality-flagged "
                  "spiked spectra (despiking off)")
            keep = [i for i in keep if not flagged[i]]
    y = [labels[i].strip() for i in keep]
    g = [groups[i] for i in keep] if groups else None

    # paired (margin) mode: paired_features takes RAW intensities and the
    # full grid; it preprocesses + crops internally (single preprocessing)
    if args.mode in ("paired", "paired-pqn"):
        import paired as pmod
        pd_ = pmod.paired_features(X_raw[keep], y, g, grid, params,
                                   use_pqn=(args.mode == "paired-pqn"))
        X, y, g = pd_.X, pd_.y, pd_.groups
        wn_cropped = pd_.wn
        print(f"[reproduce] {args.mode}: {pd_.X.shape[0]} deviation "
              f"spectra from {pd_.n_patients} patients")
    else:
        X = pp.preprocess_matrix(X_raw[keep], params, wn=grid)
        wn_cropped = np.asarray(grid)[pp.crop_mask(grid, params)]

    # ---- model comparison -------------------------------------------------
    model_names = (["PCA + LDA", "PCA + Logistic Regression",
                    "Ensemble (top-3)"] if args.mini else None)
    print("[reproduce] training…")
    try:
        _v = modeling.verify_gpu_runtime()
        print(f"[reproduce] device: mode={_v['mode']} · "
              f"CNN={_v['cnn']} · XGBoost={_v['xgboost']} · "
              f"CatBoost={_v['catboost']} · LightGBM={_v['lightgbm']} · "
              "sklearn=cpu")
    except RuntimeError as exc:
        print(f"[reproduce] DEVICE ERROR: {exc}")
        return 2
    print(f"[reproduce] {modeling.device_report()}")
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
    # B6: patient-level (mean-probability) reading — the operating
    # level the field reports (Jeng 2019; Farnesi 2025)
    if (getattr(winner, "groups", None) is not None
            and winner.oof_proba is not None
            and winner.y_true_encoded is not None):
        try:
            pm = sstats.patient_level_metrics(winner.y_true_encoded,
                                              winner.groups,
                                              winner.oof_proba)
            if pm is not None:
                lines.append(
                    f"  PATIENT-level (mean P, n={pm['n_patients']}): "
                    f"macro-F1 {pm['f1']:.3f}"
                    + (f" · AUC {pm['auc']:.3f}" if "auc" in pm else ""))
                print(f"[reproduce] patient-level: F1 {pm['f1']:.3f}"
                      + (f", AUC {pm['auc']:.3f}" if "auc" in pm else ""))
        except Exception as exc:
            print(f"[reproduce] patient-level metrics skipped: {exc}")

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
        pred_oof = np.argmax(winner.oof_proba[valid], axis=1)
        ci_lo, ci_hi = modeling.bootstrap_ci(
            yv, pred_oof,
            groups=(np.asarray(winner.groups)[valid]
                    if getattr(winner, "groups", None) else None))
        lines.append(f"  macro-F1 95% CI [{ci_lo:.2f}-{ci_hi:.2f}]"
                     " (patient-level bootstrap)")
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

    # ---- locked one-shot final exam (TRIPOD-style) ------------------------
    # split PATIENTS 70/15/15, refit the winner on the training patients
    # only, evaluate ONE-SHOT on the untouched test patients — the honest
    # headline number next to the (optimistic) pooled-CV numbers.
    if (g is not None and len(set(g)) >= 7 and winner.pipeline is not None
            and not args.no_locked):
        try:
            from sklearn.base import clone as _clone
            from sklearn.metrics import (confusion_matrix as _cm,
                                         f1_score as _f1)
            from clinical_data import patient_split
            tr_idx, _va_idx, te_idx = patient_split(
                list(g), list(y), seed=args.seed)
            te_pats = {g[i] for i in te_idx}
            rows_tr = [i for i in range(len(y)) if g[i] not in te_pats]
            rows_te = [i for i in range(len(y)) if g[i] in te_pats]
            est = _clone(winner.pipeline).fit(
                X[rows_tr], [y[i] for i in rows_tr])
            pred = est.predict(X[rows_te])
            y_te = [y[i] for i in rows_te]
            cm_ = _cm(y_te, pred, labels=list(winner.classes))
            f1_ = float(_f1(y_te, pred, average="macro"))
            lines += ["", "LOCKED FINAL EXAM (one-shot, unseen patients):",
                      f"  {len(rows_te)} spectra / {len(te_pats)} unseen "
                      f"patients — macro-F1 {f1_:.3f}"]
            if len(winner.classes) == 2:
                tp = int(cm_[1, 1])
                fn_ = int(cm_[1].sum()) - tp
                tn = int(cm_[0, 0])
                fp_ = int(cm_[0].sum()) - tn
                proba_ = est.predict_proba(X[rows_te])[:, 1]
                yv_ = (np.asarray(y_te) == winner.classes[1]).astype(int)
                a_, _se_, lo_, hi_ = clin.delong_auc_ci(yv_, proba_)
                lines.append(
                    f"  sens {tp / max(tp + fn_, 1):.3f} · "
                    f"spec {tn / max(tn + fp_, 1):.3f} · "
                    f"AUC {a_:.3f} (DeLong 95% CI {lo_:.2f}-{hi_:.2f})")
            lines.append("  (noisy at this n — run once per configuration; "
                         "re-running until it looks good defeats it)")
        except Exception as e:
            lines.append(f"\nLocked exam skipped: {e}")
    lines += ["", "Literature benchmark (pooled sens/spec):",
              "  Han 2022 meta              0.89 / 0.84",
              "  2025 OSCC meta             0.89 / 0.91",
              "  Purohit 2026 review        0.90 / 0.89",
              "  (spectrum-level splits — this study's patient-grouped "
              "numbers are the stricter standard)"]
    # Friedman + Nemenyi across models. Blocks must be independent: with
    # repeats>1 the per-fold scores are averaged across repeats first
    # (pooling repeats as blocks would violate independence).
    def _per_fold_blocks(f1s: list[float]) -> list[float]:
        if args.repeats > 1 and len(f1s) % args.repeats == 0:
            k = len(f1s) // args.repeats
            return np.asarray(f1s, dtype=float).reshape(
                args.repeats, k).mean(axis=0).tolist()
        return list(f1s)

    try:
        scores = {r.name: _per_fold_blocks(r.fold_f1) for r in results
                  if r.error is None and len(r.fold_f1) > 2}
        if len(scores) >= 2:
            fr = sstats.friedman_nemenyi(scores)
            if fr.get("ok"):
                lines += ["", f"Friedman: p={fr['p']:.3f} — differences "
                          + ("significant" if fr["significant"]
                             else "NOT significant (models tie)")
                          + f"; best by rank: {fr['best']} "
                            f"(CD {fr['cd']:.2f})"]
    except Exception as exc:
        print(f"[reproduce] Friedman skipped: {exc}")
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
        except Exception as exc:
            print(f"[reproduce] LOPO skipped: {exc}")
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
                         pqn=(args.mode == "paired-pqn"),   # deploy must PQN
                         **extras)
    print(f"[reproduce] bundle: {bundle_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plotting.apply_style()   # house style for the saved PNGs too
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
