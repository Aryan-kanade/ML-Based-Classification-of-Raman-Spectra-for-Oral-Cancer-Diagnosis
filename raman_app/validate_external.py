"""
validate_external.py — external-cohort validation of a saved model.

The TRIPOD+AI gap this project disclaims everywhere: no external
cohort.  This script closes the LOOP — load any saved bundle, apply it
to an external folder of labelled spectra, and report metrics plus a
band-area drift check (population stability index) so a shifted cohort
is visible, not silently scored.

Usage:
    python validate_external.py --data EXTERNAL_FOLDER
    python validate_external.py --data EXT --model study_run/winner.joblib
    python validate_external.py --data EXT --internal MyTrainingFolder
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modeling                              # noqa: E402
import preprocessing as pp                   # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(path: str):
    from clinical_data import load_clinical_dataset
    import dataset as ds
    cd = load_clinical_dataset(path)
    spectra = cd.spectra
    labels = [s.label for s in spectra]
    grid = ds.common_grid(spectra)
    X, _ = ds.to_matrix(spectra, grid)
    keep = [i for i, l in enumerate(labels) if l.strip()]
    if cd.flagged:
        keep = [i for i in keep if not cd.flagged[i]]
    return (X[keep], [labels[i].strip() for i in keep],
            grid, [cd.groups[i] for i in keep] if cd.groups else None)


def band_psi(wn_a, X_a, wn_b, X_b) -> float:
    """
    Mean population-stability index over coarse wavenumber bins:
    quantifies distribution drift between the internal training cohort
    and the external one (PSI < 0.1 stable, > 0.25 major shift).
    Bins are restricted to the SHARED axis range of the two cohorts —
    zero-filling out-of-range bins fabricated drift whenever the
    measurement ranges differed (fixed 2026-09-05).
    """
    wa, wb = np.asarray(wn_a, float), np.asarray(wn_b, float)
    lo = max(float(wa.min()), float(wb.min()))
    hi = min(float(wa.max()), float(wb.max()))

    def _edges():
        return [e for e in np.arange(400.0, 2400.0, 100.0)
                if e >= lo and e + 100.0 <= hi]

    def _prof(wn, X):
        wn = np.asarray(wn, float)
        # coarse 100 cm-1 bin means over the SHARED bins, normalized
        prof = []
        for e in _edges():
            m = (wn >= e) & (wn < e + 100.0)
            if m.sum():
                prof.append(np.mean(X[:, m]))
        v = np.abs(np.array(prof))
        return v / max(v.sum(), 1e-12)

    pa, pb = _prof(wa, X_a), _prof(wb, X_b)
    if len(pa) == 0 or len(pb) == 0:
        return float("nan")
    n = min(len(pa), len(pb))
    pa, pb = pa[:n] + 1e-4, pb[:n] + 1e-4
    pa, pb = pa / pa.sum(), pb / pb.sum()
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True,
                    help="external labelled spectra folder")
    ap.add_argument("--model", default=os.path.join(
        APP_DIR, "study_run", "winner.joblib"),
        help="saved bundle (default: %(default)s)")
    ap.add_argument("--internal", default=None,
                    help="internal training folder (for drift PSI)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.data):
        print(f"[external] folder not found: {args.data}")
        return 2
    if not os.path.isfile(args.model):
        print(f"[external] model not found: {args.model}")
        return 2
    bundle = modeling.load_bundle(args.model)
    params = bundle["prep_params"]
    classes = list(bundle["classes"])
    print(f"[external] model: {bundle.get('model_name', '?')} "
          f"(classes {classes})")

    X, y, grid, groups = _load(args.data)
    if len(set(y)) < 2 or not set(y) <= set(classes):
        print(f"[external] external labels {sorted(set(y))} must be a "
              f"subset of the model's {classes}")
        return 2
    # 2026-09-08 release gate: junk spectra (NaN/Inf or no signal) would
    # silently distort every printed metric — the same rule the deploy
    # guards enforce; skip with a reason instead
    keep = []
    for i, row in enumerate(X):
        if (not np.all(np.isfinite(row))
                or int(np.count_nonzero(np.abs(row) > 1e-9))
                < max(3, len(row) // 100)):
            print(f"[external] skipping spectrum {i} "
                  f"({y[i]}): NaN/Inf or no usable signal")
            continue
        keep.append(i)
    if len(keep) < len(X):
        if len(keep) < 2 * len(set(y)):
            print("[external] too few valid spectra remain — refusing")
            return 2
        X = X[keep]
        y = [y[i] for i in keep]
        if groups:
            groups = [groups[i] for i in keep]
    print(f"[external] {len(y)} spectra / "
          f"{len(set(groups)) if groups else 0} patients")

    wn_ext = np.asarray(grid, float)
    # REGRAIN onto the model's training axis (2026-09-05): every GUI
    # predict path interpolates onto bundle["wavenumbers"]; feeding the
    # external cohort's own grid crashed on different point counts and
    # silently misaligned predictions when the counts coincided.
    wn_b = np.asarray(bundle["wavenumbers"], float)
    if wn_b[0] > wn_b[-1]:                      # tolerate descending axes
        wn_b = wn_b[::-1]
    lo_b, hi_b = float(wn_b.min()), float(wn_b.max())
    if wn_ext[0] > lo_b + 1.0 or wn_ext[-1] < hi_b - 1.0:
        print(f"[external] external axis {wn_ext[0]:.1f}-"
              f"{wn_ext[-1]:.1f} cm-1 does not cover the model grid "
              f"{lo_b:.1f}-{hi_b:.1f} — refusing to fabricate the "
              f"missing region")
        return 2
    if not np.allclose(wn_ext, wn_b):
        Xr = np.stack([np.interp(wn_b, wn_ext, row) for row in X])
    else:
        Xr = X
    Xp = pp.preprocess_matrix(Xr, params, wn=wn_b)
    pipe = bundle["pipeline"]
    proba = pipe.predict_proba(Xp)
    pred = np.array(classes)[np.argmax(proba, axis=1)]
    y_arr = np.array(y)
    acc = float((pred == y_arr).mean())
    from sklearn.metrics import f1_score, roc_auc_score
    f1 = f1_score(y_arr, pred, average="macro")
    lines = [f"[external] accuracy {acc:.3f} · macro-F1 {f1:.3f}"]
    if len(classes) == 2:
        pos = classes[1]
        auc = roc_auc_score((y_arr == pos).astype(int), proba[:, 1])
        lines.append(f"[external] ROC AUC {auc:.3f} (positive = {pos})")
        n_pos = int((y_arr == pos).sum())
        n_neg = len(y_arr) - n_pos
        import clinical as clin
        pw = clin.auc_power(n_pos, n_neg, auc=auc)
        lines.append(f"[external] AUC power: detectable at 80% = "
                     f"{pw['detectable_auc_80pct']:.3f}")
    print("\n".join(lines))

    if args.internal and os.path.isdir(args.internal):
        try:
            Xi, _yi, wi_i, _g = _load(args.internal)
            # compare PREPROCESSED profiles (raw intensities let baseline
            # differences dominate the "band drift" verdict)
            pi = pp.preprocess_matrix(Xi, params, wn=wi_i)
            psi = band_psi(wi_i, pi, wn_b, Xp)
            verdict = ("stable" if psi < 0.1 else
                       "moderate shift" if psi < 0.25 else "MAJOR SHIFT")
            print(f"[external] band-profile PSI vs internal "
                  f"(preprocessed, shared range): {psi:.3f} — {verdict}")
        except Exception as e:
            print(f"[external] drift check skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
