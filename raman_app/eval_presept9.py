"""
eval_presept9.py -- faithful reproduction of the PRE-2026-09-09 study
configuration (Brain.md §13, the 2026-08-30 era numbers), evaluated
under TODAY'S honest patient-grouped protocol:

  preprocessing: crop 400-1800, deriv 0, vector norm, sym8 L4, ALS
  baseline, despike OFF (spike-flagged excluded — §13 hygiene)
  models: paired Extra Trees single (the 0.702 winner) + the old 3SSE
  chain PCA+GNB -> PCA+SVM -> Extra Trees (the 0.725 winner)

Usage:  python eval_presept9.py
Output: experiments/presept9.json + console old-vs-new table
"""

from __future__ import annotations

import json
import os

import preprocessing as pp
import sequential
import dataset as ds
from clinical_data import find_data_root, load_clinical_dataset
import paired as pmod

import exp_common

OLD = pp.PreprocessParams(
    crop_min=400.0, crop_max=1800.0, sg_deriv=0, norm="vector",
    wavelet_name="sym8", wavelet_level=4, baseline_method="als",
).validate()

OLD_CHAIN = ["PCA + Gaussian Naive Bayes", "PCA + SVM (RBF)",
             "Extra Trees"]


def prepare(mode: str):
    cd = load_clinical_dataset(find_data_root())
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)
    labels = [s.label for s in cd.spectra]
    flagged = cd.flagged or [False] * len(labels)
    keep = [i for i, lab in enumerate(labels)
            if lab.strip() and not flagged[i]]
    y = [labels[i].strip() for i in keep]
    g = [cd.groups[i] for i in keep]
    if mode == "paired":
        pd_ = pmod.paired_features(X_raw[keep], y, g, grid, OLD)
        return pd_.X, pd_.y, pd_.groups, pd_.wn
    X = pp.preprocess_matrix(X_raw[keep], OLD, grid)
    return X, y, g, grid


def main() -> int:
    results = {}
    for mode in ("paired", "standard"):
        X, y, g, wn = prepare(mode)
        facs = sequential._factories(wn)
        m = sequential.validate_arch(["Extra Trees"], facs, X, y, g,
                                     seed=42, k_outer=5, wn=wn)
        results[f"{mode}_extra_trees"] = {
            "f1": exp_common.oof_f1(m), "auc": m.get("auc"),
            "sens": m.get("sens"), "spec": m.get("spec")}
        print(f"[pre9] {mode} Extra Trees: F1 {exp_common.oof_f1(m):.3f}"
              f" · AUC {m.get('auc')}", flush=True)
        if mode == "paired":
            m2 = sequential.validate_arch(OLD_CHAIN, facs, X, y, g,
                                          seed=42, k_outer=5, wn=wn)
            results["paired_old_3sse_chain"] = {
                "f1": exp_common.oof_f1(m2), "auc": m2.get("auc"),
                "sens": m2.get("sens"), "spec": m2.get("spec")}
            print(f"[pre9] paired old chain: F1 "
                  f"{exp_common.oof_f1(m2):.3f} · AUC {m2.get('auc')}",
                  flush=True)
    results["old_record"] = {          # Brain.md §13 (2026-08-30)
        "standard_peak_bands_rf_f1": 0.594, "paired_et_f1": 0.702,
        "paired_et_auc": 0.788, "chain_f1": 0.725, "chain_auc": 0.796,
        "rule_out_sens": 0.91, "rule_in_spec": 0.91}
    path = os.path.join(exp_common.OUT_DIR, "presept9.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[pre9] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
