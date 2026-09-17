"""
prep_chain_prep_search.py -- L3 of the push-to-0.8 program (SAFETY:
pure experiment script; writes experiments/ only).

The §52 preprocessing search scored configs by SINGLE-model scorers.
This script re-scores preprocessing configs by the WINNING CHAIN's
nested validation (validate_arch) -- the protocol the deployment
actually uses.  Arms:

  baseline   d2 winner params (reference = the 0.757 config)
  crop500    crop 500-1800 instead of no-crop
  crop700    crop 700-1800
  deriv1     SG deriv 1 instead of 2
  normvec    vector normalization instead of none
  avg_reps   average deviation rows per (patient, class) -- the ~4x
             fold-variance cut, applied to PAIRED features
  cw_bal     class_weight="balanced" on the RF/ET chain layers
             (baseline features)

Best arm must also survive outer-seed stability (42/43/44) before it
can be called a win.

Usage:  python prep_chain_prep_search.py
Output: experiments/chain_prep_search.json
"""

from __future__ import annotations

import json
import os

import numpy as np

import dataset as ds
import paired as pmod
import preprocessing as pp
import sequential
from clinical_data import load_clinical_dataset
from run_3sse_d2 import WINNER

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "experiments")

CW_LAYERS = ("Random Forest", "Extra Trees")


def _p(**over) -> pp.PreprocessParams:
    d = dict(crop_min=WINNER.crop_min, crop_max=WINNER.crop_max,
             sg_deriv=WINNER.sg_deriv, norm=WINNER.norm,
             sg_window=WINNER.sg_window, sg_poly=WINNER.sg_poly,
             wavelet_name=WINNER.wavelet_name,
             wavelet_level=WINNER.wavelet_level)
    d.update(over)
    return pp.PreprocessParams(**d).validate()


ARMS = {
    "baseline": dict(params=_p()),
    "crop500": dict(params=_p(crop_min=500.0, crop_max=1800.0)),
    "crop700": dict(params=_p(crop_min=700.0, crop_max=1800.0)),
    "deriv1": dict(params=_p(sg_deriv=1)),
    "normvec": dict(params=_p(norm="vector")),
    "avg_reps": dict(params=_p(), avg_reps=True),
    "cw_bal": dict(params=_p(), cw=True),
}


def build_features(params: pp.PreprocessParams, avg_reps: bool = False):
    cd = load_clinical_dataset(sequential.DEFAULT_DATA)
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)
    labels = [s.label for s in cd.spectra]
    flagged = cd.flagged or [False] * len(labels)
    keep = [i for i, lab in enumerate(labels)
            if lab.strip() and (params.despike or not flagged[i])]
    y = [labels[i].strip() for i in keep]
    g = [cd.groups[i] for i in keep]
    pd_ = pmod.paired_features(X_raw[keep], y, g, grid, params)
    X = np.asarray(pd_.X, dtype=np.float32)
    yy, gg = list(pd_.y), list(pd_.groups)
    if avg_reps:
        agg: dict[tuple[str, str], list[int]] = {}
        for i in range(len(yy)):
            agg.setdefault((gg[i], yy[i]), []).append(i)
        X = np.vstack([X[ix].mean(axis=0) for ix in agg.values()])
        yy = [k[1] for k in agg]
        gg = [k[0] for k in agg]
        print(f"[prep] averaged replicates: {len(agg)} rows "
              f"(from {len(pd_.y)})", flush=True)
    return X, yy, gg, pd_.wn


def balanced_factories(wn) -> dict:
    facs = dict(sequential._factories(wn))

    def make_balanced(name):
        spec = next(s for s in __import__("modeling").model_specs()
                    if s["name"] == name)
        est = __import__("modeling").clone(spec["estimator"])
        est.set_params(class_weight="balanced")
        return lambda: __import__("modeling").as_env_device(
            __import__("modeling").clone(est))

    for name in CW_LAYERS:
        facs[name] = make_balanced(name)
    return facs


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]
    print(f"[prep] arch {' -> '.join(arch)}", flush=True)

    results, cache = {}, {}
    for arm, cfg in ARMS.items():
        key = (cfg["params"].__dict__ and
               json.dumps({k: cfg["params"].__dict__[k]
                           for k in sorted(cfg["params"].__dict__)},
                          default=str) + str(cfg.get("avg_reps", False)))
        if key not in cache:
            cache[key] = build_features(cfg["params"],
                                        cfg.get("avg_reps", False))
        X, y, groups, wn = cache[key]
        facs = (balanced_factories(wn) if cfg.get("cw")
                else sequential._factories(wn))
        m = sequential.validate_arch(arch, facs, X, y, groups,
                                     seed=42, k_outer=5, wn=wn)
        results[arm] = {k: (float(m[k]) if isinstance(m.get(k),
                                                      (int, float))
                            else None)
                        for k in ("f1", "sens", "spec", "auc",
                                  "pat_f1")}
        results[arm]["n_rows"] = int(X.shape[0])
        print(f"[prep] {arm:<9}: F1 {results[arm]['f1']:.3f} · "
              f"sens {results[arm]['sens']:.3f} · "
              f"spec {results[arm]['spec']:.3f} "
              f"(n={X.shape[0]})", flush=True)

    best = max((v["f1"], k) for k, v in results.items()
               if k != "baseline")
    base = results["baseline"]["f1"]
    results["_best_arm"] = best[1]
    results["_delta_vs_baseline"] = best[0] - base

    # seed stability for the best arm (42/43/44) before calling it
    cfg = ARMS[best[1]]
    X, y, groups, wn = build_features(cfg["params"],
                                      cfg.get("avg_reps", False))
    facs = (balanced_factories(wn) if cfg.get("cw")
            else sequential._factories(wn))
    f1s = []
    for seed in (42, 43, 44):
        m = sequential.validate_arch(arch, facs, X, y, groups,
                                     seed=seed, k_outer=5, wn=wn)
        f1s.append(float(m["f1"]))
    results["_best_seed_stability"] = {
        str(s): f for s, f in zip((42, 43, 44), f1s, strict=True)}
    results["_best_mean_f1"] = float(np.mean(f1s))
    print(f"[prep] best arm '{best[1]}' stability: "
          f"{[f'{f:.3f}' for f in f1s]} · mean {np.mean(f1s):.3f}",
          flush=True)
    print(f"[prep] baseline stability reference: F1 {base:.3f} "
          f"(d2 recorded 0.734 +/- 0.020 across deploy seeds)",
          flush=True)

    path = os.path.join(OUT_DIR, "chain_prep_search.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[prep] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
