"""
prep_deep_search.py -- DEEP combinatorial preprocessing search (2026-09-12).

optimize.py tests ~15 hand-picked configs, one axis at a time.  This
script runs a STAGED FACTORIAL search over EVERY PreprocessParams axis
(~350 configs per mode) on the real clinical data:

  Stage 1 backbone   crop(10) x deriv(3) x norm(5)              = 150
  Stage 2 baseline   method x lambda x p (+niter) on top-2      ~ 52
  Stage 3 denoise    wavelet name/level/rule (37) + SG win/poly
                     (21) + top3 x top3 cross (9)               ~ 67
  Stage 4 switches   despike z / wn_calibrate / detrend + joint ~ 12
  Stage 5 crop fine  min x max around the winner                ~ 25
  Stage 6 validation top-8 + defaults x 5 models x 3 seeds, then
                     nested evaluate_models for the top-2 (honest).

Scoring mirrors optimize.py: 3 fast models (PCA+SVM, RF-300,
PCA+LogReg), patient-grouped StratifiedGroupKFold(k=5), macro-F1, best
model per config.  Preprocessing runs through the same _STAGE_CACHE so
shared prefixes cost nothing; configs are ordered crop-major to keep
cache hits high.  Standard mode scores absolute spectra; paired mode
rebuilds the deviation features per config (paired.paired_features,
exactly what a paired Train run sees).  Spike-flagged spectra are
excluded unless the config despikes (GUI parity).

Selection scores are optimistic (winner's curse); stage 6 multi-seed
and nested numbers are the honest ones.

Usage:
    python prep_deep_search.py                  # both modes, full grids
    python prep_deep_search.py --mode standard
    python prep_deep_search.py --smoke          # tiny grids, minutes

Results: vit_outputs/prep_deep_search_<mode>.json + console report.
The existing preprocess_best.json is never touched (report-only).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import asdict, replace

os.environ.setdefault("RAMAN_DEVICE", "cpu")   # boosters/CNN stay CPU here

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import dataset as ds
import optimize as opt
import paired as pmod
import preprocessing as pp
from clinical_data import find_data_root, load_clinical_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "vit_outputs")

# Selection models = optimize._models() (PCA+SVM / RF / PCA+LogReg);
# validation adds the two registry winners (Extra Trees, PCA+LDA).
VALIDATION_MODELS = None


def _validation_models():
    global VALIDATION_MODELS
    if VALIDATION_MODELS is None:
        VALIDATION_MODELS = list(opt._models()) + [
            ("ExtraTrees", ExtraTreesClassifier(
                n_estimators=300, class_weight="balanced", n_jobs=-1,
                random_state=0)),
            ("PCA+LDA", Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=0)),
                ("lda", LinearDiscriminantAnalysis())])),
        ]
    return VALIDATION_MODELS


# --------------------------------------------------------------------------
# grids
# --------------------------------------------------------------------------

CROPS: list[tuple[float, float]] = [
    (500.0, 2000.0),   # tuned default
    (400.0, 1800.0),   # optimize.py historical best
    (500.0, 1800.0),
    (400.0, 2000.0),
    (600.0, 2000.0),
    (700.0, 1800.0),
    (800.0, 1800.0),
    (400.0, 2300.0),   # wide (carotenoid 1520 + amide I tail)
    (500.0, 1650.0),   # tight fingerprint
    (0.0, 0.0),        # no crop (full 15.5-3862 axis)
]
DERIVS = [0, 1, 2]
NORMS = ["vector", "snv", "area", "minmax", "none"]
BASELINE_LAMBDAS = [1e4, 1e5, 1e6, 1e7]
ALS_PS = [0.001, 0.01, 0.1]
WAVELET_NAMES = ["sym8", "sym4", "db6", "coif5"]
WAVELET_LEVELS = [2, 4, 6]
SG_WINDOWS = [5, 7, 9, 11, 15, 21, 31]
SG_POLYS = [2, 3, 4]
DESPIKE_ZS = [5.0, 7.0, 10.0]

SMOKE = False   # set by --smoke


def _crops():
    return CROPS[:2] if SMOKE else CROPS


def stage1_overrides() -> list[dict]:
    """crop x deriv x norm, crop-major (keeps the stage cache warm)."""
    out = []
    for cmin, cmax in _crops():
        for d in (DERIVS[:2] if SMOKE else DERIVS):
            for n in (NORMS[:2] if SMOKE else NORMS):
                out.append(dict(crop_min=cmin, crop_max=cmax,
                                sg_deriv=d, norm=n))
    return out


def stage2_overrides() -> list[dict]:
    """baseline method / lambda / p (deriv-0 configs only)."""
    out = []
    lams = [1e5, 1e6] if SMOKE else BASELINE_LAMBDAS
    for lam in lams:
        for p in (ALS_PS[1:2] if SMOKE else ALS_PS):
            out.append(dict(baseline_method="als", als_lambda=lam,
                            als_p=p))
    if not SMOKE:
        out.append(dict(als_niter=20))
    for method in (("arpls",) if SMOKE else ("arpls", "iarpls",
                                             "pspline")):
        for lam in lams:
            out.append(dict(baseline_method=method, als_lambda=lam))
    if not SMOKE:
        out.append(dict(baseline_method="snip"))
    return out


def stage3_wavelet_overrides() -> list[dict]:
    out = [dict(wavelet=False)]
    names = WAVELET_NAMES[:1] if SMOKE else WAVELET_NAMES
    levels = WAVELET_LEVELS[1:2] if SMOKE else WAVELET_LEVELS
    for name in names:
        for level in levels:
            out.append(dict(wavelet=True, wavelet_name=name,
                            wavelet_level=level))
            out.append(dict(wavelet=True, wavelet_name=name,
                            wavelet_level=level,
                            wavelet_threshold="bayes"))
            out.append(dict(wavelet=True, wavelet_name=name,
                            wavelet_level=level, wavelet_mode="garrote",
                            wavelet_cycle=1))
    return out


def stage3_sg_overrides() -> list[dict]:
    pairs = [(7, 2), (11, 3)] if SMOKE else list(
        itertools.product(SG_WINDOWS, SG_POLYS))
    return [dict(sg_window=w, sg_poly=p) for w, p in pairs]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def describe(p: pp.PreprocessParams) -> str:
    """Compact one-line label of a FULL param set (ASCII-safe)."""
    crop = (f"crop {p.crop_min:.0f}-{p.crop_max:.0f}"
            if p.crop_max else "no-crop")
    base = (f"{p.baseline_method} l{p.als_lambda:.0e} p{p.als_p:g}"
            f" n{p.als_niter}")
    wav = "off" if not p.wavelet else (
        f"{p.wavelet_name} L{p.wavelet_level} {p.wavelet_threshold}"
        f"/{p.wavelet_mode}" + (f" x{p.wavelet_cycle}"
                                if p.wavelet_cycle else ""))
    extra = ((" despike" if p.despike else "")
             + (" cal1003" if p.wn_calibrate else "")
             + (" detrend" if p.detrend else ""))
    return (f"{crop} | d{p.sg_deriv} | {p.norm} | {base}"
            f" | SG {p.sg_window}/{p.sg_poly} | {wav}{extra}")


class Scorer:
    """Preprocess + score one PreprocessParams under grouped CV.

    Standard mode: preprocess_matrix on kept rows.  Paired mode:
    paired_features rebuilds deviation features per config.  The keep
    set follows GUI parity: spike-flagged spectra enter training only
    when the config despikes.
    """

    def __init__(self, X_raw, grid, labels, groups, flagged, mode,
                 k=5, seed=42):
        self.X_raw = np.asarray(X_raw)
        self.grid = np.asarray(grid)
        self.labels = list(labels)
        self.groups = list(groups) if groups is not None else None
        self.flagged = list(flagged) if flagged is not None else None
        self.mode = mode
        self.k = k
        self.seed = seed
        self._cache: dict[str, tuple] = {}   # key -> (X, y, groups)
        self.n_scored = 0

    # -- feature building ------------------------------------------------
    def keep_idx(self, params: pp.PreprocessParams) -> list[int]:
        return [i for i, lab in enumerate(self.labels)
                if lab.strip()
                and (params.despike or self.flagged is None
                     or not self.flagged[i])]

    def build(self, params: pp.PreprocessParams, use_pqn: bool = False):
        key = json.dumps([asdict(params), use_pqn], sort_keys=True)
        if key in self._cache:
            return self._cache[key]
        keep = self.keep_idx(params)
        y = [self.labels[i].strip() for i in keep]
        g = ([self.groups[i] for i in keep]
             if self.groups is not None else None)
        if self.mode == "paired":
            pd_ = pmod.paired_features(self.X_raw[keep], y, g,
                                       self.grid, params,
                                       use_pqn=use_pqn)
            feat = (pd_.X, pd_.y, pd_.groups, pd_.wn)
        else:
            X = pp.preprocess_matrix(self.X_raw[keep], params,
                                     wn=self.grid)
            wn_c = self.grid[pp.crop_mask(self.grid, params)]
            feat = (X, y, g, wn_c)
        self._cache[key] = feat
        if len(self._cache) > 24:
            self._cache.pop(next(iter(self._cache)))   # ~24 matrices max
        return feat

    # -- selection scoring (optimize.py semantics) ------------------------
    def score(self, params: pp.PreprocessParams, use_pqn: bool = False,
              models=None) -> dict:
        X, y, g, _wn = self.build(params, use_pqn)
        best = None
        for mname, est in (models or opt._models()):
            mean, std = opt._cv_macro_f1(est, X, np.asarray(y), g,
                                         self.k, self.seed)
            if best is None or mean > best["mean_f1"]:
                best = {"model": mname, "mean_f1": mean, "std_f1": std,
                        "n_rows": int(X.shape[0]),
                        "n_features": int(X.shape[1])}
        self.n_scored += 1
        return best

    def score_multi_seed(self, params: pp.PreprocessParams, seeds,
                         use_pqn: bool = False) -> dict:
        """Validation: every model x every seed, per-model mean +/- std."""
        X, y, g, _wn = self.build(params, use_pqn)
        per_model = {}
        for mname, est in _validation_models():
            f1s = [opt._cv_macro_f1(est, X, np.asarray(y), g,
                                    self.k, s)[0] for s in seeds]
            per_model[mname] = [float(np.mean(f1s)), float(np.std(f1s))]
        bm = max(per_model.items(), key=lambda kv: kv[1][0])
        return {"per_model": per_model, "best_model": bm[0],
                "mean_f1": bm[1][0], "std_f1": bm[1][1]}

    def nested_eval(self, params: pp.PreprocessParams,
                    model_names: list[str], k=5, seed=42) -> dict:
        """Honest nested CV (hyperparams tuned inside folds) via
        modeling.evaluate_models — the same path a GUI Train run uses."""
        import modeling
        X, y, g, wn_c = self.build(params)
        results, winner = modeling.evaluate_models(
            X, list(y), model_names=model_names, k_folds=k, seed=seed,
            groups=g, wavenumbers=wn_c)
        rows = {}
        for r in results:
            rows[r.name] = (None if r.error else {
                "f1": r.macro_f1(),
                "sens": r.macro.get("sens", (float("nan"), 0))[0],
                "spec": r.macro.get("spec", (float("nan"), 0))[0],
                "error": r.error})
        return {"per_model": rows,
                "winner": (None if winner is None else winner.name),
                "winner_f1": (None if winner is None
                              else winner.macro_f1())}


# --------------------------------------------------------------------------
# staged search
# --------------------------------------------------------------------------

def run_stage(name: str, base: pp.PreprocessParams,
              overrides: list[dict], scorer: Scorer, out: dict,
              best: dict, log=print) -> list[dict]:
    """Score `overrides` on `base`; update global best + partial JSON."""
    rows, t0 = [], time.time()
    seen: set[str] = set()
    for i, over in enumerate(overrides, start=1):
        try:
            params = replace(base, **over).validate()
        except Exception as exc:
            log(f"  [{i}/{len(overrides)}] INVALID {over}: {exc}")
            continue
        key = json.dumps(asdict(params), sort_keys=True)
        if key in seen:
            continue                      # dedup (validate() collapses)
        seen.add(key)
        try:
            res = scorer.score(params)
        except Exception as exc:
            log(f"  [{i}/{len(overrides)}] FAILED {describe(params)}: {exc}")
            continue
        row = {"params": asdict(params), "over": over,
               "label": describe(params), **res}
        rows.append(row)
        if res["mean_f1"] > best["score"]:
            best.update(params=params, score=res["mean_f1"],
                        model=res["model"], label=row["label"],
                        stage=name)
        log(f"  [{i}/{len(overrides)}] {row['label']}: "
            f"{res['model']} F1 {res['mean_f1']:.3f} +/- {res['std_f1']:.3f}"
            f" ({res['n_rows']}x{res['n_features']})"
            + ("  <== BEST" if best["label"] == row["label"] else ""))
    rows.sort(key=lambda r: r["mean_f1"], reverse=True)
    out["stages"][name] = rows
    out["best"] = {"label": best["label"], "score": best["score"],
                   "model": best["model"],
                   "params": asdict(best["params"]), "stage": best["stage"]}
    _dump(out)
    log(f"  == stage '{name}' done in {time.time() - t0:.0f}s "
        f"({len(rows)} scored) — best so far: {best['label']} "
        f"F1 {best['score']:.3f}")
    return rows


def _dump(out: dict):
    path = os.path.join(OUT_DIR, f"prep_deep_search_{out['mode']}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


def search_mode(X_raw, grid, labels, groups, flagged, mode: str,
                k: int, seed: int, log=print) -> dict:
    pp.clear_stage_cache()
    scorer = Scorer(X_raw, grid, labels, groups, flagged, mode, k=k,
                    seed=seed)
    out = {"mode": mode, "k": k, "seed": seed,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "stages": {}, "best": None, "validation": None,
           "nested": None}
    defaults = pp.PreprocessParams().validate()
    best = {"params": defaults, "score": -1.0, "model": "-",
            "label": describe(defaults), "stage": "defaults"}

    log(f"\n{'=' * 72}\nMODE: {mode.upper()}  "
        f"({len(labels)} spectra, {len(set(groups)) if groups else '-'} "
        f"patients, {k}-fold grouped CV, seed {seed})\n{'=' * 72}")

    # baseline reference point
    res0 = scorer.score(defaults)
    log(f"[defaults] {describe(defaults)}: {res0['model']} "
        f"F1 {res0['mean_f1']:.3f} +/- {res0['std_f1']:.3f}")
    best.update(score=res0["mean_f1"], model=res0["model"])
    out["defaults"] = {"params": asdict(defaults),
                       "label": describe(defaults), **res0}

    # ---- stage 1: backbone crop x deriv x norm -------------------------
    rows1 = run_stage("1-backbone", defaults, stage1_overrides(),
                      scorer, out, best, log)
    base = best["params"]

    # ---- stage 2: baseline (deriv-0 candidates only) --------------------
    cands = [base]
    if base.sg_deriv != 0:
        d0 = [r for r in rows1
              if pp.PreprocessParams(**r["params"]).sg_deriv == 0]
        if d0:
            cands.append(pp.PreprocessParams(**d0[0]["params"]))
    st2_rows = []
    for cand in cands:
        log(f"  -- baseline sweep on: {describe(cand)}")
        st2_rows += run_stage(
            f"2-baseline[{describe(cand)}]", cand, stage2_overrides(),
            scorer, out, best, log)
    base = best["params"]

    # ---- stage 3: denoise + smoothing ------------------------------------
    wav_rows = run_stage("3-wavelet", base, stage3_wavelet_overrides(),
                         scorer, out, best, log)
    sg_rows = run_stage("3-savgol", base, stage3_sg_overrides(),
                        scorer, out, best, log)

    def _top_overs(rows, keys, n=3):
        """Top-n DISTINCT overrides restricted to `keys` fields."""
        tops, seenk = [], set()
        for r in rows:
            k2 = tuple(r["over"].get(k2) for k2 in keys)
            if k2 in seenk:
                continue
            seenk.add(k2)
            tops.append({k: r["over"][k] for k in keys if k in r["over"]})
            if len(tops) == n:
                break
        return tops

    base3 = best["params"]
    cross = []
    for wo in _top_overs(wav_rows, ["wavelet", "wavelet_name",
                                    "wavelet_level",
                                    "wavelet_threshold",
                                    "wavelet_mode", "wavelet_cycle"]):
        for so in _top_overs(sg_rows, ["sg_window", "sg_poly"]):
            cross.append({**wo, **so})
    run_stage("3-wavelet-x-savgol", base3, cross, scorer, out, best, log)
    base = best["params"]

    # ---- stage 4: switches ----------------------------------------------
    base_score4 = best["score"]
    sw = [dict(despike=True, despike_z=z) for z in DESPIKE_ZS]
    sw += [dict(wn_calibrate=True), dict(detrend=True)]
    sw_rows = run_stage("4-switches", base, sw, scorer, out, best, log)
    # joint combo of the switches that individually BEAT the base
    joint: dict = {}
    for keys in (("despike", "despike_z"), ("wn_calibrate",),
                 ("detrend",)):
        for r in sw_rows:               # sorted best-first
            if (all(k in r["over"] for k in keys)
                    and r["mean_f1"] > base_score4):
                joint.update(r["over"])
                break
    if joint:
        run_stage("4-switches-joint", base,
                  [joint, {k: v for k, v in joint.items()
                   if k != "detrend"}], scorer, out, best, log)
    base = best["params"]

    # ---- stage 5: crop fine-tune ----------------------------------------
    if base.crop_max:
        if SMOKE:
            mins, maxs = ([base.crop_min, base.crop_min + 50.0],
                          [base.crop_max - 75.0, base.crop_max])
        else:
            mins = sorted({base.crop_min + d
                           for d in (-100, -50, 0, 50, 100)})
            maxs = sorted({base.crop_max + d
                           for d in (-150, -75, 0, 75, 150)})
        fine = [dict(crop_min=float(mn), crop_max=float(mx))
                for mn, mx in itertools.product(mins, maxs)
                if 100.0 < mn < mx < 3500.0]
        run_stage("5-crop-fine", base, fine, scorer, out, best, log)
        base = best["params"]

    # ---- stage 6: honest validation -------------------------------------
    log(f"\n-- stage 6: multi-seed validation ({mode}) --")
    all_rows = [r for rows in out["stages"].values() for r in rows]
    all_rows.sort(key=lambda r: r["mean_f1"], reverse=True)
    cands, seen = [], set()
    for r in all_rows:
        key = json.dumps(r["params"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        cands.append(r)
        if len(cands) >= (3 if SMOKE else 8):
            break
    cands.append({"params": asdict(defaults),
                  "label": describe(defaults)})
    saved = opt.load_best_params()
    if saved:
        cands.append({"params": asdict(saved[0].validate()),
                      "label": describe(saved[0].validate())
                      + " [saved-best]"})
    seeds = [42, 43] if SMOKE else [42, 43, 44]
    val = []
    for i, c in enumerate(cands, start=1):
        params = pp.PreprocessParams(**c["params"])
        try:
            v = scorer.score_multi_seed(params, seeds)
        except Exception as exc:
            log(f"  [{i}] {c['label']}: FAILED {exc}")
            continue
        entry = {"label": c["label"], "params": c["params"], **v}
        val.append(entry)
        log(f"  [{i}] {c['label']}: {v['best_model']} "
            f"F1 {v['mean_f1']:.3f} +/- {v['std_f1']:.3f} over seeds {seeds}")
    # paired bonus: winner +/- PQN
    if mode == "paired":
        wparams = pp.PreprocessParams(**max(
            val, key=lambda v: v["mean_f1"])["params"])
        try:
            vq = scorer.score_multi_seed(wparams, seeds, use_pqn=True)
            val.append({"label": describe(wparams) + " +PQN",
                        "params": asdict(wparams), "use_pqn": True, **vq})
            log(f"  [PQN] +PQN variant: {vq['best_model']} "
                f"F1 {vq['mean_f1']:.3f} +/- {vq['std_f1']:.3f}")
        except Exception as exc:
            log(f"  [PQN] FAILED {exc}")
    val.sort(key=lambda v: v["mean_f1"], reverse=True)
    out["validation"] = val

    # nested CV for top-2 + defaults (the honest headline numbers)
    if not SMOKE:
        log(f"\n-- nested evaluate_models (top-2 + defaults, {mode}) --")
        nested = []
        nest_cands = val[:2] + [v for v in val
                                if v["label"] == describe(defaults)]
        model_names = ["PCA + SVM (RBF)", "Random Forest",
                       "PCA + Logistic Regression", "Extra Trees",
                       "PCA + LDA"]
        for i, c in enumerate(nest_cands, start=1):
            params = pp.PreprocessParams(**c["params"])
            log(f"  [{i}/{len(nest_cands)}] nested: {c['label']}")
            try:
                nr = scorer.nested_eval(params, model_names, k=5, seed=seed)
            except Exception as exc:
                log(f"      FAILED {exc}")
                continue
            nested.append({"label": c["label"], "params": c["params"], **nr})
            log(f"      winner {nr['winner']} "
                f"F1 {nr['winner_f1']:.3f}")
        out["nested"] = nested

    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["n_scored_configs"] = scorer.n_scored
    _dump(out)
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def print_report(out: dict, log=print):
    log(f"\n{'=' * 72}\nDEEP SEARCH RESULT — {out['mode'].upper()}\n{'=' * 72}")
    d = out.get("defaults") or {}
    log(f"defaults: {d.get('label')} -> F1 {d.get('mean_f1', float('nan')):.3f}")
    b = out["best"]
    log(f"selection best ({b['stage']}): {b['label']} -> {b['model']} "
        f"F1 {b['score']:.3f}")
    log("\nWinner — full parameter set (paste into the GUI):")
    for k, v in b["params"].items():
        log(f"  {k:<18} {v}")
    if out.get("validation"):
        log("\nMulti-seed validation (5 models x seeds):")
        for v in out["validation"][:8]:
            log(f"  {v['mean_f1']:.3f} +/- {v['std_f1']:.3f} "
                f"[{v['best_model']}] {v['label']}")
    if out.get("nested"):
        log("\nNested CV (honest, hyperparams tuned inside folds):")
        for n in out["nested"]:
            log(f"  {n['label']}: winner {n['winner']} "
                f"F1 {n['winner_f1']:.3f}")
    log("")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    global SMOKE
    ap = argparse.ArgumentParser(
        description="Deep combinatorial preprocessing search "
                    "(staged factorial, grouped CV)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--mode", choices=["standard", "paired", "both"],
                    default="both")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grids — quick sanity run")
    args = ap.parse_args()
    SMOKE = args.smoke

    root = args.data or find_data_root()
    if not root:
        raise SystemExit("No clinical data root found — pass --data PATH")
    print(f"Data root: {root}")
    cd = load_clinical_dataset(root)
    grid = ds.common_grid(cd.spectra)
    X_raw, _ = ds.to_matrix(cd.spectra, grid)
    labels = [s.label for s in cd.spectra]
    groups = cd.groups or None
    flagged = cd.flagged or None
    n_flag = sum(1 for f in (flagged or []) if f)
    print(f"{len(labels)} spectra / {len(set(groups)) if groups else '-'} "
          f"patients; {n_flag} spike-flagged (excluded unless despiking); "
          f"grid {len(grid)} pts "
          f"[{grid.min():.1f}, {grid.max():.1f}]")

    modes = (["standard", "paired"] if args.mode == "both"
             else [args.mode])
    outs = {}
    for mode in modes:
        outs[mode] = search_mode(X_raw, grid, labels, groups, flagged,
                                 mode, args.folds, args.seed)
        print_report(outs[mode])

    # combined markdown summary
    lines = [f"# Deep preprocessing search — {time.strftime('%Y-%m-%d %H:%M')}",
             f"Data: {root} · {len(labels)} spectra · "
             f"{'/'.join(modes)} mode(s) · {args.folds}-fold grouped CV",
             ""]
    for mode, out in outs.items():
        b = out["best"]
        d = out.get("defaults") or {}
        lines += [f"## {mode.upper()}",
                  f"- defaults ({d.get('label')}): selection F1 "
                  f"{d.get('mean_f1', float('nan')):.3f}",
                  f"- best ({b['stage']}): **{b['label']}** — "
                  f"{b['model']} selection F1 {b['score']:.3f}",
                  "", "| param | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in b["params"].items()]
        if out.get("validation"):
            lines += ["", "### Multi-seed validation", "",
                      "| F1 (mean±std) | best model | config |", "|---|---|---|"]
            lines += [f"| {v['mean_f1']:.3f} ± {v['std_f1']:.3f} | "
                      f"{v['best_model']} | {v['label']} |"
                      for v in out["validation"][:8]]
        if out.get("nested"):
            lines += ["", "### Nested CV (honest)", ""]
            lines += [f"- {n['label']}: winner **{n['winner']}** "
                      f"F1 **{n['winner_f1']:.3f}**"
                      for n in out["nested"]]
        lines.append("")
    md_path = os.path.join(OUT_DIR, "prep_deep_search_report.md")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nReport: {md_path}")
    print(f"JSON:   {OUT_DIR}/prep_deep_search_{{standard,paired}}.json")


if __name__ == "__main__":
    main()
