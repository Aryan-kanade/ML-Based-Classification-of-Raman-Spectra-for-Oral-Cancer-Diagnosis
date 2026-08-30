"""
sequential.py — 3SSE: Three-Stage Sequential Stacking Ensemble search.

Evaluates every ordered architecture over the model registry:

    Level 1:  N              single models
    Level 2:  N*(N-1)        two-model chains    A -> B
    Level 3:  N*(N-1)*(N-2)  three-model chains  A -> B -> C
    (N = 17 registry models  ->  17 + 272 + 4080 = 4369 architectures)

Chain semantics (probabilities only, never hard labels):

    Layer 1: X                  -> A -> OOF P1
    Layer 2: [X + OOF P1]       -> B -> OOF P2
    Layer 3: [X + OOF P1 + P2]  -> C -> final OOF probabilities

Leakage control:
  * every inter-layer prediction is OUT-OF-FOLD via the project's
    patient-grouped StratifiedGroupKFold (cross_val_predict);
  * QUICK search uses single-level OOF stacking for screening;
  * FULL validation re-generates the OOF features INSIDE each outer
    training fold only (nested) — the outer test fold never trains or
    selects anything.

Efficiency (why 4,369 architectures are feasible):
  * layer-1 OOF computed once per model and shared;
  * pair-major loop: each [X + P1 + P2] feature matrix is built once and
    all eligible third models are evaluated while it is hot;
  * pairs processed best-first with sound early-abandon: after fold i a
    triple's best reachable macro-F1 is (sum_i + (k-i))/k; below the
    current top-N cutoff it cannot enter the validated set, so the
    remaining folds are skipped (top-N ranking stays exact);
  * parallel across pairs (joblib), single-threaded inside estimators;
  * append-only JSONL checkpoint; --resume skips finished triples.

Hyperparameters stay at the registry defaults during screening —
identical treatment across all 4,369 architectures keeps the level
comparison fair (per-layer nested grids x 4,369 would be prohibitive).

Usage:
    python sequential.py --data D:/BARC/Data --mode paired --out study_run_3sse
    python sequential.py --demo --models "PCA + LDA,Random Forest" --k 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clinical as clin                      # noqa: E402
import modeling                              # noqa: E402
from modeling import fit_maybe_grouped       # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = r"D:\BARC\Data"


# --------------------------------------------------------------------------
# architecture space
# --------------------------------------------------------------------------
def architectures(names: list[str], max_len: int = 3) -> list[tuple]:
    """All ordered, repetition-free chains of length 1..max_len."""
    archs: list[tuple] = []
    for length in range(1, max_len + 1):
        archs.extend(itertools.permutations(names, length))
    return archs


def _check_space(names: list[str]) -> int:
    """17 + 17*16 + 17*16*15 == 4369 for the full registry (spec §4)."""
    n = len(names)
    total = n + n * (n - 1) + n * (n - 1) * (n - 2)
    assert len(architectures(names)) == total
    return total


# --------------------------------------------------------------------------
# model factories (fresh, clone-safe estimators from the registry)
# --------------------------------------------------------------------------
def _factories(wavenumbers=None, cnn_epochs: int = 15
               ) -> dict[str, object]:
    """
    name -> zero-arg factory returning a fresh default-params estimator.
    "Ensemble (top-3)" is resolved AFTER the singles ranking (a voting
    classifier over the best three single models) — see search().
    """
    out: dict[str, object] = {}

    def wrap(est):
        return lambda: modeling.clone(est)  # noqa: E731

    for spec in modeling.model_specs():
        name = spec["name"]
        if spec.get("ensemble"):
            continue                    # resolved post-singles
        if spec.get("needs_wn"):
            if wavenumbers is None:
                continue                # peak-bands needs the wn axis
            est, _grid = spec["make"](np.asarray(wavenumbers))
            out[name] = wrap(est)
        else:
            out[name] = wrap(spec["estimator"])
    if modeling.HAS_TORCH and "1D-CNN" in out:
        out["1D-CNN"] = lambda: modeling.CNN1DClassifier(  # noqa: E731
            epochs=cnn_epochs)
    return out


def _ensemble_factory(top3: list[str], factories: dict, wn=None):
    """VotingClassifier (soft) over the top-3 single models' default
    estimators — model 17 of the spec, itself an ensemble (nested when
    chained inside 2/3-model architectures).  A peak-bands voter is
    wrapped with a spectral slice so the ensemble also works at chained
    positions where probability columns are appended."""
    from sklearn.ensemble import VotingClassifier
    voters = []
    for i, n in enumerate(top3):
        est = factories[n]()
        if n == "Peak bands + RF" and wn is not None:
            est = _sliced(est, len(wn))
        voters.append((f"m{i}", est))
    ens = VotingClassifier(estimators=voters, voting="soft", n_jobs=1)
    return lambda: modeling.clone(ens)  # noqa: E731


class _SpectralSlice(modeling.BaseEstimator, modeling.TransformerMixin):
    """Keep only the spectral columns (drop appended probability
    columns) for estimators that need the wavenumber-aligned axis."""

    def __init__(self, n: int):
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, :self.n]


def _sliced(est, n: int):
    from sklearn.pipeline import Pipeline
    return Pipeline([("spec", _SpectralSlice(n)), ("clf", est)])


def _layer_est(name: str, factories: dict, position: int, wn):
    """
    Estimator for a chain position.  "Peak bands + RF" consumes the
    wavenumber-aligned axis, so at chained positions (>= 1, where the
    input carries appended probability columns) it is wrapped to use
    the spectral columns only (spec §9: model-specific features stay
    intact; probability chaining remains available to the others).
    """
    est = factories[name]()
    if position >= 1 and wn is not None and name == "Peak bands + RF":
        return _sliced(est, len(wn))
    return est


# --------------------------------------------------------------------------
# grouped OOF machinery
# --------------------------------------------------------------------------
def _splitter(k: int, seed: int, y, groups):
    from sklearn.model_selection import (StratifiedGroupKFold,
                                         StratifiedKFold)
    min_class = min(np.bincount(np.searchsorted(np.unique(y), y)))
    k = int(np.clip(k, 2, min(10, min_class)))
    if groups is not None:
        k = int(np.clip(k, 2, len(set(groups))))
        return StratifiedGroupKFold(n_splits=k, shuffle=True,
                                    random_state=seed)
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def _oof_proba(est, X, y, groups, k: int, seed: int) -> np.ndarray:
    """Grouped out-of-fold predict_proba for every row of X."""
    from sklearn.model_selection import cross_val_predict
    cv = _splitter(k, seed, y, groups)
    return cross_val_predict(est, X, y, groups=groups, cv=cv,
                             method="predict_proba")


def _metrics_from_oof(y, proba, classes, groups) -> dict:
    """Spectrum + patient-level metrics from pooled OOF probabilities.
    `y` must already be class-encoded (see search/validate_arch) — the
    registry's boosting models reject string labels."""
    from sklearn.metrics import (confusion_matrix, f1_score,
                                 roc_auc_score)
    ye = np.asarray(y, dtype=int)
    pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(ye, pred, labels=range(len(classes)))
    per = modeling.class_metrics_from_cm(cm)
    macro = {m: float(np.mean([per[c][m] for c in per]))
             for m in ("sens", "spec", "prec", "f1")}
    out = {"f1": macro["f1"], "sens": macro["sens"], "spec": macro["spec"],
           "prec": macro["prec"],
           "acc": float(np.trace(cm) / max(cm.sum(), 1)),
           "cm": cm.tolist()}
    if len(classes) == 2 and 0 < ye.sum() < len(ye):
        out["auc"] = float(roc_auc_score(ye, np.nan_to_num(proba[:, 1])))
    else:
        out["auc"] = float("nan")
    if groups is not None:                       # patient-level rollup
        g = np.asarray(groups)
        p_true = [int(ye[g == pat][0]) for pat in np.unique(g)]
        p_pred = [int(np.argmax(proba[g == pat].mean(axis=0)))
                  for pat in np.unique(g)]
        pcm = confusion_matrix(p_true, p_pred,
                               labels=range(len(classes)))
        pper = modeling.class_metrics_from_cm(pcm)
        out["pat_acc"] = float(np.trace(pcm) / max(pcm.sum(), 1))
        out["pat_f1"] = float(np.mean([pper[c]["f1"] for c in pper]))
        out["pat_sens"] = float(np.mean([pper[c]["sens"] for c in pper]))
        out["pat_spec"] = float(np.mean([pper[c]["spec"] for c in pper]))
    return out


def _eval_last_layer(factory, feats, y, classes, groups, k, seed, cutoff):
    """
    OOF evaluation of the FINAL layer with sound early-abandon: after
    fold i the best reachable macro-F1 is (sum_i + (k-i))/k; if that is
    below `cutoff` the architecture cannot enter the top-N and the
    remaining folds are skipped.  Returns (metrics, None) or
    (None, bound) when pruned.
    """
    from sklearn.metrics import f1_score
    y_arr = np.asarray(y)
    g_arr = np.asarray(groups) if groups is not None else None
    ye = np.searchsorted(np.asarray(classes), y_arr)
    cv = _splitter(k, seed, y_arr, groups)
    n_splits = cv.get_n_splits()
    oof = np.full((len(ye), len(classes)), np.nan)
    fold_f1: list[float] = []
    for tr, te in cv.split(feats, ye, groups):
        est = fit_maybe_grouped(factory(), feats[tr], y_arr[tr],
                                g_arr[tr] if g_arr is not None else None)
        oof[te] = est.predict_proba(feats[te])
        pred_te = np.argmax(oof[te], axis=1)
        fold_f1.append(float(f1_score(ye[te], pred_te, average="macro")))
        if cutoff is not None and len(fold_f1) < n_splits:
            bound = (sum(fold_f1) + (n_splits - len(fold_f1))) / n_splits
            if bound < cutoff:
                return None, bound
    valid = ~np.isnan(oof).any(axis=1)
    sub_g = g_arr[valid] if g_arr is not None else None
    metrics = _metrics_from_oof(y_arr[valid], oof[valid], classes, sub_g)
    return metrics, None


def _pin_threads():
    """One thread per worker process (avoid oversubscription)."""
    if modeling.HAS_TORCH:
        modeling.torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


# --------------------------------------------------------------------------
# parallel workers (self-contained args: no module globals needed)
# --------------------------------------------------------------------------
def _worker_pair(task):
    """Evaluate one ordered pair: B on [X + OOF P1_A]."""
    _pin_threads()
    (A, B), X, y, groups, classes, k, seed, p1, factories, wn = task
    feats = np.hstack([X, p1])
    est = _layer_est(B, factories, 1, wn)
    P2 = _oof_proba(est, feats, y, groups, k, seed)
    metrics = _metrics_from_oof(y, P2, classes, groups)
    return (A, B), metrics, P2.astype(np.float32)


def _worker_triples(task):
    """Pair-major: evaluate the pending third models for one pair."""
    _pin_threads()
    ((A, B), X, y, groups, classes, k, seed, p1, p2, cand, factories,
     wn, cutoff) = task
    feats = np.hstack([X, p1, p2]).astype(np.float32)
    results = []
    for C in cand:
        # fresh estimator per fold (XGBoost & co. must not be refit on
        # the same instance with re-encoded labels)
        factory = (lambda nm=C: _layer_est(nm, factories, 2, wn))
        metrics, pruned = _eval_last_layer(
            factory, feats, y, classes, groups, k, seed, cutoff)
        results.append(((A, B, C), metrics, pruned))
    return results


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------
def search(X, y, groups=None, wavenumbers=None, k: int = 3, seed: int = 42,
           top: int = 20, jobs: int = -2, model_names: list | None = None,
           prune_pairs: int = 0, cnn_epochs: int = 15,
           resume_path: str | None = None, progress_cb=None) -> dict:
    """
    QUICK search over all architectures (single-level OOF stacking,
    grouped CV).  Returns {"singles", "pairs", "triples", "total",
    "pruned"} where each entry is {"arch", "level", "metrics"} (or
    {"arch", "level", "pruned_bound"} for early-abandoned triples).
    """
    from joblib import Parallel, delayed

    X = np.asarray(X, dtype=np.float32)
    y = list(y)
    classes = sorted(set(y))
    # encode labels once at the boundary (evaluate_models parity —
    # xgboost & co. reject string labels)
    y = np.searchsorted(np.asarray(classes), np.asarray(y)).tolist()
    factories = _factories(wavenumbers, cnn_epochs)
    names = [n for n in factories
             if model_names is None or n in model_names]
    if len(names) < 2:
        raise ValueError("need at least 2 models for the search")
    total = _check_space(names)
    t0 = time.time()
    done = [0]

    def tick(best: str, best_f1: float):
        if progress_cb:
            eta = (time.time() - t0) / max(done[0], 1) * (total - done[0])
            progress_cb(done[0], total, best, best_f1, eta)

    resume: dict[str, dict] = {}
    if resume_path and os.path.exists(resume_path):
        with open(resume_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                resume[" → ".join(rec["arch"])] = rec
    ckpt = open(resume_path, "a", encoding="utf-8") \
        if resume_path else None

    def record(arch, metrics, pruned):
        if not ckpt:
            return
        rec = {"arch": list(arch), "level": 3,
               "pruned": pruned is not None,
               **({"bound": round(float(pruned), 4)}
                  if pruned is not None
                  else {m: (round(float(v), 4)
                            if isinstance(v, float) and v == v else v)
                        for m, v in metrics.items() if m != "cm"})}
        ckpt.write(json.dumps(rec) + "\n")
        ckpt.flush()

    # ---- level 1: singles (OOF kept as layer-1 features) ---------------
    singles, oof1 = [], {}
    for name in names:
        P1 = _oof_proba(factories[name](), X, y, groups, k, seed)
        oof1[name] = P1.astype(np.float32)
        m = _metrics_from_oof(y, P1, classes, groups)
        singles.append({"arch": (name,), "level": 1, "metrics": m})
        done[0] += 1
        tick(name, m["f1"])

    # model 17 resolved from the singles ranking (spec §10) whenever it
    # was requested (full run or an explicit checkbox subset)
    if model_names is None or "Ensemble (top-3)" in model_names:
        top3 = [s["arch"][0] for s in
                sorted(singles, key=lambda s: -s["metrics"]["f1"])][:3]
        factories["Ensemble (top-3)"] = _ensemble_factory(top3,
                                                          factories,
                                                          wn=wavenumbers)
        names = names + ["Ensemble (top-3)"]
        total = _check_space(names)          # now includes model 17
        # the full 17-model registry (peak-bands needs the wn axis)
        if model_names is None and wavenumbers is not None:
            assert total == 4369, total      # spec §4
        P1 = _oof_proba(factories["Ensemble (top-3)"](), X, y, groups,
                        k, seed)
        oof1["Ensemble (top-3)"] = P1.astype(np.float32)
        m = _metrics_from_oof(y, P1, classes, groups)
        singles.append({"arch": ("Ensemble (top-3)",), "level": 1,
                        "metrics": m})
        done[0] += 1
        tick("Ensemble (top-3)", m["f1"])

    # ---- level 2: ordered pairs ----------------------------------------
    tasks = [((A, B), X, y, groups, classes, k, seed, oof1[A],
              factories, wavenumbers)
             for A, B in itertools.permutations(names, 2)]
    pairs, p2 = [], {}
    for key, metrics, P2 in Parallel(n_jobs=jobs)(
            delayed(_worker_pair)(t) for t in tasks):
        p2[key] = P2
        pairs.append({"arch": key, "level": 2, "metrics": metrics})
        done[0] += 1
        tick(" → ".join(key), metrics["f1"])

    # ---- level 3: pair-major, best-first, early-abandon -----------------
    triples: list[dict] = []
    for rec in resume.values():          # replay checkpointed triples
        if rec.get("level") == 3:
            if rec.get("pruned"):
                triples.append({"arch": tuple(rec["arch"]), "level": 3,
                                "pruned_bound": rec.get("bound", 0.0)})
            else:
                triples.append({"arch": tuple(rec["arch"]), "level": 3,
                                "metrics": {m: v for m, v in rec.items()
                                            if m not in ("arch", "level",
                                                         "pruned")}})
            done[0] += 1

    def cutoff_now() -> float:
        scored = [t["metrics"]["f1"] for t in triples if "metrics" in t]
        if len(scored) < top:
            return -1.0
        return min(sorted(scored, reverse=True)[:top])

    pairs_sorted = sorted(pairs, key=lambda p: -p["metrics"]["f1"])
    if prune_pairs:
        pairs_sorted = pairs_sorted[:int(prune_pairs)]
    cutoff = cutoff_now()
    chunk = max(1, abs(jobs))
    for i in range(0, len(pairs_sorted), chunk):
        batch = pairs_sorted[i:i + chunk]
        tasks = []
        for p in batch:
            A, B = p["arch"]
            cand = [C for C in names
                    if C not in (A, B)
                    and " → ".join((A, B, C)) not in resume]
            if cand:
                tasks.append(((A, B), X, y, groups, classes, k, seed,
                              oof1[A], p2[(A, B)], cand, factories,
                              wavenumbers, cutoff))
        for results in Parallel(n_jobs=jobs)(
                delayed(_worker_triples)(t) for t in tasks):
            for arch, metrics, pruned in results:
                if metrics is None:
                    triples.append({"arch": arch, "level": 3,
                                    "pruned_bound": pruned})
                else:
                    triples.append({"arch": arch, "level": 3,
                                    "metrics": metrics})
                record(arch, metrics, pruned)
                done[0] += 1
        cutoff = cutoff_now()
        scored = [t for t in triples if "metrics" in t]
        if scored:
            b = max(scored, key=lambda t: t["metrics"]["f1"])
            tick(" → ".join(b["arch"]), b["metrics"]["f1"])
    if ckpt:
        ckpt.close()
    return {"singles": singles, "pairs": pairs, "triples": triples,
            "total": total,
            "pruned": sum(1 for t in triples if "pruned_bound" in t)}


# --------------------------------------------------------------------------
# nested full validation
# --------------------------------------------------------------------------
def validate_arch(arch, factories, X, y, groups, seed: int = 42,
                  k_outer: int = 5, wn=None) -> dict:
    """
    Honest nested evaluation of one architecture: outer grouped CV; the
    OOF chain features are regenerated INSIDE each outer training fold
    (inner grouped CV); the outer test fold only ever sees predictions.
    """
    X = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y)
    classes = sorted(set(y_arr.tolist()))
    # encode labels (evaluate_models parity — boosting models reject
    # string labels)
    y_arr = np.searchsorted(np.asarray(classes), y_arr)
    ye = y_arr
    outer = _splitter(k_outer, seed, y_arr, groups)
    oof = np.full((len(y_arr), len(classes)), np.nan)
    for tr, te in outer.split(X, ye, groups):
        g_tr = np.asarray(groups)[tr] if groups is not None else None
        y_tr = y_arr[tr]
        F_tr, F_te = X[tr], X[te]
        min_class = min(np.bincount(y_tr))
        k_in = max(2, min(3, min_class))
        if g_tr is not None:
            k_in = min(k_in, len(set(g_tr.tolist())))
        for pos, name in enumerate(arch[:-1]):
            try:                        # inner OOF within the fold only
                P_tr = _oof_proba(_layer_est(name, factories, pos, wn),
                                  F_tr, list(y_tr), g_tr, k_in, seed)
            except Exception:           # tiny fold: in-sample fallback
                est = fit_maybe_grouped(_layer_est(name, factories, pos,
                                                   wn), F_tr, y_tr, g_tr)
                P_tr = est.predict_proba(F_tr)
            est = fit_maybe_grouped(_layer_est(name, factories, pos, wn),
                                    F_tr, y_tr, g_tr)
            F_tr = np.hstack([F_tr, P_tr])
            F_te = np.hstack([F_te, est.predict_proba(F_te)])
        last = fit_maybe_grouped(
            _layer_est(arch[-1], factories, len(arch) - 1, wn),
            F_tr, y_tr, g_tr)
        oof[te] = last.predict_proba(F_te)
    valid = ~np.isnan(oof).any(axis=1)
    sub_g = (np.asarray(groups)[valid] if groups is not None else None)
    metrics = _metrics_from_oof(y_arr[valid], oof[valid], classes, sub_g)
    metrics["oof_proba"] = oof[valid]
    metrics["y_true"] = y_arr[valid]
    return metrics


# --------------------------------------------------------------------------
# SequentialChain — the deployable, bundle-compatible pipeline
# --------------------------------------------------------------------------
class SequentialChain:
    """
    Ordered chain of layer estimators.  fit() builds each layer's
    training features from the PREVIOUS layers' grouped OOF
    probabilities (leakage-free stacking) and fits every layer on the
    full data; predict_proba() chains the fitted layers, appending each
    layer's probability columns.  Speaks the sklearn protocol so the
    existing bundle / Platt / threshold / predict machinery works
    unchanged.  NOTE: predict_proba ignores any reference/paired
    arguments the bundle layer may add — preprocessing and paired
    subtraction happen BEFORE the chain, exactly like any other model.
    """

    def __init__(self, estimators: list, k: int = 3, seed: int = 42):
        self.estimators = list(estimators)     # unfitted, clone-safe
        self.k = k
        self.seed = seed

    # -- sklearn protocol --------------------------------------------------
    def get_params(self, deep=True):
        return {"estimators": list(self.estimators), "k": self.k,
                "seed": self.seed}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=np.float32)
        self.classes_ = sorted(set(y))            # public string labels
        # layers train on encoded labels (boosting models reject
        # strings); proba columns follow sorted-numeric = sorted-string
        y = np.searchsorted(np.asarray(self.classes_),
                            np.asarray(y)).tolist()
        feats = X
        self.fitted_ = []
        for est in self.estimators[:-1]:
            P = _oof_proba(modeling.clone(est), feats, y, groups,
                           self.k, self.seed)
            fitted = fit_maybe_grouped(modeling.clone(est), feats, y,
                                       groups)
            self.fitted_.append(fitted)
            feats = np.hstack([feats, P])
        self.fitted_.append(fit_maybe_grouped(
            modeling.clone(self.estimators[-1]), feats, y, groups))
        return self

    def predict_proba(self, X):
        feats = np.asarray(X, dtype=np.float32)
        for est in self.fitted_[:-1]:
            feats = np.hstack([feats, est.predict_proba(feats)])
        return self.fitted_[-1].predict_proba(feats)

    def predict(self, X):
        return np.asarray(self.classes_)[
            np.argmax(self.predict_proba(X), axis=1)]


def chain_factory(arch, factories, k: int = 3, seed: int = 42):
    """Factory for a SequentialChain over the given architecture."""
    return lambda: SequentialChain(  # noqa: E731
        [factories[n]() for n in arch], k=k, seed=seed)


# --------------------------------------------------------------------------
# dataset preparation (mirrors reproduce_study — GUI parity)
# --------------------------------------------------------------------------
def prepare_dataset(root: str, mode: str = "standard", demo: bool = False):
    """Returns (X, y, groups, wavenumbers, grid, params, meta)."""
    import dataset as ds
    import preprocessing as pp
    from clinical_data import load_clinical_dataset

    params = pp.PreprocessParams().validate()
    flagged = None
    if demo:
        import tempfile
        src = ds.find_default_source_spectrum()
        r = ds.generate_demo_data(src, os.path.join(tempfile.mkdtemp(),
                                                    "demo"))
        spectra = ds.load_folder(r)
        labels = [s.label for s in spectra]
        groups = None
    else:
        cd = load_clinical_dataset(root)
        spectra = cd.spectra
        labels = [s.label for s in spectra]
        groups = cd.groups or None
        flagged = cd.flagged or None
    grid = ds.common_grid(spectra)
    X_raw, _ = ds.to_matrix(spectra, grid)
    keep = [i for i, lab in enumerate(labels) if lab.strip()]
    if flagged is not None and not params.despike:
        keep = [i for i in keep if not flagged[i]]
    y = [labels[i].strip() for i in keep]
    g = [groups[i] for i in keep] if groups else None
    if mode in ("paired", "paired-pqn"):
        import paired as pmod
        pd_ = pmod.paired_features(X_raw[keep], y, g, grid, params,
                                   use_pqn=(mode == "paired-pqn"))
        X, y, g = pd_.X, pd_.y, pd_.groups
        wn = pd_.wn
    else:
        X = pp.preprocess_matrix(X_raw[keep], params, wn=grid)
        wn = np.asarray(grid)[pp.crop_mask(grid, params)]
    meta = {"n_spectra": len(y), "n_patients": len(set(g)) if g else 0,
            "mode": mode}
    return X, y, g, wn, grid, params, meta


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def _fmt_arch(arch) -> str:
    return " → ".join(arch)


def pick_overall(validated: dict[int, list]) -> dict | None:
    """Overall winner across levels by (F1, sens, spec) — validated
    results only (spec §15)."""
    best = None
    for level in (1, 2, 3):
        for entry in validated.get(level, []):
            m = entry["metrics"]
            key = (m["f1"], m["sens"], m["spec"])
            if best is None or key > best[0]:
                best = (key, entry)
    return best[1] if best else None


def build_report(board: dict, validated: dict, baseline: dict) -> str:
    lines = ["3SSE — Sequential Architecture Search",
             "=" * 60]
    for level, key, title in ((1, "singles", "BEST SINGLE MODEL"),
                              (2, "pairs", "BEST 2-MODEL ARCHITECTURE"),
                              (3, "triples",
                               "BEST 3-MODEL ARCHITECTURE")):
        cands = [e for e in board[key] if "metrics" in e]
        if cands:
            b = max(cands, key=lambda e: e["metrics"]["f1"])
            lines += ["", f"{title} (screening)",
                      f"  {_fmt_arch(b['arch'])}  ·  F1 "
                      f"{b['metrics']['f1']:.3f}"]
    lines += ["", "FULL NESTED VALIDATION (top architectures per level)",
              f"{'Type':<8} {'Architecture':<46} {'F1':>6} {'Sens':>6} "
              f"{'Spec':>6} {'AUC':>6} {'Acc':>6}"]
    for level in (1, 2, 3):
        for entry in validated.get(level, []):
            m = entry["metrics"]
            lines.append(
                f"{level}-Model  {_fmt_arch(entry['arch']):<46} "
                f"{m['f1']:>6.3f} {m['sens']:>6.3f} {m['spec']:>6.3f} "
                f"{m.get('auc', float('nan')):>6.3f} {m['acc']:>6.3f}")
    winner = pick_overall(validated)
    if winner:
        m = winner["metrics"]
        lines += ["", "=" * 60, "OVERALL BEST ARCHITECTURE",
                  f"  Type: {len(winner['arch'])}-Model",
                  f"  Architecture: {_fmt_arch(winner['arch'])}",
                  f"  Macro-F1: {m['f1']:.3f}   Sensitivity: "
                  f"{m['sens']:.3f}   Specificity: {m['spec']:.3f}",
                  f"  ROC-AUC: {m.get('auc', float('nan')):.3f}   "
                  f"Accuracy: {m['acc']:.3f}",
                  "=" * 60]
    if baseline:
        lines += ["", "BASELINE COMPARISON"]
        for name, (f1, auc) in baseline.items():
            lines.append(f"  {name:<34} F1 {f1:.3f} · AUC {auc:.3f}")
        if winner:
            lines.append(f"  3SSE winner {'':<20} F1 "
                         f"{winner['metrics']['f1']:.3f} · AUC "
                         f"{winner['metrics'].get('auc', float('nan')):.3f}")
    lines += ["",
              f"Architectures evaluated: {board['total']} · pruned early "
              f"(sound bound): {board.get('pruned', 0)}",
              "Screening = single-level OOF stacking, default "
              "hyperparameters, identical for every architecture;",
              "the overall winner comes from nested grouped validation."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# top-N validation + winner finalization (shared by CLI and GUI worker)
# --------------------------------------------------------------------------
def validate_top(board: dict, X, y, groups, wavenumbers, top: int = 20,
                 seed: int = 42, k_outer: int = 5, cnn_epochs: int = 40,
                 model_names: list | None = None,
                 progress=None) -> dict[int, list]:
    """Nested full validation of the top-N screening architectures per
    level.  Returns {level: [{"arch", "metrics"}]}."""
    factories = _factories(wavenumbers, cnn_epochs)
    if model_names is None and any(
            e["arch"][0] == "Ensemble (top-3)" for e in board["singles"]):
        top3 = [e["arch"][0] for e in sorted(
            (s for s in board["singles"]
             if s["arch"][0] != "Ensemble (top-3)"),
            key=lambda s: -s["metrics"]["f1"])][:3]
        factories["Ensemble (top-3)"] = _ensemble_factory(
            top3, factories, wn=wavenumbers)
    validated: dict[int, list] = {}
    for level, key in ((1, "singles"), (2, "pairs"), (3, "triples")):
        cands = sorted((e for e in board[key] if "metrics" in e),
                       key=lambda e: -e["metrics"]["f1"])
        validated[level] = []
        for entry in cands[:top]:
            arch = tuple(n for n in entry["arch"] if n in factories)
            if len(arch) != entry["level"]:
                continue
            if progress:
                progress(f"validating {len(arch)}-model "
                         f"{_fmt_arch(arch)}")
            try:
                m = validate_arch(arch, factories, X, y, groups,
                                  seed=seed, k_outer=k_outer,
                                  wn=wavenumbers)
                validated[level].append({"arch": arch, "metrics": m})
            except Exception as exc:
                if progress:
                    progress(f"validation failed for {_fmt_arch(arch)}: "
                             f"{exc}")
    return validated


def finalize_winner(validated: dict, X, y, groups, wavenumbers,
                    board: dict | None = None, k: int = 3, seed: int = 42,
                    cnn_epochs: int = 40) -> dict | None:
    """
    Pick the overall validated winner, fit its deployable
    SequentialChain on all data, and derive the binary threshold +
    Platt calibrator from its nested-OOF probabilities.
    Returns {"arch", "chain", "classes", "threshold", "calibrator",
             "metrics"} or None.
    """
    winner = pick_overall(validated)
    if winner is None:
        return None
    m = winner["metrics"]
    factories = _factories(wavenumbers, cnn_epochs)
    if "Ensemble (top-3)" in winner["arch"]:
        top3 = [e["arch"][0] for e in sorted(
            (s for s in (board or {}).get("singles", [])
             if s["arch"][0] != "Ensemble (top-3)"),
            key=lambda s: -s["metrics"]["f1"])][:3]
        factories["Ensemble (top-3)"] = _ensemble_factory(
            top3, factories, wn=wavenumbers)
    chain = SequentialChain(
        [_layer_est(n, factories, i, wavenumbers)
         for i, n in enumerate(winner["arch"])],
        k=k, seed=seed).fit(X, y, groups=groups)
    classes = sorted(set(y))
    thr, calibrator = None, None
    if m["oof_proba"].shape[1] == 2:
        # m["y_true"] is already class-encoded (validate_arch)
        yv = (np.asarray(m["y_true"]) == 1).astype(int)
        pv = m["oof_proba"][:, 1]
        thr_t, _f1, _j = modeling.best_f1_threshold(yv, pv)
        thr = float(thr_t)
        calibrator = clin.fit_platt(yv, pv)
    return {"arch": winner["arch"], "chain": chain, "classes": classes,
            "threshold": thr, "calibrator": calibrator, "metrics": m}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="3SSE architecture search")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--mode", default="paired",
                    choices=["standard", "paired", "paired-pqn"])
    ap.add_argument("--k", type=int, default=3,
                    help="screening CV folds (default 3)")
    ap.add_argument("--top", type=int, default=20,
                    help="full-validation candidates per level")
    ap.add_argument("--jobs", type=int, default=-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default=None,
                    help="comma-separated subset for quick tests")
    ap.add_argument("--prune-pairs", type=int, default=0,
                    help="beam: evaluate triples only for the top-K pairs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    out_dir = args.out or os.path.join(APP_DIR, "study_run_3sse")
    os.makedirs(out_dir, exist_ok=True)
    if args.demo and args.mode != "standard":
        print("[3sse] demo data has no patient groups — using standard "
              "mode")
        args.mode = "standard"
    X, y, g, wn, grid, params, meta = prepare_dataset(
        args.data, args.mode, demo=args.demo)
    print(f"[3sse] {meta['n_spectra']} spectra / "
          f"{meta['n_patients']} patients · mode {args.mode}")
    model_names = ([m.strip() for m in args.models.split(",")]
                   if args.models else None)
    resume_path = os.path.join(out_dir, "archs.jsonl") \
        if args.resume else None

    def prog(done, total, best, f1, eta):
        print(f"[3sse] {done}/{total} · best {best} F1 {f1:.3f} · "
              f"ETA {eta / 60:.0f} min", flush=True)

    board = search(X, y, groups=g, wavenumbers=wn, k=args.k,
                   seed=args.seed, top=args.top, jobs=args.jobs,
                   model_names=model_names,
                   prune_pairs=args.prune_pairs,
                   resume_path=resume_path, progress_cb=prog)
    with open(os.path.join(out_dir, "screening.jsonl"), "w",
              encoding="utf-8") as fh:
        for key in ("singles", "pairs", "triples"):
            for entry in board[key]:
                if "metrics" in entry:
                    fh.write(json.dumps({
                        "level": entry["level"],
                        "arch": list(entry["arch"]),
                        **{m: v for m, v in entry["metrics"].items()
                           if m != "cm"}}) + "\n")

    validated: dict[int, list] = {}
    if not args.skip_validation:
        validated = validate_top(board, X, y, g, wn, top=args.top,
                                 seed=args.seed,
                                 model_names=model_names,
                                 progress=lambda m: print(f"[3sse] {m}",
                                                          flush=True))

    baseline = ({"Extra Trees (paired baseline)": (0.702, 0.788)}
                if args.mode.startswith("paired")
                else {"Peak bands + RF (std baseline)": (0.594, 0.610)})
    report = build_report(board, validated, baseline)
    print(report)
    with open(os.path.join(out_dir, "report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(os.path.join(out_dir, "run_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"seed": args.seed, "mode": args.mode, "k": args.k,
                   "top": args.top, **meta}, fh, indent=2)

    winner = pick_overall(validated)
    if winner is not None:
        try:
            fin = finalize_winner(validated, X, y, g, wn, board=board,
                                  k=args.k, seed=args.seed)
            extras = ({"calibrator": fin["calibrator"]}
                      if fin["calibrator"] else {})
            winner_ns = SimpleNamespace(
                name="3SSE: " + _fmt_arch(winner["arch"]),
                pipeline=fin["chain"], classes=fin["classes"],
                threshold=fin["threshold"],
                macro={"f1": (fin["metrics"]["f1"], 0.0),
                       "sens": (fin["metrics"]["sens"], 0.0),
                       "spec": (fin["metrics"]["spec"], 0.0)})
            modeling.save_bundle(
                os.path.join(out_dir, "winner.joblib"), winner_ns, grid,
                params, dataset_name=args.data,
                paired=args.mode in ("paired", "paired-pqn"), **extras)
            print(f"[3sse] winner bundle: {_fmt_arch(winner['arch'])}")
        except Exception as exc:
            print(f"[3sse] winner bundle not saved: {exc}")
    print("[3sse] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
