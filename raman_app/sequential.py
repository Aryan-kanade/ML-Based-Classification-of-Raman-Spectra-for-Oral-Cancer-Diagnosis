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
    triple's best reachable MEAN per-fold macro-F1 is (sum_i + (k-i))/k;
    below the current top-N cutoff (also expressed as mean per-fold F1,
    `metrics["f1_mean"]`) it cannot enter the validated set, so the
    remaining folds are skipped (the abandon bound exactly upper-bounds
    the mean-fold metric the ranking uses; pooled-OOF F1 is reported for
    display but is NOT prunable).  Exactness caveat: with the default
    beam (--prune-pairs 50) triples whose base pair ranks below the
    top-50 pairs are never evaluated, so the triple top-N is exact only
    within the beam, not globally;
  * parallel across pairs (joblib), single-threaded inside estimators;
  * append-only JSONL checkpoint; --resume skips finished triples.

Hyperparameters stay at the registry defaults during screening —
identical treatment across all 4,369 architectures keeps the level
comparison fair (per-layer nested grids x 4,369 would be prohibitive).

Usage:
    python sequential.py --mode paired --out study_run_3sse
        (--data defaults to the auto-detected dataset root — see
         clinical_data.find_data_root; override with --data or the
         RAMAN_DATA_DIR environment variable)
    python sequential.py --models "PCA + LDA,Random Forest" --k 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clinical as clin                      # noqa: E402
import clinical_data as cdata                # noqa: E402
import modeling                              # noqa: E402
from modeling import fit_maybe_grouped       # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# auto-detected per device — no hardcoded machine path to edit when the
# project moves to another computer (see clinical_data.find_data_root)
DEFAULT_DATA = cdata.find_data_root() or os.path.join(APP_DIR, "..", "Data")
# models skipped by the GUI's "fast screening" toggle (slowest first)
SLOW_MODELS = ("1D-CNN", "CatBoost", "XGBoost")
# study reference baselines (2026-08-30 run, seed 42, 5-fold grouped ×3)
# — ONE home for the numbers so the CLI report, the GUI winner tab and
# the HTML export can never drift apart; shown as "study reference",
# not as a per-dataset result
BASELINES = {
    "paired": ("paired Extra Trees", 0.702, 0.788),
    "standard": ("Peak bands + RF", 0.594, 0.610),
}


def baseline_for(mode: str):
    """(name, f1, auc) study baseline for a run mode — paired modes get
    the paired reference, everything else the standard one."""
    return BASELINES["paired" if "paired" in (mode or "") else "standard"]


class SearchCancelled(Exception):
    """Raised inside search() when cancel_check() turns true.  The JSONL
    checkpoint (if any) keeps everything finished so far."""


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
        # clone + RAMAN_DEVICE guard: the parent builds estimators with GPU
        # params baked in; 3SSE loky workers force them back to CPU (one
        # CUDA context per child process would blow VRAM)
        return lambda: modeling.as_env_device(modeling.clone(est))

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
        if n in WN_ALIGNED_MODELS and wn is not None:
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


# wn-aligned models (their band/peak features index the spectral axis,
# so at chained positions — where probability columns are appended — they
# must be wrapped to use the spectral columns only)
WN_ALIGNED_MODELS = ("Peak bands + RF", "Spectral + band features")


def _layer_est(name: str, factories: dict, position: int, wn):
    """
    Estimator for a chain position.  WN-aligned models consume the
    wavenumber-aligned axis, so at chained positions (>= 1, where the
    input carries appended probability columns) they are wrapped to use
    the spectral columns only (spec §9: model-specific features stay
    intact; probability chaining remains available to the others).
    """
    est = factories[name]()
    if position >= 1 and wn is not None and name in WN_ALIGNED_MODELS:
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
    from sklearn.metrics import (confusion_matrix, roc_auc_score)
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
    # supplementary metrics (2026-09-08 formula audit) — ADDITIVE keys
    # on the same pooled OOF; old screening.jsonl rows simply lack them
    if len(classes) == 2 and 0 < ye.sum() < len(ye):
        p1 = np.nan_to_num(proba[:, 1])
        out["bal_acc"] = modeling.balanced_accuracy_from_cm(cm)
        out["mcc"] = modeling.mcc_from_cm(cm)
        out["brier"] = modeling.brier_score(ye, p1)
        out["pr_auc"] = float(modeling.pr_points(ye, p1)[2])
    else:
        out["bal_acc"] = float("nan")
        out["mcc"] = float("nan")
    if groups is not None:                       # patient-level rollup
        g = np.asarray(groups)
        # patient truth = DOMINANT class of their spectra (not the first
        # row — paired patients legitimately carry both classes and row
        # order is arbitrary; fixed 2026-09-05)
        uniq_p = np.unique(g)
        p_true, p_pred = [], []
        n_tied = 0
        for pat in uniq_p:
            lab = ye[g == pat]
            vals, cnts = np.unique(lab, return_counts=True)
            if len(vals) == 2 and cnts[0] == cnts[1]:
                n_tied += 1        # tie-break to the lowest code (audit)
            p_true.append(int(vals[np.argmax(cnts)]))
            p_pred.append(int(np.argmax(proba[g == pat].mean(axis=0))))
        pcm = confusion_matrix(p_true, p_pred,
                               labels=range(len(classes)))
        pper = modeling.class_metrics_from_cm(pcm)
        out["pat_acc"] = float(np.trace(pcm) / max(pcm.sum(), 1))
        out["pat_f1"] = float(np.mean([pper[c]["f1"] for c in pper]))
        out["pat_sens"] = float(np.mean([pper[c]["sens"] for c in pper]))
        out["pat_spec"] = float(np.mean([pper[c]["spec"] for c in pper]))
        out["pat_tied"] = n_tied
    return out


def _screen_f1(metrics: dict) -> float:
    """Ranking key for PRUNED screening levels (triples): the mean
    per-fold F1 the abandon bound actually upper-bounds.  Falls back to
    pooled F1 for records that predate the field (old checkpoints)."""
    return float(metrics.get("f1_mean", metrics.get("f1", 0.0)))


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
    # y arrives PRE-ENCODED as ints (search() encodes once at the
    # boundary).  Re-encoding against the string class list cast the
    # ints to '0'/'1' strings that sort BEFORE every alphabetic label,
    # making ye all-zeros — every f1_mean in level-3 screening was
    # computed against a constant truth (2026-09-12 audit; fixed).
    ye = np.asarray(y, dtype=int)
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
    # Screen/prune metric: MEAN of the completed per-fold macro-F1s.
    # The early-abandon bound below upper-bounds exactly this quantity;
    # pooled-OOF F1 (metrics["f1"], F1 of the summed confusion matrix) is
    # NOT bounded by any fold-mean, so ranking pruned candidates on it
    # made "top-N stays exact" false (fixed 2026-09-05).
    metrics["f1_mean"] = float(np.mean(fold_f1))
    return metrics, None


def _pin_threads():
    """One thread per worker process (avoid oversubscription + memory
    pressure); CPU-only torch — one CUDA context per loky child would
    blow VRAM."""
    if modeling.HAS_TORCH:
        modeling.torch.set_num_threads(1)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    os.environ.setdefault("RAMAN_DEVICE", "cpu")


# --------------------------------------------------------------------------
# parallel workers (self-contained args: no module globals needed)
# --------------------------------------------------------------------------
def _worker_single(task):
    """Evaluate one single model: OOF probabilities + metrics.
    Errors are RETURNED (name, None, None, msg) — one broken model must
    never kill the whole screening run (CatBoost 'bad allocation',
    2026-09-05)."""
    _pin_threads()
    name, X, y, groups, classes, k, seed, factories = task
    try:
        P1 = _oof_proba(factories[name](), X, y, groups, k, seed)
        m = _metrics_from_oof(y, P1, classes, groups)
        return name, P1.astype(np.float32), m, None
    except Exception as exc:
        return name, None, None, f"{type(exc).__name__}: {exc}"


def _worker_pair(task):
    """Evaluate one ordered pair: B on [X + OOF P1_A]."""
    _pin_threads()
    (A, B), X, y, groups, classes, k, seed, p1, factories, wn = task
    try:
        feats = np.hstack([X, p1])
        est = _layer_est(B, factories, 1, wn)
        P2 = _oof_proba(est, feats, y, groups, k, seed)
        metrics = _metrics_from_oof(y, P2, classes, groups)
        return (A, B), metrics, P2.astype(np.float32), None
    except Exception as exc:
        return (A, B), None, None, f"{type(exc).__name__}: {exc}"


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
        try:
            factory = (lambda nm=C: _layer_est(nm, factories, 2, wn))
            metrics, pruned = _eval_last_layer(
                factory, feats, y, classes, groups, k, seed, cutoff)
            results.append(((A, B, C), metrics, pruned, None))
        except Exception as exc:
            results.append(((A, B, C), None, None,
                            f"{type(exc).__name__}: {exc}"))
    return results


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------
def _available_gb() -> float:
    """Available physical RAM in GB (Windows via ctypes GlobalMemoryStatusEx;
    other platforms / any failure -> conservative 8.0)."""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = _MS(dwLength=ctypes.sizeof(_MS))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return float(ms.ullAvailPhys) / (1024 ** 3)
    except Exception:
        pass
    return 8.0


def _safe_jobs(requested: int) -> int:
    """
    Resolve a joblib-style job count (negative = CPUs+N) and size it to
    the AVAILABLE RAM: every loky child imports the full modeling stack
    (torch + CatBoost/XGBoost/LightGBM, ~1.5-2 GB each) — a fixed CPU
    count OOM'd a 16 GB laptop with ~7.7 GB free (CatBoostError 'bad
    allocation' / TerminatedWorkerError at the pairs loop, 2026-09-05).
    Budget 2 GB per worker, hard ceiling 4, floor 1.
    """
    cpus = os.cpu_count() or 2
    n = (cpus + 1 + requested) if requested < 0 else requested
    by_ram = int(_available_gb() / 2.0)
    return max(1, min(int(n), 4, max(1, by_ram)))


@contextmanager
def _child_cpu_only():
    """Hide the GPU from loky children for the duration: torch (cu
    build) and CatBoost each probe/init CUDA at import inside every
    child, and on a busy 6 GB card those probes OOM'd whole runs
    (CUDA error 2 aborts, 2026-09-05 evening). The parent's own CUDA
    context is already initialized and unaffected by the env change;
    restore the previous value afterwards."""
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if _cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _cvd


def search(X, y, groups=None, wavenumbers=None, k: int = 3, seed: int = 42,
           top: int = 20, jobs: int = -2, model_names: list | None = None,
           prune_pairs: int = 50, cnn_epochs: int = 15,
           resume_path: str | None = None, progress_cb=None,
           cancel_check=None, leaderboard_cb=None) -> dict:
    """(see _search_impl) — thin wrapper: loky children spawn with the
    GPU HIDDEN (_child_cpu_only), so torch/CatBoost never probe CUDA
    inside a worker even when the parent's VRAM is busy."""
    with _child_cpu_only():
        return _search_impl(X, y, groups=groups, wavenumbers=wavenumbers,
                            k=k, seed=seed, top=top, jobs=jobs,
                            model_names=model_names,
                            prune_pairs=prune_pairs,
                            cnn_epochs=cnn_epochs, resume_path=resume_path,
                            progress_cb=progress_cb,
                            cancel_check=cancel_check,
                            leaderboard_cb=leaderboard_cb)


def _search_impl(X, y, groups=None, wavenumbers=None, k: int = 3,
                 seed: int = 42, top: int = 20, jobs: int = -2,
                 model_names: list | None = None, prune_pairs: int = 50,
                 cnn_epochs: int = 15, resume_path: str | None = None,
                 progress_cb=None, cancel_check=None,
                 leaderboard_cb=None) -> dict:
    """
    QUICK search over all architectures (single-level OOF stacking,
    grouped CV).  Returns {"singles", "pairs", "triples", "total",
    "pruned"} where each entry is {"arch", "level", "metrics"} (or
    {"arch", "level", "pruned_bound"} for early-abandoned triples).

    Speed engine (2026-09-05 program):
      * SUCCESSIVE-HALVING FIDELITY — big spaces screen at 2-fold
        (k_s = min(k, 2)); the nested validate_top() re-ranks the top-N
        at full fidelity, so screening noise never reaches the winner.
      * BEAM (prune_pairs, default 50) — triples only for the top-K
        pairs (~8x fewer third-model evaluations).
      * EARLY-ABANDON cutoff from completed triples' f1_mean — the
        bound guarantees top-N exactness within the beam.  (A WARM
        cutoff seeded from the pairs' pooled F1 was removed 2026-09-12:
        it mixed metric scales and could prune valid triples.)
      * CHECKPOINTED at every level (singles/pairs carry their OOF
        arrays; resume replays them instead of refitting).
      * joblib memmapping (max_nbytes) shares X across workers once.

    jobs is clamped by _safe_jobs (≤4 workers — more OOMs the machine).

    cancel_check: zero-arg callable; when it returns True the search
    stops promptly with SearchCancelled (the JSONL checkpoint keeps
    everything finished so far for --resume).
    leaderboard_cb: optional callback receiving the current top-5
    [(arch_str, f1), ...] at batch boundaries.
    """
    from joblib import Parallel, delayed

    # parent-process pin: factories wrap estimators with as_env_device(),
    # which caps CatBoost thread_count/GPU and XGB/LGBM/RF n_jobs — but
    # only under RAMAN_DEVICE=cpu.  Without this the GUI/CLI process
    # itself fit CatBoost with ALL cores -> native OOM "bad allocation"
    # that killed whole runs (2026-09-05).  Power users: set the env var
    # to "gpu" BEFORE starting to keep GPU CatBoost.
    os.environ.setdefault("RAMAN_DEVICE", "cpu")

    jobs = _safe_jobs(jobs)

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

    def check_cancel():
        if cancel_check is not None and cancel_check():
            raise SearchCancelled()

    def top5_of(pool):
        scored = sorted((e for e in pool if "metrics" in e),
                        key=lambda e: -_screen_f1(e["metrics"]))[:5]
        # display the metric the ranking actually uses (_screen_f1:
        # f1_mean for triples, pooled f1 otherwise) — the old display
        # always showed pooled f1, so order and number disagreed
        return [((" → ".join(e["arch"])), _screen_f1(e["metrics"]))
                for e in scored]

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
    if resume_path:        # parent may not exist yet (GUI resume path)
        os.makedirs(os.path.dirname(resume_path) or ".", exist_ok=True)
    ckpt = open(resume_path, "a", encoding="utf-8") \
        if resume_path else None

    def record(arch, metrics, pruned, level: int = 3, oof=None):
        if not ckpt:
            return
        rec = {"arch": list(arch), "level": level,
               "pruned": pruned is not None,
               **({"bound": round(float(pruned), 4)}
                  if pruned is not None
                  else {m: (round(float(v), 4)
                            if isinstance(v, float) and v == v else v)
                        for m, v in metrics.items() if m != "cm"})}
        if oof is not None and pruned is None:
            rec["oof"] = np.asarray(oof, dtype=np.float32).tolist()
        ckpt.write(json.dumps(rec) + "\n")
        ckpt.flush()

    def record_error(arch, level: int, msg: str):
        """A failed architecture is still 'done' (resume must not retry
        it) — recorded with an `error` field, skipped by every
        ranking/cutoff/winner filter."""
        if ckpt:
            ckpt.write(json.dumps({"arch": list(arch), "level": level,
                                   "error": msg[:300]}) + "\n")
            ckpt.flush()
        print(f"3SSE: {_fmt_arch(tuple(arch))} failed — {msg}")

    # successive-halving: big spaces screen at 2-fold — validate_top()
    # re-ranks the survivors at full fidelity, so cheap screening noise
    # never reaches the winner
    k_s = min(k, 2) if total > 600 else k

    def _metrics_from_rec(rec):
        return {m: v for m, v in rec.items()
                if m not in ("arch", "level", "pruned", "oof")}

    # ---- level 1: singles (OOF kept as layer-1 features) ---------------
    singles, oof1 = [], {}
    done_names: set[str] = set()
    for rec in resume.values():            # replay checkpointed singles
        if rec.get("level") == 1 and len(rec["arch"]) == 1 \
                and rec["arch"][0] in names:
            name = rec["arch"][0]
            done_names.add(name)
            done[0] += 1
            if rec.get("error"):
                continue               # failed before: don't retry
            singles.append({"arch": (name,), "level": 1,
                            "metrics": _metrics_from_rec(rec)})
            if "oof" in rec:
                oof1[name] = np.asarray(rec["oof"], dtype=np.float32)
    pending = [n for n in names if n not in done_names]
    batch1 = max(1, jobs) * 2
    for i in range(0, len(pending), batch1):
        check_cancel()
        chunk = pending[i:i + batch1]
        for name, P1, m, err in Parallel(n_jobs=jobs, max_nbytes=100)(
                delayed(_worker_single)(
                    (name, X, y, groups, classes, k_s, seed, factories))
                for name in chunk):
            if m is None:
                record_error((name,), 1, err)
                done[0] += 1
                continue
            oof1[name] = P1
            singles.append({"arch": (name,), "level": 1, "metrics": m})
            record((name,), m, None, level=1, oof=P1)
            done[0] += 1
            tick(name, m["f1"])
    if leaderboard_cb:
        leaderboard_cb(top5_of(singles))

    # model 17 resolved from the singles ranking (spec §10) whenever it
    # was requested (full run or an explicit checkbox subset)
    if model_names is None or "Ensemble (top-3)" in model_names:
        top3 = [s["arch"][0] for s in
                sorted(singles, key=lambda s: -s["metrics"]["f1"])][:3]
        factories["Ensemble (top-3)"] = _ensemble_factory(top3,
                                                          factories,
                                                          wn=wavenumbers)
        try:                    # parent-process fit — guarded like workers
            P1 = _oof_proba(factories["Ensemble (top-3)"](), X, y, groups,
                            k_s, seed)
        except Exception as exc:
            record_error(("Ensemble (top-3)",), 1,
                         f"{type(exc).__name__}: {exc}")
        else:
            names = names + ["Ensemble (top-3)"]
            total = _check_space(names)          # now includes model 17
            oof1["Ensemble (top-3)"] = P1.astype(np.float32)
            m = _metrics_from_oof(y, P1, classes, groups)
            singles.append({"arch": ("Ensemble (top-3)",), "level": 1,
                            "metrics": m})
            record(("Ensemble (top-3)",), m, None, level=1, oof=P1)
            done[0] += 1
            tick("Ensemble (top-3)", m["f1"])

    # ---- level 2: ordered pairs (checkpointed, memmapped) --------------
    pairs, p2 = [], {}
    done_pairs: set[str] = set()
    for rec in resume.values():            # replay checkpointed pairs
        if rec.get("level") == 2 and len(rec["arch"]) == 2 \
                and set(rec["arch"]) <= set(names):
            key = tuple(rec["arch"])
            done_pairs.add(" → ".join(rec["arch"]))
            done[0] += 1
            if rec.get("error"):
                continue               # failed before: don't retry
            pairs.append({"arch": key, "level": 2,
                          "metrics": _metrics_from_rec(rec)})
            if "oof" in rec:
                p2[key] = np.asarray(rec["oof"], dtype=np.float32)
    pair_permutations = [ab for ab in itertools.permutations(names, 2)
                         if " → ".join(ab) not in done_pairs]
    for A, B in pair_permutations:
        if A not in oof1:      # its level-1 OOF failed — pair impossible
            record_error((A, B), 2, f"level-1 features unavailable "
                                    f"({A} failed)")
            done_pairs.add(" → ".join((A, B)))
            done[0] += 1
    tasks = [((A, B), X, y, groups, classes, k_s, seed, oof1[A],
              factories, wavenumbers)
             for A, B in pair_permutations
             if " → ".join((A, B)) not in done_pairs]
    batch = max(1, abs(jobs)) * 2
    for i in range(0, len(tasks), batch):
        check_cancel()
        for key, metrics, P2, err in Parallel(n_jobs=jobs, max_nbytes=100)(
                delayed(_worker_pair)(t) for t in tasks[i:i + batch]):
            if metrics is None:
                record_error(key, 2, err)
                done[0] += 1
                continue
            p2[key] = P2
            pairs.append({"arch": key, "level": 2, "metrics": metrics})
            record(key, metrics, None, level=2, oof=P2)
            done[0] += 1
            tick(" → ".join(key), metrics["f1"])
        if leaderboard_cb:
            leaderboard_cb(top5_of(singles + pairs))

    # ---- level 3: pair-major, best-first, early-abandon -----------------
    triples: list[dict] = []
    name_set = set(names)
    for rec in resume.values():          # replay checkpointed triples
        if rec.get("level") == 3 and set(rec["arch"]) <= name_set:
            done[0] += 1
            if rec.get("error"):
                continue               # failed before: don't retry
            if rec.get("pruned"):
                triples.append({"arch": tuple(rec["arch"]), "level": 3,
                                "pruned_bound": rec.get("bound", 0.0)})
            else:
                triples.append({"arch": tuple(rec["arch"]), "level": 3,
                                "metrics": {m: v for m, v in rec.items()
                                            if m not in ("arch", "level",
                                                         "pruned")}})

    def cutoff_now() -> float:
        scored = [_screen_f1(t["metrics"]) for t in triples
                  if "metrics" in t]
        if len(scored) < top:
            return -1.0
        return min(sorted(scored, reverse=True)[:top])

    # single-model F1 ranking: third candidates are tried best-first so
    # the early-abandon cutoff tightens as fast as possible
    single_f1 = {s["arch"][0]: s["metrics"]["f1"] for s in singles}
    pairs_sorted = sorted(pairs, key=lambda p: -p["metrics"]["f1"])
    beam_applied = 0
    if prune_pairs and prune_pairs < len(pairs_sorted):
        beam_applied = len(pairs_sorted) - int(prune_pairs)
        pairs_sorted = pairs_sorted[:int(prune_pairs)]
    # WARM cutoff REMOVED (2026-09-12 audit): it was seeded from the
    # pairs' POOLED F1 minus 0.10 while triple pruning/ranking operates
    # on f1_mean (mean per-fold F1) — a different scale, so a triple
    # bound in f1_mean units could fall below a pooled-F1 cutoff and be
    # wrongly abandoned in the first batch.  The cutoff now starts open
    # (-1) and tightens from the first batch's actual f1_mean values via
    # cutoff_now(); the abandon bound then guarantees top-N exactness.
    cutoff = -1.0
    # beam-skipped triples count as done so progress reaches 100%
    n_names = len(names)
    done[0] += beam_applied * max(0, n_names - 2)
    chunk = max(1, abs(jobs))
    for i in range(0, len(pairs_sorted), chunk):
        check_cancel()
        batch = pairs_sorted[i:i + chunk]
        tasks = []
        for p in batch:
            A, B = p["arch"]
            cand = sorted((C for C in names
                           if C not in (A, B)
                           and " → ".join((A, B, C)) not in resume),
                          key=lambda c: -single_f1.get(c, 0.0))
            if cand:
                tasks.append(((A, B), X, y, groups, classes, k_s, seed,
                              oof1[A], p2[(A, B)], cand, factories,
                              wavenumbers, cutoff))
        for results in Parallel(n_jobs=jobs, max_nbytes=100)(
                delayed(_worker_triples)(t) for t in tasks):
            for arch, metrics, pruned, err in results:
                if err is not None:
                    record_error(arch, 3, err)
                    done[0] += 1
                elif metrics is None:
                    triples.append({"arch": arch, "level": 3,
                                    "pruned_bound": pruned})
                    record(arch, metrics, pruned, level=3)
                    done[0] += 1
                else:
                    triples.append({"arch": arch, "level": 3,
                                    "metrics": metrics})
                    record(arch, metrics, pruned, level=3)
                    done[0] += 1
        cutoff = cutoff_now()
        scored = [t for t in triples if "metrics" in t]
        if scored:
            b = max(scored, key=lambda t: _screen_f1(t["metrics"]))
            tick(" → ".join(b["arch"]), b["metrics"]["f1"])
        if leaderboard_cb:
            leaderboard_cb(top5_of(singles + pairs + triples))
    # screening at reduced fidelity: report the effective fold count
    board = {"singles": singles, "pairs": pairs, "triples": triples,
             "total": total, "pruned": sum(1 for t in triples
                                           if "pruned_bound" in t),
             "screen_folds": k_s}
    if ckpt:
        ckpt.close()
    return board


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
            except Exception as exc:
                # A tiny/degenerate inner fold must NOT fall back to
                # IN-SAMPLE probabilities: those leak the training rows
                # into the chaining features and inflate this fold's
                # OOF metrics — the overall winner could then be picked
                # partly on a leaky fold (fixed 2026-09-05).  Fail the
                # architecture instead; the error-tolerant wrapper
                # records it and the rankings skip it.
                raise RuntimeError(
                    f"inner OOF failed for layer {name!r}: {exc}") from exc
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
        # stored VERBATIM: sklearn's clone() requires __init__ to not
        # modify params (list(...) copies failed clone's identity check
        # and killed every GUI diagnostic that clones the winner —
        # learning curve / seeds / noise / locked / LOPO, 2026-09-06).
        # Mutation-safety comes from fit() cloning each layer anyway.
        self.estimators = estimators
        self.k = k
        self.seed = seed

    # -- sklearn protocol --------------------------------------------------
    def get_params(self, deep=True):
        return {"estimators": self.estimators, "k": self.k,
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


class AveragedChain:
    """
    B4 (2026-09-05 program): the deployable winner refit at several
    seeds whose predict_proba is the SEED-AVERAGE — the cheap
    robustness boost for fold-dependent chains.  Speaks the sklearn
    protocol so bundles / Platt / threshold / predict keep working.
    """

    def __init__(self, estimators: list, k: int = 3, seed: int = 42,
                 n_seeds: int = 3):
        # stored VERBATIM (sklearn clone contract — see SequentialChain)
        self.estimators = estimators
        self.k = k
        self.seed = seed
        self.n_seeds = n_seeds

    def get_params(self, deep=True):
        return {"estimators": self.estimators, "k": self.k,
                "seed": self.seed, "n_seeds": self.n_seeds}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y, groups=None):
        self.fitted_ = []
        for i in range(self.n_seeds):
            chain = SequentialChain(
                [modeling.clone(e) for e in self.estimators],
                k=self.k, seed=self.seed + i).fit(X, y, groups=groups)
            self.fitted_.append(chain)
        self.classes_ = self.fitted_[0].classes_
        return self

    def predict_proba(self, X):
        probas = [c.predict_proba(X) for c in self.fitted_]
        return np.mean(probas, axis=0)

    def predict(self, X):
        return np.asarray(self.classes_)[
            np.argmax(self.predict_proba(X), axis=1)]


def tune_chain(arch, X, y, groups, wavenumbers, seed: int = 42,
               cnn_epochs: int = 40, progress=None) -> dict:
    """
    B1 of the 2026-09-05 Speed&Accuracy program: greedy PER-LAYER
    hyperparameter tuning of a chain.  Layer i is grid-tuned with the
    project's grouped inner CV (modeling._tune_hyperparams — the same
    machinery evaluate_models uses) on the features it actually sees:
    X plus the TUNED previous layers' grouped-OOF probabilities, i.e.
    the exact stacking context the chain runs in.  Leakage-free: each
    layer is tuned on its own features only.

    Returns {"estimators": [unfitted, tuned, clone-safe per layer],
             "params": [best-params dict per layer]}.
    """
    X = np.asarray(X, dtype=np.float32)
    classes = sorted(set(y))
    ye = np.searchsorted(np.asarray(classes), np.asarray(y)).tolist()
    min_class = int(min(np.bincount(np.asarray(ye))))
    specs = {s["name"]: s for s in modeling.model_specs()}
    factories = _factories(wavenumbers, cnn_epochs)
    g = list(groups) if groups is not None else None
    tuned, params = [], []
    feats = X
    for pos, name in enumerate(arch):
        if progress:
            progress(f"tuning chain layer {pos + 1}/{len(arch)}: {name}")
        # WN-aligned models at chained positions tune AND deploy on the
        # spectral columns only — feats carries appended probability
        # columns that band/peak features cannot read (same wrap rule as
        # _layer_est; before 2026-09-05 this was missing here and made
        # tuning of any band-feature chain raise -> silently skipped).
        wn_aligned = (pos >= 1 and wavenumbers is not None
                      and name in WN_ALIGNED_MODELS)
        tune_feats = (feats[:, :len(wavenumbers)] if wn_aligned else feats)
        spec = specs.get(name)
        est = grid = None
        if spec is not None and not spec.get("ensemble"):
            if spec.get("needs_wn"):
                if wavenumbers is not None:
                    est, grid = spec["make"](np.asarray(wavenumbers))
            else:
                est, grid = spec["estimator"], spec.get("grid")
        if est is None or not grid:
            # 1D-CNN (empty grid by design) / Ensemble — keep defaults
            est_t, bp = _layer_est(name, factories, pos, wavenumbers), {}
        else:
            _fitted, bp = modeling._tune_hyperparams(
                modeling.as_env_device(modeling.clone(est)), grid,
                tune_feats, ye, min_class, g)
            base = (modeling.as_env_device(
                modeling.clone(est).set_params(**bp)) if bp else
                    _layer_est(name, factories, pos, wavenumbers))
            est_t = _sliced(base, len(wavenumbers)) if wn_aligned else base
        tuned.append(est_t)
        params.append(bp)
        if pos < len(arch) - 1:
            P = _oof_proba(modeling.clone(est_t), feats, ye, g,
                           min(3, min_class), seed)
            feats = np.hstack([feats, P])
    return {"estimators": tuned, "params": params}


# --------------------------------------------------------------------------
# dataset preparation (mirrors reproduce_study — GUI parity)
# --------------------------------------------------------------------------
def prepare_dataset(root: str, mode: str = "standard"):
    """Returns (X, y, groups, wavenumbers, grid, params, meta)."""
    import dataset as ds
    import preprocessing as pp
    from clinical_data import load_clinical_dataset

    params = pp.PreprocessParams().validate()
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


def build_report(board: dict, validated: dict, baseline: dict,
                 significance: dict | None = None) -> str:
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
    if significance:
        f1s = significance["seed_f1s"]
        mean = sum(f1s) / len(f1s)
        sd = (sum((x - mean) ** 2 for x in f1s)
              / max(len(f1s) - 1, 1)) ** 0.5
        verdict = ("SIGNIFICANT" if significance["mcnemar_p"] < 0.05
                   else "NOT significant")
        lines += ["", "SIGNIFICANCE",
                  f"  vs best single ({significance['baseline']}, F1 "
                  f"{significance['baseline_f1']:.3f}): McNemar "
                  f"b={significance['mcnemar_b']} "
                  f"c={significance['mcnemar_c']} -> "
                  f"p={significance['mcnemar_p']:.3f} — the winner's "
                  f"improvement is {verdict}.",
                  f"  Seed stability ({len(f1s)} seeds): F1 "
                  f"{mean:.3f} ± {sd:.3f} "
                  f"({', '.join(f'{x:.3f}' for x in f1s)})"]
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
def _worker_validate(task):
    """Nested-validate one architecture (validate_top's pool worker)."""
    _pin_threads()
    arch, factories, X, y, groups, seed, k_outer, wn = task
    try:
        m = validate_arch(arch, factories, X, y, groups, seed=seed,
                          k_outer=k_outer, wn=wn)
        return arch, m, None
    except Exception as exc:
        return arch, None, f"{type(exc).__name__}: {exc}"


def validate_top(board: dict, X, y, groups, wavenumbers, top: int = 20,
                 seed: int = 42, k_outer: int = 5, cnn_epochs: int = 40,
                 model_names: list | None = None, jobs: int = -2,
                 progress=None) -> dict[int, list]:
    """Nested full validation of the top-N screening architectures per
    level.  Returns {level: [{"arch", "metrics"}]}.

    The architectures are independent — they validate in a joblib
    process pool (clamped by _safe_jobs, CPU-pinned workers; jobs=1
    keeps the old serial loop).  Results are collected back into the
    original screening-rank order, so the returned structure is
    identical to the serial version."""
    factories = _factories(wavenumbers, cnn_epochs)
    if model_names is None and any(
            e["arch"][0] == "Ensemble (top-3)" for e in board["singles"]):
        top3 = [e["arch"][0] for e in sorted(
            (s for s in board["singles"]
             if s["arch"][0] != "Ensemble (top-3)"),
            key=lambda s: -s["metrics"]["f1"])][:3]
        factories["Ensemble (top-3)"] = _ensemble_factory(
            top3, factories, wn=wavenumbers)
    cand: dict[int, list] = {}
    for level, key in ((1, "singles"), (2, "pairs"), (3, "triples")):
        archs = [tuple(n for n in e["arch"] if n in factories)
                 for e in sorted((e for e in board[key] if "metrics" in e),
                                 key=lambda e: -_screen_f1(
                                     e["metrics"]))[:top]]
        cand[level] = [a for a in archs if len(a) == level]
    tasks = [(a, factories, X, y, groups, seed, k_outer, wavenumbers)
             for level in (1, 2, 3) for a in cand[level]]
    by_arch: dict = {}
    jobs = _safe_jobs(jobs)
    if jobs > 1:
        from joblib import Parallel, delayed
        with _child_cpu_only():
            for arch, m, err in Parallel(
                    n_jobs=jobs, max_nbytes=100,
                    return_as="generator")(
                    delayed(_worker_validate)(t) for t in tasks):
                if err and progress:
                    progress(f"validation failed for "
                             f"{_fmt_arch(arch)}: {err}")
                elif progress:
                    progress(f"validated {len(arch)}-model "
                             f"{_fmt_arch(arch)}")
                if m is not None:
                    by_arch[arch] = m
    else:
        for t in tasks:
            arch, m, err = _worker_validate(t)
            if err and progress:
                progress(f"validation failed for {_fmt_arch(arch)}: {err}")
            elif progress:
                progress(f"validating {len(arch)}-model {_fmt_arch(arch)}")
            if m is not None:
                by_arch[arch] = m
    validated: dict[int, list] = {}
    for level in (1, 2, 3):
        validated[level] = [{"arch": a, "metrics": by_arch[a]}
                            for a in cand[level] if a in by_arch]
    return validated


def finalize_winner(validated: dict, X, y, groups, wavenumbers,
                    board: dict | None = None, k: int = 3, seed: int = 42,
                    cnn_epochs: int = 40, tune: bool = True,
                    progress=None) -> dict | None:
    """
    Pick the overall validated winner, fit its deployable
    SequentialChain on all data, and derive the binary threshold +
    Platt calibrator from its nested-OOF probabilities.
    tune=True (B1): first greedily grid-tune each layer of the winning
    chain on its own stacking context, nested-validate the tuned chain,
    and deploy it when it is not worse than the untuned one (both F1s
    and the per-layer best params are returned).
    Returns {"arch", "chain", "classes", "threshold", "calibrator",
             "metrics", "tuned", "tuned_metrics", "tune_params"} or None.
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
    tuned = False
    tuned_metrics = None
    tune_params: list = []
    ests = [_layer_est(n, factories, i, wavenumbers)
            for i, n in enumerate(winner["arch"])]
    if tune and len(winner["arch"]) >= 2:
        try:
            if progress:
                progress("tuning the winning chain (per layer)")
            tc = tune_chain(winner["arch"], X, y, groups, wavenumbers,
                            seed=seed, cnn_epochs=cnn_epochs,
                            progress=progress)
            ests = tc["estimators"]
            tune_params = tc["params"]
            tm = validate_arch(winner["arch"],
                               {n: (lambda e=e: modeling.clone(e))
                                for n, e in zip(winner["arch"], ests)},
                               X, y, groups, seed=seed, wn=wavenumbers)
            tuned_metrics = tm
            if tm["f1"] >= m["f1"] - 1e-9:
                tuned = True          # deploy the tuned chain
            else:
                ests = [_layer_est(n, factories, i, wavenumbers)
                        for i, n in enumerate(winner["arch"])]
        except Exception as exc:
            if progress:
                progress(f"chain tuning skipped: {exc}")
            ests = [_layer_est(n, factories, i, wavenumbers)
                    for i, n in enumerate(winner["arch"])]
    chain = AveragedChain(ests, k=k, seed=seed,
                          n_seeds=3).fit(X, y, groups=groups)
    classes = sorted(set(y))
    # threshold + calibrator derive from the UNTUNED nested OOF: the
    # tuned variant's OOF is selection-biased (tune_chain tuned on all
    # rows, then validate_arch scored the same rows), so its
    # probabilities would give an optimistic threshold/calibrator
    # (2026-09-12 audit).  tuned_metrics is flagged accordingly.
    oof_m = m
    thr, calibrator = None, None
    if oof_m["oof_proba"].shape[1] == 2:
        # oof_m["y_true"] is already class-encoded (validate_arch)
        yv = (np.asarray(oof_m["y_true"]) == 1).astype(int)
        pv = oof_m["oof_proba"][:, 1]
        thr_t, _f1, _j = modeling.best_f1_threshold(yv, pv)
        thr = float(thr_t)
        calibrator = clin.fit_platt(yv, pv)
    if tuned_metrics is not None:
        tuned_metrics = dict(tuned_metrics)
        tuned_metrics["selection_biased"] = True   # tuned+validated on
        tuned_metrics["selection_bias_note"] = (   # the same rows
            "hyperparameters were tuned on ALL rows before this "
            "nested validation — treat as an optimistic upper bound; "
            "the honest number is 'metrics'")
    return {"arch": winner["arch"], "chain": chain, "classes": classes,
            "threshold": thr, "calibrator": calibrator, "metrics": m,
            "tuned": tuned, "tuned_metrics": tuned_metrics,
            "tune_params": tune_params}


# --------------------------------------------------------------------------
# significance of the winner (shared by GUI worker, CLI and back-fills)
# --------------------------------------------------------------------------
def significance_of(validated, winner, board, X, y, groups, wavenumbers,
                    seed: int = 42, extra_seeds: int = 2,
                    cnn_epochs: int = 15, progress=None) -> dict | None:
    """
    Is the chain winner really better than the best single model?
    Exact McNemar on the two architectures' nested-OOF predictions
    (identical outer folds + seed -> a valid paired comparison) plus a
    small seed-stability rerun of the winner.  Returns a dict (see the
    GUI winner tab) or None when not applicable (1-model winner,
    multi-class, missing OOF).
    """
    try:
        if winner is None or len(winner["arch"]) < 2:
            return None
        singles = validated.get(1) or []
        if not singles:
            return None
        best_single = max(singles, key=lambda e: e["metrics"]["f1"])
        wm, sm = winner["metrics"], best_single["metrics"]
        if (wm.get("oof_proba") is None
                or sm.get("oof_proba") is None
                or wm["oof_proba"].shape[1] != 2):
            return None
        yt = np.asarray(wm["y_true"])
        pw = np.argmax(wm["oof_proba"], axis=1)
        ps = np.argmax(sm["oof_proba"], axis=1)
        b, c, p = modeling.mcnemar_test(yt, pw, ps)
        facs = _factories(wavenumbers, cnn_epochs=cnn_epochs)
        if "Ensemble (top-3)" in winner["arch"]:
            top3 = [e["arch"][0] for e in sorted(
                (s for s in board.get("singles", [])
                 if s["arch"][0] != "Ensemble (top-3)"),
                key=lambda s: -s["metrics"]["f1"])][:3]
            facs["Ensemble (top-3)"] = _ensemble_factory(
                top3, facs, wn=wavenumbers)
        seed_f1s = [float(wm["f1"])]
        for extra in range(1, extra_seeds + 1):
            m = validate_arch(winner["arch"], facs, X, y, groups,
                              seed=seed + extra)
            seed_f1s.append(float(m["f1"]))
            if progress:
                progress(f"seed stability {seed + extra}: "
                         f"F1 {m['f1']:.3f}")
        return {"baseline": _fmt_arch(best_single["arch"]),
                "baseline_f1": float(sm["f1"]),
                "mcnemar_b": int(b), "mcnemar_c": int(c),
                "mcnemar_p": float(p), "seed_f1s": seed_f1s}
    except Exception as exc:
        if progress:
            progress(f"significance testing skipped: {exc}")
        return None


def _dumpable(m):
    """JSON-safe metrics dict (ndarrays → lists)."""
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in m.items()}


def persist_run(out_dir: str, board: dict, validated: dict, winner,
                significance: dict | None = None):
    """Write validated.json / winner.json / significance.json so the
    next app start can restore the winner into the Train page.
    (2026-09-06: validated/winner were accepted but never written — a
    CLI run left the PREVIOUS run's artifacts behind, mixing
    generations at restore time.)"""

    def _num(v):
        if isinstance(v, dict):
            return {str(k): _num(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_num(x) for x in v]
        if isinstance(v, np.ndarray):
            return v.tolist()          # oof_proba / cm / y_true in the
        if isinstance(v, np.generic):  # GUI worker's validated+winner
            v = v.item()               # metrics (2026-09-06 regression:
        if isinstance(v, float) and v != v:   # ndarray killed finished
            return None                         # searches at persist time)
        return v

    if validated:
        with open(os.path.join(out_dir, "validated.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(_num(validated), fh)
    if winner:
        wcopy = {k: _num(v) for k, v in winner.items()
                 if k not in ("chain", "calibrator")}
        with open(os.path.join(out_dir, "winner.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(wcopy, fh, indent=2)
    if significance is not None:
        with open(os.path.join(out_dir, "significance.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(significance, fh, indent=2)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="3SSE architecture search")
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="dataset root (default: auto-detected, "
                         "currently %(default)s)")
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
    ap.add_argument("--prune-pairs", type=int, default=50,
                    help="beam: evaluate triples only for the top-K "
                         "pairs (0 = all)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if not os.path.isdir(args.data):
        print(f"[3sse] data folder not found: {args.data} "
              "(pass --data PATH or set RAMAN_DATA_DIR)")
        return 2

    out_dir = args.out or os.path.join(APP_DIR, "study_run_3sse")
    os.makedirs(out_dir, exist_ok=True)
    # device summary (2026-09-08): REAL backends + mode; strict GPU mode
    # fails loudly here rather than silently running CPU
    try:
        _mode = modeling.resolve_device_mode()
    except RuntimeError as exc:
        print(f"[3sse] DEVICE ERROR: {exc}")
        return 2
    _v = modeling.verify_gpu_runtime()
    print(f"[3sse] device: mode={_v['mode']} · CNN={_v['cnn']} · "
          f"XGBoost={_v['xgboost']} · CatBoost={_v['catboost']} · "
          f"LightGBM={_v['lightgbm']} · sklearn=cpu"
          + (f" (torch probe: {_v['torch_probe_device']})"
             if _v.get("torch_probe_device") else ""))
    X, y, g, wn, grid, params, meta = prepare_dataset(args.data, args.mode)
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

    _bname, _bf1, _bauc = baseline_for(args.mode)
    baseline = {f"{_bname} (study baseline)": (_bf1, _bauc)}

    with open(os.path.join(out_dir, "validated.json"), "w",
              encoding="utf-8") as fh:
        json.dump({str(lv): [{"arch": list(e["arch"]),
                              "metrics": _dumpable(e["metrics"])}
                             for e in validated.get(lv, [])]
                   for lv in (1, 2, 3)}, fh, indent=2)
    winner = pick_overall(validated)
    sig = None
    if winner is not None:
        sig = significance_of(validated, winner, board, X, y, g, wn,
                              seed=args.seed,
                              progress=lambda m: print(f"[3sse] {m}",
                                                       flush=True))
    # winner entry for persist_run: attach finalize's threshold/calib
    persist_winner = None
    if winner is not None:
        persist_winner = {
            "arch": winner["arch"], "metrics": winner["metrics"],
            "threshold": None, "calibrator": None}
    try:
        persist_run(out_dir, board, validated, persist_winner,
                    significance=sig)
    except Exception:
        # persistence must never kill a COMPLETED search: the report,
        # run meta and winner bundle below still get written (2026-09-06)
        print(f"[3sse] WARNING: result persistence failed:\n"
              f"{traceback.format_exc()}", flush=True)
    report = build_report(board, validated, baseline, significance=sig)
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
                paired=args.mode in ("paired", "paired-pqn"),
                pqn=(args.mode == "paired-pqn"),   # deploy must PQN too
                **extras)
            print(f"[3sse] winner bundle: {_fmt_arch(winner['arch'])}")
        except Exception as exc:
            print(f"[3sse] winner bundle not saved: {exc}")
    print("[3sse] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
