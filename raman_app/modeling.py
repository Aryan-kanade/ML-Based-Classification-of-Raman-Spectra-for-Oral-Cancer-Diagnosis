"""
modeling.py — model suite, leakage-free stratified cross-validation with
per-class sensitivity / specificity / F1, binary threshold optimization,
auto-selection of the best model, and persistence.

Strategy used to MAXIMIZE sensitivity / specificity / F1:
  * nested tuning: per outer fold, hyperparameters are chosen with an inner
    GridSearchCV on the training part only (no leakage),
  * binary problems: the decision threshold is tuned on an inner holdout of
    each training fold to maximize F1 (Youden's J also recorded),
  * the final winner is the model with the best cross-validated macro-F1
    (sensitivity / specificity as tie-breakers), then refit on all data.
"""

from __future__ import annotations

import traceback
from collections import Counter
from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.base import (BaseEstimator, ClassifierMixin, TransformerMixin,
                          clone)
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# PLS-DA classifier wrapper
# --------------------------------------------------------------------------
class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """PLS-DA: PLSRegression on one-hot targets, class = argmax of scores."""

    def __init__(self, n_components: int = 5):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Y = np.zeros((len(y), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0
        nc = int(min(self.n_components, len(y), X.shape[1], Y.shape[1]))
        nc = max(1, nc)
        self.pls_ = PLSRegression(n_components=nc, scale=False)
        self.pls_.fit(X, Y)
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_scores(X), axis=1)]

    def decision_scores(self, X):
        return self.pls_.predict(X)

    def predict_proba(self, X):
        s = self.decision_scores(X)
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(np.clip(s, -50, 50))
        return p / p.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Calibrated SVM (sklearn >= 1.9 deprecates SVC(probability=True))
# --------------------------------------------------------------------------
class CalibratedSVC(BaseEstimator, ClassifierMixin):
    """SVC with predict_proba; calibration folds adapt to small classes."""

    def __init__(self, C=1.0, gamma="scale", class_weight="balanced",
                 random_state=RANDOM_STATE):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.calibration import CalibratedClassifierCV
        self.classes_ = np.unique(y)
        base = SVC(C=self.C, gamma=self.gamma,
                   class_weight=self.class_weight,
                   random_state=self.random_state)
        counts = np.bincount(np.searchsorted(self.classes_, y),
                             minlength=len(self.classes_))
        cv = int(min(3, counts.min()))
        if cv >= 2 and len(self.classes_) >= 2:
            skf = StratifiedKFold(n_splits=cv, shuffle=True,
                                  random_state=self.random_state)
            self.model_ = CalibratedClassifierCV(
                base, method="sigmoid", ensemble=False, cv=skf).fit(X, y)
        else:  # tiny data: fall back to decision_function transforms
            self.model_ = base.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)
        d = self.model_.decision_function(X)
        if d.ndim == 1:
            p = 1.0 / (1.0 + np.exp(-np.clip(d, -500, 500)))
            return np.column_stack([1.0 - p, p])
        e = np.exp(np.clip(d - d.max(axis=1, keepdims=True), -500, 500))
        return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Isolation Forest classifier (one-vs-rest anomaly scoring)
# --------------------------------------------------------------------------
class IsolationForestOvR(BaseEstimator, ClassifierMixin):
    """
    Isolation Forest used as a classifier: one anomaly detector is fitted
    per class; a sample is assigned to the class whose detector considers
    it least anomalous (highest score_samples).  predict_proba comes from
    a temperature-scaled softmax over the per-class normality scores
    (monotone per class, so ROC / threshold ranking is preserved).
    """

    def __init__(self, n_estimators: int = 200,
                 random_state: int = RANDOM_STATE):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.ensemble import IsolationForest
        self.classes_ = np.unique(y)
        self.forests_ = {}
        for i, c in enumerate(self.classes_):
            forest = IsolationForest(
                n_estimators=self.n_estimators, contamination="auto",
                n_jobs=-1, random_state=self.random_state)
            forest.fit(X[y == c])
            self.forests_[i] = forest
        return self

    def _scores(self, X):
        return np.column_stack(
            [self.forests_[i].score_samples(X)
             for i in range(len(self.classes_))])

    def predict(self, X):
        return self.classes_[np.argmax(self._scores(X), axis=1)]

    def predict_proba(self, X):
        s = self._scores(X)
        e = np.exp(np.clip((s - s.max(axis=1, keepdims=True)) * 20.0,
                           -500, 500))
        return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
# Peak-band feature extractor (interpretable biochemistry)
# --------------------------------------------------------------------------
class PeakIntensityFeatures(BaseEstimator, TransformerMixin):
    """
    Mean intensity in +/- `width` cm-1 windows around biochemically known
    Raman bands — a 16-feature, human-readable summary of a spectrum.

    Bands (cm-1): 480 glycogen · 525 S–S proteins · 640 C–S · 725 DNA/RNA
    · 830/880 collagen · 1003 phenylalanine · 1095 phosphate · 1130 C–C
    lipids · 1205/1240 amide III · 1335 nucleic acids · 1450 CH2/CH3
    lipids · 1555 tryptophan · 1580 nucleic acids · 1655 amide I.
    """

    CENTERS = np.array([480, 525, 640, 725, 830, 880, 1003, 1095, 1130,
                        1205, 1240, 1335, 1450, 1555, 1580, 1655])

    def __init__(self, wn=None, width: float = 30.0):
        self.wn = wn
        self.width = width

    def fit(self, X, y=None):
        if self.wn is None:
            raise ValueError("PeakIntensityFeatures needs the wavenumber "
                             "axis (wn=...).")
        wn = np.asarray(self.wn, dtype=float)
        X = np.asarray(X)
        if X.ndim == 2 and len(wn) != X.shape[1]:
            raise ValueError(
                f"PeakIntensityFeatures: wn has {len(wn)} points but X "
                f"has {X.shape[1]} features — pass the wavenumber axis "
                "that matches the (possibly cropped) feature columns.")
        self.masks_ = [np.abs(wn - c) <= self.width
                       for c in self.CENTERS]
        if not all(m.any() for m in self.masks_):
            raise ValueError("The wavenumber range does not cover all "
                             "Raman bands.")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([X[:, m].mean(axis=1) for m in self.masks_])


# --------------------------------------------------------------------------
def model_specs() -> list[dict]:
    """Candidate models: {name, estimator, grid}."""
    specs = [
        {
            "name": "PCA + SVM (RBF)",
            "estimator": Pipeline([
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                ("clf", CalibratedSVC()),
            ]),
            "grid": {"clf__C": [1, 10, 100],
                     "clf__gamma": ["scale", 0.01]},
        },
        {
            "name": "PCA + LDA",
            "estimator": Pipeline([
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                ("clf", LinearDiscriminantAnalysis()),
            ]),
            "grid": [
                {"clf__solver": ["lsqr"], "clf__shrinkage": [None, "auto"]},
                {"clf__solver": ["svd"], "clf__shrinkage": [None]},
            ],
        },
        {
            "name": "PCA + Logistic Regression",
            "estimator": Pipeline([
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                ("clf", LogisticRegression(max_iter=5000,
                                           class_weight="balanced",
                                           random_state=RANDOM_STATE)),
            ]),
            "grid": {"clf__C": [0.1, 1, 10]},
        },
        {
            "name": "PCA + KNN",
            "estimator": Pipeline([
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                ("clf", KNeighborsClassifier()),
            ]),
            "grid": {"clf__n_neighbors": [3, 5, 7],
                     "clf__weights": ["uniform", "distance"]},
        },
        {
            "name": "PCA + Gaussian Naive Bayes",
            "estimator": Pipeline([
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                ("clf", GaussianNB()),
            ]),
            "grid": {"clf__var_smoothing": [1e-9, 1e-7]},
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(
                n_estimators=250, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1),
            "grid": {"max_depth": [None, 12],
                     "min_samples_leaf": [1, 3]},
        },
        {
            "name": "Extra Trees",
            "estimator": ExtraTreesClassifier(
                n_estimators=250, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1),
            "grid": {"max_depth": [None, 12],
                     "min_samples_leaf": [1, 3]},
        },
        {
            "name": "Hist Gradient Boosting",
            "estimator": Pipeline([
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                ("clf", HistGradientBoostingClassifier(
                    learning_rate=0.1, class_weight="balanced",
                    random_state=RANDOM_STATE)),
            ]),
            "grid": {"clf__max_iter": [100, 300]},
        },
        {
            "name": "PLS-DA",
            "estimator": Pipeline([
                ("sc", StandardScaler()),
                ("clf", PLSDAClassifier()),
            ]),
            "grid": {"clf__n_components": [2, 3, 5, 8]},
        },
        {
            "name": "PCA + MLP (neural net)",
            "estimator": Pipeline([
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                ("clf", MLPClassifier(hidden_layer_sizes=(64,),
                                      max_iter=1500,
                                      random_state=RANDOM_STATE)),
            ]),
            "grid": {"clf__alpha": [1e-3, 1e-1]},
        },
        {
            "name": "Isolation Forest (one-vs-rest)",
            "estimator": IsolationForestOvR(),
            "grid": {"n_estimators": [100, 300]},
        },
    ]
    if HAS_XGB:
        specs.append({
            "name": "XGBoost",
            "estimator": XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                tree_method="hist", eval_metric="mlogloss",
                random_state=RANDOM_STATE, n_jobs=-1),
            "grid": {"max_depth": [3, 5], "learning_rate": [0.05, 0.2]},
        })
    # interpretable 16-band model (needs the wavenumber axis at fit time)
    specs.append({
        "name": "Peak bands + RF",
        "needs_wn": True,
        "make": lambda wn: (
            Pipeline([
                ("peaks", PeakIntensityFeatures(wn=wn)),
                ("clf", RandomForestClassifier(
                    n_estimators=250, class_weight="balanced",
                    random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            {"clf__max_depth": [None, 8]}),
    })
    # resolved by evaluate_models AFTER the individual models: soft voting
    # over the three best cross-validated pipelines
    specs.append({
        "name": "Ensemble (top-3)",
        "estimator": None,
        "grid": {},
        "ensemble": True,
    })
    return specs


ALL_MODEL_NAMES = [s["name"] for s in model_specs()]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def class_metrics_from_cm(cm: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-class sensitivity/specificity/precision/F1 from a confusion matrix."""
    total = cm.sum()
    out = {}
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        out[str(i)] = {"sens": sens, "spec": spec, "prec": prec, "f1": f1}
    return out


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray):
    """Threshold maximizing F1 (and Youden's J) for score -> positive class."""
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best = (0.5, -1.0, -1.0)  # (thr, f1, j)
    for t in thresholds:
        pred = scores >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        tn = int(np.sum(~pred & (y_true == 0)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        j = sens + spec - 1.0
        if f1 > best[1] or (f1 == best[1] and j > best[2]):
            best = (float(t), f1, j)
    return best


@dataclass
class ModelResult:
    name: str
    classes: list[str]
    per_class: dict = field(default_factory=dict)     # class -> metric -> (mean, std)
    macro: dict = field(default_factory=dict)         # metric -> (mean, std)
    cm: np.ndarray | None = None                      # pooled confusion matrix
    oof_proba: np.ndarray | None = None               # out-of-fold probabilities
    y_true_encoded: np.ndarray | None = None
    best_params: Counter = field(default_factory=Counter)
    thresholds: list = field(default_factory=list)    # binary only
    threshold: float | None = None                    # binary only (median)
    fold_f1: list = field(default_factory=list)       # macro-F1 per fold/repeat
    groups: list | None = None                        # eval group (patient) per row
    pipeline: object = None                           # refit on all data
    error: str | None = None

    def macro_f1(self) -> float:
        return self.macro.get("f1", (0.0, 0.0))[0]

    def summary_row(self) -> list[str]:
        def ms(key):
            m, s = self.macro.get(key, (float("nan"), 0.0))
            return f"{m:.3f} \u00b1 {s:.3f}"
        return [self.name, ms("sens"), ms("spec"), ms("f1")]


# --------------------------------------------------------------------------
# Cross-validated evaluation of the full suite
# --------------------------------------------------------------------------
def _encode(y: list[str], classes: list[str]) -> np.ndarray:
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut[v] for v in y], dtype=int)


def _tune_hyperparams(estimator, grid, Xtr, ytr, min_class: int, groups=None):
    """Inner GridSearchCV; returns (best_estimator_fitted_on_Xtr, best_params).

    When patient `groups` are given the inner search is grouped too, so no
    subject appears in both train and validation of a tuning fold.
    """
    inner_cv = int(min(3, min_class))
    n_groups = len(set(groups)) if groups is not None else None
    if inner_cv < 2 or not grid or (groups is not None
                                    and (n_groups or 0) < 2):
        est = clone(estimator).fit(Xtr, ytr)
        return est, {}
    if groups is not None and n_groups >= inner_cv:
        cv = StratifiedGroupKFold(n_splits=inner_cv, shuffle=True,
                                  random_state=RANDOM_STATE)
        gs = GridSearchCV(clone(estimator), grid, cv=cv,
                          scoring="f1_macro", n_jobs=-1)
        gs.fit(Xtr, ytr, groups=groups)
    else:
        gs = GridSearchCV(clone(estimator), grid, cv=inner_cv,
                          scoring="f1_macro", n_jobs=-1)
        gs.fit(Xtr, ytr)
    best = clone(estimator).set_params(**gs.best_params_).fit(Xtr, ytr)
    return best, gs.best_params_


def fit_maybe_grouped(est, X, y, groups=None):
    """fit(X, y, groups=...) when the estimator's fit accepts `groups`
    (our StackedEnsemble does; plain sklearn estimators do not).  Uses
    signature inspection — no try/except, so real fit errors surface."""
    if groups is not None:
        import inspect
        try:
            params = inspect.signature(est.fit).parameters
        except (TypeError, ValueError):
            params = {}
        if "groups" in params:
            return est.fit(X, y, groups=groups)
    return est.fit(X, y)


class StackedEnsemble:
    """
    Top-k base pipelines + a logistic meta-learner stacked on their
    GROUPED inner out-of-fold probabilities (no patient leaks between
    the meta-training folds).  Clone-safe: get_params/set_params follow
    the sklearn protocol, so `run_cv` can clone and refit it per fold.
    """

    def __init__(self, specs, seed: int = RANDOM_STATE, k_inner: int = 3):
        # params stored UNMODIFIED (identity) so sklearn clone accepts us
        self.specs = specs          # [(name, estimator template), ...]
        self.seed = seed
        self.k_inner = k_inner

    # --- sklearn protocol -------------------------------------------------
    def get_params(self, deep: bool = True):
        return {"specs": self.specs, "seed": self.seed,
                "k_inner": self.k_inner}

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self

    # --- fitting ----------------------------------------------------------
    def _meta_features(self, X, fitted):
        X = np.asarray(X)
        n_cls = len(self.classes_)
        meta = np.zeros((len(X), len(fitted) * n_cls))
        for j, est in enumerate(fitted):
            meta[:, j * n_cls:(j + 1) * n_cls] = est.predict_proba(X)
        return meta

    def fit(self, X, y, groups=None):
        from sklearn.linear_model import LogisticRegression

        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        groups = np.asarray(groups) if groups is not None else None
        n_cls = len(self.classes_)
        n_groups = len(set(groups.tolist())) if groups is not None else 0
        inner_k = int(min(self.k_inner, len(set(y.tolist()))))
        if groups is not None:
            inner_k = int(min(inner_k, n_groups))
        if inner_k >= 2:
            if groups is not None and n_groups >= inner_k:
                inner = StratifiedGroupKFold(
                    n_splits=inner_k, shuffle=True,
                    random_state=self.seed)
                splits = inner.split(X, y, groups)
            else:
                from sklearn.model_selection import StratifiedKFold
                inner = StratifiedKFold(n_splits=inner_k, shuffle=True,
                                        random_state=self.seed)
                splits = inner.split(X, y)
            meta = np.zeros((len(y), len(self.specs) * n_cls))
            for tr, te in splits:
                for j, (_nm, est) in enumerate(self.specs):
                    m = fit_maybe_grouped(clone(est), X[tr], y[tr],
                                          groups[tr] if groups is not None
                                          else None)
                    meta[te, j * n_cls:(j + 1) * n_cls] = \
                        m.predict_proba(X[te])
        else:
            # too little data for inner CV: fit bases on all, meta on
            # their (in-sample) probabilities — weaker, flagged by the
            # small fold counts in the results anyway
            fitted = [clone(est).fit(X, y) for _nm, est in self.specs]
            meta = self._meta_features(X, fitted)
        self.meta_ = LogisticRegression(max_iter=2000)
        self.meta_.fit(meta, y)
        self.bases_ = [clone(est).fit(X, y) for _nm, est in self.specs]
        return self

    def predict_proba(self, X):
        return self.meta_.predict_proba(self._meta_features(X, self.bases_))

    def predict(self, X):
        return self.meta_.predict(self._meta_features(X, self.bases_))


def evaluate_models(X: np.ndarray, y: list[str],
                    model_names: list[str] | None = None,
                    k_folds: int = 5, seed: int = RANDOM_STATE,
                    progress_cb=None,
                    groups: list[str] | None = None,
                    repeats: int = 1,
                    wavenumbers: np.ndarray | None = None
                    ) -> tuple[list[ModelResult], ModelResult]:
    """
    Evaluate all selected models with stratified k-fold CV.
    Returns (results, winner). winner = best macro-F1 (sens, spec tiebreak).

    When `groups` (e.g. patient IDs, one per spectrum) are given, every
    outer fold and every inner hyperparameter/threshold split is
    GROUP-stratified: all spectra of one subject stay on the same side of
    a split — required for paired clinical data (same subject in more
    than one class), otherwise CV scores leak.

    `repeats` > 1 repeats the whole CV with different fold shuffles and
    pools the folds — the winner is picked on the pooled mean, which is
    far less fold-luck-dependent.

    "Ensemble (top-3)" is resolved after the individual models: a soft
    voting classifier over the three best cross-validated pipelines.
    """
    classes = sorted(set(y))
    if len(classes) < 2:
        raise ValueError("Need at least 2 distinct classes to train.")
    counts = Counter(y)
    min_class = min(counts.values())
    if min_class < 2:
        raise ValueError(
            f"Every class needs >= 2 samples (smallest class has {min_class}).")
    k = int(np.clip(k_folds, 2, min(10, min_class)))
    if groups is not None:
        if len(groups) != len(y):
            raise ValueError("groups must align with y (one per spectrum).")
        k = int(np.clip(k, 2, len(set(groups))))  # <= distinct subjects
    ye = _encode(y, classes)
    binary = len(classes) == 2
    pos_idx = 1 if binary else None  # positive class = sorted classes[1]
    repeats = int(max(1, repeats))

    # all CV splits across repeats (a different shuffle each repeat)
    all_splits: list = []
    for rep in range(repeats):
        rseed = seed + rep * 100
        if groups is not None:
            outer = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                         random_state=rseed)
            all_splits.extend(outer.split(X, ye, np.asarray(groups)))
        else:
            skf = StratifiedKFold(n_splits=k, shuffle=True,
                                  random_state=rseed)
            all_splits.extend(skf.split(X, ye))
    n_folds_total = len(all_splits)

    specs = [s for s in model_specs()
             if model_names is None or s["name"] in model_names]
    ens_spec = next((s for s in specs if s.get("ensemble")), None)
    if ens_spec and len(specs) == 1:
        raise ValueError("The ensemble needs the individual models "
                         "selected too.")
    if not specs:
        raise ValueError("No models selected.")

    results: list[ModelResult] = []
    base_specs: list[dict] = []
    for spec in specs:
        if spec.get("ensemble"):
            continue
        if spec.get("needs_wn"):
            if wavenumbers is None:
                res = ModelResult(name=spec["name"], classes=classes)
                res.error = ("needs the wavenumber axis — pass "
                             "wavenumbers=... to evaluate_models")
                results.append(res)
                continue
            est, grid = spec["make"](wavenumbers)
            base_specs.append({"name": spec["name"], "estimator": est,
                               "grid": grid})
        else:
            base_specs.append(spec)
    total_work = len(specs) * n_folds_total
    done = 0

    # ------------------------------------------------------------------ CV
    def run_cv(name: str, estimator, grid) -> tuple[ModelResult, tuple]:
        """One model through every fold; returns (result, template)."""
        nonlocal done
        res = ModelResult(name=name, classes=classes)
        fold_metrics: list[dict[str, dict[str, float]]] = []
        cm_total = np.zeros((len(classes), len(classes)), dtype=int)
        # OOF probabilities pooled (averaged) across repeats, matching the
        # pooled confusion matrices — a plain `oof[te] = proba` would keep
        # only the last repeat's probabilities
        oof_sum = np.zeros((len(ye), len(classes)))
        oof_cnt = np.zeros(len(ye))
        param_list: list[dict] = []
        param_counter: Counter = Counter()
        best_tpl: tuple | None = None
        try:
            for fold_i, (tr, te) in enumerate(all_splits, start=1):
                if progress_cb:
                    progress_cb(
                        int(100 * done / total_work),
                        f"{name}: fold {fold_i}/{n_folds_total}"
                        + (" (grouped)" if groups is not None else ""))
                Xtr, ytr, Xte, yte = X[tr], ye[tr], X[te], ye[te]
                groups_tr = (np.asarray(groups)[tr]
                             if groups is not None else None)

                min_tr = min(Counter(ytr.tolist()).values())
                params: dict = {}
                thr = None
                # inner split for hyperparams + (binary) threshold tuning
                # (grouped as well when patient groups are given)
                inner_k = int(min(3, min_tr))
                if groups_tr is not None:
                    inner_k = int(min(inner_k, len(set(groups_tr.tolist()))))
                if min_tr >= 2 and inner_k >= 2:
                    if groups_tr is not None:
                        inner = StratifiedGroupKFold(
                            n_splits=inner_k, shuffle=True,
                            random_state=seed)
                    else:
                        inner = StratifiedKFold(n_splits=inner_k,
                                                shuffle=True,
                                                random_state=seed)
                    # hyperparameters: grouped inner CV over the whole
                    # training fold (no throwaway holdout)
                    est, params = _tune_hyperparams(
                        estimator, grid, Xtr, ytr, min_tr, groups=groups_tr)
                    param_list.append(params)
                    param_counter[str(sorted(params.items()))] += 1
                    if binary:
                        # threshold: tuned on POOLED out-of-fold inner
                        # predictions, and only kept if it beats the 0.5
                        # default there (unstable tiny-holdout thresholds
                        # hurt more than they help on small datasets)
                        try:
                            from sklearn.model_selection import cross_val_predict
                            proba = cross_val_predict(
                                est, Xtr, ytr, groups=groups_tr, cv=inner,
                                method="predict_proba")
                            scores = proba[:, pos_idx]
                            yv = (ytr == pos_idx).astype(int)
                            if 0 < yv.sum() < len(yv):
                                cand, _, _ = best_f1_threshold(yv, scores)
                                pred_c = np.where(scores >= cand,
                                                  pos_idx, 1 - pos_idx)
                                pred_h = np.where(scores >= 0.5,
                                                  pos_idx, 1 - pos_idx)
                                if (f1_score(ytr, pred_c, average="macro")
                                        > f1_score(ytr, pred_h,
                                                   average="macro")):
                                    thr = float(cand)
                        except Exception:
                            thr = None
                        res.thresholds.append(thr)
                else:
                    est, params = _tune_hyperparams(
                        estimator, grid, Xtr, ytr, min_tr, groups=groups_tr)
                    param_list.append(params)
                    param_counter[str(sorted(params.items()))] += 1

                # refit best hyperparams on the whole training fold
                final = fit_maybe_grouped(
                    clone(estimator).set_params(**params), Xtr, ytr,
                    groups_tr)
                proba = final.predict_proba(Xte)
                oof_sum[te] += proba
                oof_cnt[te] += 1
                if binary and thr is not None:
                    pred = np.where(proba[:, pos_idx] >= thr, pos_idx, 1 - pos_idx)
                else:
                    # classes encoded 0..C-1, proba columns follow that order
                    pred = np.argmax(proba, axis=1)

                cm = confusion_matrix(yte, pred, labels=range(len(classes)))
                cm_total += cm
                fold_metrics.append(class_metrics_from_cm(cm))
                res.fold_f1.append(float(f1_score(yte, pred,
                                                  average="macro")))
                done += 1
            _aggregate(res, fold_metrics, cm_total)
            oof = np.full((len(ye), len(classes)), np.nan)
            seen = oof_cnt > 0
            oof[seen] = oof_sum[seen] / oof_cnt[seen, None]
            res.oof_proba = oof
            res.y_true_encoded = ye
            res.groups = list(groups) if groups is not None else None
            if binary and res.thresholds:
                valid = [t for t in res.thresholds if t is not None]
                res.threshold = float(np.median(valid)) if valid else None
            # refit on all data with the most frequently chosen hyperparams
            if param_list:
                best_key = param_counter.most_common(1)[0][0]
                best_params = next(p for p in param_list
                                   if str(sorted(p.items())) == best_key)
                res.pipeline = fit_maybe_grouped(
                    clone(estimator).set_params(**best_params), X, ye,
                    np.asarray(groups) if groups is not None else None)
                best_tpl = (estimator, best_params)
        except Exception:
            res.error = traceback.format_exc()
            if progress_cb:
                progress_cb(int(100 * done / total_work), f"{name}: FAILED")
        return res, best_tpl

    pipelines: dict[str, tuple] = {}
    for spec in base_specs:
        res, tpl = run_cv(spec["name"], spec["estimator"], spec["grid"])
        if tpl:
            pipelines[spec["name"]] = tpl
        results.append(res)

    # soft-voting ensemble over the three best individual models, plus a
    # stacked variant (logistic meta-learner over grouped inner-OOF base
    # probabilities) — both compete honestly under the same grouped CV
    if ens_spec is not None and len(pipelines) >= 2:
        from sklearn.ensemble import VotingClassifier
        top = sorted((r for r in results if r.error is None),
                     key=lambda r: r.macro_f1(), reverse=True)[:3]
        voters = [(f"m{i}", clone(pipelines[r.name][0]).set_params(
                       **pipelines[r.name][1]))
                  for i, r in enumerate(top)]
        ens = VotingClassifier(estimators=voters, voting="soft", n_jobs=1)
        res, _ = run_cv(ens_spec["name"], ens, {})
        results.append(res)
        try:
            stack = StackedEnsemble(
                [(nm, clone(est)) for nm, est in voters], seed=seed)
            res_s, _ = run_cv("Stacked (top-3)", stack, {})
            results.append(res_s)
        except Exception:
            results.append(ModelResult(name="Stacked (top-3)",
                                       classes=classes,
                                       error=traceback.format_exc()))

    ok = [r for r in results if r.error is None]
    if not ok:
        raise RuntimeError("All models failed:\n" +
                           "\n".join(f"{r.name}: {r.error}" for r in results))
    winner = max(ok, key=lambda r: (r.macro_f1(),
                                    r.macro.get("sens", (0, 0))[0],
                                    r.macro.get("spec", (0, 0))[0]))
    if progress_cb:
        progress_cb(100, f"Best model: {winner.name} "
                         f"(macro-F1 {winner.macro_f1():.3f})")
    return results, winner


# named biochemical Raman bands for interpretability annotations
BAND_NAMES = {
    782: "DNA/RNA phosphate", 788: "DNA", 830: "collagen", 853: "tyrosine",
    880: "tryptophan", 938: "collagen", 1003: "phenylalanine",
    1032: "phenylalanine", 1095: "phosphate / DNA", 1130: "C–C lipids",
    1209: "collagen", 1240: "amide III", 1335: "nucleic acids",
    1450: "CH2 lipids/proteins", 1555: "tryptophan", 1580: "nucleic acids",
    1615: "tyrosine/tryptophan", 1655: "amide I",
}


def _band_name(center: float) -> str:
    if not BAND_NAMES:
        return ""
    best = min(BAND_NAMES, key=lambda c: abs(c - center))
    return (f" ({BAND_NAMES[best]})" if abs(best - center) <= 25 else "")


def region_importance_shap(X, y, groups, wn, seed: int = RANDOM_STATE,
                           smooth: int = 21, n_bands: int = 5):
    """
    SIGNED explainability: TreeSHAP values of a RandomForest mapped back
    onto the wavenumber axis (Contreras et al. 2024 pattern).  Returns
    (wn, signed_importance, bands) where positive importance pushes the
    prediction toward the positive class (sorted classes[1]) and bands
    are (center_cm1, share, name) of the top contributing regions.
    """
    from scipy.ndimage import uniform_filter1d

    classes_ = sorted(set(y))
    ye = _encode(list(y), classes_)
    est = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1,
        random_state=seed).fit(np.asarray(X), ye)

    import shap
    expl = shap.TreeExplainer(est)
    sv = expl.shap_values(np.asarray(X), check_additivity=False)
    if isinstance(sv, list):                       # classic list layout
        sv_pos = sv[1] if len(classes_) == 2 else sv[-1]
    else:                                          # (n, features, classes)
        sv = np.asarray(sv)
        sv_pos = sv[..., -1]
    signed = np.asarray(sv_pos).mean(axis=0)       # mean signed SHAP
    wn = np.asarray(wn, dtype=float)
    smooth = int(max(1, min(smooth, len(signed))))
    signed_s = uniform_filter1d(signed, smooth)

    magnitude = np.abs(signed_s)
    band_mass = magnitude * (wn[1] - wn[0] if len(wn) > 1 else 1.0)
    order = np.argsort(band_mass)[::-1]
    picked: list[int] = []
    min_gap = max(1, smooth)
    for idx in order:
        if all(abs(int(idx) - p) >= min_gap for p in picked):
            picked.append(int(idx))
        if len(picked) >= n_bands:
            break
    total = band_mass.sum() if band_mass.sum() > 0 else 1.0
    bands = [(float(wn[i]), float(band_mass[i] / total),
              _band_name(float(wn[i])), float(np.sign(signed_s[i])))
             for i in sorted(picked)]
    return wn, signed_s, bands


def region_importance(X, y, groups, wn, seed: int = RANDOM_STATE,
                      smooth: int = 21, n_bands: int = 5):
    """
    Which spectral regions drive the classification?

    Fits a RandomForest on the (cropped) features, maps its feature
    importances back onto the wavenumber axis, smooths them, and finds
    the top discriminative bands.  Returns (wn, importance, bands) with
    bands = list of (center_cm1, width_cm1, importance_share).

    NOTE: importance is a descriptive diagnostic (what a tree model uses),
    not a causal claim.
    """
    from scipy.ndimage import uniform_filter1d

    est = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", n_jobs=-1,
        random_state=seed).fit(np.asarray(X), _encode(list(y),
                                                      sorted(set(y))))
    imp = np.asarray(est.feature_importances_, dtype=float)
    wn = np.asarray(wn, dtype=float)
    smooth = int(max(1, min(smooth, len(imp))))
    imp_s = uniform_filter1d(imp, smooth)
    # top bands: local maxima of the smoothed curve, ranked by mass
    band_mass = imp_s * (wn[1] - wn[0] if len(wn) > 1 else 1.0)
    order = np.argsort(band_mass)[::-1]
    picked: list[int] = []
    min_gap = max(1, smooth)
    for idx in order:
        if all(abs(int(idx) - p) >= min_gap for p in picked):
            picked.append(int(idx))
        if len(picked) >= n_bands:
            break
    total = band_mass.sum() if band_mass.sum() > 0 else 1.0
    bands = [(float(wn[i]), float((wn[1] - wn[0]) * smooth if len(wn) > 1
                                  else smooth),
              float(band_mass[i] / total)) for i in sorted(picked)]
    return wn, imp_s, bands


def bootstrap_ci(y_true, y_pred, n: int = 1000, alpha: float = 0.05,
                 seed: int = RANDOM_STATE, groups=None):
    """95% bootstrap confidence interval for macro-F1 -> (lo, hi).

    When `groups` (one id per sample, e.g. patient) is given, the resample
    is drawn over GROUPS so all spectra of a patient move together —
    spectrum-level resampling on grouped data would understate the CI
    width (pseudo-replication).
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups)
        rows_by = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        n_blocks = len(rows_by)
    scores = []
    for _ in range(n):
        if groups is None:
            idx = rng.integers(0, len(y_true), len(y_true))
        else:
            idx = np.concatenate(
                [rows_by[b] for b in rng.integers(0, n_blocks, n_blocks)])
        if len(set(y_true[idx])) < 2:
            continue                      # degenerate resample
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    if not scores:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lo, hi


def mcnemar_test(y_true, pred_a, pred_b):
    """
    Exact McNemar test: are models A and B significantly different on the
    same samples?  Returns (b, c, p) where b = A right/B wrong,
    c = A wrong/B right.
    """
    from math import comb

    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return b, c, min(1.0, p)


def evaluate_pipeline(X_raw: np.ndarray, wn: np.ndarray, y: list[str],
                      groups: list[str] | None = None, k: int = 5,
                      seed: int = RANDOM_STATE, progress=None
                      ) -> dict:
    """
    UNBIASED end-to-end evaluation: inside every outer patient-grouped
    fold the PREPROCESSING is re-chosen on the training patients only
    (mini optimizer grid), then the best model is scored on the held-out
    patients.  Removes the selection bias of picking preprocessing on
    the same data it is scored on.

    Returns {mean_f1, std_f1, fold_f1s, fold_choices}.
    """
    from sklearn.metrics import f1_score
    import optimize
    import preprocessing as pp

    # compact grid: the three configurations that dominated the full
    # optimization runs (crop alone / crop+deriv / no crop)
    grid = [dict(crop_min=400.0, crop_max=1800.0, sg_deriv=0,
                 norm="vector"),
            dict(crop_min=400.0, crop_max=1800.0, sg_deriv=1,
                 norm="vector"),
            dict(crop_min=0.0, crop_max=0.0, sg_deriv=0, norm="vector")]
    model_by_name = dict(optimize._models())

    splitter = (StratifiedGroupKFold(n_splits=k, shuffle=True,
                                     random_state=seed) if groups
                is not None else StratifiedKFold(n_splits=k, shuffle=True,
                                                 random_state=seed))
    y_arr = np.asarray(y)
    g_arr = np.asarray(groups) if groups is not None else None
    fold_f1s, fold_choices = [], []
    for fi, (tr, te) in enumerate(splitter.split(X_raw, y_arr, g_arr),
                                  start=1):
        if progress:
            progress(f"nested fold {fi}/{k}: choosing preprocessing on "
                     "training patients…")
        sub = optimize.optimize_preprocessing(
            X_raw[tr], wn, [y[i] for i in tr],
            groups=(None if g_arr is None else g_arr[tr].tolist()),
            grid=grid, k=3, seed=seed, progress=lambda m: None)
        params, model_name = sub["params"], sub["best"]["model"]
        Xtr = pp.preprocess_matrix(X_raw[tr], params, wn=wn)
        Xte = pp.preprocess_matrix(X_raw[te], params, wn=wn)
        est = clone(model_by_name[model_name])
        est.fit(Xtr, y_arr[tr])
        f1 = f1_score(y_arr[te], est.predict(Xte), average="macro")
        fold_f1s.append(float(f1))
        fold_choices.append({"fold": fi, "preprocess": sub["best"]["label"],
                             "model": model_name, "f1": float(f1)})
        if progress:
            progress(f"nested fold {fi}/{k}: {sub['best']['label']} + "
                     f"{model_name} -> F1 {f1:.3f}")
    return {"mean_f1": float(np.mean(fold_f1s)),
            "std_f1": float(np.std(fold_f1s)),
            "fold_f1s": fold_f1s, "fold_choices": fold_choices}


def learning_curve_by_groups(X, y_encoded, groups, estimator,
                             k: int = 5, seed: int = RANDOM_STATE,
                             fractions=(0.25, 0.50, 0.75, 1.0)):
    """
    Grouped-CV macro-F1 at a growing number of patients — the classic
    diagnostic for 'how much would more data be worth?'.

    Returns (n_patients_list, mean_f1_list, std_f1_list).
    """
    from sklearn.metrics import f1_score

    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    sizes, means, stds = [], [], []
    for frac in fractions:
        take = uniq[:max(2, int(round(len(uniq) * frac)))]
        m = np.isin(groups, take)
        Xs, ys, gs = X[m], np.asarray(y_encoded)[m], groups[m]
        kk = int(min(k, len(take)))
        if kk < 2 or min(Counter(ys.tolist()).values()) < 2:
            continue                      # not enough patients for k folds
        sgkf = StratifiedGroupKFold(n_splits=kk, shuffle=True,
                                    random_state=seed)
        f1s = []
        for tr, te in sgkf.split(Xs, ys, gs):
            est = clone(estimator)
            est.fit(Xs[tr], ys[tr])
            f1s.append(f1_score(ys[te], est.predict(Xs[te]),
                                average="macro"))
        sizes.append(len(take))
        means.append(float(np.mean(f1s)))
        stds.append(float(np.std(f1s)))
    return sizes, means, stds


def _aggregate(res: ModelResult, fold_metrics, cm_total):
    keys = ["sens", "spec", "prec", "f1"]
    n_cls = cm_total.shape[0]
    res.cm = cm_total
    agg = {k: [] for k in keys}
    for ci in range(n_cls):
        per = {k: (float(np.mean([fm[str(ci)][k] for fm in fold_metrics])),
                   float(np.std([fm[str(ci)][k] for fm in fold_metrics])))
               for k in keys}
        res.per_class[res.classes[ci]] = per
        for k in keys:
            agg[k].append(per[k][0])
    res.macro = {k: (float(np.mean(agg[k])), float(np.std(agg[k])))
                 for k in keys}


# --------------------------------------------------------------------------
# Persistence + prediction
# --------------------------------------------------------------------------
def save_bundle(path: str, winner: ModelResult, wavenumbers: np.ndarray,
                prep_params, dataset_name: str = "",
                paired: bool = False, **extra) -> str:
    bundle = {
        "model_name": winner.name,
        "pipeline": winner.pipeline,
        "classes": winner.classes,
        "threshold": winner.threshold,
        "wavenumbers": np.asarray(wavenumbers),
        "prep_params": prep_params,
        "macro": winner.macro,
        "dataset_name": dataset_name,
        "paired": paired,          # predict-time needs a normal reference
    }
    cal = extra.get("calibrator")
    if (cal and len(bundle["classes"]) == 2
            and bundle["threshold"] is not None):
        # the stored threshold must live in the SAME (calibrated)
        # probability space as predict-time outputs
        import clinical as clin
        bundle["threshold"] = clin.apply_platt(bundle["threshold"], cal)
    bundle.update(extra)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str) -> dict:
    return joblib.load(path)


def predict_with_bundle(bundle: dict, wavenumbers: np.ndarray,
                        intensities: np.ndarray,
                        reference: np.ndarray | None = None) -> dict:
    """Predict one raw spectrum (any wavenumber order) with a saved bundle.

    `reference` (preprocessed, cropped vector) subtracts a paired
    normal-reference before prediction — required for bundles trained in
    paired mode.  When the bundle carries a Platt `calibrator`, the
    probabilities (and the stored threshold, which is saved in
    CALIBRATED units) are mapped through it first.
    """
    from preprocessing import (PreprocessParams, crop_mask,
                               preprocess_spectrum)

    def _as_params(p):
        # bundles store PreprocessParams, but tolerate plain dicts at this
        # trust boundary (hand-made or future writers)
        return p if hasattr(p, "validate") else PreprocessParams(**p)

    order = np.argsort(wavenumbers)
    wn, it = np.asarray(wavenumbers)[order], np.asarray(intensities)[order]
    grid = bundle["wavenumbers"]
    y = np.interp(grid, wn, it)
    params = _as_params(bundle["prep_params"])
    # apply the crop range exactly as at training time (the model expects
    # the cropped feature count); fall back to the uncropped vector for
    # bundles that were trained without cropping
    m = crop_mask(grid, params)
    proba = None
    for vec in ((y[m], y) if int(m.sum()) != len(grid) else (y,)):
        xc = preprocess_spectrum(vec, params)
        if reference is not None:
            if len(reference) != len(xc):
                raise ValueError(
                    f"paired reference has {len(reference)} points but "
                    f"the spectrum has {len(xc)} after preprocessing")
            xc = xc - reference
        try:
            proba = bundle["pipeline"].predict_proba(xc.reshape(1, -1))[0]
            break
        except ValueError:
            if reference is not None:
                raise                    # paired mismatch is a real error
            continue
    if proba is None:
        raise ValueError(
            "Spectrum length does not match the saved model "
            f"(expected {int(m.sum())} or {len(grid)} points).")
    classes = list(bundle["classes"])
    cal = bundle.get("calibrator")
    if cal and len(classes) == 2:
        import clinical as clin
        p1 = clin.apply_platt(proba[1], cal)
        proba = np.array([1.0 - p1, p1])
    thr = bundle.get("threshold")
    if len(classes) == 2 and thr is not None:
        pos = 1
        pred = classes[pos] if proba[pos] >= thr else classes[1 - pos]
    else:
        pred = classes[int(np.argmax(proba))]
    return {"prediction": pred,
            "probabilities": {c: float(p) for c, p in zip(classes, proba, strict=True)},
            "threshold": thr}


def roc_points(y_true_encoded: np.ndarray, proba_pos: np.ndarray):
    """ROC curve + AUC helper (for plotting)."""
    fpr, tpr, _ = roc_curve(y_true_encoded, proba_pos)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr, fpr))
    return fpr, tpr, auc
