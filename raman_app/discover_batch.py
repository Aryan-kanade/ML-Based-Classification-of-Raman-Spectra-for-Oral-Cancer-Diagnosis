"""
discover_batch.py -- runner for the 25-method discovery loop.

Each method = (optional stateless per-row feature map) x (estimator
factory) evaluated by sequential.validate_arch (patient-grouped nested
5-fold) at seeds 42/43/44.  Row maps are stateless (no fitting) and
fitted transforms live INSIDE the pipeline, so the harness cannot leak
test rows.  Every result lands in experiments/method_bank.jsonl
immediately; re-runs skip done ids.

Usage:  python discover_batch.py --batch A|B|C|all|report
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import exp_common
import method_bank
import sequential

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- data
_CACHE: dict = {}


def data():
    if "X" not in _CACHE:
        _arch, X, y, g, wn, _meta = exp_common.load_data()
        _cls = sorted(set(y))
        _CACHE.update(X=X, y=np.asarray([_cls.index(v) for v in y]),
                      g=list(g), wn=np.asarray(wn))
    return _CACHE["X"], _CACHE["y"], _CACHE["g"], _CACHE["wn"]


# ------------------------------------------------- stateless row maps
def _windows(wn, n=16):
    edges = np.linspace(wn[0], wn[-1], n + 1)
    return [(np.searchsorted(wn, edges[i]),
             np.searchsorted(wn, edges[i + 1])) for i in range(n)]


# ------------------------------------------------------------ QC views
_QC_CACHE: dict = {}


def qc_data(drop_pct: float = 0.0, drop_keratin: bool = False,
            drop_suspect: bool = False, min_reps: int = 0,
            ref_agg: str = "mean"):
    """d2-winner paired features on a QUALITY-FILTERED spectrum set
    (PUSH-0.8 Layer 1).  Filters apply BEFORE paired_features, so the
    patient references are recomputed on kept spectra.  drop_pct: drop
    the worst N% of spectra by spike score (fixed dataset percentiles,
    not fitted on labels).  ref_agg: normal-reference aggregation
    ('mean' matches production; 'median'/'trimmed' = robustification
    implemented by per-patient median/trimmed-mean of the kept Normal
    spectra — passed through paired_features' own reference builder is
    not switchable, so robust refs are applied by pre-averaging each
    patient's normals into ONE representative normal row (reference
    then equals that row; deviations unchanged for tumor rows)."""
    key = (drop_pct, drop_keratin, drop_suspect, min_reps, ref_agg)
    if key in _QC_CACHE:
        return _QC_CACHE[key]
    import dataset as ds
    import paired as pmod
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

    if drop_suspect:                        # eval_label_errors verdicts
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

    if ref_agg != "mean":
        # robust reference: replace each patient's Normal block by its
        # median / trimmed-mean row (one representative normal), then
        # let paired_features build references from it
        idx_by = {}
        for pos, i in enumerate(keep):
            idx_by.setdefault((cd.groups[i], labels[i].strip()),
                              []).append(pos)
        Xk = np.asarray(X_raw[keep], dtype=np.float64)
        drop_pos = set()
        for (pat, cls_), pos_list in idx_by.items():
            if cls_.lower() == "normal" and len(pos_list) > 1:
                block = Xk[pos_list]
                if ref_agg == "median":
                    rep = np.median(block, axis=0)
                else:
                    k_ = max(1, len(block) // 5)
                    rep = np.sort(block, axis=0)[k_:len(block) - k_] \
                        .mean(axis=0) if len(block) - 2 * k_ > 0 else \
                        block.mean(axis=0)
                Xk[pos_list[0]] = rep
                drop_pos.update(pos_list[1:])
        keep2 = [i for p_, i in enumerate(keep) if p_ not in drop_pos]
        y = [labels[i].strip() for i in keep2]
        g = [cd.groups[i] for i in keep2]
        Xk = Xk[[p_ for p_ in range(len(keep)) if p_ not in drop_pos]]
        pd_ = pmod.paired_features(Xk, y, g, grid, WINNER)
    else:
        pd_ = pmod.paired_features(X_raw[keep], y, g, grid, WINNER)

    out = (np.asarray(pd_.X, dtype=np.float32), list(pd_.y),
           list(pd_.groups), np.asarray(pd_.wn), len(keep))
    _QC_CACHE[key] = out
    print(f"[qc] pct={drop_pct} ker={drop_keratin} sus={drop_suspect} "
          f"reps>={min_reps} ref={ref_agg} -> {len(keep)} spectra / "
          f"{len(set(g))} patients", flush=True)
    return out


def rowmap_bands(X, wn):
    """17 trapezoid band areas + 16 window areas.  Every band MUST keep
    >= 2 grid points — _monotone_xgb builds one constraint per band, so
    a silent skip would desync the constraint vector (audit finding)."""
    import biochemistry as bio
    cols = [np.trapezoid(X[:, a:b], wn[a:b], axis=1)
            for a, b in _windows(wn)]
    n_bands = 0
    for c, hw, _m, _a, _d in bio.BANDS:
        m = (wn >= c - hw) & (wn <= c + hw)
        assert m.sum() >= 2, f"band {c} empty on grid — callers rely " \
                             f"on all 17 bands being present"
        cols.append(np.trapezoid(X[:, m], wn[m], axis=1))
        n_bands += 1
    return np.stack(cols, 1).astype(np.float32)


def rowmap_ratios(X, wn):
    """Classical band ratios + keratin/thiocyanate style markers."""
    def area(c, hw=10):
        m = (wn >= c - hw) & (wn <= c + hw)
        return np.trapezoid(X[:, m], wn[m], axis=1)
    num = {k: area(k, 12) for k in (785, 854, 938, 1090, 1335, 1445,
                                    1578, 1655)}
    den = area(1003) + 1e-6
    cols = [num[k] / den for k in num] + [
        area(1655) / (area(1240) + 1e-6), area(938) / den]
    return np.stack(cols, 1).astype(np.float32)


def rowmap_wpt(X, wn, level=4):
    """Wavelet-packet energies per node (db4)."""
    import pywt
    def one(row):
        wp = pywt.WaveletPacket(row, "db4", mode="symmetric",
                                maxlevel=level)
        return [np.sqrt(np.mean(n.data ** 2)) for n in wp.get_level(level)]
    return np.stack([one(r) for r in X]).astype(np.float32)


def rowmap_fft(X, wn):
    """Low-order |FFT| magnitudes (first 64 coefficients)."""
    f = np.abs(np.fft.rfft(X, axis=1))[:, :64]
    return f.astype(np.float32)


def rowmap_entropy(X, wn):
    """Per-window: spectral entropy + Higuchi FD + PSD slope + std."""
    def hig(v, k=10):
        n = len(v)
        ls = np.arange(1, k + 1)
        L = []
        for kk in ls:
            m = int(np.ceil(n / kk))
            Lm = 0.0
            for i in range(1, kk + 1):
                seg = v[i - 1::kk]
                if len(seg) > 1:
                    Lm += (len(seg) - 1) / (len(seg) ** 0) * np.abs(
                        np.diff(seg)).sum() / (len(seg) - 1) / kk
            L.append(Lm / m if m else 0.0)
        L = np.maximum(L, 1e-12)
        sl = np.polyfit(np.log(1.0 / ls), np.log(L), 1)[0]
        return float(sl)
    cols = []
    for a, b in _windows(wn):
        W = X[:, a:b]
        p = np.abs(W) / (np.abs(W).sum(1, keepdims=True) + 1e-12)
        ent = -(p * np.log(p + 1e-12)).sum(1)
        cols += [ent, W.std(1)]
        cols.append(np.stack([hig(r) for r in W]))
    return np.stack(cols, 1).astype(np.float32)


def rowmap_wincorr(X, wn):
    """Upper-triangle correlations between window means (2D-COS-lite)."""
    mus = np.stack([X[:, a:b].mean(1) for a, b in _windows(wn)], 1)
    mu = mus.mean(1, keepdims=True)
    sd = mus.std(1, keepdims=True) + 1e-12
    z = (mus - mu) / sd
    n = z.shape[1]
    iu = np.triu_indices(n, 1)
    return (z[:, iu[0]] * z[:, iu[1]]).astype(np.float32)


def rowmap_quantiles(X, wn):
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    cols = [np.quantile(X[:, a:b], q, axis=1)
            for a, b in _windows(wn) for q in qs]
    return np.stack(cols, 1).astype(np.float32)


# ------------------------------------------------------- estimators
def et(n=300):
    from sklearn.ensemble import ExtraTreesClassifier
    return lambda: ExtraTreesClassifier(n_estimators=n, random_state=42,
                                        n_jobs=1, class_weight="balanced")


def xgb():
    from xgboost import XGBClassifier
    return lambda: XGBClassifier(n_estimators=200, max_depth=4,
                                 learning_rate=0.1, random_state=42,
                                 n_jobs=1, eval_metric="logloss")


def pca_pipe(clf_fn, comp=0.95):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return lambda: Pipeline([("s", StandardScaler()),
                             ("p", PCA(n_components=comp,
                                       random_state=42)),
                             ("c", clf_fn())])


def svm_kernel(kernel, gamma="scale"):
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return lambda: Pipeline([
        ("s", StandardScaler()),
        ("c", CalibratedClassifierCV(
            SVC(kernel=kernel, gamma=gamma, C=1.0),
            cv=3, ensemble=False))])


def gp_pipe():
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return lambda: Pipeline([
        ("s", StandardScaler()), ("p", PCA(n_components=50,
                                           random_state=42)),
        ("c", GaussianProcessClassifier(kernel=1.0 * RBF(),
                                         random_state=42, max_iter_predict=200))])


def subspace_bag(base_fn, features=0.3):
    from sklearn.ensemble import BaggingClassifier
    return lambda: BaggingClassifier(base_fn(), n_estimators=100,
                                     max_features=features, random_state=42,
                                     n_jobs=1)


def rotation_forest_cls(n_estimators=25, sub_features=200):
    """Compact Rotation Forest: per tree, PCA on a random feature
    subset fitted on a bootstrap sample; majority of tree probabilities."""
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.decomposition import PCA
    from sklearn.tree import ExtraTreeClassifier

    class RotationForest(BaseEstimator, ClassifierMixin):
        def __init__(self, n_estimators=n_estimators,
                     sub_features=sub_features, random_state=42):
            self.n_estimators = n_estimators
            self.sub_features = sub_features
            self.random_state = random_state

        def fit(self, X, y):
            rng = np.random.default_rng(self.random_state)
            self.classes_ = np.unique(y)
            self.trees_, self.pcas_, self.cols_ = [], [], []
            for _ in range(self.n_estimators):
                boot = rng.choice(len(X), len(X), replace=True)
                cols = rng.choice(X.shape[1],
                                  min(self.sub_features, X.shape[1]),
                                  replace=False)
                p = PCA(random_state=int(rng.integers(1 << 30)))
                Z = p.fit_transform(X[np.ix_(boot, cols)])
                t = ExtraTreeClassifier(random_state=int(
                    rng.integers(1 << 30)), class_weight="balanced",
                    max_depth=None, min_samples_leaf=2)
                t.fit(Z, y[boot])
                self.cols_.append(cols)
                self.pcas_.append(p)
                self.trees_.append(t)
            return self

        def predict_proba(self, X):
            out = 0.0
            for t, p, c in zip(self.trees_, self.pcas_, self.cols_):
                out = out + t.predict_proba(p.transform(X[:, c]))
            out /= len(self.trees_)
            return out
    return lambda: RotationForest()


def mixup_et(alpha=0.3):
    """In-fold mixup: fit() accepts groups (fit_maybe_grouped) and mixes
    within-class across patients on the TRAIN rows it was given only."""
    from sklearn.base import BaseEstimator, ClassifierMixin

    class MixupET(BaseEstimator, ClassifierMixin):
        def __init__(self, alpha=alpha, n_estimators=300):
            self.alpha = alpha
            self.n_estimators = n_estimators

        def fit(self, X, y, groups=None):
            from sklearn.ensemble import ExtraTreesClassifier
            rng = np.random.default_rng(42)
            Xa = [X]
            ya = [y]
            if groups is not None:
                for cls_ in np.unique(y):
                    idx = np.where(y == cls_)[0]
                    if len(idx) < 4:
                        continue
                    for _ in range(len(idx) // 2):
                        i, j = rng.choice(idx, 2, replace=False)
                        # never mix the same patient with itself
                        if groups[i] == groups[j]:
                            continue
                        lam = rng.beta(self.alpha, self.alpha)
                        Xa.append((lam * X[i] + (1 - lam) * X[j])[None])
                        ya.append([cls_])
            Xa = np.vstack(Xa)
            ya = np.concatenate([np.atleast_1d(v) for v in ya])
            self.classes_ = np.unique(y)
            self.clf = ExtraTreesClassifier(
                n_estimators=self.n_estimators, random_state=42,
                n_jobs=1, class_weight="balanced").fit(Xa, ya)
            return self

        def predict_proba(self, X):
            return self.clf.predict_proba(X)
    return lambda: MixupET()


# ------------------------------------------------------------ batches
def batch_defs():
    """id -> (mechanism, family, rowmap|None, factory)."""
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import (
        AdaBoostClassifier, HistGradientBoostingClassifier,
        RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def qda_pipe():
        return lambda: Pipeline([("s", StandardScaler()),
                                 ("p", _pca50()),
                                 ("c", QuadraticDiscriminantAnalysis(
                                     reg_param=0.5))])

    def _pca50():
        from sklearn.decomposition import PCA
        return PCA(n_components=50, random_state=42)

    A = {  # feature space (stateless row maps)
        "MA1": ("band+window area vector -> ET", "feature",
                rowmap_bands, et()),
        "MA2": ("band-ratio/keratin markers -> ET", "feature",
                rowmap_ratios, et()),
        "MA3": ("wavelet-packet node energies -> ET", "feature",
                rowmap_wpt, et()),
        "MA4": ("FFT magnitude band -> ET", "feature",
                rowmap_fft, et()),
        "MA5": ("entropy/FD/PSD window features -> ET", "feature",
                rowmap_entropy, et()),
        "MA6": ("window correlation (2D-COS-lite) -> ET", "feature",
                rowmap_wincorr, et()),
        "MA7": ("window quantile summaries -> ET", "feature",
                rowmap_quantiles, et()),
        "MA8": ("band+ratio+quantile concat -> ET", "feature",
                lambda X, wn: np.hstack([rowmap_bands(X, wn),
                                         rowmap_ratios(X, wn),
                                         rowmap_quantiles(X, wn)]), et()),
    }
    B = {  # learners on winner features
        "MB1": ("Gaussian Process (PCA-50)", "learner", None, gp_pipe()),
        "MB2": ("Rotation Forest (PCA subsets)", "learner", None,
                rotation_forest_cls()),
        "MB3": ("random-subspace bagging ET", "learner", None,
                subspace_bag(et(150))),
        "MB4": ("SVM polynomial kernel (calibrated)", "learner", None,
                svm_kernel("poly")),
        "MB5": ("elastic-net logistic (saga)", "learner", None,
                lambda: Pipeline([("s", StandardScaler()),
                                  ("c", LogisticRegression(
                                      penalty="elasticnet", l1_ratio=0.5,
                                      C=1.0, solver="saga", max_iter=5000,
                                      class_weight="balanced"))])),
        "MB6": ("QDA on PCA-50 (regularized)", "learner", None, qda_pipe()),
        "MB7": ("GNB on PCA-50", "learner", None,
                lambda: Pipeline([("s", StandardScaler()),
                                  ("p", _pca50()),
                                  ("c", GaussianNB())])),
        "MB8": ("NCA + kNN (PCA-50)", "learner", None,
                lambda: Pipeline([("s", StandardScaler()),
                                  ("p", _pca50()),
                                  ("n", NeighborhoodComponentsAnalysis(
                                      random_state=42, max_iter=100)),
                                  ("c", KNeighborsClassifier(n_neighbors=5))])),
        "MB9": ("AdaBoost (shallow-tree boosting)", "learner", None,
                lambda: AdaBoostClassifier(n_estimators=200,
                                           random_state=42)),
    }
    C = {  # augmentation + hybrids on winner features
        "MC2": ("within-class cross-patient mixup + ET", "augmentation",
                None, mixup_et()),
        "MC3": ("band features -> monotone XGBoost (BANDS priors)",
                "hybrid", rowmap_bands, _monotone_xgb()),
        "MC4": ("quantile+entropy concat -> rotation forest", "hybrid",
                lambda X, wn: np.hstack([rowmap_quantiles(X, wn),
                                         rowmap_entropy(X, wn)]),
                rotation_forest_cls()),
        "MC5": ("FFT features -> GP(PCA-50)", "hybrid", rowmap_fft,
                gp_pipe()),
        "MC6": ("window-corr features -> subspace bag", "hybrid",
                rowmap_wincorr, subspace_bag(et(150))),
        "MC7": ("HGB boosting on band features", "hybrid",
                rowmap_bands,
                lambda: HistGradientBoostingClassifier(
                    max_iter=300, random_state=42)),
        "MC8": ("RF on quantile features", "hybrid", rowmap_quantiles,
                lambda: RandomForestClassifier(n_estimators=300,
                                               random_state=42, n_jobs=1,
                                               class_weight="balanced")),
    }
    P, R, E = "PLS + XGBoost", "Random Forest", "Extra Trees"
    D = {  # winner-chain composition variants (the WORKS-tier family)
        "MD1": (f"{P} -> {R} -> XGBoost (L3 swap)", "chain", None, None,
                [P, R, "XGBoost"]),
        "MD2": (f"{P} -> {R} -> LightGBM (L3 swap)", "chain", None, None,
                [P, R, "LightGBM"]),
        "MD3": (f"{P} -> {R} -> CatBoost (L3 swap)", "chain", None, None,
                [P, R, "CatBoost"]),
        "MD4": (f"{P} -> {R} -> HGB (L3 swap)", "chain", None, None,
                [P, R, "Hist Gradient Boosting"]),
        "MD5": (f"{P} -> {R} -> PCA+LogReg (L3 swap)", "chain", None,
                None, [P, R, "PCA + Logistic Regression"]),
        "MD6": (f"{P} -> {R} -> {P} (self-stack L3)", "chain", None, None,
                [P, R, P]),
        "MD7": (f"{P} -> Extra Trees -> Extra Trees (L2 swap)", "chain",
                None, None, [P, E, E]),
        "MD8": (f"{P} -> Extra Trees (2-layer)", "chain", None, None,
                [P, E]),
        "MD9": (f"{P} -> Random Forest (2-layer)", "chain", None, None,
                [P, R]),
        "MD10": (f"{P} -> {R} -> mixup-ET (augmented L3)", "chain",
                 None, mixup_et(), [P, R, "mix"]),
        "MD11": ("Random Forest -> Extra Trees (2-layer, no PLS base)",
                 "chain", None, None, [R, E]),
        "MD12": (f"{P} -> CatBoost -> Extra Trees (L2 swap)", "chain",
                 None, None, [P, "CatBoost", E]),
    }
    F = {  # deeper chains + unseen layer families (fast learners only)
        "MF2": (f"{P} -> {R} -> {E} -> {E} (4-layer)", "chain", None,
                None, [P, R, E, E]),
        "MF3": (f"{P} -> {R} -> {E} -> XGBoost (4-layer)", "chain",
                None, None, [P, R, E, "XGBoost"]),
        "MF4": (f"{P} -> PCA + SVM (RBF) -> {E}", "chain", None, None,
                [P, "PCA + SVM (RBF)", E]),
        "MF5": (f"{P} -> t-test filter + XGBoost -> {E}", "chain",
                None, None, [P, "t-test filter + XGBoost", E]),
        "MF6": (f"{P} -> LightGBM -> {E}", "chain", None, None,
                [P, "LightGBM", E]),
        "MF7": (f"{P} -> Hist Gradient Boosting -> {E}", "chain",
                None, None, [P, "Hist Gradient Boosting", E]),
        "MF8": (f"Peak bands + RF -> {R} -> {E}", "chain", None, None,
                ["Peak bands + RF", R, E]),
        "MF9": (f"{P} -> PCA + KNN -> {E}", "chain", None, None,
                [P, "PCA + KNN", E]),
    }
    G = {  # layer-1 swaps + remaining families (fast learners)
        "MG1": (f"{P} -> Extra Trees -> XGBoost", "chain", None, None,
                [P, E, "XGBoost"]),
        "MG2": (f"XGBoost -> {R} -> {E} (L1 swap)", "chain", None, None,
                ["XGBoost", R, E]),
        "MG3": (f"PCA + XGBoost -> {R} -> {E}", "chain", None, None,
                ["PCA + XGBoost", R, E]),
        "MG4": (f"t-test filter + XGBoost -> {R} -> {E}", "chain",
                None, None, ["t-test filter + XGBoost", R, E]),
        "MG5": (f"{P} -> PCA + Gaussian Naive Bayes -> {E}", "chain",
                None, None,
                [P, "PCA + Gaussian Naive Bayes", E]),
        "MG6": (f"{P} -> PCA + MLP -> {E}", "chain", None, None,
                [P, "PCA + MLP (neural net)", E]),
        "MG7": (f"Peak bands + RF -> {E} (2-layer)", "chain", None,
                None, ["Peak bands + RF", E]),
        "MG8": (f"Spectral + band features -> {R} -> {E}", "chain",
                None, None, ["Spectral + band features", R, E]),
        "MG9": (f"{P} -> {R} -> Hist Gradient Boosting", "chain",
                None, None, [P, R, "Hist Gradient Boosting"]),
        "MG10": (f"{P} -> Isolation Forest (one-vs-rest) -> {E}",
                 "chain", None, None,
                 [P, "Isolation Forest (one-vs-rest)", E]),
    }
    return {"A": A, "B": B, "C": C, "D": D, "F": F, "G": G}


def _monotone_xgb():
    """XGBoost with monotone constraints from BANDS directions (+1 tumor
    band must push toward tumor).  Feature order = rowmap_bands: 16
    window areas (no prior) then the 17 BAND areas (direction prior)."""
    from xgboost import XGBClassifier
    import biochemistry as bio

    def factory():
        n_win = 16
        cons = tuple([0] * n_win + [int(d)
                                    for _c, _h, _m, _a, d in bio.BANDS])
        return XGBClassifier(n_estimators=200, max_depth=3,
                             learning_rate=0.1, random_state=42, n_jobs=1,
                             monotone_constraints=cons,
                             eval_metric="logloss")
    return factory


# ------------------------------------------------------------- runner
def run_method(mid: str, mech: str, family: str, rowmap, factory,
               seeds=(42, 43, 44), arch=None, data_tuple=None) -> None:
    if mid in method_bank.bank_ids():
        print(f"[{mid}] already banked — skip", flush=True)
        return
    if data_tuple is not None:                   # QC view (PUSH-0.8 L1)
        X, y, g, wn = data_tuple[:4]
        y = list(y)
    else:
        X, y, g, wn = data()
    t0 = time.time()
    try:
        if rowmap is not None:
            Xm = rowmap(X, wn)
        else:
            Xm = X
        if arch is not None:                     # chain variant
            factories = dict(sequential._factories(wn))
            if factory is not None:
                factories["mix"] = factory
            use_arch = list(arch)
        else:
            factories = {"m": factory}
            use_arch = ["m"]
        f1s, pfs = [], []
        oof_first = None
        for s in seeds:
            m = sequential.validate_arch(use_arch, factories,
                                         Xm, list(y), list(g),
                                         seed=s, k_outer=5, wn=wn)
            f1s.append(exp_common.oof_f1(m))
            pfs.append(float(m.get("pat_f1", float("nan"))))
            if oof_first is None:            # seed-42 OOF for fusion arms
                oof_first = (np.asarray(m["y_true"]),
                             np.asarray(m["oof_proba"], dtype=np.float32))
        method_bank.add({"id": mid, "mechanism": mech, "family": family,
                         "f1_mean": float(np.mean(f1s)),
                         "f1_std": float(np.std(f1s)),
                         "pat_f1": float(np.nanmean(pfs)),
                         "f1s": [round(f, 3) for f in f1s],
                         "seeds": len(seeds),
                         "runtime_s": round(time.time() - t0, 1)})
        oof_dir = os.path.join(exp_common.OUT_DIR, "oof_bank")
        os.makedirs(oof_dir, exist_ok=True)
        gv = np.asarray(g)[~np.isnan(oof_first[1]).any(axis=1)]
        np.savez_compressed(os.path.join(oof_dir, f"{mid}.npz"),
                            y_true=oof_first[0], oof=oof_first[1],
                            groups=gv)
        print(f"[{mid}] F1 {np.mean(f1s):.3f}±{np.std(f1s):.3f} "
              f"pat {np.nanmean(pfs):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    except Exception as exc:                     # error-tolerant (gotcha 23b)
        method_bank.add({"id": mid, "mechanism": mech, "family": family,
                         "error": repr(exc)[:300],
                         "f1_mean": 0.0, "f1_std": 0.0, "pat_f1": None,
                         "seeds": 0,
                         "runtime_s": round(time.time() - t0, 1)})
        print(f"[{mid}] ERROR {exc!r}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="report",
                    help="A|B|C|all|report")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    if args.batch == "report":
        print(method_bank.render(), flush=True)
        return 0
    seeds = tuple(int(s) for s in args.seeds.split(","))
    defs = batch_defs()
    batches = list(defs) if args.batch == "all" else [args.batch]
    method_bank.harvest_queue()
    for b in batches:
        for mid, spec in defs[b].items():
            if len(spec) == 5:
                mech, fam, rowmap, factory, arch = spec
                run_method(mid, mech, fam, rowmap, factory, seeds=seeds,
                           arch=arch)
            else:
                mech, fam, rowmap, factory = spec
                run_method(mid, mech, fam, rowmap, factory, seeds=seeds)
    print(method_bank.render(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
