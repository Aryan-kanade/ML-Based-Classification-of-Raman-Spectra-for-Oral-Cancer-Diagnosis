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
import os
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
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    # GPU-trained boosters predict from CPU numpy on purpose (GUI batches
    # are tiny) — xgboost warns about the DMatrix device fallback; expected
    import warnings
    warnings.filterwarnings(
        "ignore", message=".*Falling back to prediction using DMatrix.*")
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except Exception:                      # ImportError OR loader errors
    HAS_TABPFN = False

try:
    # NOTE: import torch BEFORE any Qt binding loads (see qt_compat.py)
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception:                      # ImportError OR DLL/loader errors
    HAS_TORCH = False


def torch_device():
    """cuda when a CUDA torch build + GPU are present, else cpu.
    RAMAN_DEVICE=cpu forces CPU — the 3SSE loky workers do exactly
    that (one CUDA context per child process would blow VRAM)."""
    if not HAS_TORCH:
        return None
    if os.environ.get("RAMAN_DEVICE", "").lower() == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# GPU detection — capability-based, never card-name-based (any CUDA GPU
# works: RTX 2050/3050/4050/…; the name in reports comes from the driver)
# --------------------------------------------------------------------------
_GPU_OK = None


def _xgb_cuda_canary() -> bool:
    """BUILD-capability check only: xgboost was compiled with CUDA.
    NOT proof of GPU execution — with no visible device, device='cuda'
    fits SILENTLY fall back to CPU (probed 2026-09-08: no warning,
    build_info USE_CUDA=True, CUDA_VISIBLE_DEVICES='' -> CPU run).
    Actual driver presence is proven by the torch CUDA probe in
    _real_cuda_present(); this canary only gates the 'the wheel itself
    supports CUDA' half."""
    if not HAS_XGB:
        return False
    try:
        import xgboost as _xgb
        return bool(_xgb.build_info().get("USE_CUDA"))
    except Exception:
        return False


def _real_cuda_present() -> bool:
    """A CUDA backend proven at RUNTIME. torch.cuda.is_available()
    actually initializes a CUDA context (fails without a driver/GPU);
    xgboost cannot serve as the probe because its device='cuda' fits
    silently fall back to CPU when no device is visible.  Without a
    CUDA torch build we conservatively report no GPU."""
    return bool(HAS_TORCH and torch.cuda.is_available())


def gpu_ok() -> bool:
    """Boosters on GPU ONLY by explicit opt-in (RAMAN_DEVICE=gpu).
    Default CPU for three reasons: at n≈300×1600 GPU boosters LOSE to
    CPU (kernel-launch/transfer overhead dominates); their CUDA
    contexts fill VRAM and OOM'd loky children mid-import of catboost
    during full-registry training (CUDA error 2 abort, 2026-09-05);
    and the 3SSE workers already run CPU-pinned. The torch path
    (1D-CNN/ViT via torch_device()) is independent and stays
    GPU-by-default."""
    if os.environ.get("RAMAN_DEVICE", "").lower() != "gpu":
        return False
    global _GPU_OK
    if _GPU_OK is None:
        # driver presence via the torch CUDA probe ONLY — xgboost's
        # device='cuda' silently falls back to CPU without a device
        # (see _xgb_cuda_canary), so it can never prove a GPU
        _GPU_OK = (HAS_TORCH and torch.cuda.is_available()
                   and _xgb_cuda_canary())
    return _GPU_OK


def boost_device() -> str:
    """XGBoost device: 'cuda' on any usable GPU, else 'cpu'."""
    return "cuda" if gpu_ok() else "cpu"


def as_env_device(est):
    """Re-point a GPU-built boosted estimator to CPU when RAMAN_DEVICE=cpu
    (estimators are constructed once in the parent — where the GPU is
    visible — then pickled into 3SSE workers where the env forces CPU).
    In that worker mode also CAP native thread pools (thread_count=1 /
    n_jobs=1): every worker spawning all-core pools multiplied memory
    pressure until fits died with 'bad allocation' (2026-09-05)."""
    if os.environ.get("RAMAN_DEVICE", "").lower() == "cpu":
        try:
            p = est.get_params(deep=False)
            if p.get("device") == "cuda":
                est.set_params(device="cpu")
            if p.get("task_type") == "GPU":
                est.set_params(task_type="CPU")
            if "thread_count" in p or "task_type" in p:   # CatBoost
                est.set_params(thread_count=1)
            if "n_jobs" in p:                  # XGBoost / LightGBM / RF / ET
                est.set_params(n_jobs=1)
        except AttributeError:
            pass
    return est


def device_report() -> str:
    """One-line accelerator summary from the hardware actually present."""
    parts = []
    if HAS_TORCH:
        dev = torch_device()
        if dev is not None and dev.type == "cuda":
            try:
                parts.append(f"1D-CNN: cuda ({torch.cuda.get_device_name()})")
            except Exception:
                parts.append("1D-CNN: cuda")
        else:
            parts.append("1D-CNN: cpu")
    if HAS_XGB:
        parts.append(f"XGBoost: {boost_device()}")
    if HAS_CATBOOST:
        parts.append(f"CatBoost: {'GPU' if gpu_ok() else 'CPU'}")
    cpu_fams = "scikit-learn" + (" + LightGBM" if HAS_LGBM else "")
    parts.append(f"{cpu_fams}: CPU-only")
    return "Accelerators — " + " · ".join(parts)


# --------------------------------------------------------------------------
# Device MODE layer (2026-09-08 GPU work): RAMAN_DEVICE = auto|gpu|cpu.
# auto (default): torch models (1D-CNN/ViT/TabPFN) on GPU when CUDA is
#   real; boosters stay CPU (measured: they lose at n≈300 and auto-GPU
#   boosters OOM'd loky children — REPORTED in the summary, not silent).
# gpu: STRICT — a real CUDA backend is required or we fail loudly.
# cpu: force everything to CPU (checked on every gpu_ok/torch_device call).
# --------------------------------------------------------------------------
def _real_cuda_present() -> bool:
    """A CUDA backend proven at runtime — the torch probe initializes a
    real CUDA context; never a mere 'library has a GPU option' claim
    (xgboost device='cuda' silently CPU-falls-back, see canary)."""
    return bool(HAS_TORCH and torch.cuda.is_available())


def resolve_device_mode() -> str:
    """Validate RAMAN_DEVICE and return 'auto' | 'gpu' | 'cpu'.
    'gpu' is strict: no real CUDA -> RuntimeError (NEVER a silent CPU
    fallback); an unknown value is also an error."""
    mode = (os.environ.get("RAMAN_DEVICE") or "auto").strip().lower()
    mode = {"gpu": "gpu", "cuda": "gpu", "auto": "auto", "": "auto",
            "cpu": "cpu"}.get(mode)
    if mode is None:
        raise RuntimeError(
            f"RAMAN_DEVICE={os.environ.get('RAMAN_DEVICE')!r} is invalid "
            "— use auto, gpu or cpu.")
    if mode == "gpu" and not _real_cuda_present():
        raise RuntimeError(
            "RAMAN_DEVICE=gpu was requested but no usable CUDA backend "
            "was found (torch CUDA unavailable and the XGBoost CUDA "
            "canary fit failed). Fix the driver/install, or use "
            "RAMAN_DEVICE=auto.")
    return mode


def verify_gpu_runtime() -> dict:
    """Runtime PROOF, not availability claims: runs an actual forward
    pass on the selected torch device and records the device the
    output tensor landed on; boosters report what gpu_ok()/boost_device
    actually configure; LightGBM/scikit are confirmed CPU."""
    out: dict = {"mode": resolve_device_mode(), "torch": None,
                 "torch_probe_device": None, "cnn": "absent",
                 "xgboost": "absent", "catboost": "absent",
                 "lightgbm": "absent", "sklearn": "cpu"}
    if HAS_TORCH:
        dev = torch_device()
        lin = torch.nn.Linear(8, 4).to(dev)
        x = torch.randn(4, 8, device=dev)
        with torch.inference_mode():
            y = lin(x)
        out["torch"] = torch.__version__
        out["torch_probe_device"] = str(y.device)      # e.g. 'cuda:0'
        out["cnn"] = dev.type
    if HAS_XGB:
        out["xgboost"] = boost_device()
        if out["mode"] == "gpu":
            # build capability (NOT execution proof — see canary note);
            # execution-path config is proven by boost_device()=='cuda'
            out["xgboost_cuda_build"] = _xgb_cuda_canary()
    if HAS_CATBOOST:
        out["catboost"] = "GPU" if gpu_ok() else "CPU"
    if HAS_LGBM:
        out["lightgbm"] = "cpu"   # standard Windows wheel: no GPU build
    return out

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# PLS-DA classifier wrapper
# --------------------------------------------------------------------------
class PLSDAClassifier(ClassifierMixin, BaseEstimator):
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


class TTestSelect(TransformerMixin, BaseEstimator):
    """
    Univariate spectral feature selection after the internship report
    (§3.3.1): keep wavenumbers whose Welch t-test p < `p_max` AND
    |Cohen's d| > `d_min` between the two classes.  Binary only; fitted
    INSIDE the CV pipeline so the selection never sees test spectra
    (Cawley & Talbot 2010).  Falls back to all features when nothing
    passes (tiny/unaligned folds) so pipelines still train.

    SELECTION HEURISTIC, NOT INFERENCE: the per-wavenumber p-values are
    uncorrected (~1e3 tests -> ~50 expected false positives at p<0.05);
    set `fdr=True` to apply Benjamini-Hochberg before thresholding.
    """

    def __init__(self, p_max: float = 0.05, d_min: float = 0.1,
                 fdr: bool = False):
        self.p_max = p_max
        self.d_min = d_min
        self.fdr = fdr

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)          # list labels silently disabled the
        classes = np.unique(y)     # filter (2026-09-12 audit)
        if len(classes) != 2:      # multiclass: keep everything (report
            self.mask_ = np.ones(X.shape[1], dtype=bool)  # is binary)
            self.n_selected_ = int(X.shape[1])
            return self
        a = X[y == classes[0]]
        b = X[y == classes[1]]
        from scipy.stats import ttest_ind
        t, p = ttest_ind(a, b, axis=0, equal_var=False)   # Welch
        if self.fdr:               # Benjamini-Hochberg step-up
            o = np.argsort(p, kind="stable")
            ranked = p[o] * len(p) / np.arange(1, len(p) + 1)
            p = np.empty_like(p)
            p[o] = np.minimum.accumulate(ranked[::-1])[::-1]
        na, nb = len(a), len(b)
        sp = np.sqrt(((na - 1) * a.var(axis=0, ddof=1)
                      + (nb - 1) * b.var(axis=0, ddof=1)) / (na + nb - 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.abs((a.mean(axis=0) - b.mean(axis=0)) / np.maximum(sp,
                                                                     1e-12))
        mask = (p < self.p_max) & (d > self.d_min)
        mask = np.nan_to_num(mask, nan=0.0).astype(bool)
        self.mask_ = mask if mask.any() else np.ones(X.shape[1], dtype=bool)
        self.n_selected_ = int(self.mask_.sum())
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.mask_]


class PLSScores(TransformerMixin, BaseEstimator):
    """
    PLS latent-variable scores as FEATURES for a downstream classifier
    (the internship report's PLS+SVM / PLS+XGB pipelines).  PLSRegression
    is fitted on one-hot class codes, so string labels work and multiclass
    needs no extra handling.
    """

    def __init__(self, n_components: int = 5):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            # binary: 1-D class codes, so n_components is not capped at
            # n_targets (the report uses 5 components on 2 classes)
            Y = (y == self.classes_[1]).astype(float)
        else:
            Y = np.zeros((len(y), len(self.classes_)))
            for i, c in enumerate(self.classes_):
                Y[y == c, i] = 1.0
        nc = max(1, int(min(self.n_components, len(y), X.shape[1])))
        self.pls_ = PLSRegression(n_components=nc, scale=False).fit(X, Y)
        return self

    def transform(self, X):
        return self.pls_.transform(np.asarray(X, dtype=float))


class SparsePLSDA(ClassifierMixin, BaseEstimator):
    """
    Sparse PLS-DA (mixOmics-style variable selection): fit PLS on class
    codes, keep the `n_select` wavenumbers with the largest summed
    |loading| across components, refit PLS on just those.  Sparse
    loadings are stabler than VIP for feature selection (Lê Cao 2008).
    """

    def __init__(self, n_components: int = 5, n_select: int = 50):
        self.n_components = n_components
        self.n_select = n_select

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        probe = PLSScores(n_components=self.n_components).fit(X, y)
        load = np.abs(probe.pls_.x_loadings_).sum(axis=1)
        k = int(max(2, min(self.n_select, X.shape[1])))
        self.sel_ = np.argsort(-load)[:k]
        self.clf_ = PLSDAClassifier(n_components=self.n_components
                                    ).fit(X[:, self.sel_], y)
        return self

    def predict(self, X):
        return self.clf_.predict(np.asarray(X, dtype=float)[:, self.sel_])

    def predict_proba(self, X):
        return self.clf_.predict_proba(
            np.asarray(X, dtype=float)[:, self.sel_])


# --------------------------------------------------------------------------
# 1D-CNN over the raw (cropped) spectrum — torch wrapper, sklearn-compatible
# --------------------------------------------------------------------------
if HAS_TORCH:

    class _CNN1DNet(nn.Module):
        """Conv1d(1→32, k7) → pool → Conv1d(32→64, k5) → global avg pool
        → dropout → linear head.  ~25k parameters; CPU-fast."""

        def __init__(self, n_classes: int, length: int, dropout: float):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(64, n_classes))

        def forward(self, x):            # x: (batch, 1, length)
            h = self.features(x).squeeze(-1)
            return self.head(h)

    class CNN1DClassifier(ClassifierMixin, BaseEstimator):
        """
        Small 1D-CNN classifier: fits inside the CV/bundle machinery
        like any sklearn estimator (clone, GridSearchCV with an empty
        grid, joblib pickling of the torch state).

        Training: class-weighted cross-entropy, Adam, fixed epoch budget
        with early stopping on a stratified 15 % validation split
        (best-weights restored).  Seeded for reproducibility.
        """

        def __init__(self, epochs: int = 25, batch_size: int = 32,
                     lr: float = 1e-3, dropout: float = 0.2,
                     seed: int = RANDOM_STATE, augment: bool = True,
                     synth: float = 0.0):
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.dropout = dropout
            self.seed = seed
            self.augment = augment
            # B2: fraction of the TRAINING rows to additionally
            # synthesize as within-class spectral blends (0 = off)
            self.synth = synth

        def _to_tensor(self, X):
            X = np.asarray(X, dtype=np.float32)
            if X.ndim != 3:
                X = X[:, None, :]                    # (n, 1, length)
            return torch.from_numpy(X)

        def fit(self, X, y, groups=None):
            from sklearn.model_selection import train_test_split
            torch.manual_seed(self.seed)
            # seeded numpy Generator for mixup (the legacy global RNG made
            # CNN fits nondeterministic despite the configured seed)
            np_rng = np.random.default_rng(self.seed)
            self.classes_ = np.unique(y)
            ye = np.searchsorted(self.classes_, y)
            dev = torch_device()
            amp = dev.type == "cuda"
            if amp and tuple(torch.cuda.get_device_capability(dev)) >= (8, 0):
                # TF32 matmuls: free speedup on Ampere+ GPUs only
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            Xt = self._to_tensor(X).to(dev)
            self.n_features_in_ = Xt.shape[-1]
            if len(ye) >= 20:
                if groups is not None and len(set(groups)) >= 8:
                    # GROUPED early-stop split (2026-09-05): hold out
                    # whole PATIENTS — a per-spectrum split let the val
                    # patients' other spectra train the net, making
                    # val-loss optimistic (late, biased stopping).
                    from sklearn.model_selection import GroupShuffleSplit
                    idx_tr, idx_va = next(GroupShuffleSplit(
                        n_splits=1, test_size=0.15,
                        random_state=self.seed).split(X, ye, groups))
                else:
                    idx_tr, idx_va = train_test_split(
                        np.arange(len(ye)), test_size=0.15, stratify=ye,
                        random_state=self.seed)
            else:                                   # too small to split
                idx_tr = idx_va = np.arange(len(ye))
            # B2: physics-safe within-class spectral blends (see
            # lorentzian_synthesize — convex pairs + smooth gain/tilt)
            # of the TRAINING rows only — the early-stop validation
            # split stays 100% real, so the stopping signal stays honest
            tr_X, tr_y = Xt[idx_tr], ye[idx_tr]
            if self.synth > 0 and len(idx_tr) >= 4:
                X_tr_np = np.asarray(X, dtype=np.float32)[idx_tr]
                xs, ys = [], []
                uq = np.unique(tr_y)
                n_new = max(1, int(self.synth * len(idx_tr)) // len(uq))
                for c in uq:
                    m = tr_y == c
                    if m.sum() >= 2:
                        S = lorentzian_synthesize(X_tr_np[m], n_new,
                                                  seed=self.seed)
                        xs.append(S)
                        ys.extend([c] * len(S))
                if xs:
                    tr_X = torch.cat([tr_X, self._to_tensor(
                        np.vstack(xs)).to(dev)])
                    tr_y = np.concatenate([tr_y, np.asarray(ys)])
            counts = np.bincount(ye, minlength=len(self.classes_))
            weights = torch.tensor(
                len(ye) / (len(self.classes_) * np.maximum(counts, 1)),
                dtype=torch.float32, device=dev)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            self.net_ = _CNN1DNet(len(self.classes_), Xt.shape[-1],
                                  self.dropout).to(dev)
            opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
            gen = torch.Generator().manual_seed(self.seed)
            best_state, best_loss, best_epoch = None, np.inf, -1
            for epoch in range(self.epochs):
                self.net_.train()
                perm = torch.randperm(len(tr_X), generator=gen).numpy()
                for s in range(0, len(perm), self.batch_size):
                    b = perm[s:s + self.batch_size]
                    opt.zero_grad()
                    xb, yb = tr_X[b], torch.from_numpy(
                        tr_y[b]).to(dev)
                    if self.augment:
                        # ViT-parity regularization for small spectral
                        # sets: noise + roll + mixup + band-mask
                        xb = xb + torch.randn_like(xb) * (0.05 * xb.std())
                        xb = torch.roll(xb, int(torch.randint(-2, 3, (1,))),
                                        dims=2)
                        if torch.rand(1).item() < 0.5:
                            lam = float(np_rng.beta(0.2, 0.2))
                            pm = torch.randperm(len(xb), device=xb.device)
                            xb = lam * xb + (1 - lam) * xb[pm]
                            oh = torch.nn.functional.one_hot(
                                yb.long(), num_classes=len(self.classes_)
                            ).float()
                            yb = lam * oh + (1 - lam) * oh[pm]
                        if torch.rand(1).item() < 0.5:   # SpecAugment mask
                            L = xb.shape[2]
                            w = max(4, int(0.05 * L))
                            st = int(torch.randint(0, max(1, L - w), (1,)))
                            xb[:, :, st:st + w] = 0.0
                    with torch.autocast(dev.type, enabled=amp):
                        loss = loss_fn(self.net_(xb), yb)
                    if amp:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                self.net_.eval()
                with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
                    va_loss = float(loss_fn(self.net_(Xt[idx_va]),
                                            torch.from_numpy(
                                                ye[idx_va]).to(dev)))
                if va_loss < best_loss - 1e-4:
                    best_loss, best_epoch = va_loss, epoch
                    best_state = {k: v.clone()
                                  for k, v in self.net_.state_dict().items()}
                elif epoch - best_epoch >= 6:       # early stop
                    break
            if best_state is not None:
                self.net_.load_state_dict(best_state)
            self.net_.eval()
            # store fitted weights on CPU: pickled bundles stay portable
            # to machines without CUDA (predict re-selects the device)
            self.net_.to("cpu")
            return self

        def predict_proba(self, X):
            # CPU predict on purpose: batches are tiny (n≈hundreds) and
            # self.net_ must never be left holding CUDA tensors (it gets
            # joblib-pickled into saved model bundles)
            self.net_.to("cpu")
            with torch.no_grad():
                out = []
                for s in range(0, len(X), 256):
                    logits = self.net_(self._to_tensor(
                        np.asarray(X)[s:s + 256]))
                    out.append(torch.softmax(logits, dim=1).numpy())
                return np.vstack(out)

        def predict(self, X):
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

        def predict_proba_tta(self, X, n_aug: int = 8
                              ) -> np.ndarray:
            """
            Test-time augmentation: average predict_proba over n_aug
            augmented copies (same recipe as training: noise, roll,
            band-mask).  Free accuracy on small data — the prediction
            becomes invariant to acquisition jitter.
            """
            rng = np.random.default_rng(self.seed)
            X = np.asarray(X, dtype=np.float32)
            L = X.shape[1] if X.ndim == 2 else X.shape[-1]
            acc = self.predict_proba(X)
            for _ in range(int(n_aug)):
                noisy = X + rng.normal(0, 0.05 * X.std(), X.shape
                                       ).astype(np.float32)
                shift = int(rng.integers(-2, 3))
                noisy = np.roll(noisy, shift, axis=-1)
                w = max(4, int(0.05 * L))
                st = int(rng.integers(0, max(1, L - w)))
                noisy[..., st:st + w] = 0.0
                acc = acc + self.predict_proba(noisy)
            return acc / (int(n_aug) + 1)

        def grad_cam(self, X, class_idx: int | None = None
                     ) -> np.ndarray:
            """
            1D Grad-CAM over the last Conv1d layer: for each spectrum,
            the wavenumber-length saliency of the predicted (or chosen)
            class — gradient-weighted activation energy, ReLU'd and
            max-normalized.  (Selvaraju 2017 adapted to 1D; no captum.)
            """
            self.net_.to("cpu")
            self.net_.eval()
            Xt = self._to_tensor(np.asarray(X, dtype=np.float32))
            store = {}

            def fwd(_m, _i, out):
                store["act"] = out.detach()

            def bwd(_m, _grad_in, grad_out):
                store["grad"] = grad_out[0].detach()   # d loss / d act

            last_conv = self.net_.features[4]      # second Conv1d block
            hf = last_conv.register_forward_hook(fwd)
            hb = last_conv.register_full_backward_hook(bwd)
            try:
                cams = []
                for i in range(0, len(Xt), 64):
                    xb = Xt[i:i + 64].clone().requires_grad_(True)
                    logits = self.net_(xb)
                    if class_idx is None:
                        cls = logits.argmax(dim=1)
                    else:
                        cls = torch.full((len(xb),), int(class_idx),
                                         dtype=torch.long)
                    score = logits.gather(1, cls.view(-1, 1)).sum()
                    self.net_.zero_grad()
                    score.backward()
                    act = store["act"]                       # (b, C, L')
                    grad = store.get("grad", torch.zeros_like(act))
                    w = grad.mean(dim=(0, 2), keepdim=True)   # GAP of grad
                    cam = torch.relu((w * act).sum(dim=1))    # (b, L')
                    b, L = cam.shape
                    cam = torch.nn.functional.interpolate(
                        cam.view(b, 1, L), size=Xt.shape[2],
                        mode="linear").view(b, -1)
                    cams.append(cam.detach().numpy())
                out = np.vstack(cams)
                mx = out.max(axis=1, keepdims=True)
                return out / np.maximum(mx, 1e-9)
            finally:
                hf.remove()
                hb.remove()

        def predict_proba_mc(self, X, n_passes: int = 20):
            """
            MC-dropout uncertainty: enable dropout at inference, sample
            n stochastic forwards, return (mean probs, std of probs) —
            the std is an epistemic-uncertainty proxy (per class).
            """
            self.net_.to("cpu")
            for m in self.net_.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
            try:
                with torch.no_grad():
                    runs = []
                    for _ in range(int(n_passes)):
                        out = []
                        for s in range(0, len(X), 256):
                            logits = self.net_(self._to_tensor(
                                np.asarray(X)[s:s + 256]))
                            out.append(torch.softmax(logits, dim=1).numpy())
                        runs.append(np.vstack(out))
                stacked = np.stack(runs)
                return stacked.mean(axis=0), stacked.std(axis=0)
            finally:
                self.net_.eval()      # always leave eval mode behind

    class CNNEnsemble(ClassifierMixin, BaseEstimator):
        """
        Deep-ensemble uncertainty for the 1D-CNN: `n_seeds` networks with
        different seeds; predict_proba is the mean (predictive mean),
        `predict_proba_std` the disagreement across members.  Lakshmin-
        arayanan 2017; the cheap, reliable deep UQ baseline.
        """

        def __init__(self, n_seeds: int = 5, epochs: int = 40,
                     batch_size: int = 32, lr: float = 1e-3,
                     dropout: float = 0.2, seed: int = RANDOM_STATE,
                     augment: bool = True, synth: float = 0.0):
            self.n_seeds = n_seeds
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.dropout = dropout
            self.seed = seed
            self.augment = augment
            self.synth = synth

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self.members_ = [
                CNN1DClassifier(epochs=self.epochs,
                                batch_size=self.batch_size, lr=self.lr,
                                dropout=self.dropout,
                                seed=self.seed + i, augment=self.augment,
                                synth=self.synth
                                ).fit(X, y)
                for i in range(int(self.n_seeds))]
            return self

        def predict_proba(self, X):
            return np.mean([m.predict_proba(X) for m in self.members_],
                           axis=0)

        def predict_proba_std(self, X):
            """Member disagreement — epistemic uncertainty per class."""
            return np.std([m.predict_proba(X) for m in self.members_],
                          axis=0)

        def predict(self, X):
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# --------------------------------------------------------------------------
# Calibrated SVM (sklearn >= 1.9 deprecates SVC(probability=True))
# --------------------------------------------------------------------------
class CalibratedSVC(ClassifierMixin, BaseEstimator):
    """SVC with predict_proba; calibration folds adapt to small classes."""

    def __init__(self, C=1.0, gamma="scale", class_weight="balanced",
                 random_state=RANDOM_STATE):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        from sklearn.calibration import CalibratedClassifierCV
        self.classes_ = np.unique(y)
        base = SVC(C=self.C, gamma=self.gamma,
                   class_weight=self.class_weight,
                   random_state=self.random_state)
        counts = np.bincount(np.searchsorted(self.classes_, y),
                             minlength=len(self.classes_))
        cv = int(min(3, counts.min()))
        n_groups = len(set(groups)) if groups is not None else None
        if groups is not None and cv >= 2 and (n_groups or 0) >= cv:
            # GROUPED calibration (2026-09-05): same-patient spectra must
            # never straddle a calibration split — plain StratifiedKFold
            # here leaked patients into the sigmoid fit and biased every
            # downstream probability.  Precomputed index splits avoid
            # routing `groups` through CalibratedClassifierCV.fit.
            from sklearn.model_selection import StratifiedGroupKFold
            gkf = StratifiedGroupKFold(n_splits=cv, shuffle=True,
                                       random_state=self.random_state)
            splits = list(gkf.split(X, y, groups))
            self.model_ = CalibratedClassifierCV(
                base, method="sigmoid", ensemble=False,
                cv=splits).fit(X, y)
        elif cv >= 2 and len(self.classes_) >= 2:
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
class IsolationForestOvR(ClassifierMixin, BaseEstimator):
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
# Physics-based synthetic spectra (safe "generative augmentation")
# --------------------------------------------------------------------------
def phantom_cohort(n_patients: int = 40, spectra_per: int = 4,
                   seed: int = 0):
    """
    Ground-truth synthetic OSCC-like cohort for CI smoke data, method
    demos and teaching: Lorentzian bands with KNOWN class directions
    (tumor: +nucleic-acid 785/1090/1335, +Phe 1003, −carotenoid
    1155/1520; normal: lipid/carotenoid heavy), patient-level random
    effects on band amplitudes, session baseline humps, Gaussian noise
    and occasional cosmic spikes.  Returns (X, y, groups, wn,
    planted) where planted = list of (center, direction_in_tumor).

    Also the ground-truth test of the explainability stack: a profile
    whose top-k lands on `planted` proves the profiler works.
    """
    rng = np.random.default_rng(seed)
    wn = np.linspace(400.0, 2300.0, 1900)

    def lor(center: float, width: float) -> np.ndarray:
        return 1.0 / (1.0 + ((wn - center) / width) ** 2)

    planted = [(785.0, +1), (1003.0, +1), (1090.0, +1), (1335.0, +1),
               (1155.0, -1), (1520.0, -1), (1445.0, 0), (1655.0, +1)]
    X, y, groups = [], [], []
    for p in range(n_patients):
        tumor = (p % 2 == 1)
        # patient random effects (±25%) — the realistic between-subject
        # variability that makes patient-grouped CV the honest protocol
        amp = {c: rng.uniform(0.75, 1.25) for c, _d in planted}
        base = (0.8 * lor(1445.0, 18.0)                # CH2 always on
                + 0.5 * amp[1655.0] * lor(1655.0, 20.0))
        for center, d in planted:
            if center == 1445.0:
                continue
            a = amp[center]
            if d == +1:
                base = base + a * (0.9 if tumor else 0.25) * lor(center, 9)
            elif d == -1:
                base = base + a * (0.15 if tumor else 0.9) * lor(center, 11)
        hump = rng.uniform(0.0, 0.4) * np.exp(-((wn - rng.uniform(500.,
                                                                   900.))
                                                / 350.0) ** 2)
        for s in range(spectra_per):
            spec = base + hump * rng.uniform(0.7, 1.3) \
                + rng.normal(0, 0.02, len(wn))
            if rng.random() < 0.05:                    # occasional spike
                spec[int(rng.integers(0, len(wn)))] += rng.uniform(2, 8)
            X.append(spec)
            y.append("Tumor" if tumor else "Normal")
            groups.append(f"P{p:03d}")
    return (np.asarray(X), np.asarray(y), np.asarray(groups), wn,
            planted)


def lorentzian_synthesize(X: np.ndarray, n_new: int, wn=None,
                          seed: int = RANDOM_STATE) -> np.ndarray:
    """
    Physics-safe augmentation (no GAN): bootstrap `n_new` synthetic
    spectra as CONVEX BLENDS of random same-class pairs plus smooth
    random gain curves and a linear tilt — a perturbation that stays
    on the data manifold by construction.  (Naming note, 2026-09-05:
    despite the name there is NO Lorentzian peak fitting here; the
    original docstring claimed amplitude/width/position perturbation of
    fitted peaks, which the code never did.)

    Use INSIDE training folds only (augmenting before a split leaks).
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    if len(X) < 2 or n_new <= 0:
        return X[:0]
    out = []
    for _ in range(int(n_new)):
        i, j = rng.integers(0, len(X), 2)
        lam = rng.uniform(0.2, 0.8)
        base = lam * X[i] + (1.0 - lam) * X[j]        # manifold blend
        # small spectral perturbations: smooth random gain curve + tilt
        if wn is not None and len(wn) == X.shape[1]:
            wn_ = np.asarray(wn, dtype=float)
            anchor = rng.uniform(wn_.min(), wn_.max())
            width = rng.uniform(50.0, 400.0)
            gain = 1.0 + rng.uniform(-0.15, 0.15) * np.exp(
                -((wn_ - anchor) / width) ** 2)
        else:
            gain = 1.0 + rng.uniform(-0.08, 0.08)
        tilt = np.linspace(-1, 1, X.shape[1]) * rng.uniform(-0.05, 0.05)
        out.append(base * gain + tilt * np.abs(base).mean() * 0.5)
    return np.vstack(out)


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
            "name": "Sparse PLS-DA",
            "estimator": Pipeline([
                ("sc", StandardScaler()),
                ("clf", SparsePLSDA()),
            ]),
            "grid": {"clf__n_select": [30, 100]},
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
                tree_method="hist", device=boost_device(),
                eval_metric="mlogloss",
                random_state=RANDOM_STATE, n_jobs=-1),
            "grid": {"max_depth": [3, 5], "learning_rate": [0.05, 0.2]},
        })

        def _xgb():
            return XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                tree_method="hist", device=boost_device(),
                eval_metric="mlogloss",
                random_state=RANDOM_STATE, n_jobs=-1)
        # the internship report's winner (PLS+XGB) and its PCA twin —
        # PLS/PCA latent-variable scores feeding gradient boosting
        specs.append({
            "name": "PLS + XGBoost",
            "estimator": Pipeline([("pls", PLSScores(n_components=5)),
                                   ("xgb", _xgb())]),
            "grid": {"pls__n_components": [3, 5, 8],
                     "xgb__max_depth": [3, 5],
                     "xgb__learning_rate": [0.05, 0.2]},
        })
        specs.append({
            "name": "PCA + XGBoost",
            "estimator": Pipeline([
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=5, random_state=RANDOM_STATE)),
                ("xgb", _xgb())]),
            "grid": {"pca__n_components": [5, 10],
                     "xgb__max_depth": [3, 5],
                     "xgb__learning_rate": [0.05, 0.2]},
        })
        # the report's univariate filter (Welch t + Cohen's d), selected
        # INSIDE each tuning fold — leakage-proof — XGBoost on the
        # surviving wavenumbers directly
        specs.append({
            "name": "t-test filter + XGBoost",
            "estimator": Pipeline([("sel", TTestSelect()),
                                   ("xgb", _xgb())]),
            "grid": {"sel__d_min": [0.1, 0.3],
                     "xgb__max_depth": [3, 5],
                     "xgb__learning_rate": [0.05, 0.2]},
        })
    if HAS_LGBM:
        specs.append({
            "name": "LightGBM",
            "estimator": LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                class_weight="balanced", verbosity=-1,
                random_state=RANDOM_STATE, n_jobs=-1),
            "grid": {"max_depth": [3, 5], "learning_rate": [0.05, 0.2]},
        })
    if HAS_CATBOOST:
        specs.append({
            "name": "CatBoost",
            "estimator": CatBoostClassifier(
                iterations=200, depth=4, learning_rate=0.1,
                auto_class_weights="Balanced", verbose=0,
                task_type=("GPU" if gpu_ok() else "CPU"),
                allow_writing_files=False, random_seed=RANDOM_STATE),
            "grid": {"depth": [3, 5], "learning_rate": [0.05, 0.2]},
        })
    if HAS_TORCH:
        specs.append({
            "name": "1D-CNN",
            # empty grid: one seeded fit per fold (grid over epochs
            # would multiply an already-iterative training)
            "estimator": CNN1DClassifier(),
            "grid": {},
        })
        # B1+B2 (2026-09-05): deep (seed) ensemble — the small-n gold
        # standard (Lakshminarayanan 2017) — plus 30% physics-safe
        # synthetic within-class blends INSIDE the training split. The
        # plain 1D-CNN above stays single-seed/unaugmented as the
        # controlled variant.
        specs.append({
            "name": "1D-CNN ensemble (5 seeds)",
            "estimator": CNNEnsemble(n_seeds=5, epochs=25, synth=0.3),
            "grid": {},
        })
    if HAS_TABPFN:
        # C1 (2026-09-05): prior-fitted tabular foundation model —
        # zero tuning, seconds per fit; RamanBench 2026 + TabPFN v2
        # (Nature 2025) show foundation models winning exactly in this
        # n≈300 tabular/spectral regime. PCA first: v2.5's pretraining
        # limit is ~500 features and the cropped spectrum has 780-1600.
        # Weights auto-download from HuggingFace once (research license);
        # runs on the torch GPU when present.
        specs.append({
            "name": "TabPFN (foundation model)",
            "estimator": Pipeline([
                ("pca", PCA(n_components=0.95,
                            random_state=RANDOM_STATE)),
                ("clf", TabPFNClassifier(
                    n_estimators=2, random_state=RANDOM_STATE,
                    device=("cuda" if (HAS_TORCH
                                       and torch_device().type == "cuda")
                            else "cpu"))),
            ]),
            "grid": {},
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
    # B3 (2026-09-05 program): fuse the full spectrum (via PCA) with the
    # 16 literature band intensities — the field's standard recipe; band
    # features alone already score near the top (0.736 vs 0.766).
    specs.append({
        "name": "Spectral + band features",
        "needs_wn": True,
        "make": lambda wn: (
            Pipeline([
                ("union", FeatureUnion([
                    ("spec", Pipeline([
                        ("sc", StandardScaler()),
                        ("pca", PCA(n_components=0.95,
                                    random_state=RANDOM_STATE))])),
                    ("bands", PeakIntensityFeatures(wn=wn)),
                ])),
                ("clf", ExtraTreesClassifier(
                    n_estimators=250, class_weight="balanced",
                    random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            {"clf__max_depth": [None, 12],
             "union__spec__pca__n_components": [0.95, 8]}),
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


def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    """Balanced accuracy = mean per-class recall (= (sens+spec)/2 for
    binary).  Supplementary metric (2026-09-08 formula audit): keep
    Accuracy alongside — this only ADDS information."""
    cm = np.asarray(cm)
    recalls = []
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        recalls.append(cm[i, i] / denom if denom > 0 else 0.0)
    return float(np.mean(recalls)) if recalls else float("nan")


def mcc_from_cm(cm: np.ndarray) -> float:
    """Matthews correlation coefficient from a BINARY confusion
    matrix [[TN,FP],[FN,TP]]:
        (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    Zero denominator -> 0.0 (safe).  Non-binary -> NaN (the binary
    formula does not generalize; documented limitation)."""
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        return float("nan")
    tn, fp, fn, tp = (float(cm[0, 0]), float(cm[0, 1]),
                      float(cm[1, 0]), float(cm[1, 1]))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den <= 0:
        return 0.0
    return float((tp * tn - fp * fn) / den)


def brier_score(y_true01, p_pos) -> float:
    """Brier score for the positive-class probability:
    mean((p_i - y_i)^2) over rows with finite p.  Probabilities only —
    never hard labels.  Lower is better (0 = perfect)."""
    y = np.asarray(y_true01, dtype=float)
    p = np.asarray(p_pos, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return float("nan")
    return float(np.mean((p[ok] - y[ok]) ** 2))


def threshold_stats(thresholds) -> dict:
    """Spread of the per-fold binary thresholds (stored in
    ModelResult.thresholds; None = that fold kept the 0.5/argmax
    rule).  ADDITIVE reporting (2026-09-08): the deployment threshold
    itself never changes — this only makes its stability visible.
    `unstable` fires on a >10x max/min spread OR SD > 0.2, computed
    from the data (never hardcoded)."""
    vals = [float(t) for t in (thresholds or []) if t is not None]
    out = {"n_folds": len(list(thresholds or [])), "n_kept": len(vals),
           "min": None, "max": None, "mean": None, "median": None,
           "sd": None, "iqr": None, "spread_ratio": None,
           "unstable": False, "values": vals}
    if not vals:
        return out
    v = np.asarray(vals)
    q1, q3 = np.percentile(v, [25, 75])
    out.update({"min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "median": float(np.median(v)),
                "sd": float(v.std()), "iqr": float(q3 - q1)})
    vmin = max(v.min(), 1e-6)
    out["spread_ratio"] = float(v.max() / vmin)
    out["unstable"] = bool(out["spread_ratio"] > 10.0 or out["sd"] > 0.2)
    return out


def patient_level_evaluation(y_encoded, groups, oof_proba,
                             threshold: float | None = None
                             ) -> dict | None:
    """Patient-level evaluation (supplementary, 2026-09-08).

    Aggregation is UNCHANGED from the deployed rule: each patient's
    probability = MEAN of their spectrum probabilities; the patient is
    positive when that mean >= the given (existing) threshold (0.5
    fallback).  Returns the full metric set on the patient confusion
    matrix + rank metrics on the patient mean-probabilities, or None
    when grouping is unavailable.  `n_tied_patients` counts patients
    with an EXACT Normal/Tumor spectrum tie — their dominant-class
    truth is an arbitrary first-value tie-break (2026-09-12 audit)."""
    if groups is None or oof_proba is None or y_encoded is None:
        return None
    ye = np.asarray(y_encoded)
    g = np.asarray(groups)
    P = np.asarray(oof_proba)
    ok = ~np.isnan(P).any(axis=1)
    ye, g, P = ye[ok], g[ok], P[ok]
    if len(P) == 0 or P.shape[1] != 2:
        return None
    thr = float(threshold) if threshold is not None else 0.5
    pats = []
    n_tied = 0
    for gu in np.unique(g):
        m = g == gu
        p_mean = float(P[m, 1].mean())
        # patient truth = DOMINANT class among their spectra (paired
        # patients contribute both classes; same rule as the existing
        # per-patient rollup)
        vals, cnts = np.unique(ye[m], return_counts=True)
        if len(vals) == 2 and cnts[0] == cnts[1]:
            n_tied += 1
        pats.append((int(vals[np.argmax(cnts)]), p_mean))
    y_pat = np.array([p[0] for p in pats])
    p_pat = np.array([p[1] for p in pats])
    pred = (p_pat >= thr).astype(int)
    cm = np.zeros((2, 2), int)
    for t, q in zip(y_pat, pred):
        cm[t, q] += 1
    per = class_metrics_from_cm(cm)
    return {
        "n": int(len(pats)),
        "n_tied_patients": n_tied,
        "tp": int(cm[1, 1]), "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]), "fn": int(cm[1, 0]),
        "sens": per["1"]["sens"], "spec": per["0"]["sens"],
        "precision": per["1"]["prec"], "f1": per["1"]["f1"],
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "balanced_accuracy": balanced_accuracy_from_cm(cm),
        "mcc": mcc_from_cm(cm),
        "roc_auc": float(roc_points(y_pat, p_pat)[2]),
        "pr_auc": float(pr_points(y_pat, p_pat)[2]),
        "brier": brier_score(y_pat, p_pat),
        "threshold": thr,
    }


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray):
    """Threshold maximizing F1 (and Youden's J) for score -> positive
    class.  EXACT: candidates are every distinct score and every
    midpoint between neighbours (every achievable confusion matrix);
    the old 1-99% quantile subsample could miss the optimum (2026-09-12
    audit).  Falls back to the quantile grid above 5000 candidates."""
    uniq = np.unique(scores)
    if len(uniq) > 5000:
        thresholds = np.unique(np.quantile(scores,
                                           np.linspace(0.01, 0.99, 99)))
    else:
        thresholds = np.concatenate(
            [uniq, (uniq[:-1] + uniq[1:]) / 2.0]) if len(uniq) > 1 else uniq
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
    # GPU-trained boosted models search serially: n_jobs=-1 loky children
    # would each spawn their own CUDA context (~300 MB) and swamp VRAM.
    # The grids are tiny (2x2), so serial costs nothing.
    gs_jobs = -1
    try:
        p = estimator.get_params(deep=False)
        if p.get("device") == "cuda" or p.get("task_type") == "GPU":
            gs_jobs = 1
    except AttributeError:
        pass
    if groups is not None and n_groups >= inner_cv:
        cv = StratifiedGroupKFold(n_splits=inner_cv, shuffle=True,
                                  random_state=RANDOM_STATE)
        gs = GridSearchCV(clone(estimator), grid, cv=cv,
                          scoring="f1_macro", n_jobs=gs_jobs)
        gs.fit(Xtr, ytr, groups=groups)
        # refit the best params on the FULL training fold (groups-aware
        # for estimators that consume them, e.g. StackedEnsemble)
        best = fit_maybe_grouped(
            clone(estimator).set_params(**gs.best_params_), Xtr, ytr, groups)
    else:
        gs = GridSearchCV(clone(estimator), grid, cv=inner_cv,
                          scoring="f1_macro", n_jobs=gs_jobs)
        gs.fit(Xtr, ytr)
        best = clone(estimator).set_params(**gs.best_params_).fit(Xtr, ytr)
    return best, gs.best_params_


def _NOT_THREAD_SAFE(name: str) -> bool:
    """Models whose fits must NOT run concurrently in threads: the
    boosters keep global native pools (rabit / CatBoost), and torch
    seeds the process-global RNG at fit start.  Matched on the registry
    NAME because the estimator may be wrapped in pipelines/unions.
    Ensemble/stacked wrappers are included: their VOTERS are routinely
    boosters or the CNN and the wrapper's own name hides that."""
    return any(s in name for s in ("XGBoost", "CatBoost", "LightGBM",
                                   "1D-CNN", "TabPFN",
                                   "Ensemble", "Stacked"))


def _tune_inner(estimator, grid, Xtr, ytr, inner, pos_idx=None,
                groups=None, serial=False, halve=False):
    """
    ONE inner-CV pass for both jobs run_cv used to do separately
    (2026-09-05): every param combo is fit on the inner training folds
    and scored on POOLED out-of-fold validation probabilities; the
    WINNING combo's pooled probabilities also tune the binary
    threshold.  Replaces GridSearchCV + a second threshold-only
    cross_val_predict (~4 fewer fits per outer fold per model) and the
    redundant refit inside _tune_hyperparams (run_cv refits the fold
    model itself).

    Parallelism: THREAD backend (sklearn fits release the GIL) — never
    loky processes: every child would load the torch-CUDA + booster
    DLLs (~GBs each) just to run a 0.1 s sklearn fit, which OOM'd the
    machine (OpenBLAS allocation aborts, 2026-09-05). `serial=True`
    (boosters / torch models — global C++ pools, per-process RNG) runs
    the loop in-process.

    `halve=True` (TURBO only — approximate): grids with >= 4 combos
    are successive-halved — the first inner split ranks the combos,
    the best half finish the remaining splits.  Selection can differ
    from the exact protocol; every turbo output is captioned as such.

    Semantics note: hyperparameter scoring changed from mean-of-inner-
    folds to pooled-OOF macro-F1 — the same objective the threshold
    step optimizes (documented in Brain.md).  Fits are plain .fit
    (GridSearchCV never passed groups to fit either — only to the
    splitter, which is what `inner` already encodes).  Returns
    (best_params, threshold|None).
    """
    from joblib import Parallel, delayed
    from sklearn.model_selection import ParameterGrid

    combos = list(ParameterGrid(grid)) if grid else [{}]
    splits = list(inner.split(Xtr, ytr, groups))

    def _fit_proba(params, tr, va):
        est = clone(estimator)
        if params:
            est = est.set_params(**params)
        # grouped fit: estimators with internal probability calibration
        # (CalibratedSVC) must not calibrate across a patient boundary
        # (2026-09-12 audit); the outer refit already used the grouped
        # helper, the inner loop now matches it
        fit_maybe_grouped(est, Xtr[tr], ytr[tr],
                          groups[tr] if groups is not None else None)
        return est.predict_proba(Xtr[va])

    def _run_tasks(tasks):
        if serial or len(tasks) == 1:
            return [_fit_proba(*t) for t in tasks]
        return Parallel(n_jobs=min(4, os.cpu_count() or 1),
                        prefer="threads")(
            delayed(_fit_proba)(*t) for t in tasks)

    y_by_split = [ytr[va] for _t, va in splits]
    y_all = np.concatenate(y_by_split) if y_by_split else ytr
    if halve and len(combos) >= 4 and len(splits) >= 2:
        keep_n = max(2, len(combos) // 2)
        pr1 = _run_tasks([(combo, tr, va)
                          for combo in combos for tr, va in splits[:1]])
        rank = [f1_score(y_by_split[0], np.argmax(P, axis=1),
                         average="macro") for P in pr1]
        surv = sorted(np.argsort(rank)[::-1][:keep_n].tolist())
        pr1 = [pr1[i] for i in surv]
        combos = [combos[i] for i in surv]
        pr2 = _run_tasks([(combo, tr, va)
                          for combo in combos for tr, va in splits[1:]])
        rest = len(splits) - 1
        probas = []
        for ci in range(len(combos)):
            probas.append(pr1[ci])
            probas.extend(pr2[ci * rest:(ci + 1) * rest])
    else:
        tasks = [(combo, tr, va) for combo in combos for tr, va in splits]
        probas = _run_tasks(tasks)
    best_i, best_f1, pooled = 0, -np.inf, None
    for ci in range(len(combos)):
        P = np.vstack(probas[ci * len(splits):(ci + 1) * len(splits)])
        f1 = f1_score(y_all, np.argmax(P, axis=1), average="macro")
        if f1 > best_f1 + 1e-12:            # first combo wins ties
            best_f1, best_i, pooled = f1, ci, P
    thr = None
    if pos_idx is not None and pooled is not None:
        try:
            scores = pooled[:, pos_idx]
            yb = (y_all == pos_idx).astype(int)
            if 0 < yb.sum() < len(yb):
                cand, _, _ = best_f1_threshold(yb, scores)
                pred_c = np.where(scores >= cand, pos_idx, 1 - pos_idx)
                pred_h = np.where(scores >= 0.5, pos_idx, 1 - pos_idx)
                if (f1_score(y_all, pred_c, average="macro")
                        > f1_score(y_all, pred_h, average="macro")):
                    thr = float(cand)
        except Exception:
            thr = None
    return dict(combos[best_i]), thr


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
        # nearly-free meta-OOF probabilities for run_cv's threshold
        # tuning: the meta features are already inner-OOF, so a small
        # logistic CV on them costs milliseconds — and saves three
        # FULL StackedEnsemble refits per outer fold (A5, 2026-09-05)
        self.meta_oof_ = None
        if inner_k >= 2:
            try:
                from sklearn.model_selection import cross_val_predict
                self.meta_oof_ = cross_val_predict(
                    LogisticRegression(max_iter=2000), meta, y,
                    groups=groups, cv=inner, method="predict_proba")
            except Exception:
                self.meta_oof_ = None
        self.bases_ = [clone(est).fit(X, y) for _nm, est in self.specs]
        return self

    def predict_proba(self, X):
        return self.meta_.predict_proba(self._meta_features(X, self.bases_))

    def predict(self, X):
        return self.meta_.predict(self._meta_features(X, self.bases_))


def _run_cv_model(name: str, estimator, grid, X, ye, groups_arr,
                  all_splits, classes, binary, pos_idx, seed,
                  n_folds_total, progress=None, turbo=False
                  ) -> tuple[ModelResult, tuple]:
    """One model through every fold; returns (result, template).
    Module-level (not a closure) so the concurrent-models loky pool can
    pickle it as a task (2026-09-12 speed program) — results identical
    to the previous in-function closure (pinned by
    test_parallel_models_match_serial)."""
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
            if progress:
                progress(f"{name}: fold {fold_i}/{n_folds_total}"
                         + (" (grouped)" if groups_arr is not None else ""))
            Xtr, ytr, Xte, yte = X[tr], ye[tr], X[te], ye[te]
            groups_tr = (groups_arr[tr]
                         if groups_arr is not None else None)

            min_tr = min(Counter(ytr.tolist()).values())
            params: dict = {}
            thr = None
            stacked_thr_pending = False
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
                # ONE inner-CV pass: hyperparams AND the binary
                # threshold both come from pooled inner-OOF
                # predictions (no second cross_val_predict, no
                # throwaway refit — see _tune_inner)
                if binary and isinstance(estimator, StackedEnsemble):
                    # A5: the stacked's own meta-OOF (computed in
                    # fit, nearly free) tunes the threshold after
                    # the fold refit — skips 3 full stacked refits
                    params, thr = {}, None
                    stacked_thr_pending = True
                else:
                    params, thr = _tune_inner(
                        estimator, grid, Xtr, ytr, inner,
                        pos_idx=pos_idx if binary else None,
                        groups=groups_tr,
                        serial=_NOT_THREAD_SAFE(name),
                        halve=turbo)
                param_list.append(params)
                param_counter[str(sorted(params.items()))] += 1
                if binary and not stacked_thr_pending:
                    res.thresholds.append(thr)
            else:
                params, thr = {}, None
                param_list.append(params)
                param_counter[str(sorted(params.items()))] += 1

            # refit best hyperparams on the whole training fold
            final = fit_maybe_grouped(
                clone(estimator).set_params(**params), Xtr, ytr,
                groups_tr)
            if binary and stacked_thr_pending:
                # A5: threshold from the fitted stack's meta-OOF
                mo = getattr(final, "meta_oof_", None)
                if mo is not None:
                    try:
                        scores = mo[:, pos_idx]
                        yv = (ytr == pos_idx).astype(int)
                        if 0 < int(yv.sum()) < len(yv):
                            cand, _, _ = best_f1_threshold(yv, scores)
                            pred_c = np.where(scores >= cand, pos_idx,
                                              1 - pos_idx)
                            pred_h = np.where(scores >= 0.5, pos_idx,
                                              1 - pos_idx)
                            if (f1_score(ytr, pred_c, average="macro")
                                    > f1_score(ytr, pred_h,
                                               average="macro")):
                                thr = float(cand)
                    except Exception:
                        thr = None
                res.thresholds.append(thr)
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
        _aggregate(res, fold_metrics, cm_total)
        oof = np.full((len(ye), len(classes)), np.nan)
        seen = oof_cnt > 0
        oof[seen] = oof_sum[seen] / oof_cnt[seen, None]
        res.oof_proba = oof
        res.y_true_encoded = ye
        res.groups = list(groups_arr) if groups_arr is not None else None
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
                groups_arr)
            best_tpl = (estimator, best_params)
    except Exception:
        res.error = traceback.format_exc()
        if progress:
            progress(f"{name}: FAILED")
    return res, best_tpl


def evaluate_models(X: np.ndarray, y: list[str],
                    model_names: list[str] | None = None,
                    k_folds: int = 5, seed: int = RANDOM_STATE,
                    progress_cb=None,
                    groups: list[str] | None = None,
                    repeats: int = 1,
                    wavenumbers: np.ndarray | None = None,
                    turbo: bool = False,
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


    pipelines: dict[str, tuple] = {}
    groups_arr = (np.asarray(groups) if groups is not None else None)

    def run_cv(name: str, estimator, grid) -> tuple[ModelResult, tuple]:
        """Serial path: exact % progress via the shared fold counter."""
        nonlocal done

        def _pct(msg: str):
            progress_cb(int(100 * done / total_work), msg)
        res, tpl = _run_cv_model(name, estimator, grid, X, ye, groups_arr,
                                 all_splits, classes, binary, pos_idx,
                                 seed, n_folds_total, progress=_pct
                                 if progress_cb else None, turbo=turbo)

        def _count(res):       # fold count for the % progress baseline
            return len(res.fold_f1)
        done += _count(res)
        return res, tpl

    # ---- concurrent model evaluation (2026-09-12 speed program) -----
    # Models WITHOUT internal thread pools (the linear/algebra family:
    # PCA+SVM/LDA/LogReg/KNN/NB/MLP/PLS) cannot use the cores alone —
    # those run their whole CV CONCURRENTLY in threads (the
    # _tune_inner pattern, proven since 2026-09-05).  Models WITH
    # internal pools (RF/ET/HGB/boosters/CNN) stay SERIAL: they
    # already saturate every core by themselves — capping them to make
    # room for concurrency measured 0.94-0.96x (bench 2026-09-12), and
    # loky children pay ~2 GB of torch+booster imports each for 1.0x.
    # Big multi-model selections gain the most (the linear block hides
    # behind nothing instead of stacking); results are identical to
    # serial (pinned by test_parallel_models_match_serial).
    _INTERNAL_POOL = ("Random Forest", "Extra Trees", "Peak bands",
                      "Hist Gradient", "Isolation", "CatBoost",
                      "XGBoost", "LightGBM", "1D-CNN", "TabPFN",
                      "Ensemble", "Stacked")
    threadable = [s for s in base_specs
                  if not any(k in s["name"] for k in _INTERNAL_POOL)]
    ran: dict[str, tuple[ModelResult, tuple]] = {}
    if len(threadable) > 1:
        n_threads = min(4, os.cpu_count() or 1)
        from joblib import Parallel, delayed
        gen = Parallel(n_jobs=n_threads, prefer="threads",
                       return_as="generator")(
            delayed(_run_cv_model)(
                s["name"], s["estimator"], s["grid"], X, ye, groups_arr,
                all_splits, classes, binary, pos_idx, seed, n_folds_total,
                progress=None, turbo=turbo)
            for s in threadable)
        n_thr = len(threadable)
        for i, out in enumerate(gen, start=1):
            ran[out[0].name] = out
            if progress_cb:
                progress_cb(int(100 * i / n_thr),
                            f"{out[0].name}: done "
                            f"({i}/{n_thr} models, concurrent)")
    for spec in base_specs:
        if spec["name"] in ran:
            res, tpl = ran[spec["name"]]
        else:
            res, tpl = run_cv(spec["name"], spec["estimator"],
                              spec["grid"])
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
    # aligned with biochemistry.BANDS (the Result-page biochemistry card
    # and the keratin guard read THAT table — 938 IS the keratin marker
    # and 853/854 proline/hydroxyproline collagen there)
    782: "DNA/RNA phosphate", 788: "DNA", 830: "collagen",
    853: "collagen", 880: "tryptophan", 938: "keratin",
    1003: "phenylalanine",
    1032: "phenylalanine", 1095: "phosphate / DNA", 1130: "C–C lipids",
    1209: "collagen", 1240: "amide III", 1335: "nucleic acids",
    1450: "CH2 lipids/proteins", 1555: "tryptophan", 1580: "nucleic acids",
    1615: "tyrosine/tryptophan", 1655: "amide I",
}


def _band_name(center: float) -> str:
    if not BAND_NAMES:
        return ""
    best = min(BAND_NAMES, key=lambda c: abs(c - center))
    # no leading space — consumers join with their own separator
    return (f"({BAND_NAMES[best]})" if abs(best - center) <= 25 else "")


def region_importance_shap(X, y, groups, wn, seed: int = RANDOM_STATE,
                           smooth: int = 21, n_bands: int = 5):
    """
    SIGNED explainability: TreeSHAP values of a RandomForest mapped back
    onto the wavenumber axis (Contreras et al. 2024 pattern).  Returns
    (wn, signed_importance, bands) where positive importance pushes the
    prediction toward the positive class (sorted classes[1]) and bands
    are (center_cm1, share, name) of the top contributing regions.

    The surrogate RF + TreeExplainer + mean signed SHAP are CACHED per
    (X, seed, n_estimators): the auto battery used to refit the same
    RF-300 + full TreeSHAP up to three times back-to-back (regions ->
    band agreement -> biochemistry) and once more per local-explain
    click (2026-09-12 speed program).
    """
    from scipy.ndimage import uniform_filter1d

    classes_ = sorted(set(y))
    ye = _encode(list(y), classes_)
    est, _expl, sv_pos_mean = _surrogate_shap(
        X, ye, seed=seed)
    signed = sv_pos_mean                       # mean signed SHAP (cached)
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


# session-level surrogate-SHAP cache: key -> (RF, TreeExplainer, mean
# signed SHAP over all rows).  Small (a fitted RF-300 + tree paths);
# three training-matrices deep is plenty for regions + band agreement
# + biochemistry + local explain on the same winner.
_SURROGATE_CACHE: dict[tuple, tuple] = {}
_SURROGATE_CACHE_MAX = 3


def clear_surrogate_cache():
    _SURROGATE_CACHE.clear()


def _surrogate_key(X, seed: int, n_estimators: int) -> tuple:
    return (hash(np.asarray(X).tobytes()), int(seed), int(n_estimators))


def _surrogate_shap(X, y_encoded, seed: int = RANDOM_STATE,
                    n_estimators: int = 300):
    """Fit (or reuse) the tree surrogate + TreeExplainer for X.  One fit
    per training matrix — shared by region importance, band agreement,
    biochemistry and the Result-page local explanation.  n_jobs=1
    deliberately (gotcha #25: loky pools beside the live Qt loop are
    the native-crash race; the cache makes the single fit pay off)."""
    key = _surrogate_key(X, seed, n_estimators)
    hit = _SURROGATE_CACHE.get(key)
    if hit is not None:
        return hit
    # n_jobs=1: these diagnostics run from a GUI QThread — an all-core
    # loky pool next to the live Qt main loop is the native-crash race
    # that killed the app on 2026-09-12 (gotcha #25; fits are seconds)
    est = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", n_jobs=1,
        random_state=seed).fit(np.asarray(X), y_encoded)
    import shap
    expl = shap.TreeExplainer(est)
    sv = expl.shap_values(np.asarray(X), check_additivity=False)
    if isinstance(sv, list):                       # classic list layout
        sv_pos = sv[-1]
    else:                                          # (n, features, classes)
        sv = np.asarray(sv)
        sv_pos = sv[..., -1]
    mean_signed = np.asarray(sv_pos).mean(axis=0)
    if len(_SURROGATE_CACHE) >= _SURROGATE_CACHE_MAX:
        _SURROGATE_CACHE.pop(next(iter(_SURROGATE_CACHE)))
    _SURROGATE_CACHE[key] = (est, expl, mean_signed)
    return est, expl, mean_signed


def surrogate_explainer(X, y, seed: int = RANDOM_STATE,
                        n_estimators: int = 300):
    """Fitted surrogate RF + shap.TreeExplainer for (X, y) — the shared
    cache behind 'Explain this prediction': the Result page used to
    refit RF-300 + rebuild the explainer on every click."""
    classes_ = sorted(set(y))
    ye = _encode(list(y), classes_)
    return _surrogate_shap(X, ye, seed=seed, n_estimators=n_estimators)[:2]


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
        n_estimators=400, class_weight="balanced", n_jobs=1,
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
                      seed: int = RANDOM_STATE, progress=None,
                      turbo: bool = False
                      ) -> dict:
    """
    UNBIASED end-to-end evaluation: inside every outer patient-grouped
    fold the PREPROCESSING is re-chosen on the training patients only
    (mini optimizer grid), then the best model is scored on the held-out
    patients.  Removes the selection bias of picking preprocessing on
    the same data it is scored on.

    Returns {mean_f1, std_f1, fold_f1s, fold_choices} plus POOLED
    honest metrics over all outer test rows (2026-09-08):
    {sens, spec, acc, auc, cm} — macro sens/spec from one pooled
    confusion matrix; AUC binary-only (NaN otherwise).
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
    # pooled test predictions across all outer folds -> the honest
    # sens/spec/AUC the GUI banner and the reports display (2026-09-08:
    # mean_f1 alone left the honest estimate incomplete next to the
    # full-metric optimistic numbers)
    pool_y: list = []
    pool_pred: list = []
    pool_p: list = []
    pool_p_proba: list = []       # per fold: p_col was a true probability?
    for fi, (tr, te) in enumerate(splitter.split(X_raw, y_arr, g_arr),
                                  start=1):
        if progress:
            progress(f"nested fold {fi}/{k}: choosing preprocessing on "
                     "training patients…")
        sub = optimize.optimize_preprocessing(
            X_raw[tr], wn, [y[i] for i in tr],
            groups=(None if g_arr is None else g_arr[tr].tolist()),
            grid=grid, k=(2 if turbo else 3), seed=seed,
            progress=lambda m: None)
        params, model_name = sub["params"], sub["best"]["model"]
        Xtr = pp.preprocess_matrix(X_raw[tr], params, wn=wn)
        Xte = pp.preprocess_matrix(X_raw[te], params, wn=wn)
        est = clone(model_by_name[model_name])
        est.fit(Xtr, y_arr[tr])
        pred = est.predict(Xte)
        f1 = f1_score(y_arr[te], pred, average="macro")
        fold_f1s.append(float(f1))
        fold_choices.append({"fold": fi, "preprocess": sub["best"]["label"],
                             "model": model_name, "f1": float(f1)})
        pool_y.extend(y_arr[te])
        pool_pred.extend(pred)
        p_col = None
        p_is_proba = False
        if hasattr(est, "predict_proba"):
            try:
                proba = np.asarray(est.predict_proba(Xte))
                if proba.ndim == 2 and proba.shape[1] == 2:
                    p_col = proba[:, 1]
                    p_is_proba = True
            except Exception:
                p_col = None
        if p_col is None and hasattr(est, "decision_function"):
            # AUC needs a SCORE, not a calibrated probability — plain
            # SVC (the optimizer's frequent pick) only exposes
            # decision_function
            try:
                dec = np.asarray(est.decision_function(Xte))
                if dec.ndim == 1:
                    p_col = dec
            except Exception:
                p_col = None
        pool_p.extend(p_col if p_col is not None
                      else [None] * len(pred))
        pool_p_proba.append(p_is_proba)
        if progress:
            progress(f"nested fold {fi}/{k}: {sub['best']['label']} + "
                     f"{model_name} -> F1 {f1:.3f}")
    labels = sorted(set(y_arr.tolist()))
    cm = confusion_matrix(pool_y, pool_pred, labels=labels)
    per = class_metrics_from_cm(cm)
    sens = float(np.mean([m["sens"] for m in per.values()]))
    spec = float(np.mean([m["spec"] for m in per.values()]))
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    auc = float("nan")
    p_arr = np.asarray(pool_p, dtype=float)
    if len(pool_p) == len(pool_y) and len(labels) == 2 \
            and bool(np.all(np.isfinite(p_arr))):
        enc = np.asarray([0 if v == labels[0] else 1 for v in pool_y])
        try:
            auc = float(roc_points(enc, p_arr)[2])
        except Exception:
            auc = float("nan")
    # supplementary metrics (2026-09-08 formula audit) — ADDITIVE keys,
    # computed from the SAME pooled rows/probabilities
    supp = {}
    if len(labels) == 2:
        enc = np.asarray([0 if v == labels[0] else 1 for v in pool_y])
        supp["balanced_accuracy"] = balanced_accuracy_from_cm(cm)
        supp["mcc"] = mcc_from_cm(cm)
        if bool(np.all(np.isfinite(p_arr))) and len(p_arr) == len(enc):
            # Brier is only defined on PROBABILITIES: decision margins
            # (plain SVC) are unbounded — a margin "brier" of 8.6 was
            # reported before the guard (2026-09-12 audit).  AUC/PR-AUC
            # stay valid on margins (rank-based).
            supp["brier"] = (brier_score(enc, p_arr)
                             if all(pool_p_proba) else float("nan"))
            supp["pr_auc"] = float(pr_points(enc, p_arr)[2])
        else:
            supp["brier"] = float("nan")
            supp["pr_auc"] = float("nan")
    return {"mean_f1": float(np.mean(fold_f1s)),
            "std_f1": float(np.std(fold_f1s)),
            "fold_f1s": fold_f1s, "fold_choices": fold_choices,
            # pooled honest metrics (additive — old consumers only read
            # the four keys above)
            "sens": sens, "spec": spec, "acc": acc, "auc": auc,
            "cm": cm, **supp}


def _lc_fraction(X, y_encoded, groups, frac_index: int, frac: float,
                 k: int, seed: int, estimator):
    """One learning-curve point (patient subset + grouped CV), runnable
    in a loky child (2026-09-12 speed program).  Bit-identical to the
    serial loop: the serial code shuffles the patient list ONCE with a
    seeded rng and every fraction takes a prefix of that permutation —
    this child replays the same single shuffle."""
    import study_stats as ss
    from sklearn.metrics import f1_score

    ss._pin_child()
    est0 = ss._cap_native_threads(clone(estimator),
                                  ss._threads_budget())
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    take = uniq[:max(2, int(round(len(uniq) * frac)))]
    m = np.isin(groups, take)
    Xs, ys, gs = X[m], np.asarray(y_encoded)[m], groups[m]
    kk = int(min(k, len(take)))
    if kk < 2 or min(Counter(ys.tolist()).values()) < 2:
        return None                      # not enough patients for k folds
    sgkf = StratifiedGroupKFold(n_splits=kk, shuffle=True,
                                random_state=seed)
    f1s = []
    for tr, te in sgkf.split(Xs, ys, gs):
        est = clone(est0)
        est.fit(Xs[tr], ys[tr])
        f1s.append(f1_score(ys[te], est.predict(Xs[te]),
                            average="macro"))
    return len(take), float(np.mean(f1s)), float(np.std(f1s))


def learning_curve_by_groups(X, y_encoded, groups, estimator,
                             k: int = 5, seed: int = RANDOM_STATE,
                             fractions=(0.25, 0.50, 0.75, 1.0),
                             cancel_check=None, progress=None, jobs: int = 1):
    """
    Grouped-CV macro-F1 at a growing number of patients — the classic
    diagnostic for 'how much would more data worth?'.

    Returns (n_patients_list, mean_f1_list, std_f1_list).
    `jobs` > 1 evaluates the fractions concurrently in the thread-
    pinned loky pool (bit-identical to serial — same rng trajectory,
    same seeds; 2026-09-12 speed program).
    `cancel_check` aborts between points with RuntimeError('diagnostics
    cancelled by user'); `progress(msg)` fires per fraction
    (2026-09-06: chain winners make this slow — the user must see
    movement).
    """
    from sklearn.metrics import f1_score

    if jobs > 1 and len(fractions) > 1:
        from joblib import Parallel, delayed
        from study_stats import (_clear_threads_budget, _drain,
                                 _set_threads_budget)
        _set_threads_budget(jobs)
        try:
            gen = Parallel(n_jobs=jobs, max_nbytes=100,
                           prefer="processes", return_as="generator")(
                delayed(_lc_fraction)(X, y_encoded, groups, i, f, k, seed,
                                      estimator)
                for i, f in enumerate(fractions))
            pts = _drain(gen, len(fractions), cancel_check, progress,
                         fmt=lambda i, r: (
                             f"{r[0]} patients: F1 {r[1]:.3f} ± {r[2]:.3f}"
                             if r is not None else f"point {i} skipped"))
        finally:
            _clear_threads_budget()
        pts = [p for p in pts if p is not None]
        return ([p[0] for p in pts], [p[1] for p in pts],
                [p[2] for p in pts])

    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    sizes, means, stds = [], [], []
    for frac in fractions:
        if cancel_check is not None and cancel_check():
            raise RuntimeError("diagnostics cancelled by user")
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
            if cancel_check is not None and cancel_check():
                raise RuntimeError("diagnostics cancelled by user")
            est = clone(estimator)
            est.fit(Xs[tr], ys[tr])
            f1s.append(f1_score(ys[te], est.predict(Xs[te]),
                                average="macro"))
        sizes.append(len(take))
        means.append(float(np.mean(f1s)))
        stds.append(float(np.std(f1s)))
        if progress is not None:
            progress(f"{len(take)} patients: F1 "
                     f"{means[-1]:.3f} ± {stds[-1]:.3f}")
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
    # macro ± is now the FOLD-TO-FOLD variability (std of fold macro-F1,
    # population ddof=0) — the old between-class spread read like a CV
    # error bar but was dispersion across classes (2026-09-12 audit);
    # the per-class spread stays in res.per_class
    res.macro = {k: (float(np.mean(agg[k])), float(np.std(agg[k])))
                 for k in keys}
    if res.fold_f1:
        res.macro["f1"] = (float(np.mean(res.fold_f1)),
                           float(np.std(res.fold_f1)))


# --------------------------------------------------------------------------
# Persistence + prediction
# --------------------------------------------------------------------------
def save_bundle(path: str, winner: ModelResult, wavenumbers: np.ndarray,
                prep_params, dataset_name: str = "",
                paired: bool = False, pqn: bool = False, **extra) -> str:
    bundle = {
        "model_name": winner.name,
        "pipeline": winner.pipeline,
        "classes": winner.classes,
        "threshold": winner.threshold,
        "wavenumbers": np.asarray(wavenumbers),
        "prep_params": prep_params,
        "macro": winner.macro,
        # SimpleNamespace winners (3SSE dialog / Model Lab finalize)
        # carry no per-class table — getattr, never attribute access
        # (2026-09-06: bare winner.per_class broke saving 3SSE bundles)
        "per_class": getattr(winner, "per_class", None),
        "dataset_name": dataset_name,
        "paired": paired,          # predict-time needs a normal reference
        "pqn": pqn,                # Margin+PQN: PQN-normalize against the
        #                          # reference BEFORE the deviation
        #                          # (2026-09-06: the flag never existed and
        #                          # every deploy path skipped the PQN step)
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
    # bundles saved by CLI scripts pickle their classes as __main__.*
    # (e.g. sequential.SequentialChain saved from `python sequential.py`)
    # — register them on __main__ so unpickling works everywhere
    import __main__ as _main
    try:
        import sequential as _seq
    except ImportError:
        _seq = None
    if _seq is not None:
        for _name in ("SequentialChain", "_SpectralSlice"):
            if hasattr(_seq, _name) and not hasattr(_main, _name):
                setattr(_main, _name, getattr(_seq, _name))
    return joblib.load(path)


def _apply_reference(xc: np.ndarray, reference: np.ndarray,
                     bundle: dict) -> np.ndarray:
    """Finish a paired feature EXACTLY like training (paired.py):
    PQN-normalize against the reference when the bundle was trained in
    Margin+PQN mode, then subtract the reference.  Shared by the single
    and batched predict paths so they cannot drift apart (2026-09-06:
    the PQN step was silently skipped at deploy for every PQN bundle —
    same failure class as the wn_calibrate mismatch)."""
    from preprocessing import pqn_normalize
    if len(reference) != len(xc):
        raise ValueError(
            f"paired reference has {len(reference)} points but "
            f"the spectrum has {len(xc)} after preprocessing")
    if bundle.get("pqn"):
        xc = pqn_normalize(xc, reference)
    return xc - reference


def _reject_raw_degenerate(it: np.ndarray):
    """BUG-1 raw-input check: junk spectra (empty, lone spike, subnormal
    scale) must be rejected BEFORE preprocessing — the wavelet stage
    smears a single spike into many tiny values, hiding it afterwards.
    Real files carry intensity ~1e0..1e5 across essentially all points."""
    nz = int(np.count_nonzero(np.abs(it) > 1e-9))
    if nz < max(3, len(it) // 100):
        raise ValueError(
            f"spectrum has no usable signal ({nz} of {len(it)} points "
            "carry intensity — empty, single-point or numerically "
            "degenerate); fix or re-measure the file")


def _reject_degenerate(xc: np.ndarray):
    """Deploy-boundary check on the PREPROCESSED vector (BUG-1):
    non-finite or subnormal-scale features (float-underflow territory
    for PCA) must never become a silent confident prediction."""
    if (not np.all(np.isfinite(xc))
            or float(np.abs(xc).max(initial=0.0)) < 1e-9):
        raise ValueError(
            "spectrum has no usable signal in the model's wavenumber "
            "range (empty, all-zero or numerically degenerate after "
            "preprocessing)")


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
    from preprocessing import (PreprocessParams, align_to_grid, crop_mask,
                               preprocess_spectrum)

    def _as_params(p):
        # bundles store PreprocessParams, but tolerate plain dicts at this
        # trust boundary (hand-made or future writers)
        return p if hasattr(p, "validate") else PreprocessParams(**p)

    # release-gate: mismatched/empty arrays crashed at the sort itself
    # (IndexError) before any guard could speak — reject cleanly FIRST
    if len(wavenumbers) != len(intensities) or len(intensities) == 0:
        raise ValueError(
            f"empty or mismatched spectrum arrays ({len(wavenumbers)} "
            f"wavenumbers vs {len(intensities)} intensities) — the "
            "file is corrupt")
    order = np.argsort(wavenumbers)
    wn, it = np.asarray(wavenumbers)[order], np.asarray(intensities)[order]
    # non-finite guard (2026-09-08 audit BUG-1): an all-NaN spectrum used
    # to reach the classifier as zeros and came back with MAXIMAL
    # confidence (Tumor p=1.0) — silent garbage.  Fail loudly instead.
    n_bad = int(np.count_nonzero(~np.isfinite(it)))
    if n_bad:
        raise ValueError(
            f"spectrum contains {n_bad} NaN/Inf intensity points — fix "
            "or re-measure the file; predicting on it would fabricate a "
            "confident answer")
    grid = np.asarray(bundle["wavenumbers"], dtype=float)
    # overlap guard (2026-09-05): np.interp CONSTANT-extrapolates, so a
    # spectrum whose measured axis does not span the training grid would
    # silently be filled with edge intensities and confidently
    # mispredicted.  1 cm-1 tolerance for boundary rounding.
    lo, hi = float(grid.min()), float(grid.max())
    if float(wn[0]) > lo + 1.0 or float(wn[-1]) < hi - 1.0:
        raise ValueError(
            f"spectrum axis {float(wn[0]):.1f}-{float(wn[-1]):.1f} cm-1 "
            f"does not cover the model's wavenumber range "
            f"{lo:.1f}-{hi:.1f} — interpolation would fabricate the "
            f"missing region")
    _reject_raw_degenerate(it)
    y = np.interp(grid, wn, it)
    params = _as_params(bundle["prep_params"])
    # 2026-09-06: bundles trained with wn_calibrate=True were predicted
    # WITHOUT the Phe-1003 alignment — every deploy-time feature was
    # shifted/out-of-distribution and predictions saturated to one class
    # (all-Normal→Tumor).  align_to_grid mirrors preprocess_matrix's
    # calibrate-then-crop order so deploy features match training.
    if params.wn_calibrate:
        y = align_to_grid(wn, it, grid, params)
    # apply the crop range exactly as at training time (the model expects
    # the cropped feature count); fall back to the uncropped vector for
    # bundles that were trained without cropping
    m = crop_mask(grid, params)
    proba = None
    for vec in ((y[m], y) if int(m.sum()) != len(grid) else (y,)):
        xc = preprocess_spectrum(vec, params)
        _reject_degenerate(xc)
        if reference is not None:
            xc = _apply_reference(xc, reference, bundle)
        try:
            # B3: winners that support test-time augmentation (the
            # 1D-CNN) predict from an augmented-view average — small
            # robustness gain at deploy time, zero training change
            _clf = getattr(bundle["pipeline"], "steps", None)
            _clf = _clf[-1][1] if _clf else bundle["pipeline"]
            if hasattr(_clf, "predict_proba_tta"):
                proba = _clf.predict_proba_tta(xc.reshape(1, -1))[0]
            else:
                proba = bundle["pipeline"].predict_proba(
                    xc.reshape(1, -1))[0]
            break
        except ValueError:
            if reference is not None:
                raise                    # paired mismatch is a real error
            continue
    if proba is None:
        raise ValueError(
            "Spectrum length does not match the saved model "
            f"(expected {int(m.sum())} or {len(grid)} points).")
    return _postprocess_prediction(bundle, proba)


def _postprocess_prediction(bundle: dict, proba: np.ndarray) -> dict:
    """Shared result assembly for single + batched prediction: Platt
    mapping (binary), tuned-threshold class decision."""
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


def _expected_features(bundle: dict) -> int | None:
    """Feature count the fitted pipeline was trained on (walks chain
    wrappers); None when undetectable."""
    c = bundle["pipeline"]
    for _ in range(4):
        n = getattr(c, "n_features_in_", None)
        if n:
            return int(n)
        nxt = getattr(c, "fitted_", None)
        c = nxt[0] if isinstance(nxt, list) and nxt else None
        if c is None:
            return None
    return None


def predict_with_bundle_many(bundle: dict, spectra: list,
                             references: list | None = None
                             ) -> tuple[list, dict]:
    """
    Batched twin of predict_with_bundle: identical per-spectrum result
    dicts, but ONE pipeline call per feature-length group instead of
    one call per spectrum.  Exists because per-CALL cost dominates for
    some voters (TabPFN ≈ seconds per call on CPU): file-by-file
    prediction of a folder took minutes, batched takes seconds
    (2026-09-06).  Per-row predictors are batch-equivalent (measured
    max |Δp| ≈ 8e-8 on the TabPFN voter, i.e. float noise).

    spectra: list of (wavenumbers, intensities); references: list of
    preprocessed reference vectors or None per spectrum.  Returns
    (results, errors): results[i] is the result dict or None; errors
    maps i -> message for the failed rows.
    """
    from preprocessing import (PreprocessParams, align_to_grid, crop_mask,
                               preprocess_spectrum)

    def _as_params(p):
        return p if hasattr(p, "validate") else PreprocessParams(**p)

    spectra = list(spectra)
    refs = list(references) if references is not None \
        else [None] * len(spectra)
    grid = np.asarray(bundle["wavenumbers"], dtype=float)
    params = _as_params(bundle["prep_params"])
    m = crop_mask(grid, params)
    exp = _expected_features(bundle)
    groups: dict[int, list[tuple[int, np.ndarray]]] = {}
    errors: dict[int, str] = {}
    lo, hi = float(grid.min()), float(grid.max())
    for i, ((wn, it), ref) in enumerate(zip(spectra, refs, strict=True)):
        try:
            # release-gate: mismatched/empty arrays crashed at the sort
            # itself before any guard could speak — reject cleanly FIRST
            if len(wn) != len(it) or len(it) == 0:
                raise ValueError(
                    f"empty or mismatched spectrum arrays ({len(wn)} "
                    f"wavenumbers vs {len(it)} intensities) — the file "
                    "is corrupt")
            order = np.argsort(wn)
            w, x = np.asarray(wn)[order], np.asarray(it)[order]
            # BUG-1 guards (2026-09-08): identical to the single-spectrum
            # path — non-finite input and degenerate features must land
            # in errors[i], never in a confident prediction
            n_bad = int(np.count_nonzero(~np.isfinite(x)))
            if n_bad:
                raise ValueError(
                    f"spectrum contains {n_bad} NaN/Inf intensity points "
                    "— fix or re-measure the file; predicting on it "
                    "would fabricate a confident answer")
            if float(w[0]) > lo + 1.0 or float(w[-1]) < hi - 1.0:
                raise ValueError(
                    f"spectrum axis {float(w[0]):.1f}-{float(w[-1]):.1f} "
                    f"cm-1 does not cover the model's wavenumber range "
                    f"{lo:.1f}-{hi:.1f} — interpolation would fabricate "
                    f"the missing region")
            _reject_raw_degenerate(x)
            y = np.interp(grid, w, x)
            if params.wn_calibrate:
                y = align_to_grid(w, x, grid, params)
            for vec in ((y[m], y) if int(m.sum()) != len(grid) else (y,)):
                xc = preprocess_spectrum(vec, params)
                _reject_degenerate(xc)
                if ref is not None:
                    xc = _apply_reference(xc, ref, bundle)
                if exp is None or len(xc) == exp:
                    groups.setdefault(len(xc), []).append((i, xc))
                    break
            else:
                raise ValueError(
                    "Spectrum length does not match the saved model "
                    f"(expected {exp or int(m.sum())} points).")
        except ValueError as exc:
            errors[i] = str(exc)
    results: list = [None] * len(spectra)
    for _n, items in groups.items():
        X = np.vstack([xc for _i, xc in items])
        _clf = getattr(bundle["pipeline"], "steps", None)
        _clf = _clf[-1][1] if _clf else bundle["pipeline"]
        try:
            if hasattr(_clf, "predict_proba_tta"):
                proba = np.vstack([
                    _clf.predict_proba_tta(X[r:r + 1])[0]
                    for r in range(len(X))])
            else:
                proba = bundle["pipeline"].predict_proba(X)
        except ValueError:
            # unexpected length: fall back to the proven single path
            for i, _xc in items:
                try:
                    results[i] = predict_with_bundle(
                        bundle, spectra[i][0], spectra[i][1],
                        reference=refs[i])
                except ValueError as exc:
                    errors[i] = str(exc)
            continue
        for (i, _xc), p in zip(items, proba, strict=True):
            results[i] = _postprocess_prediction(bundle, p)
    return results, errors


def roc_points(y_true_encoded: np.ndarray, proba_pos: np.ndarray):
    """ROC curve + AUC helper (for plotting)."""
    fpr, tpr, _ = roc_curve(y_true_encoded, proba_pos)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def pr_points(y_true_encoded: np.ndarray, proba_pos: np.ndarray):
    """Precision-recall curve + average precision — the imbalance-aware
    companion to ROC (a rare-disease screen can look fine on ROC and
    still have poor PPV)."""
    from sklearn.metrics import (average_precision_score,
                                 precision_recall_curve)
    prec, rec, _ = precision_recall_curve(y_true_encoded, proba_pos)
    ap = float(average_precision_score(y_true_encoded, proba_pos))
    return prec, rec, ap


# --------------------------------------------------------------------------
# Grouped permutation test of the winner's OOF AUC
# --------------------------------------------------------------------------
def grouped_oof_auc(est, X, y, groups, k: int = 5,
                    seed: int = RANDOM_STATE) -> float:
    """Pooled grouped-CV out-of-fold AUC of a (fitted-params) estimator."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    y = np.asarray(y)
    if len(np.unique(y)) != 2:
        raise ValueError("permutation AUC test is binary-only")
    cv = StratifiedGroupKFold(n_splits=int(k), shuffle=True,
                              random_state=seed)
    proba = cross_val_predict(clone(est), X, y, cv=cv, n_jobs=1,
                              method="predict_proba", groups=groups)
    classes = np.unique(y)
    pos = classes[1]
    return float(roc_auc_score((y == pos).astype(int), proba[:, 1]))


def permutation_auc_p(est, X, y, groups, n_perm: int = 100, k: int = 5,
                      seed: int = RANDOM_STATE) -> dict:
    """
    Permutation test with PATIENT-level label shuffling: build the null
    by reassigning whole patients' labels, re-running the grouped CV.
    Empirical p = (1 + #{null >= observed}) / (1 + n_perm).  This is the
    hard answer to "is the AUC real at n<500?".

    Patients carrying BOTH classes (paired design) break the
    patient-level label shuffle — their single first-row label would be
    assigned arbitrarily.  They are EXCLUDED and counted; but when the
    cohort is (almost) FULLY paired and exclusion would remove a class
    entirely, the null switches to WITHIN-PATIENT label shuffling
    (shuffle which spectra of each patient carry which label — the
    natural paired-design null) (2026-09-12 audit).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    mixed = [g for g in np.unique(groups)
             if len(np.unique(y[groups == g])) > 1]
    within_patient = False
    if mixed:
        keep = ~np.isin(groups, mixed)
        yk, gk = y[keep], groups[keep]
        n_keep = len(np.unique(gk))
        cls_counts = (np.unique(yk, return_counts=True)[1] if len(yk)
                      else np.array([0]))
        too_small = (len(mixed) == len(np.unique(groups))
                     or len(np.unique(yk)) < 2
                     or n_keep < k + 1
                     or int(cls_counts.min()) < 2)
        if not too_small:
            Xk = X[keep]
            try:                       # even a big-enough kept set can
                obs = grouped_oof_auc(est, Xk, yk, gk, k=k, seed=seed)
            except Exception:           # still fold into one class —
                within_patient = True   # fall back to the paired null
            else:
                X, y, groups = Xk, yk, gk
                print(f"[perm] excluding {len(mixed)} mixed-label "
                      f"patient(s) from the permutation test")
        else:
            within_patient = True
    if within_patient:
        try:
            obs = grouped_oof_auc(est, X, y, groups, k=k, seed=seed)
        except Exception:
            raise ValueError(
                "permutation_auc_p: within-patient null needs at least "
                "one patient with both classes and >=2 labels overall")
    elif not mixed:
        obs = grouped_oof_auc(est, X, y, groups, k=k, seed=seed)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    null = []
    for _ in range(int(n_perm)):
        if within_patient:
            y_perm = y.copy()
            for g in uniq:
                idx = np.where(groups == g)[0]
                y_perm[idx] = rng.permutation(y[idx])
        else:
            perm_lab = dict(zip(uniq, rng.permutation(
                [y[groups == g][0] for g in uniq])))
            y_perm = np.array([perm_lab[g] for g in groups])
        if len(np.unique(y_perm)) < 2:
            continue
        try:
            null.append(grouped_oof_auc(est, X, y_perm, groups, k=k,
                                        seed=seed))
        except Exception:
            continue
    p = (1 + sum(1 for v in null if v >= obs)) / (1 + len(null))
    return {"auc": obs, "p": float(p), "n_perm": len(null),
            "n_mixed_excluded": 0 if within_patient else len(mixed),
            "null": ("within-patient shuffle" if within_patient
                     else "patient-label shuffle"),
            "null_mean": float(np.mean(null)) if null else float("nan")}


# --------------------------------------------------------------------------
# Model-specific wavenumber-importance profiles (explainability)
# --------------------------------------------------------------------------
def pls_vip(pls_fitted) -> np.ndarray:
    """VIP scores of a fitted PLSRegression (p weights vs explained Y)."""
    w = pls_fitted.x_weights_                    # (p, a)
    t = pls_fitted.x_scores_                     # (n, a)
    q = pls_fitted.y_loadings_                   # (a, ) or (a, m)
    q = np.asarray(q).reshape(t.shape[1], -1)
    ssy = (t ** 2).sum(axis=0) * (q ** 2).sum(axis=1)   # per component
    w2 = w ** 2
    p = w.shape[0]
    vip = np.sqrt(p * (w2 @ ssy) / max(ssy.sum(), 1e-12))
    return np.asarray(vip).ravel()


def winner_importance(winner_pipeline, X, y=None, wn=None) -> np.ndarray:
    """
    Best-effort per-wavenumber importance profile of a fitted winner:
    PLS-family → VIP; 1D-CNN → Grad-CAM energy; tree/other → RF-surrogate
    SHAP (when y is given) or a |mean| profile as last resort.  Returns a
    non-negative (n_wavenumbers,) profile aligned with X's columns.
    """
    est = winner_pipeline
    steps = getattr(est, "steps", None)
    for s in (dict(steps).values() if steps else [est]):
        if hasattr(s, "pls_"):
            return pls_vip(s.pls_)
    named = getattr(est, "named_steps", None) or {}
    clf = named.get("clf", est)
    if HAS_TORCH and isinstance(clf, CNN1DClassifier):
        return clf.grad_cam(X).mean(axis=0)
    if y is not None and wn is not None:
        # groups slot is unused inside region_importance_shap; it used
        # to receive wn positionally -> "missing 1 required positional
        # argument: 'wn'" killed every Band-agreement run (2026-09-06)
        _, signed, _ = region_importance_shap(X, y, None, wn)
        return np.abs(signed)
    return np.abs(np.asarray(X, dtype=float)).mean(axis=0)


# --------------------------------------------------------------------------
# Clinical covariate fusion (spectra + tobacco/age/sex/subsite)
# --------------------------------------------------------------------------
def covariate_fusion_cv(proba_pos: np.ndarray, covariates, y, groups,
                        k: int = 5, seed: int = RANDOM_STATE) -> dict:
    """
    Does adding clinical covariates (tobacco pack-years, age, sex,
    subsite…) to the winner's OOF probability improve discrimination?
    Logistic meta-model over [p, covariates] vs [p] alone, patient-
    grouped CV both times — Hanna 2024's confirmed open gap: nobody has
    formally merged Raman with clinical risk factors.  covariates is an
    (n, m) numeric/one-hot matrix aligned with proba_pos.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    y = np.asarray(y)
    if len(np.unique(y)) != 2:
        return {}
    p = np.asarray(proba_pos, dtype=float).reshape(-1, 1)
    C = np.asarray(covariates, dtype=float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    C = np.nan_to_num(C)
    cv = StratifiedGroupKFold(n_splits=int(k), shuffle=True,
                              random_state=seed)
    lr = LogisticRegression(max_iter=2000)
    base = cross_val_predict(lr, p, y, cv=cv, groups=groups,
                             method="predict_proba")[:, 1]
    fused = cross_val_predict(lr, np.hstack([p, C]), y, cv=cv,
                              groups=groups,
                              method="predict_proba")[:, 1]
    classes = np.unique(y)
    yb = (y == classes[1]).astype(int)
    a0 = float(roc_auc_score(yb, base))
    a1 = float(roc_auc_score(yb, fused))
    return {"auc_base": a0, "auc_fused": a1, "delta": a1 - a0}


def label_error_report(oof_proba: np.ndarray, y,
                       threshold_strong: float = 0.10,
                       threshold_soft: float = 0.25) -> list[dict]:
    """
    Confident-learning-style label-error flags from the winner's pooled
    OUT-OF-FOLD probabilities (cleanlab approach, hand-rolled):
    a spectrum whose GIVEN class got < 10% while another class got > 90%
    is 'likely mislabeled'; < 25%/> 75% is 'review'.  No new fit — pure
    post-processing, so it is cheap and leakage-free by construction.
    Returns [{'i', 'given', 'p_self', 'other', 'p_other', 'tier'}, ...]
    sorted by (p_self ascending).
    """
    p = np.asarray(oof_proba, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    lut = {c: i for i, c in enumerate(classes)}
    out = []
    for i in range(len(y)):
        row = p[i]
        gi = lut.get(y[i])
        if gi is None or np.isnan(row).all():
            continue
        p_self = float(row[gi])
        rest = np.delete(row, gi)
        if len(rest) == 0 or np.isnan(rest).all():
            continue
        jo = int(np.nanargmax(rest))
        others = [c for k, c in enumerate(classes) if k != gi]
        other = str(others[jo])
        p_other = float(rest[jo])
        if p_self < threshold_strong and p_other > 1 - threshold_strong:
            tier = "likely"
        elif p_self < threshold_soft and p_other > 1 - threshold_soft:
            tier = "review"
        else:
            continue
        out.append({"i": i, "given": str(y[i]), "p_self": p_self,
                    "other": other, "p_other": p_other, "tier": tier})
    return sorted(out, key=lambda r: r["p_self"])


def band_stability(profiles: list[np.ndarray], top_k: int = 20
                   ) -> dict:
    """
    Stability selection for wavenumber importance: per-fold/repeat
    profiles → per-profile top-k sets → pairwise Jaccard + the bands
    that appear in EVERY top-k set (the defensible ones).  Returns
    {'jaccard_mean', 'jaccard_min', 'stable'} with stable as a boolean
    consensus mask (all-True when only one profile is given).
    """
    if not profiles:
        return {"jaccard_mean": 0.0, "jaccard_min": 0.0,
                "stable": np.zeros(0, dtype=bool)}
    p = min(map(len, profiles))
    tops = [set(np.argsort(-np.asarray(pr, dtype=float)[:p])[:top_k])
            for pr in profiles]
    consensus = set.intersection(*tops)
    stable_mask = np.isin(np.arange(p), list(consensus))
    if len(tops) < 2:
        return {"jaccard_mean": 1.0, "jaccard_min": 1.0,
                "stable": stable_mask}
    jac = []
    for a in range(len(tops)):
        for b in range(a + 1, len(tops)):
            u = tops[a] | tops[b]
            jac.append(len(tops[a] & tops[b]) / len(u) if u else 1.0)
    return {"jaccard_mean": float(np.mean(jac)),
            "jaccard_min": float(np.min(jac)),
            "stable": stable_mask}


# --------------------------------------------------------------------------
# Model card export (TRIPOD+AI-flavored)
# --------------------------------------------------------------------------
def export_model_card(path: str, winner, params=None,
                      dataset_name: str = "", k_folds: int = 5,
                      repeats: int = 1, grouped: bool = True,
                      nested: tuple[float, float] | None = None) -> str:
    """
    Publication/report-ready model card: development data, full
    preprocessing, CV protocol, discrimination + calibration, AUC power
    and the limitations boilerplate a TRIPOD+AI reviewer expects.
    `nested` (mean, std) is the honest nested-evaluation macro-F1 —
    printed next to the selection-optimistic CV number so the card
    cannot overstate expected performance.
    """
    from dataclasses import asdict as _asdict
    w = winner
    lines = ["# Model card — Raman Spectra Classifier", ""]
    lines += [f"**Model**: {w.name}",
              f"**Development data**: {dataset_name or '(unnamed cohort)'}",
              "**Validation**: "
              + (f"{k_folds}-fold patient-grouped CV" if grouped
                 else f"{k_folds}-fold SPECTRUM-LEVEL CV (no patient "
                      "information — patient leaks possible)")
              + (f" × {repeats}" if repeats > 1 else "")
              + " (nested hyperparameter tuning inside folds)", ""]
    if params is not None:
        try:
            lines += ["## Preprocessing", "```",
                      "\n".join(f"{k} = {v}" for k, v in
                                _asdict(params.validate()).items()),
                      "```", ""]
        except Exception:
            pass
    lines += ["## Performance (out-of-fold, pooled)"]
    try:
        lines.append(f"- macro-F1: **{w.macro_f1():.3f}**")
    except Exception:
        pass
    if nested is not None:
        lines.append(
            f"- Honest (nested) macro-F1: **{nested[0]:.3f} ± "
            f"{nested[1]:.3f}** — the number above is selection-"
            "optimistic (winner chosen on the same CV that scores it); "
            "expect the nested estimate on unseen patients")
    try:
        per = getattr(w, "per_class", None)
        if per:
            lines.append("- Per-class (out-of-fold, mean ± std):")

            def _ms(m, k):
                v = m.get(k)
                return f"{v[0]:.3f} ± {v[1]:.3f}" if v else "–"
            for cls in w.classes:
                m = per.get(cls) or {}
                lines.append(
                    f"  - **{cls}**: sens {_ms(m, 'sens')} · "
                    f"spec {_ms(m, 'spec')} · F1 {_ms(m, 'f1')}")
    except Exception:
        pass
    try:
        ye = np.asarray(w.y_true_encoded)
        valid = ~np.isnan(w.oof_proba[:, 1])
        fpr, tpr, auc = roc_points(ye[valid], w.oof_proba[valid, 1])
        _prec, _rec, ap = pr_points(ye[valid], w.oof_proba[valid, 1])
        lines.append(f"- ROC AUC: **{auc:.3f}** · AUPRC: **{ap:.3f}**")
        import clinical as _clin
        n_pos = int((ye[valid] == 1).sum())
        n_neg = int((ye[valid] == 0).sum())
        pw = _clin.auc_power(n_pos, n_neg, auc=auc)
        n_needed = pw.get("n_per_group_for_target")
        n_txt = (f"~{n_needed} per group needed to reach the observed "
                 f"AUC {auc:.3f} at 80% power"
                 if n_needed else
                 "the observed AUC is not reachable at 80% power even "
                 "with 100k per group")
        lines.append(
            f"- AUC power: detectable AUC at 80% power = "
            f"{pw['detectable_auc_80pct']:.3f} "
            f"(n+ {n_pos} / n− {n_neg}; {n_txt})")
    except Exception:
        pass
    lines += ["", "## Limitations",
              "- Single-center development; no external cohort "
              "validation yet (TRIPOD+AI gap).",
              "- Spectrum-level CV numbers can exceed patient-level "
              "reality; patient-grouped CV is the reported standard.",
              "- Triage-to-biopsy aid, not a diagnosis.",
              "- Conformal sets abstain on ambiguous spectra; coverage "
              "is marginal (~90%), not per-patient.",
              "", "_Auto-generated by the Raman Spectra Classifier._"]
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text
