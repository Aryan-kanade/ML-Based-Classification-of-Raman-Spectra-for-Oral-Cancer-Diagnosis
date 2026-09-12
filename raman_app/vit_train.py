"""
vit_train.py — train the spectral Vision Transformer (vit_model.SpectralViT)
on a folder of Raman spectra (2-column .txt/.dat/.csv files, class parsed
from the `C<number>` filename token).

Pipeline: load -> common grid -> preprocess (preprocessing.py defaults) ->
resample to a fixed length -> stratified train/val/test split -> class-
weighted training with AdamW.

Outputs (in --out, default `vit_outputs/`):
  vit_model.pt               checkpoint (weights + config + split + history)
  vit_training_curves.png    loss & accuracy over epochs
  vit_confusion_matrix.png   confusion matrix on the held-out test set

Examples:
    python vit_train.py --data D:\\data\\my_spectra --epochs 25
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")            # headless: figures are saved, not shown
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import dataset as ds
import preprocessing as pp
from clinical_data import (find_data_root, is_clinical_layout,
                           load_clinical_dataset, patient_split,
                           write_report)
from optimize import load_best_params
from plotting import COL_MAIN, COL_RESULT, apply_style, plot_confusion_matrix
from vit_model import (SpectraDataset, SpectralViT, ViTConfig, require_torch,
                       resample_to_len, save_checkpoint)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def resolve_data_folder(explicit: str | None) -> str:
    """--data folder, else ../Data (clinical layout), else the
    auto-detected dataset root (clinical_data.find_data_root)."""
    if explicit:
        if not os.path.isdir(explicit):
            raise SystemExit(f"Data folder not found: {explicit}")
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    real = os.path.join(os.path.dirname(here), "Data")
    if is_clinical_layout(real):
        return real
    root = find_data_root()
    if root:
        return root
    raise SystemExit(
        "No --data folder given and no clinical dataset found.  "
        "Re-run with:  python vit_train.py --data <folder>"
    )


def load_xy(folder: str):
    """Load a folder -> (spectra, X on common grid, labels, grid)."""
    spectra = ds.load_folder(folder)
    spectra = [s for s in spectra if s.label]
    if len(spectra) < 6:
        raise SystemExit(
            f"Only {len(spectra)} labelled spectra in {folder!r} - "
            "need at least 6 (2 per class) to train the ViT."
        )
    counts: dict[str, int] = {}
    for s in spectra:
        counts[s.label] = counts.get(s.label, 0) + 1
    tiny = {c: n for c, n in counts.items() if n < 3}
    if tiny:
        raise SystemExit(
            f"Classes with fewer than 3 spectra cannot be split "
            f"train/val/test: {tiny}"
        )
    grid = ds.common_grid(spectra)
    X, y = ds.to_matrix(spectra, grid)
    return spectra, X, y, grid


def stratified_split(names, labels, test_size, val_size, seed):
    """-> (train_idx, val_idx, test_idx) with a per-class balanced split."""
    idx = np.arange(len(names))
    train_val, test = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed)
    val_frac = val_size / max(1e-9, 1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_frac,
        stratify=[labels[i] for i in train_val], random_state=seed)
    return train, val, test


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device,
              optimizer=None, scheduler=None, augment=False, scaler=None
              ) -> tuple[float, float]:
    """One pass over `loader` -> (mean loss, accuracy). Optimizer=None: eval.

    With `augment`, training batches get gaussian noise (sigma = 5% of the
    feature std) plus a random wavenumber shift of up to 2 points — cheap
    regularization for small spectral datasets. CUDA runs under AMP
    (autocast + GradScaler) for the tensor-core speedup.
    """
    import torch

    training = optimizer is not None
    amp = device.type == "cuda"
    model.train() if training else model.eval()
    total_loss = correct = n = 0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = (X.to(device, non_blocking=amp),
                    y.to(device, non_blocking=amp))
            if augment:
                X = X + torch.randn_like(X) * (0.05 * X.std())
                X = torch.roll(X, int(torch.randint(-2, 3, (1,))), dims=1)
                if torch.rand(1).item() < 0.5:
                    # SpecAugment band-masking (Park 2019): zero one
                    # random wavenumber window — forces peak redundancy
                    L = X.shape[1]
                    w = max(4, int(0.05 * L))
                    st = int(torch.randint(0, max(1, L - w), (1,)))
                    X[:, st:st + w] = 0.0
                if torch.rand(1).item() < 0.5:
                    # mixup (Zhang 2018): interpolate pairs; targets become
                    # one-hot probability vectors so CrossEntropyLoss
                    # accepts them
                    lam = float(np.random.beta(0.2, 0.2))
                    perm = torch.randperm(X.size(0), device=X.device)
                    X = lam * X + (1 - lam) * X[perm]
                    n_cls = getattr(getattr(model, "cfg", None), "n_classes",
                                    None) or int(y.max().item()) + 1
                    oh = torch.nn.functional.one_hot(
                        y.long(), num_classes=n_cls).float()
                    y = lam * oh + (1 - lam) * oh[perm]
            with torch.autocast(device.type, enabled=amp):
                logits = model(X)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * len(y)
            if y.dim() == 2:                    # one-hot mixup targets
                tgt = y.argmax(1)
            elif y.is_floating_point():
                tgt = y.round().long()
            else:
                tgt = y
            correct += int((logits.argmax(1) == tgt).sum())
            n += len(y)
    if scheduler is not None:
        scheduler.step()
    return total_loss / n, correct / n


def evaluate(model, X: np.ndarray, y: np.ndarray, device, batch_size: int):
    """Predict in eval mode -> (y_true, y_pred)."""
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    preds = []
    amp = device.type == "cuda"
    loader = DataLoader(SpectraDataset(X, y), batch_size=batch_size,
                        pin_memory=amp)
    with torch.no_grad(), torch.autocast(device.type, enabled=amp):
        for Xb, _ in loader:
            preds.append(model(Xb.to(device, non_blocking=amp))
                         .argmax(1).cpu().numpy())
    return np.asarray(y), np.concatenate(preds)


def tta_predict(model, X: np.ndarray, device, batch_size: int = 16,
                n_aug: int = 8) -> np.ndarray:
    """
    Test-time augmentation for the ViT: mean softmax over n_aug jittered
    copies (gaussian noise + wavenumber roll + one masked window) —
    acquisition-invariant predictions at inference time.
    """
    import torch

    model = model.to(device).eval()
    rng = np.random.default_rng(0)
    X = np.asarray(X, dtype=np.float32)
    L = X.shape[1]

    def _proba(x):
        probs = []
        amp = device.type == "cuda"
        with torch.no_grad(), torch.autocast(device.type, enabled=amp):
            for s in range(0, len(x), batch_size):
                xb = torch.as_tensor(x[s:s + batch_size], device=device)
                probs.append(torch.softmax(model(xb), dim=1)
                             .float().cpu().numpy())
        return np.concatenate(probs)

    acc = _proba(X)
    for _ in range(int(n_aug)):
        j = X + rng.normal(0, 0.05 * X.std(), X.shape).astype(np.float32)
        j = np.roll(j, int(rng.integers(-2, 3)), axis=1)
        w = max(4, int(0.05 * L))
        st = int(rng.integers(0, max(1, L - w)))
        j[:, st:st + w] = 0.0
        acc = acc + _proba(j)
    return acc / (int(n_aug) + 1)


def gradient_saliency(model, X: np.ndarray, device) -> np.ndarray:
    """
    Input-gradient saliency for the ViT: |d top-logit / d input| per
    wavenumber, batch-averaged — the deep-model counterpart of SHAP/VIP
    band profiles (attention-rollout substitute; robust, 10 lines).
    Returns (n, seq_len), row-max normalized.
    """
    import torch

    model = model.to(device).eval()
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    Xt.requires_grad_(True)
    logits = model(Xt)
    score = logits.gather(1, logits.argmax(dim=1).view(-1, 1)).sum()
    model.zero_grad()
    score.backward()
    sal = Xt.grad.detach().abs().squeeze(1).cpu().numpy()
    mx = sal.max(axis=1, keepdims=True)
    return sal / np.maximum(mx, 1e-9)


def mc_dropout_predict(model, X: np.ndarray, device, batch_size: int = 16,
                       n_passes: int = 20):
    """
    MC-dropout uncertainty for the ViT: enable the dropout modules at
    inference, sample `n_passes` stochastic forward passes, and return
    (mean probs, per-class std) — the std is an epistemic-uncertainty
    proxy usable for abstention alongside conformal sets.
    """
    import torch
    from torch.utils.data import DataLoader

    model = model.to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    try:
        amp = device.type == "cuda"
        loader = DataLoader(
            SpectraDataset(X, np.zeros(len(X), dtype=int)),
            batch_size=batch_size, pin_memory=amp)
        runs = []
        with torch.no_grad(), torch.autocast(device.type, enabled=amp):
            for _ in range(int(n_passes)):
                probs = []
                for Xb, _ in loader:
                    probs.append(torch.softmax(
                        model(Xb.to(device, non_blocking=amp)),
                        dim=1).float().cpu().numpy())
                runs.append(np.concatenate(probs))
        stacked = np.stack(runs)
        return stacked.mean(axis=0), stacked.std(axis=0)
    finally:
        model.eval()


def standardize_fit(X: np.ndarray, idx) -> tuple[np.ndarray, np.ndarray]:
    """Per-point mean/std from the TRAIN split only (no leakage)."""
    mu = X[idx].mean(axis=0)
    sd = X[idx].std(axis=0) + 1e-8
    return mu, sd


def train_vit(X_tr, y_tr, X_va, y_va, args, device, cfg, verbose=True):
    """Train one SpectralViT with augmentation + cosine LR + early stopping.

    X_* are already standardized; y_* are integer class codes.
    Returns (model, history, best_f1, epochs_run).
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    model = SpectralViT(cfg).to(device)
    n_classes = cfg.n_classes
    cnt = np.bincount(y_tr, minlength=n_classes)
    w = len(y_tr) / np.maximum(cnt, 1) / n_classes
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(w, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    mk = lambda Xa, ya, sh: DataLoader(  # noqa: E731 (local helper)
        SpectraDataset(Xa, ya), batch_size=args.batch_size, shuffle=sh,
        pin_memory=amp)
    loaders = {"train": mk(X_tr, y_tr, True), "val": mk(X_va, y_va, False)}
    history = {"train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": [], "val_f1": []}
    warmup = min(5, max(1, args.epochs // 10))

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        t = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if verbose:
        print("-" * 78)
    t0 = time.time()
    best_f1, best_state, best_epoch, epochs_run = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion,
                                    device, optimizer, scheduler,
                                    augment=True, scaler=scaler)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion,
                                    device)
        _, yv_pred = evaluate(model, X_va, y_va, device, args.batch_size)
        va_f1 = float(f1_score(y_va, yv_pred, average="macro"))
        for key, val in (("train_loss", tr_loss), ("train_acc", tr_acc),
                         ("val_loss", va_loss), ("val_acc", va_acc),
                         ("val_f1", va_f1)):
            history[key].append(val)
        epochs_run = epoch
        marker = ""
        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            marker = "  *"
        if verbose:
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss "
                  f"{tr_loss:.4f}  train_acc {tr_acc:.4f} | val_loss "
                  f"{va_loss:.4f}  val_acc {va_acc:.4f}  "
                  f"val_F1 {va_f1:.4f}{marker}")
        if epoch - best_epoch >= args.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no val-F1 gain "
                      f"for {args.patience} epochs)")
            break
    if best_state is not None:
        model.load_state_dict(best_state)     # restore the best weights
    if verbose:
        print("-" * 78)
        print(f"Training finished in {time.time() - t0:.1f}s "
              f"({epochs_run} epochs, best val_F1 {best_f1:.4f} "
              f"@ epoch {best_epoch})")
    return model, history, best_f1, epochs_run


def grouped_cv(X, y_cls, groups, classes, args, device, cfg):
    """
    Patient-grouped k-fold CV of the ViT: every patient is predicted by a
    model that never saw them; the pooled out-of-fold report is the
    honest performance estimate for small clinical datasets.
    Returns (y_true, y_pred).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    groups = np.asarray(groups)
    sgkf = StratifiedGroupKFold(n_splits=args.cv, shuffle=True,
                                random_state=args.seed)
    y_true, y_pred = [], []
    for fi, (tr, te) in enumerate(sgkf.split(X, y_cls, groups), start=1):
        # grouped inner split: first fold of a 5-fold SGKF becomes val
        inner = StratifiedGroupKFold(n_splits=min(5, len(set(groups[tr]))),
                                     shuffle=True,
                                     random_state=args.seed + fi)
        itr, iva = next(inner.split(X[tr], y_cls[tr], groups[tr]))
        mu, sd = standardize_fit(X, tr[itr])
        Xt = (X[tr] - mu) / (sd + 1e-8)
        model, _, best_f1, _ = train_vit(
            Xt[itr], y_cls[tr][itr], Xt[iva], y_cls[tr][iva],
            args, device, cfg, verbose=False)
        Xte = (X[te] - mu) / (sd + 1e-8)
        yt, yp = evaluate(model, Xte, y_cls[te], device, args.batch_size)
        y_true.append(yt)
        y_pred.append(yp)
        fold_f1 = f1_score(yt, yp, average="macro")
        print(f"  fold {fi}/{args.cv}: {len(te)} spectra "
              f"({len(set(groups[te]))} patients)  macro-F1 "
              f"{fold_f1:.3f}  (best val_F1 {best_f1:.3f})")
    return np.concatenate(y_true), np.concatenate(y_pred)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def save_training_curves(path: str, history: dict):
    """Two-panel figure: loss and accuracy over epochs."""
    apply_style()
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5),
                                   constrained_layout=True)
    ax1.plot(ep, history["train_loss"], marker="o", ms=3.5, lw=1.4,
             color=COL_MAIN, label="Training loss")
    ax1.plot(ep, history["val_loss"], marker="s", ms=3.5, lw=1.4,
             color=COL_RESULT, label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss over epochs")
    ax1.legend()
    ax2.plot(ep, history["train_acc"], marker="o", ms=3.5, lw=1.4,
             color=COL_MAIN, label="Training accuracy")
    ax2.plot(ep, history["val_acc"], marker="s", ms=3.5, lw=1.4,
             color=COL_RESULT, label="Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Accuracy over epochs")
    ax2.legend()
    fig.savefig(path)
    plt.close(fig)


def save_confusion_matrix_png(path: str, cm: np.ndarray, classes, title: str):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    plot_confusion_matrix(ax, cm, classes, title=title)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Train the spectral Vision Transformer on Raman spectra")
    ap.add_argument("--data", default=None,
                    help="folder of labelled spectra (default: "
                         "auto-detected dataset root)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=10,
                    help="early-stopping patience on validation macro-F1")
    ap.add_argument("--preset", choices=["best", "default"], default="best",
                    help="preprocessing: auto-tuned winner from optimize.py "
                         "(if available) or library defaults")
    ap.add_argument("--cv", type=int, default=0,
                    help="patient-grouped k-fold CV (e.g. --cv 5): pooled "
                         "out-of-fold report — the honest estimate for "
                         "small clinical datasets")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--test-size", type=float, default=0.15,
                    help="held-out test fraction (default 0.15)")
    ap.add_argument("--val-size", type=float, default=0.15,
                    help="validation fraction (default 0.15)")
    # model shape (defaults = the compact ViT from vit_model.ViTConfig)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"],
                    default="auto",
                    help="auto = use any CUDA GPU that is present; "
                         "RAMAN_DEVICE=cpu forces CPU regardless")
    ap.add_argument("--out", default=os.path.join(here, "vit_outputs"))
    ap.add_argument("--model-name", default="vit_model.pt")
    return ap.parse_args()


def main():
    import torch

    require_torch()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # device: never hardcoded to a card — whatever CUDA GPU the driver
    # exposes; RAMAN_DEVICE=cpu is the same kill-switch the app uses
    if (os.environ.get("RAMAN_DEVICE", "").lower() == "cpu"
            or args.device == "cpu"):
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")   # explicit: fail loudly if absent
    else:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
    if device.type == "cuda":
        try:
            if tuple(torch.cuda.get_device_capability(device)) >= (8, 0):
                # TF32 matmuls: free speedup on Ampere+ GPUs only
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    print("=" * 78)
    print("Vision Transformer (ViT) - Raman spectra classification")
    print("=" * 78)

    # preprocessing preset first (drives the spiked-spectrum decision)
    params = pp.PreprocessParams().validate()
    preset_exclude_flagged: bool | None = None
    if args.preset == "best":
        loaded = load_best_params()
        if loaded is not None:
            params, meta = loaded
            params = params.validate()
            # selection-CV score: max over the sweep — optimistic upper
            # bound (winner's curse), NOT a performance estimate
            # (2026-09-12 audit)
            print(f"Preprocess : auto-tuned preset from optimize.py "
                  f"({meta.get('model', '?')}, selection-CV macro-F1 "
                  f"{meta.get('score_macro_f1', 0):.3f} — optimistic "
                  "upper bound)")
            # honor the flagged-spectra policy the preset was SELECTED
            # under (optimize.py excludes spiked by default); missing
            # key = legacy preset, keep the despike-based rule below
            preset_exclude_flagged = meta.get("exclude_flagged")
            if preset_exclude_flagged is not None:
                preset_exclude_flagged = bool(preset_exclude_flagged)
        else:
            print("Preprocess : library defaults (no optimize.py result "
                  "found - run python optimize.py first)")

    folder = resolve_data_folder(args.data)
    groups: list[str] | None = None
    report = None
    if is_clinical_layout(folder):
        # clinical layout: <root>/<class>/<patient folder>/*.csv
        cd = load_clinical_dataset(folder)
        report = cd.report
        spectra = cd.spectra
        y = [s.label for s in spectra]
        groups = list(cd.groups)
        # spiked spectra are RECOVERED by Whitaker-Hayes despiking in the
        # pipeline; exclude them when despiking is disabled — and always
        # when the selected preset was optimized without them
        if cd.flagged and any(cd.flagged) and (
                preset_exclude_flagged is True or not params.despike):
            keep = [i for i, bad in enumerate(cd.flagged) if not bad]
            n_bad = len(spectra) - len(keep)
            spectra = [spectra[i] for i in keep]
            y = [y[i] for i in keep]
            groups = [groups[i] for i in keep]
            why = ("preset selected without them"
                   if preset_exclude_flagged is True
                   else "quality flag, despiking off")
            print(f"  cleaned   : {n_bad} spiked spectra excluded ({why})")
        classes = sorted(set(y))
        if cd.grid is not None:      # all files share one grid (verified)
            grid = cd.grid
            X_raw, _ = ds.to_matrix(spectra, grid)
        else:
            grid = ds.common_grid(spectra)
            X_raw, y = ds.to_matrix(spectra, grid)
        counts = {c: y.count(c) for c in classes}
        print(f"Data folder : {folder}  (clinical layout)")
        print(f"Spectra     : {len(spectra)}  ({report['n_subjects']} "
              f"subjects)   classes: "
              + ", ".join(f"{c} ({n})" for c, n in counts.items()))
        for key, label in (
                ("references_excluded", "reference spectra excluded"),
                ("duplicates_dropped", "duplicate copies dropped"),
                ("cross_class_dropped",
                 "cross-class identical spectra dropped")):
            if report[key]:
                print(f"  cleaned   : {len(report[key])} {label} "
                      "(details in data_report.txt)")
    else:
        spectra, X_raw, y, grid = load_xy(folder)
        classes = sorted(set(y))
        counts = {c: y.count(c) for c in classes}
        print(f"Data folder : {folder}")
        print(f"Spectra     : {len(spectra)}   classes: "
              + ", ".join(f"{c} ({n})" for c, n in counts.items()))

    # preprocessing: auto-tuned winner (optimize.py) or library defaults;
    # fixed per-spectrum transforms - no cross-sample fit, no leakage
    # (params were resolved at the top of main())
    X = pp.preprocess_matrix(X_raw, params, wn=grid)
    X = resample_to_len(X, args.seq_len)

    # split by spectrum NAME so vit_test.py can rebuild the exact test set;
    # clinical data is PAIRED (same subject in both classes) -> the split
    # is grouped by patient so no subject lands on two sides
    names = [s.name for s in spectra]
    y_cls = np.array([classes.index(l) for l in y])
    if groups is not None:
        tr, va, te = patient_split(groups, y, args.seed)
        split_desc = "grouped by patient"
    else:
        tr, va, te = stratified_split(names, y, args.test_size,
                                      args.val_size, args.seed)
        split_desc = "stratified"
    print(f"Split       : train {len(tr)} / val {len(va)} / test {len(te)}"
          f"  ({split_desc})")

    # standardize per spectral point with TRAIN-split statistics — the ViT
    # patch embedding needs inputs of roughly unit scale to learn
    # (vector-normalized spectra are ~2 orders of magnitude smaller)
    mu, sd = standardize_fit(X, tr)
    X = (X - mu) / sd

    cfg = ViTConfig(seq_len=args.seq_len, patch_size=args.patch_size,
                    dim=args.dim, depth=args.depth, heads=args.heads,
                    dropout=args.dropout, n_classes=len(classes)).validate()
    n_par = sum(p.numel() for p in SpectralViT(cfg).parameters())
    print(f"Model       : {cfg.n_patches} patches x {cfg.patch_size} | "
          f"dim {cfg.dim} | depth {cfg.depth} | heads {cfg.heads} | "
          f"{n_par / 1e3:.1f}k params")
    print(f"Device      : {device}")

    # optional: patient-grouped k-fold CV for an honest pooled estimate
    if args.cv and groups is not None:
        print("-" * 78)
        print(f"Grouped {args.cv}-fold cross-validation "
              f"(every patient predicted by an unseen model):")
        y_cv_true, y_cv_pred = grouped_cv(X, y_cls, groups, classes,
                                          args, device, cfg)
        cv_f1 = float(f1_score(y_cv_true, y_cv_pred, average="macro"))
        cv_acc = float((y_cv_true == y_cv_pred).mean())
        print(f"\nPooled out-of-fold accuracy: {cv_acc:.2%}  "
              f"({int((y_cv_true == y_cv_pred).sum())}/{len(y_cv_true)})")
        print(f"Pooled out-of-fold macro-F1: {cv_f1:.4f}")
        print("\nClassification report (pooled out-of-fold):")
        print(classification_report(y_cv_true, y_cv_pred,
                                    target_names=classes, digits=4,
                                    zero_division=0))
        os.makedirs(args.out, exist_ok=True)
        cv_png = os.path.join(args.out, "vit_cv_confusion_matrix.png")
        save_confusion_matrix_png(
            cv_png,
            confusion_matrix(y_cv_true, y_cv_pred,
                             labels=range(len(classes))), classes,
            f"ViT confusion matrix ({args.cv}-fold grouped CV, pooled)")
        print(f"Saved: {cv_png}")
        print("-" * 78)
        print("Training the deployable single-split model next "
              "(checkpoint + curves):")

    # single train/val/test run -> deployable checkpoint
    model, history, best_f1, epochs_run = train_vit(
        X[tr], y_cls[tr], X[va], y_cls[va], args, device, cfg, verbose=True)

    # held-out test evaluation
    y_true, y_pred = evaluate(model, X[te], y_cls[te], device,
                              args.batch_size)
    acc = float((y_true == y_pred).mean())
    print(f"\nTest accuracy: {acc:.2%}  "
          f"({int((y_true == y_pred).sum())}/{len(y_true)})")
    print("\nClassification report (test set):")
    print(classification_report(y_true, y_pred, target_names=classes,
                                digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    # save everything
    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, args.model_name)
    save_checkpoint(
        model_path, model, classes, grid,
        asdict(params), {
            "train": [names[i] for i in tr],
            "val": [names[i] for i in va],
            "test": [names[i] for i in te],
            "test_groups": [groups[i] for i in te] if groups else [],
        }, history, data_folder=os.path.abspath(folder), seed=args.seed,
        input_mean=mu, input_std=sd, best_val_f1=best_f1,
        epochs_run=epochs_run)
    curves = os.path.join(args.out, "vit_training_curves.png")
    cm_png = os.path.join(args.out, "vit_confusion_matrix.png")
    save_training_curves(curves, history)
    save_confusion_matrix_png(cm_png, cm, classes,
                              "ViT confusion matrix (test set)")
    print(f"\nSaved: {model_path}")
    print(f"Saved: {curves}")
    print(f"Saved: {cm_png}")
    if report is not None:
        rep_path = os.path.join(args.out, "data_report.txt")
        write_report(rep_path, report)
        print(f"Saved: {rep_path}")
    print("\nEvaluate again later with:  python vit_test.py")


if __name__ == "__main__":
    main()
