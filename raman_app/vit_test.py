"""
vit_test.py — evaluate a trained spectral ViT checkpoint (from vit_train.py).

By default it re-evaluates exactly the held-out test spectra recorded in
the checkpoint, reusing the wavenumber grid and preprocessing settings
stored at training time — so the numbers match the end-of-training report.

Outputs (in --out, default `vit_outputs/`):
  vit_test_confusion_matrix.png   confusion matrix of the test set
  vit_test_report.txt             classification report + accuracy

Examples:
    python vit_test.py
    python vit_test.py --model vit_outputs/vit_model.pt --data my_folder
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import dataset as ds
import preprocessing as pp
from clinical_data import (is_clinical_layout, load_clinical_dataset)
from vit_model import load_checkpoint, require_torch, resample_to_len
from vit_train import evaluate, save_confusion_matrix_png


def resolve_data_folder(explicit: str | None, recorded: str | None) -> str:
    """--data, else the folder recorded in the checkpoint, else demo_data."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [explicit, recorded, os.path.join(here, "demo_data")]
    for cand in candidates:
        if cand and os.path.isdir(cand):
            return cand
    raise SystemExit(
        "Could not find the spectra folder.  Re-run with:\n"
        "    python vit_test.py --data <folder>"
    )


def main():
    import torch

    require_torch()
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Evaluate a trained spectral ViT on its test set")
    ap.add_argument("--model",
                    default=os.path.join(here, "vit_outputs", "vit_model.pt"))
    ap.add_argument("--data", default=None,
                    help="folder of spectra (default: the checkpoint's "
                         "training folder, else demo_data)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(here, "vit_outputs"))
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit(
            f"Checkpoint not found: {args.model}\n"
            "Train one first:  python vit_train.py"
        )
    model, ckpt = load_checkpoint(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    classes = list(ckpt["classes"])
    test_names = list(ckpt["splits"].get("test", []))
    grid = np.asarray(ckpt["grid"], dtype=float)
    params = pp.PreprocessParams(**ckpt["preprocess_params"]).validate()
    seq_len = int(ckpt["model_config"]["seq_len"])

    folder = resolve_data_folder(args.data, ckpt.get("data_folder"))
    if is_clinical_layout(folder):
        # clinical layout: same loader/rules as training (flagged spectra
        # were already excluded at training time, so they are not in the
        # recorded test names either)
        cd = load_clinical_dataset(folder)
        spectra = {s.name: s for s in cd.spectra}
        groups_by_name = dict(zip([s.name for s in cd.spectra], cd.groups,
                             strict=True))
    else:
        spectra = {s.name: s for s in ds.load_folder(folder) if s.label}
        groups_by_name = {}
    missing = [n for n in test_names if n not in spectra]
    if missing:
        raise SystemExit(
            f"{len(missing)} of the {len(test_names)} checkpoint test "
            f"files are missing from {folder!r} (e.g. {missing[0]!r}). "
            "Point --data at the original training folder."
        )

    print("=" * 78)
    print("Vision Transformer (ViT) - testing phase")
    print("=" * 78)
    n_subj = len(set(groups_by_name.get(n, n) for n in test_names))
    print(f"Checkpoint  : {args.model}")
    print(f"Data folder : {folder}")
    print(f"Test set    : {len(test_names)} spectra from {n_subj} subjects "
          f"({len(classes)} classes: {', '.join(classes)})")

    # same transform chain as training: common grid -> preprocess -> resample
    # -> standardize with the train-split statistics stored in the checkpoint
    X_raw, labels = ds.to_matrix([spectra[n] for n in test_names], grid,
                                 [spectra[n].label for n in test_names])
    X = pp.preprocess_matrix(X_raw, params)
    X = resample_to_len(X, seq_len)
    mu = np.asarray(ckpt.get("input_mean", 0.0), dtype=float)
    sd = np.asarray(ckpt.get("input_std", 1.0), dtype=float) + 1e-8
    X = (X - mu) / sd
    y = np.array([classes.index(l) for l in labels])

    y_true, y_pred = evaluate(model, X, y, device, args.batch_size)
    acc = float((y_true == y_pred).mean())
    report = classification_report(y_true, y_pred, target_names=classes,
                                   digits=4, zero_division=0)
    print(f"\nTest accuracy: {acc:.2%}  "
          f"({int((y_true == y_pred).sum())}/{len(y_true)})")
    print("\nClassification report (test set):")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    os.makedirs(args.out, exist_ok=True)
    cm_png = os.path.join(args.out, "vit_test_confusion_matrix.png")
    save_confusion_matrix_png(cm_png, cm, classes,
                              "ViT confusion matrix (test set)")
    rep_txt = os.path.join(args.out, "vit_test_report.txt")
    with open(rep_txt, "w", encoding="utf-8") as fh:
        fh.write(f"ViT test evaluation - {args.model}\n"
                 f"Data folder: {folder}\n"
                 f"Test spectra: {len(test_names)}\n\n"
                 f"Test accuracy: {acc:.4f}\n\n{report}\n")
    print(f"Saved: {cm_png}")
    print(f"Saved: {rep_txt}")


if __name__ == "__main__":
    main()
