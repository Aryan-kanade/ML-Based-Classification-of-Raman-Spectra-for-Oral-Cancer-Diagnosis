"""
eval_pretrain.py -- L51 self-supervised pretraining on SYNTHETIC
Raman (SAFETY: pure experiment script -> experiments/ only).

The 2025 literature recipe for tiny labeled cohorts (PMC12264939,
arXiv 2601.16107): pretrain an encoder on large physics-simulated
spectra, then fine-tune on the few real labeled rows.  This app
already ships a physics simulator (modeling.phantom_cohort) and an
augmenter (lorentzian_synthesize) — pretraining needs no internet.

Phases:
  1. phantom_cohort -> 6k synthetic 2-class spectra (3x500x4) (patient
     random effects, baselines, spikes) at the clinical grid.
  2. Pretrain the encoder supervised on the synthetic labels
     (classification pretraining is the strongest simple recipe).
  3. Fine-tune on each outer fold of the REAL 287 rows (head reset,
     encoder kept at lr/10), grouped 5-fold — identical protocol to
     every other arm.
"""

from __future__ import annotations

import os

import numpy as np

import modeling
import sequential
from exp_common import OUT_DIR, load_data, record, stamp

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arch, X, y, groups, wn, _meta = load_data()
    X = np.asarray(X, dtype=np.float32)
    y_arr = np.searchsorted(sorted(set(y)), np.asarray(y))
    groups = np.asarray(groups)
    classes = sorted(set(y))
    n_feat = X.shape[1]
    print(f"[pre] {stamp()} {n_feat} features · dev={dev}", flush=True)

    # ---- 1. synthetic cohort at our axis ------------------------------
    # (phantom_cohort is fully synthetic — no clinical data needed; the
    # old load_clinical_dataset call here was dead, audit finding 16)
    synth_X, synth_y = [], []
    for i in range(3):
        coh = modeling.phantom_cohort(n_patients=500, spectra_per=4,
                                      seed=i)
        Xs = coh[0]
        ys = coh[1]
        synth_X.append(np.asarray(Xs, dtype=np.float32))
        synth_y.append(np.asarray(ys))
    SX = np.vstack(synth_X)
    Sy = np.concatenate(synth_y)
    if SX.shape[1] != n_feat:            # resample to our grid
        from scipy.interpolate import interp1d
        xs_old = np.linspace(0, 1, SX.shape[1])
        xs_new = np.linspace(0, 1, n_feat)
        SX = np.vstack([interp1d(xs_old, row)(xs_new)
                        for row in SX])
    print(f"[pre] synthetic corpus: {SX.shape}", flush=True)

    # ---- 2. encoder ---------------------------------------------------
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv1d(1, 32, 15, padding=7), nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 9, padding=4), nn.ReLU(),
                nn.AdaptiveMaxPool1d(32), nn.Flatten(),
                nn.Linear(64 * 32, 256), nn.ReLU())
            self.head = nn.Linear(256, 2)

        def forward(self, x):
            return self.head(self.body(x))

    enc = Enc().to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3,
                            weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    tX = torch.tensor(SX, dtype=torch.float32, device=dev).unsqueeze(1)
    ty = torch.tensor((Sy == sorted(set(Sy))[-1]).astype(np.int64),
                      device=dev)
    enc.train()
    for epoch in range(6):
        perm = torch.randperm(len(tX), device=dev)[:8192]
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(enc(tX[idx]), ty[idx])
            loss.backward()
            opt.step()
    print("[pre] pretraining done", flush=True)

    # ---- 3. fine-tune per outer fold (encoder lr/10) ------------------
    # LEAKAGE FIX (2026-09-16): deep-copy the PRETRAINED encoder per
    # fold.  Reusing one encoder let fold-k fine-tune start from
    # weights already tuned on earlier folds' training data — which
    # contains fold-k's TEST patients (measured inflation 0.839 ->
    # honest 0.647, McNemar p=1e-4 vs baseline; see
    # eval_pretrain_adj.py + experiments/pretrain_adjudicated.json).
    import copy
    from sklearn.metrics import f1_score
    outer = sequential._splitter(5, 42, y_arr, groups)
    oof = np.full((len(y_arr), len(classes)), np.nan)
    for tr, te in outer.split(X, y_arr, groups):
        enc_fresh = copy.deepcopy(enc)
        ft = nn.Sequential(enc_fresh.body, nn.Linear(256, 2)).to(dev)
        for p_ in enc_fresh.body.parameters():
            p_.requires_grad_(True)
        o_ = torch.optim.AdamW(
            [{"params": enc_fresh.body.parameters(), "lr": 1e-4},
             {"params": ft[-1].parameters(), "lr": 1e-3}],
            weight_decay=1e-4)
        tXf = torch.tensor(X[tr], dtype=torch.float32,
                           device=dev).unsqueeze(1)
        tyf = torch.tensor(y_arr[tr], dtype=torch.long, device=dev)
        ft.train()
        for _ in range(120):
            perm = torch.randperm(len(tXf), device=dev)
            for i in range(0, len(perm), 32):
                idx = perm[i:i + 32]
                o_.zero_grad()
                loss = lossf(ft(tXf[idx]), tyf[idx])
                loss.backward()
                o_.step()
        ft.eval()
        with torch.no_grad():
            oof[te] = ft(torch.tensor(X[te], dtype=torch.float32,
                                      device=dev).unsqueeze(1)
                         ).softmax(-1).cpu().numpy()
    v = ~np.isnan(oof).any(axis=1)
    f1 = float(f1_score(y_arr[v], oof[v].argmax(axis=1),
                        average="macro"))
    print(f"[pre] L51 pretrained+finetuned F1 {f1:.3f} "
          "(baseline 0.753)", flush=True)
    record("L51 pretrain->finetune",
           "6k synthetic phantom spectra -> grouped 5-fold", f1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
