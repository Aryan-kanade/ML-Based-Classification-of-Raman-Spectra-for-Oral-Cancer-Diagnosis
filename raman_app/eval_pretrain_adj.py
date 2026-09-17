"""
eval_pretrain_adj.py -- adjudication for the L51 phantom-pretrain
breakthrough candidate (F1 0.839 single-seed): outer seeds 42/43/44
plus exact McNemar vs the d2-baseline OOF (same rows).  The encoder is
pretrained ONCE on the physics corpus (data-independent prior — no
leakage), then fine-tuned inside every outer fold.

Usage:  python eval_pretrain_adj.py
Output: experiments/pretrain_adjudicated.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

import exp_common
import modeling
import pat_common
import sequential

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def main() -> int:
    if not HAS_TORCH:
        print("[adj] torch unavailable", flush=True)
        return 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _arch, X, y_str, groups, wn, _meta = exp_common.load_data()
    cls = sorted(set(y_str))
    y = np.asarray([cls.index(v) for v in y_str])
    groups = np.asarray(groups)
    classes = np.unique(y)

    # 1) physics corpus + encoder (identical to eval_pretrain.py)
    synth_X, synth_y = [], []
    for i in range(15):
        Xs, ys, _g, _w, _p = modeling.phantom_cohort(4, 4, seed=i)
        synth_X.append(np.asarray(Xs, dtype=np.float32))
        synth_y.append(np.asarray(ys))
    SX = np.vstack(synth_X)
    Sy = np.concatenate(synth_y)
    if SX.shape[1] != X.shape[1]:
        from scipy.interpolate import interp1d
        old = np.linspace(0, 1, SX.shape[1])
        new = np.linspace(0, 1, X.shape[1])
        SX = np.vstack([interp1d(old, row)(new) for row in SX])

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
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    tX = torch.tensor(SX, dtype=torch.float32, device=dev).unsqueeze(1)
    ty = torch.tensor((Sy == sorted(set(Sy))[-1]).astype(np.int64),
                      device=dev)
    enc.train()
    for _epoch in range(6):
        perm = torch.randperm(len(tX), device=dev)[:8192]
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            lossf(enc(tX[idx]), ty[idx]).backward()
            opt.step()

    # 2) fine-tune under grouped CV at 3 outer seeds
    import copy
    f1s = []
    oof_by_seed = {}
    for seed in (42, 43, 44):
        outer = sequential._splitter(5, seed, y, groups)
        oof = np.full((len(y), len(classes)), np.nan)
        for tr, te in outer.split(X, y, groups):
            enc_s = copy.deepcopy(enc)
            ft = nn.Sequential(enc_s.body, nn.Linear(256, 2)).to(dev)
            for p_ in enc_s.body.parameters():
                p_.requires_grad_(True)
            o_ = torch.optim.AdamW(
                [{"params": enc_s.body.parameters(), "lr": 1e-4},
                 {"params": ft[-1].parameters(), "lr": 1e-3}],
                weight_decay=1e-4)
            tXf = torch.tensor(X[tr], dtype=torch.float32,
                               device=dev).unsqueeze(1)
            tyf = torch.tensor(y[tr], dtype=torch.long, device=dev)
            ft.train()
            for _ep in range(120):
                perm = torch.randperm(len(tXf), device=dev)
                for i in range(0, len(perm), 32):
                    idx = perm[i:i + 32]
                    o_.zero_grad()
                    lossf(ft(tXf[idx]), tyf[idx]).backward()
                    o_.step()
            ft.eval()
            with torch.no_grad():
                oof[te] = ft(torch.tensor(
                    X[te], dtype=torch.float32,
                    device=dev).unsqueeze(1)).softmax(-1).cpu().numpy()
        v = ~np.isnan(oof).any(axis=1)
        f1 = float(f1_score(y[v], oof[v].argmax(axis=1), average="macro"))
        f1s.append(f1)
        oof_by_seed[seed] = oof.copy()
        print(f"[adj] seed {seed}: F1 {f1:.3f}", flush=True)

    # 3) exact McNemar vs baseline OOF on identical rows (seed 42)
    ye_b, proba_b, g_b = pat_common.winner_oof()
    oof = oof_by_seed[42]
    keep = ~np.isnan(oof).any(axis=1)
    pred_n = oof[keep].argmax(axis=1)
    pred_b = proba_b[keep].argmax(axis=1)
    yv = y[keep]
    b_ = int(((pred_n == yv) & (pred_b != yv)).sum())   # new right, base wrong
    c_ = int(((pred_n != yv) & (pred_b == yv)).sum())   # base right, new wrong
    p = float(binomtest(b_, b_ + c_, 0.5).pvalue) if b_ + c_ else 1.0

    out = {"f1s": f1s, "f1_mean": float(np.mean(f1s)),
           "f1_std": float(np.std(f1s)),
           "mcnemar": {"b": b_, "c": c_, "p": p},
           "verdict": (f"phantom-pretrain F1 {np.mean(f1s):.3f}"
                       f"±{np.std(f1s):.3f} · McNemar b={b_} c={c_} "
                       f"p={p:.4f}")}
    print(f"[adj] {out['verdict']}", flush=True)
    path = os.path.join(exp_common.OUT_DIR, "pretrain_adjudicated.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[adj] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
