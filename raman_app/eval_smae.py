"""
eval_smae.py -- Stage 2c of the push-to-0.8 program v10.

Masked-autoencoder pretraining (SMAE, Ren 2025) on the RSClass
oral-cancer corpus (2,830 binary spectra), then fine-tune on OUR
64-patient paired features.  External spectra are interpolated onto
our feature grid and SNV-normalized so the encoder sees comparable
input scales.  Honest evaluation: StratifiedGroupKFold patient-grouped
outer folds; all pretraining happens inside the loop on external data
only (no leakage — external corpus carries none of our patients).

Usage:  python eval_smae.py [--seed 42] [--mask 0.4] [--pt-epochs 60]
Output: experiments/smae.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import exp_common

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

EXT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "_external", "processed",
    "oral_binary.npz")


class Encoder(nn.Module):
    def __init__(self, d_in, width=512, dim=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.ReLU(), nn.Dropout(p),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, dim))

    def forward(self, x):
        return self.net(x)


class MAE(nn.Module):
    """Linear-spectrum MAE: mask random 40% of points, reconstruct."""

    def __init__(self, d_in, dim=128):
        super().__init__()
        self.enc = Encoder(d_in, dim=dim)
        self.dec = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, d_in))

    def forward(self, x, mask):
        z = self.enc(x)
        rec = self.dec(z)
        # loss only on masked points
        loss = ((rec - x) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        return loss, rec


def _snv(a):
    mu = a.mean(-1, keepdims=True)
    sd = a.std(-1, keepdims=True) + 1e-8
    return (a - mu) / sd


def main() -> int:
    if not HAS_TORCH:
        print("[2c] torch unavailable — skip", flush=True)
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask", type=float, default=0.4)
    ap.add_argument("--pt-epochs", type=int, default=60)
    ap.add_argument("--ft-epochs", type=int, default=60)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    _arch, X, y, groups, wn, _meta = exp_common.load_data()
    y = np.asarray(y)
    _cls = sorted(np.unique(y))          # ['Normal','Tumor']
    y = np.asarray([_cls.index(v) for v in y])
    groups = np.asarray(groups)

    z = np.load(EXT)
    Xe_raw, wne = z["X"], z["wn"]
    Xe = np.stack([np.interp(wn, wne, row) for row in Xe_raw])
    Xe = _snv(Xe.astype(np.float32))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = X.shape[1]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                random_state=args.seed)
    oof = np.full(len(y), np.nan)

    for f, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        # 1) pretrain MAE on external corpus (train side uses no our-data)
        mae = MAE(d).to(device)
        opt = torch.optim.Adam(mae.parameters(), lr=1e-3)
        Xt_e = torch.tensor(Xe, dtype=torch.float32, device=device)
        g = torch.Generator(device="cpu").manual_seed(args.seed + f)
        for ep in range(args.pt_epochs):
            mae.train()
            perm = torch.randperm(len(Xe), generator=g)
            tot = 0.0
            for s in range(0, len(Xe), 128):
                idx = perm[s:s + 128]
                xb = Xt_e[idx.to(device)]
                mask = (torch.rand_like(xb) < args.mask).float()
                loss, _rec = mae(xb, mask)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(idx)
            if ep == args.pt_epochs - 1:
                print(f"[2c] fold {f + 1} pretrain recon MSE "
                      f"{tot / len(Xe):.4f}", flush=True)

        # 2) fine-tune encoder + linear head on OUR training patients
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        enc = mae.enc
        head = nn.Linear(128, 1).to(device)
        clf = nn.Sequential(enc, head).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-3,
                               weight_decay=1e-4)
        Xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32,
                          device=device)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=device)
        pos_w = torch.tensor([(y[tr] == 0).sum()
                              / max((y[tr] == 1).sum(), 1)],
                             dtype=torch.float32, device=device)
        for _ep in range(args.ft_epochs):
            clf.train()
            perm = torch.randperm(len(tr), device=device)
            for s in range(0, len(tr), 32):
                idx = perm[s:s + 32]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    clf(Xt[idx]).squeeze(-1), yt[idx], pos_weight=pos_w)
                opt.zero_grad()
                loss.backward()
                opt.step()
        clf.eval()
        with torch.no_grad():
            Xs = torch.tensor((X[te] - mu) / sd, dtype=torch.float32,
                              device=device)
            oof[te] = torch.sigmoid(clf(Xs).squeeze(-1)).cpu().numpy()
        print(f"[2c] fold {f + 1}/5 done", flush=True)

    m = ~np.isnan(oof)
    sf1 = float(f1_score(y[m], (oof[m] >= 0.5).astype(int),
                         average="macro"))
    out = {"seed": args.seed, "mask": args.mask,
           "spectrum_f1_at_0.5": sf1,
           "verdict": f"SMAE pretrain->finetune spectrum F1 {sf1:.3f} "
                      f"(baseline 0.753)"}
    print(f"[2c] {out['verdict']}", flush=True)
    path = os.path.join(exp_common.OUT_DIR, "smae.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    exp_common.record("external-data", "SMAE pretrain->finetune", sf1)
    print(f"[2c] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
