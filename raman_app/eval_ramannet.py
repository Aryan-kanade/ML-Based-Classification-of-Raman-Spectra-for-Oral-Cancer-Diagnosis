"""
eval_ramannet.py -- Stage 2d of the push-to-0.8 program v10.

RamanNet-style segment architecture (Ibtehaz & Chowdhury 2023): the
spectrum is split into overlapping segments, each with its OWN weights
(non-shared shifted MLPs) so peak POSITION carries meaning — exactly
what our failed weight-shared CNN (F1 ~0.50) could not learn.
Efficient implementation: one Conv1d(groups=n_segments) is a bank of
independent per-segment linear layers.  Honest grouped CV as ever.

Usage:  python eval_ramannet.py [--seed 42] [--epochs 80]
Output: experiments/ramannet.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import exp_common
import pat_common

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


class RamanNet(nn.Module):
    def __init__(self, d_in: int, seg: int = 50, stride: int = 25,
                 dim: int = 64, p: float = 0.2):
        super().__init__()
        self.seg, self.stride = seg, stride
        self.n_seg = (d_in - seg) // stride + 1
        # groups = n_seg -> independent weights per segment (RamanNet)
        self.seg_proj = nn.Conv1d(self.n_seg, self.n_seg * dim,
                                  kernel_size=seg, groups=self.n_seg)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(p),
            nn.Linear(self.n_seg * dim, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 1))

    def forward(self, x):                       # x: (B, d_in)
        u = x.unfold(-1, self.seg, self.stride)  # (B, n_seg, seg)
        z = self.seg_proj(u).flatten(1)          # (B, n_seg*dim)
        return self.head(z).squeeze(-1)


def main() -> int:
    if not HAS_TORCH:
        print("[2d] torch unavailable — skip", flush=True)
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    _arch, X, y, groups, _wn, _meta = exp_common.load_data()
    y = np.asarray(y)
    _cls = sorted(np.unique(y))          # ['Normal','Tumor']
    y = np.asarray([_cls.index(v) for v in y])
    groups = np.asarray(groups)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                random_state=args.seed)
    oof = np.full(len(y), np.nan)

    for f, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        model = RamanNet(X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                               weight_decay=1e-4)
        Xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32,
                          device=device)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=device)
        pos_w = torch.tensor([(y[tr] == 0).sum()
                              / max((y[tr] == 1).sum(), 1)],
                             dtype=torch.float32, device=device)
        for _ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(tr), device=device)
            for s in range(0, len(tr), 32):
                idx = perm[s:s + 32]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model(Xt[idx]), yt[idx], pos_weight=pos_w)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            Xs = torch.tensor((X[te] - mu) / sd, dtype=torch.float32,
                              device=device)
            oof[te] = torch.sigmoid(model(Xs)).cpu().numpy()
        print(f"[2d] fold {f + 1}/5 done", flush=True)

    m = ~np.isnan(oof)
    sf1 = float(f1_score(y[m], (oof[m] >= 0.5).astype(int),
                         average="macro"))
    yp, pp, _ = pat_common.patient_table(y, np.c_[1 - oof, oof], groups,
                                         "mean")
    pred = np.empty_like(yp)
    for i in range(yp.size):                     # LOO-tuned threshold
        trm = np.arange(yp.size) != i
        best_t, best_f1 = 0.5, -1.0
        for t in np.unique(pp[trm]):
            f1 = f1_score(yp[trm], (pp[trm] >= t).astype(int),
                          average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        pred[i] = int(pp[i] >= best_t)
    pf1 = float(f1_score(yp, pred, average="macro"))

    out = {"seed": args.seed, "spectrum_f1_at_0.5": sf1,
           "patient_f1": pf1,
           "verdict": f"RamanNet spectrum F1 {sf1:.3f} · patient F1 "
                      f"{pf1:.3f} (our CNN was ~0.50)"}
    print(f"[2d] {out['verdict']}", flush=True)
    path = os.path.join(exp_common.OUT_DIR, "ramannet.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    exp_common.record("external-data", "RamanNet segment-MLP", sf1,
                      extra=f"patient F1 {pf1:.3f}")
    print(f"[2d] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
