"""
eval_mil.py -- Stage 1c of the push-to-0.8 program v10.

Multi-Instance Learning with gated attention pooling (Ilse et al. 2018):
each (patient, class) replicate group is a BAG.  A shared encoder
scores every spectrum; attention learns which replicates to trust and
a bag head predicts the bag label.  Instance head (per-spectrum CE) is
trained jointly, so OOF spectrum probabilities + patient roll-up both
come out honest (patients never straddle folds).

Usage:  python eval_mil.py [--seed 42] [--epochs 80]
Output: experiments/mil.json
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


class MILNet(nn.Module):
    def __init__(self, d_in, width=512, dim=128, p=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, width), nn.ReLU(), nn.Dropout(p),
            nn.Linear(width, dim), nn.ReLU())
        # gated attention (Ilse 2018): a * sigmoid(b), V dims
        self.att_V = nn.Linear(dim, 128)
        self.att_U = nn.Linear(dim, 128)
        self.att_w = nn.Linear(128, 1)
        self.bag_head = nn.Linear(dim, 1)
        self.inst_head = nn.Linear(dim, 1)

    def forward(self, x, bag_ids):
        z = self.enc(x)
        a = self.att_w(torch.tanh(self.att_V(z))
                       * torch.sigmoid(self.att_U(z))).squeeze(-1)
        pooled = []
        for b in torch.unique(bag_ids):
            m = bag_ids == b
            w = torch.softmax(a[m], dim=0)
            pooled.append((z[m] * w.unsqueeze(-1)).sum(0))
        pooled = torch.stack(pooled)
        return (self.bag_head(pooled).squeeze(-1),   # bag logits
                self.inst_head(z).squeeze(-1))       # instance logits


def _loo_threshold(y, p):
    pred = np.empty_like(y)
    for i in range(y.size):
        tr = np.arange(y.size) != i
        best_t, best_f1 = 0.5, -1.0
        for t in np.unique(p[tr]):
            f1 = f1_score(y[tr], (p[tr] >= t).astype(int), average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        pred[i] = int(p[i] >= best_t)
    return pred


def main() -> int:
    if not HAS_TORCH:
        print("[1c] torch unavailable — skip", flush=True)
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
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                random_state=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oof = np.full(len(y), np.nan)

    for f, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32,
                          device=device)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=device)
        # bag id = (patient, class)
        keys = [f"{g}_{c}" for g, c in zip(groups[tr], y[tr])]
        uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
        bt = torch.tensor([uniq[k] for k in keys], device=device)
        model = MILNet(X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                               weight_decay=1e-4)
        n = len(tr)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            tot = 0.0
            for s in range(0, n, 32):
                idx = perm[s:s + 32]
                bag_logit, inst_logit = model(Xt[idx], bt[idx])
                # instance CE on the sampled spectra
                li = nn.functional.binary_cross_entropy_with_logits(
                    inst_logit, yt[idx])
                loss = li
                if s == 0:  # bag CE on full bags, once per epoch
                    bag_keys = sorted(set(keys))
                    # bag label = its class (last field of "patient_class")
                    bag_y = torch.tensor(
                        [float(k.rsplit("_", 1)[1]) for k in bag_keys],
                        dtype=torch.float32, device=device)
                    bl, _il = model(Xt, bt)
                    lb = nn.functional.binary_cross_entropy_with_logits(
                        bl, bag_y)
                    loss = loss + lb
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(li)
        model.eval()
        with torch.no_grad():
            Xe = torch.tensor((X[te] - mu) / sd, dtype=torch.float32,
                              device=device)
            ke = [f"{g}_{c}" for g, c in zip(groups[te], y[te])]
            be = torch.tensor(
                [{k: i for i, k in enumerate(sorted(set(ke)))}[k]
                 for k in ke], device=device)
            _bag, inst = model(Xe, be)
            oof[te] = torch.sigmoid(inst).cpu().numpy()
        print(f"[1c] fold {f + 1}/5 done (mean inst CE {tot:.3f})",
              flush=True)

    m = ~np.isnan(oof)
    sf1 = float(f1_score(y[m], (oof[m] >= 0.5).astype(int),
                         average="macro"))
    yp, pp, _ = pat_common.patient_table(y, np.c_[1 - oof, oof], groups,
                                         "mean")
    pred = _loo_threshold(yp, pp)
    pf1 = float(f1_score(yp, pred, average="macro"))
    sens = float(((pred == 1) & (yp == 1)).sum() / max((yp == 1).sum(), 1))
    spec = float(((pred == 0) & (yp == 0)).sum() / max((yp == 0).sum(), 1))

    out = {"seed": args.seed, "spectrum_f1_at_0.5": sf1,
           "patient": {"f1": pf1, "sens": sens, "spec": spec},
           "verdict": f"MIL spectrum F1 {sf1:.3f} · patient F1 {pf1:.3f} "
                      f"(sens {sens:.3f} spec {spec:.3f})"}
    print(f"[1c] {out['verdict']}", flush=True)
    path = os.path.join(exp_common.OUT_DIR, "mil.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    exp_common.record("patient-level", "MIL attention bag", pf1,
                      extra=f"spectrum F1 {sf1:.3f}")
    print(f"[1c] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
