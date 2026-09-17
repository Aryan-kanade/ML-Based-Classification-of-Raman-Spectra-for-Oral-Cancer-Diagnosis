"""
eval_ranking.py -- Stage 1b of the push-to-0.8 program v10.

Within-patient pairwise RANKING objective (RankNet): the encoder must
score every Tumor spectrum of a patient ABOVE that patient's own Normal
spectra.  64 patients give ~hundreds of clean within-patient contrasts
-- no absolute thresholds, no between-patient amplitude confounds.
Honest evaluation: StratifiedGroupKFold patient-grouped outer folds;
the decision threshold is tuned leave-one-patient-out inside each
TRAINING fold only.

Usage:  python eval_ranking.py [--seed 42] [--epochs 120]
Output: experiments/ranking.json
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

HERE = os.path.dirname(os.path.abspath(__file__))


class RankEncoder(nn.Module):
    def __init__(self, d_in: int, width: int = 512, dim: int = 128,
                 p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.ReLU(), nn.Dropout(p),
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(p),
            nn.Linear(width, dim))
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)   # scalar score


def _patient_folds(y, groups, seed, k=5):
    """Patient-level stratified folds: label = dominant class."""
    pats = np.unique(groups)
    ydom = []
    for pat in pats:
        m = groups == pat
        vals, cnts = np.unique(y[m], return_counts=True)
        ydom.append(int(vals[np.argmax(cnts)]))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups)), np.asarray(ydom)


def _loo_threshold(y, p):
    pred = np.empty_like(y)
    for i in range(y.size):
        tr = np.arange(y.size) != i
        grid = np.unique(p[tr])
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y[tr], (p[tr] >= t).astype(int), average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        pred[i] = int(p[i] >= best_t)
    return pred, best_f1


def main() -> int:
    if not HAS_TORCH:
        print("[1b] torch unavailable — skip", flush=True)
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    _arch, X, y, groups, _wn, _meta = exp_common.load_data()
    y = np.asarray(y)
    _cls = sorted(np.unique(y))          # ['Normal','Tumor']
    y = np.asarray([_cls.index(v) for v in y])
    groups = np.asarray(groups)
    folds, _ = _patient_folds(y, groups, args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    oof_score = np.full(len(y), np.nan)
    fold_thr = []                        # per-fold TRAIN-median threshold
    for f, (tr, te) in enumerate(folds):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32,
                          device=device)
        yt = y[tr]
        gt = groups[tr]
        model = RankEncoder(X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # within-patient (tumor, normal) pairs
        pairs = []
        for pat in np.unique(gt):
            m = gt == pat
            tu = np.where(m & (yt == 1))[0]
            no = np.where(m & (yt == 0))[0]
            pairs.extend((i, j) for i in tu for j in no)
        pairs = np.asarray(pairs) if pairs else np.zeros((0, 2), int)
        if len(pairs) == 0:                    # degenerate fold guard
            oof_score[te] = 0.0
            print(f"[1b] fold {f + 1}/5 skipped (no within-patient "
                  "pairs)", flush=True)
            continue
        for ep in range(args.epochs):
            model.train()
            perm = rng.permutation(len(pairs))
            losses = []
            for s in range(0, len(pairs), 64):
                b = perm[s:s + 64]
                i, j = pairs[b, 0], pairs[b, 1]
                si = model(Xt[i])
                sj = model(Xt[j])
                # pairwise logistic loss  log(1+exp(s_n - s_t))
                loss = nn.functional.softplus(sj - si).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        # train-side median score = the spectrum-level decision threshold
        # (audit fix 2026-09-16: the old full-OOF median included test
        # rows)
        with torch.no_grad():
            train_scores = model(Xt).cpu().numpy()
        fold_thr.append(float(np.median(train_scores)))
        # OOF scores for the test patients
        model.eval()
        with torch.no_grad():
            Xe = torch.tensor((X[te] - mu) / sd, dtype=torch.float32,
                              device=device)
            oof_score[te] = model(Xe).cpu().numpy()
        print(f"[1b] fold {f + 1}/5 done (final loss "
              f"{np.mean(losses):.3f}, {len(pairs)} pairs)", flush=True)

    # patient-level roll-up + LOO-tuned threshold (honest)
    yp, sp, _pats = pat_common.patient_table(y, oof_score, groups, "mean")
    pred, _ = _loo_threshold(yp, sp)
    sens = float(((pred == 1) & (yp == 1)).sum() / max((yp == 1).sum(), 1))
    spec = float(((pred == 0) & (yp == 0)).sum() / max((yp == 0).sum(), 1))
    pf1 = float(f1_score(yp, pred, average="macro"))

    # spectrum-level reference: each fold's TRAIN-median threshold
    # applied to that fold's test rows (audit fix 2026-09-16 — the old
    # full-OOF median peeked at the test distribution)
    pred_spec = np.zeros(len(y), dtype=int)
    scored = ~np.isnan(oof_score)
    for (_tr, te), thr in zip(folds, fold_thr):
        m = te[scored[te]]
        pred_spec[m] = (oof_score[m] >= thr).astype(int)
    sf1 = float(f1_score(y[scored], pred_spec[scored], average="macro"))

    out = {"seed": args.seed, "epochs": args.epochs,
           "patient": {"f1": pf1, "sens": sens, "spec": spec,
                       "n_correct": int((pred == yp).sum()),
                       "n_patients": int(yp.size)},
           "spectrum_median_binned_f1": sf1,
           "verdict": f"RankNet patient F1 {pf1:.3f} (sens {sens:.3f} "
                      f"spec {spec:.3f}) vs Stage-1a baseline"}
    print(f"[1b] {out['verdict']}", flush=True)
    path = os.path.join(exp_common.OUT_DIR, "ranking.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    exp_common.record("patient-level", "RankNet within-patient", pf1,
                      extra=f"patient sens {sens:.3f} spec {spec:.3f}")
    print(f"[1b] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
