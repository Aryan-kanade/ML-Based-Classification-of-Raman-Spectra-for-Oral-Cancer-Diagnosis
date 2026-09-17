"""
eval_deep_arms.py -- L54 Bayesian search (Optuna TPE or random), L55
TabPFN retest on cleaned features, L56 RamanNet-style 1D CNN probe,
L58 adversarial-validation diagnostic (SAFETY: pure experiment script
-> experiments/ only).
"""

from __future__ import annotations

import os

import numpy as np

import sequential
from exp_common import OUT_DIR, load_data, record, stamp

HERE = os.path.dirname(os.path.abspath(__file__))


def try_optuna():
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna
    except ImportError:
        return None


# ---------------------------------------------------------------- L54
def arm_bayes_search(arch, X, y, groups, wn, classes):
    optuna = try_optuna()
    rng = np.random.default_rng(0)
    facs = sequential._factories(wn)

    def suggest(trial_like, name):
        # returns param dict for one layer config
        if optuna is not None and hasattr(trial_like, "suggest_int"):
            t = trial_like
            return {"max_depth": t.suggest_int("max_depth", 3, 20),
                    "min_samples_leaf": t.suggest_int("min_samples_leaf",
                                                      1, 8),
                    "n_estimators": t.suggest_int("n_estimators", 150,
                                                  600, step=50)}
        return {"max_depth": (None if rng.random() < 0.2 else
                              int(rng.choice([3, 6, 10, 16]))),
                "min_samples_leaf": int(rng.integers(1, 8)),
                "n_estimators": int(rng.choice([150, 250, 400, 600]))}

    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    # tune an estimator of the SAME family as the actual final layer
    # (audit fix 2026-09-16: the old `if arch[-1] == "rf"` guard could
    # never fire — the d2 final layer is "Extra Trees", so every trial
    # silently re-evaluated the untuned chain)
    tree_cls = (ExtraTreesClassifier if "Extra" in arch[-1]
                else RandomForestClassifier)

    trial_configs: list = []

    def objective(t):
        params = suggest(t, "rf")
        trial_configs.append(dict(params))
        est = tree_cls(random_state=0, n_jobs=1, **params)
        tuned_facs = dict(facs)
        tuned_facs[arch[-1]] = lambda: est
        f1s = []
        outer = sequential._splitter(5, 42, y, groups)
        for tr_i, te_i in outer.split(X, y, groups):
            g_tr = groups[tr_i]
            k_in = max(2, min(3, int(min(np.bincount(y[tr_i])))))
            k_in = min(k_in, len(set(g_tr.tolist())))
            F_tr, F_te = X[tr_i], X[te_i]
            for pos, nm in enumerate(arch[:-1]):
                P_tr = sequential._oof_proba(
                    sequential._layer_est(nm, facs, pos, wn),
                    F_tr, list(y[tr_i]), g_tr, k_in, 42)
                e = sequential.fit_maybe_grouped(
                    sequential._layer_est(nm, facs, pos, wn),
                    F_tr, y[tr_i], g_tr)
                F_tr = np.hstack([F_tr, P_tr])
                F_te = np.hstack([F_te, e.predict_proba(F_te)])
            from sklearn.metrics import f1_score
            f1s.append(f1_score(y[te_i],
                                sequential.fit_maybe_grouped(
                                    sequential._layer_est(
                                        arch[-1], tuned_facs,
                                        len(arch) - 1, wn),
                                    F_tr, y[tr_i], g_tr).predict(F_te),
                                average="macro"))
        return float(np.mean(f1s))

    if optuna is not None:
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(
                                        seed=0))
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        best, tag = study.best_value, "TPE-30"
    else:
        vals = [objective(r) for r in range(15)]
        best, tag = max(vals), "random-15"
    print(f"[deep] L54 bayes({tag}) F1 {best:.3f} (selection folds, "
          f"seed 42 — optimistic)", flush=True)
    # audit fix (2026-09-16): the search above SELECTS on the seed-42
    # folds it reports; the honest number evaluates the chosen config
    # on untouched seed-43 folds.
    if optuna is not None:
        honest_params = dict(study.best_params)
    else:
        honest_params = dict(trial_configs[int(np.argmax(vals))])
    honest = _l54_eval_honest(tree_cls(**honest_params), arch, facs, X,
                              y, groups, wn, seed=43)
    print(f"[deep] L54 honest (seed-43 folds) F1 {honest:.3f}",
          flush=True)
    record("L54 bayes-search", f"final-layer {tag} (seed-43 honest)",
           honest)


def _l54_eval_honest(est, arch, facs, X, y, groups, wn, seed=43):
    from sklearn.metrics import f1_score
    tuned_facs = dict(facs)
    tuned_facs[arch[-1]] = lambda: est
    f1s = []
    outer = sequential._splitter(5, seed, y, groups)
    for tr_i, te_i in outer.split(X, y, groups):
        g_tr = groups[tr_i]
        k_in = max(2, min(3, int(min(np.bincount(y[tr_i])))))
        k_in = min(k_in, len(set(g_tr.tolist())))
        F_tr, F_te = X[tr_i], X[te_i]
        for pos, nm in enumerate(arch[:-1]):
            P_tr = sequential._oof_proba(
                sequential._layer_est(nm, facs, pos, wn),
                F_tr, list(y[tr_i]), g_tr, k_in, 42)
            e = sequential.fit_maybe_grouped(
                sequential._layer_est(nm, facs, pos, wn),
                F_tr, y[tr_i], g_tr)
            F_tr = np.hstack([F_tr, P_tr])
            F_te = np.hstack([F_te, e.predict_proba(F_te)])
        f1s.append(f1_score(y[te_i],
                            sequential.fit_maybe_grouped(
                                sequential._layer_est(
                                    arch[-1], tuned_facs, len(arch) - 1,
                                    wn),
                                F_tr, y[tr_i], g_tr).predict(F_te),
                            average="macro"))
    return float(np.mean(f1s))


# ---------------------------------------------------------------- L55
def arm_tabpfn(arch, X, y, groups, wn, classes):
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        print("[deep] L55 tabpfn not installed — skipped", flush=True)
        return
    from sklearn.metrics import f1_score
    outer = sequential._splitter(5, 42, y, groups)
    oof = np.full((len(y), len(classes)), np.nan)
    for tr, te in outer.split(X, y, groups):
        est = TabPFNClassifier()
        est.fit(X[tr], y[tr])
        oof[te] = est.predict_proba(X[te])[:, :len(classes)]
    v = ~np.isnan(oof).any(axis=1)
    f1 = float(f1_score(y[v], oof[v].argmax(axis=1), average="macro"))
    print(f"[deep] L55 TabPFN F1 {f1:.3f}", flush=True)
    record("L55 TabPFN-v2", "single model, grouped 5-fold", f1)


# ---------------------------------------------------------------- L56
def arm_ramannet(arch, X, y, groups, wn, classes):
    import torch
    import torch.nn as nn

    class MultiScaleNet(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.branches = nn.ModuleList([
                nn.Sequential(nn.Conv1d(1, 16, k, padding=k // 2),
                              nn.ReLU(), nn.MaxPool1d(4))
                for k in (5, 15, 41)])
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * (n_in // 4) * 3, 128), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(128, 2))

        def forward(self, x):           # x: (B, n)
            z = [b(x.unsqueeze(1)) for b in self.branches]
            m = min(z_.shape[-1] for z_ in z)
            z = torch.cat([z_[..., :m] for z_ in z], dim=1)
            return self.head(z)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outer = sequential._splitter(5, 42, y, groups)
    oof = np.full((len(y), len(classes)), np.nan)
    for tr, te in outer.split(X, y, groups):
        net = MultiScaleNet(X.shape[1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3,
                               weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        Xtr, ytr = X[tr], y[tr]
        # L13-style in-fold augmentation x2 — audit fix (2026-09-16):
        # lorentzian_synthesize blends RANDOM pairs (cross-class), so
        # synthetic rows must be built PER CLASS and labelled with that
        # class; the old random labels were uncorrelated with content.
        import modeling
        aug_rows, aug_labels = [], []
        for cls_ in np.unique(ytr):
            rows = Xtr[ytr == cls_]
            aug_rows.append(modeling.lorentzian_synthesize(
                rows, len(rows), seed=7 + int(cls_)))
            aug_labels.append(np.full(len(aug_rows[-1]), cls_))
        Xtr = np.vstack([Xtr] + aug_rows)
        ytr = np.concatenate([ytr] + aug_labels)
        tX = torch.tensor(Xtr, dtype=torch.float32, device=dev)
        ty = torch.tensor(ytr, dtype=torch.long, device=dev)
        net.train()
        for _ in range(60):
            perm = torch.randperm(len(tX), device=dev)[:64]
            opt.zero_grad()
            loss = lossf(net(tX[perm]), ty[perm])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            p = net(torch.tensor(X[te], dtype=torch.float32,
                                 device=dev)).softmax(-1).cpu().numpy()
        oof[te] = p
    from sklearn.metrics import f1_score
    v = ~np.isnan(oof).any(axis=1)
    f1 = float(f1_score(y[v], oof[v].argmax(axis=1), average="macro"))
    print(f"[deep] L56 RamanNet F1 {f1:.3f}", flush=True)
    record("L56 RamanNet-style", "multi-scale 1D CNN + aug, GPU", f1)


# ---------------------------------------------------------------- L58
def arm_adversarial(X, groups, names_dates=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import re
    if names_dates is None:
        return
    b = np.array([re.match(r"(\d{8})", nm).group(1)
                  if re.match(r"(\d{8})", nm) else "?" for nm
                  in names_dates])
    top = max(set(b), key=list(b).count)
    yv = (b == top).astype(int)
    if len(set(yv)) < 2:
        return
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True,
                                  random_state=0).split(X, yv):
        c = RandomForestClassifier(n_estimators=200, random_state=0,
                                   n_jobs=1).fit(X[tr], yv[tr])
        p = c.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(yv[te], p))
    a = float(np.mean(aucs))
    print(f"[deep] L58 adversarial date-AUC {a:.3f} "
          f"({'drift present' if a > 0.7 else 'no strong drift'})",
          flush=True)
    # NOTE: this is an AUC, not an F1 — exp_common.record() computes
    # KEEP/tie/drop against the F1 baseline, so label it explicitly
    # (audit finding 13).
    record("L58 adversarial", "date-batch AUC (NOT F1; ignore verdict)",
           a)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    arch, X, y, groups, wn, _meta = load_data()
    y_arr = np.searchsorted(sorted(set(y)), np.asarray(y))
    groups = np.asarray(groups)
    classes = sorted(set(y))
    print(f"[deep] {stamp()} arch {' -> '.join(arch)}", flush=True)
    arm_bayes_search(arch, X, y_arr, groups, wn, classes)
    arm_tabpfn(arch, X, y_arr, groups, wn, classes)
    arm_ramannet(arch, X, y_arr, groups, wn, classes)
    try:
        from eval_signals import load_named
        _X, _y, _g, _wn, names = load_named()
        arm_adversarial(np.asarray(X, dtype=np.float32), groups, names)
    except Exception as e:
        print(f"[deep] L58 skipped: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
