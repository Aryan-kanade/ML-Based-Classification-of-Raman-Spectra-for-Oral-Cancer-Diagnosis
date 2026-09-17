"""
exp_common.py -- shared helpers for the push-to-0.8 experiment scripts.
SAFETY: read-only over app artifacts; writes go to experiments/ only.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

import sequential
from run_3sse_d2 import _prepare_winner

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "experiments")
SUMMARY = os.path.join(OUT_DIR, "SUMMARY.md")
BASELINE_F1 = 0.753          # L4 multi-seed mean of the d2 winner


def load_data():
    """Paired d2 features + winner arch (cached at module level by the
    orchestrator's separate processes — each script loads once)."""
    with open(os.path.join(HERE, "study_run_3sse_d2", "winner.json"),
              encoding="utf-8") as fh:
        arch = json.load(fh)["arch"]
    X, y, groups, wn, _grid, _params, meta = _prepare_winner(
        sequential.DEFAULT_DATA)
    return arch, np.asarray(X, dtype=np.float32), list(y), list(groups), \
        np.asarray(wn), meta


def record(lever: str, arm: str, f1: float, extra: str = "",
           baseline: float = BASELINE_F1) -> None:
    """L22: append one row to the experiments dashboard."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(SUMMARY):
        with open(SUMMARY, "w", encoding="utf-8") as fh:
            fh.write("# Push-to-0.8 program — results dashboard\n\n"
                     f"Baseline = L4 multi-seed mean of the d2 winner "
                     f"(PLS + XGBoost -> RF -> ET): **{baseline:.3f}**\n\n"
                     f"| lever | arm | F1 | delta | verdict | notes |\n"
                     f"|---|---|---|---|---|---|\n")
    if f1 >= baseline + 0.005:
        verdict = "KEEP"
    elif f1 >= baseline - 0.005:
        verdict = "tie"
    else:
        verdict = "drop"
    with open(SUMMARY, "a", encoding="utf-8") as fh:
        fh.write(f"| {lever} | {arm} | {f1:.3f} | {f1 - baseline:+.3f} "
                 f"| {verdict} | {extra} |\n")


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def oof_f1(m: dict) -> float:
    """Mean per-fold F1 from a validate_arch-style result (f1_mean
    preferred, pooled 'f1' fallback)."""
    v = m.get("f1_mean", m.get("f1", float("nan")))
    return float(v)
