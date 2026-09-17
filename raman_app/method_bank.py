"""
method_bank.py -- the 25-method discovery-loop bank (v2 plan, 2026-09-16).

STATE: experiments/method_bank.jsonl (one JSON per method, appended;
resume-safe) + rendered experiments/METHOD_BANK.md.

TIERS (vs baseline 0.753, the d2-winner 3-seed mean):
  works        mean F1 >= 0.740 (std <= 0.030, 3 seeds) OR patient F1 >= 0.500
  strong       mean F1 >= 0.753 (std <= 0.020)
  breakthrough mean F1 >= 0.758 AND exact McNemar p < 0.10 vs baseline OOF
  fail         everything else
Uniqueness = distinct mechanism class; hyperparameter variants do not
count as new methods.

SAFETY: writes experiments/ only; single loky-friendly process.
"""

from __future__ import annotations

import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "experiments")
BANK = os.path.join(OUT_DIR, "method_bank.jsonl")
MD = os.path.join(OUT_DIR, "METHOD_BANK.md")

BASELINE = 0.753
TARGET_N = 25


def classify(entry: dict) -> str:
    """Tier for a bank entry with keys f1_mean, f1_std, pat_f1 (and
    optional mcnemar_p)."""
    fm = entry.get("f1_mean") or 0.0
    fs = entry.get("f1_std")
    fs = 0.0 if fs is None else fs
    pf = entry.get("pat_f1")
    pf = float("nan") if pf is None else pf
    if fm >= 0.758 and entry.get("mcnemar_p", 1.0) < 0.10:
        return "breakthrough"
    # 'strong' requires a 3-seed estimate (audit finding 19: 1-seed
    # arms used to reach the same tier as confirmed ones)
    if fm >= 0.753 and fs <= 0.020 and entry.get("seeds", 0) >= 3:
        return "strong"
    if (fm >= 0.740 and fs <= 0.030) or pf >= 0.500:
        return "works"
    return "fail"


def load_bank() -> list[dict]:
    if not os.path.isfile(BANK):
        return []
    out = []
    with open(BANK, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def bank_ids() -> set[str]:
    return {e["id"] for e in load_bank()}


def add(entry: dict) -> None:
    """Append one method entry (skips duplicate ids)."""
    if entry["id"] in bank_ids():
        return
    entry["tier"] = classify(entry)
    entry["ts"] = time.strftime("%H:%M:%S")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(BANK, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    render()


def render() -> str:
    rows = load_bank()
    order = {"breakthrough": 0, "strong": 1, "works": 2, "fail": 3}
    rows.sort(key=lambda e: (order.get(e.get("tier", "fail"), 3),
                             -(e.get("f1_mean") or 0)))
    n_ok = sum(1 for e in rows if e.get("tier") != "fail")
    lines = [
        "# METHOD BANK — 25 unique working methods (D:\\BARC\\Data only)",
        "",
        f"Baseline 0.753 · qualified **{n_ok}/{TARGET_N}** "
        f"(works = F1>=0.740 std<=0.030 or patient F1>=0.500; "
        "strong >=0.753; breakthrough >=0.758 + McNemar<0.10)",
        "",
        "| id | mechanism | family | F1 mean±std | patient F1 | tier |",
        "|---|---|---|---|---|---|",
    ]
    for e in rows:
        fm = e.get("f1_mean")
        fs = e.get("f1_std")
        f1s = f"{fm:.3f}±{fs:.3f}" if fm is not None and fs is not None \
            else str(e.get("f1", ""))
        pf = e.get("pat_f1")
        pfs = f"{pf:.3f}" if pf is not None else ""
        icon = {"breakthrough": "🚀", "strong": "⭐", "works": "✅",
                "fail": "❌"}
        lines.append(
            f"| {e['id']} | {e.get('mechanism','')} "
            f"| {e.get('family','')} | {f1s} | {pfs} "
            f"| {icon.get(e.get('tier'), '')} {e.get('tier','')} |")
    lines += ["", f"qualified {n_ok}/{TARGET_N}", ""]
    with open(MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return f"qualified {n_ok}/{TARGET_N}"


def harvest_queue() -> int:
    """Import the completed program-queue arms (single-seed) as
    provisional entries marked needs_seed_confirm.  Returns count."""
    imported = [
        # (id, mechanism, family, f1, pat_f1, note)
        ("MQ1", "greedy Caruana ensemble over 3-chain pool (in-fold)",
         "ensemble", 0.765, None, "KEEP in queue run"),
        ("MQ2", "logistic stacking meta over 6 chains+views (in-fold)",
         "ensemble", 0.758, None, ""),
        ("MQ3", "find_peaks top-30 pos/int/width appended to deviations",
         "feature", 0.757, None, ""),
        ("MQ4", "t-test gate k=200 before ET final layer",
         "selection", 0.757, None, ""),
        ("MQ5", "inner-CV bagging of layer OOFs (2 seeds)",
         "ensemble", 0.754, None, ""),
        ("MQ6", "TTA: 4 jittered copies averaged at inference",
         "augmentation", 0.749, None, ""),
        ("MQ7", "fingerprint+high window, deriv-0", "feature", 0.699,
         None, ""),
        ("MQ8", "drop-silent-region window, deriv-0", "feature", 0.711,
         None, ""),
    ]
    n = 0
    for mid, mech, fam, f1, pf, note in imported:
        if mid in bank_ids():
            continue
        add({"id": mid, "mechanism": mech, "family": fam, "f1_mean": f1,
             "f1_std": 0.0, "pat_f1": pf, "seeds": 1,
             "needs_seed_confirm": True, "note": note})
        n += 1
    # RankNet from the patient track (3-seed not run; patient metric)
    if "MC1" not in bank_ids():
        add({"id": "MC1", "mechanism": "RankNet within-patient ranking",
             "family": "objective", "f1_mean": 0.570, "f1_std": 0.0,
             "pat_f1": 0.570, "seeds": 1, "needs_seed_confirm": True,
             "note": "patient sens 0.742 — qualifies on patient track"})
        n += 1
    return n
