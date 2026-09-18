#!/usr/bin/env python3
"""check_metrics.py — low-value alert for BARC evaluation outputs.

Scans eval result files under raman_app/vit_outputs/ (JSON metric files and
text reports), flags every metric that falls below its threshold, and (with
--ai) asks the Hermes agent (D:\\hermes-agent) to explain WHY the values are
low using the project's own memory (Brain.md).

Usage:
    python check_metrics.py          # plain threshold check (any Python, stdlib only)
    python check_metrics.py --ai     # + Hermes AI diagnosis (run via check-low-values.bat)

Exit codes: 0 = all metrics ok, 1 = low values found, 2 = usage/IO error.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "raman_app" / "vit_outputs"

# Thresholds: (regex on metric key, minimum acceptable value, human label).
# Tune these to the study's targets — they are the definition of "low".
THRESHOLDS = [
    (r"sens", 0.70, "sensitivity"),
    (r"spec", 0.70, "specificity"),
    (r"auc", 0.70, "AUC"),
    (r"f1", 0.60, "F1"),
    (r"acc", 0.65, "accuracy"),
]

SKIP_KEYS = ("params",)  # preprocessing parameter dicts are not metrics


def threshold_for(key: str):
    if re.search(r"std|dev|var\b|range", key, re.IGNORECASE):
        return None, None  # spread statistics are not performance metrics
    for pat, lo, label in THRESHOLDS:
        if re.search(pat, key, re.IGNORECASE):
            return lo, label
    return None, None


def collect_json_metrics(obj, prefix="", out=None):
    """Recursively collect float metrics + 2x2 confusion matrices from a JSON tree."""
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if k in SKIP_KEYS:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                lo, label = threshold_for(k)
                if lo is not None and 0.0 <= float(v) <= 1.0:
                    out.append((path, float(v), lo, label))
            elif isinstance(v, list) and len(v) == 2 and all(
                isinstance(r, list) and len(r) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in r)
                for r in v
            ):
                tn, fp = v[0]
                fn, tp = v[1]
                sens = tp / (tp + fn) if (tp + fn) else float("nan")
                spec = tn / (tn + fp) if (tn + fp) else float("nan")
                out.append((f"{path}.sensitivity(from confusion)", sens, 0.70, "sensitivity"))
                out.append((f"{path}.specificity(from confusion)", spec, 0.70, "specificity"))
            elif isinstance(v, dict):
                collect_json_metrics(v, path, out)
    return out


def collect_txt_metrics(path: Path):
    """Parse overall accuracy / macro-F1 from sklearn-style text reports."""
    out = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    m = re.search(r"(?:test\s+)?accuracy[:\s]+([01]?\.\d+)", text, re.IGNORECASE)
    if m:
        out.append(("test accuracy", float(m.group(1)), 0.65, "accuracy"))
    m = re.search(r"^\s*macro avg.*?([\d.]+)\s+\d+\s*$", text, re.MULTILINE)
    if m:
        out.append(("macro avg f1", float(m.group(1)), 0.60, "F1"))
    return out


def main() -> int:
    ai = "--ai" in sys.argv
    if not OUT_DIR.is_dir():
        print(f"[error] result directory not found: {OUT_DIR}")
        return 2

    rows = []  # (file, metric, value, threshold, label, is_low)
    files = sorted(OUT_DIR.glob("*.json")) + sorted(OUT_DIR.glob("*report*.txt"))
    for f in files:
        if f.suffix == ".json":
            try:
                data = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"[warn] could not parse {f.name}: {e}")
                continue
            for path, val, lo, label in collect_json_metrics(data):
                rows.append((f.name, path, val, lo, label, val < lo))
        else:
            for path, val, lo, label in collect_txt_metrics(f):
                rows.append((f.name, path, val, lo, label, val < lo))

    if not rows:
        print(f"no metrics found under {OUT_DIR}")
        return 2

    print(f"BARC metric check — {datetime.now():%Y-%m-%d %H:%M} — {len(files)} file(s), {len(rows)} metric(s)")
    print("-" * 78)
    current_file = None
    lows = []
    for fname, metric, val, lo, label, is_low in rows:
        if fname != current_file:
            print(f"[{fname}]")
            current_file = fname
        flag = "LOW " if is_low else "ok  "
        detail = f"< {lo:.2f} ({label})" if is_low else f"(min {lo:.2f})"
        print(f"  {flag} {metric:55s} {val:6.3f}  {detail}")
        if is_low:
            lows.append((fname, metric, val, lo, label))
    print("-" * 78)
    if lows:
        print(f"ALERT: {len(lows)} low value(s) detected.")
    else:
        print("All metrics above thresholds.")
        return 0

    if not ai:
        print("\nFor an AI explanation of the low values, run:  check-low-values.bat")
        return 1
    return ai_explain(lows)


def ai_explain(lows) -> int:
    """Ask the Hermes agent (from D:\\hermes-agent) to diagnose the low values."""
    table = "\n".join(f"- {f} :: {m} = {v:.3f} (threshold {lo:.2f}, {label})" for f, m, v, lo, label in lows)
    prompt = (
        "You are embedded in the BARC project (Raman spectroscopy Normal-vs-Tumor "
        "classification) at D:\\BARC. The metric checker just flagged these LOW values:\n\n"
        f"{table}\n\n"
        "Read D:\\BARC\\Brain.md (the project memory: architecture, methodology, "
        "gotchas, results history) and the flagged result files under "
        "D:\\BARC\\raman_app\\vit_outputs\\ if helpful. Then:\n"
        "1. Say plainly which low values matter most for a medical screening use case.\n"
        "2. Give the most likely causes, citing the relevant Brain.md gotcha numbers "
        "or methodology sections where they apply.\n"
        "3. Recommend at most 3 concrete next experiments/fixes, ranked.\n"
        "Be specific to this project; no generic ML advice."
    )
    try:
        from run_agent import AIAgent  # only importable inside the hermes uv env

        agent = AIAgent(
            quiet_mode=True,
            enabled_toolsets=["file"],   # read-only: can read Brain.md/results, no terminal
            max_iterations=15,
            skip_memory=True,
        )
        print("\nAsking Hermes to analyze the low values...\n")
        print(agent.chat(prompt))
        return 1
    except ImportError:
        print(
            "\n[ai unavailable] The Hermes library was not importable. Run this via:\n"
            "  check-low-values.bat   (or: uv run --project D:\\hermes-agent python check_metrics.py --ai)"
        )
        return 1
    except Exception as e:
        print(
            f"\n[ai error] {e}\n"
            "Hermes needs a configured model/API key. Run `hermes setup` once "
            "(set OPENROUTER_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY), then retry."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
