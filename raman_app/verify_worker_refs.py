"""
verify_worker_refs.py — 2026-09-06 acceptance check for the margin-mode
auto-reference fix.

Replays the user's failing GUI run through the REAL PredictWorker: the
Tumor class folder of the real clinical tree + the saved 3SSE Extra
Trees bundle, empty manual reference (exactly the 13:32 session).
Expectation after the fix: per-patient NORMAL references are auto-built
("Margin mode: built N ..." finally appears), patients with a Normal
side predict, Tumor-only patients are skipped with a logged reason.

Run:  python verify_worker_refs.py [data_root] [bundle.joblib]
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qt_compat import QtWidgets            # noqa: E402
import ui_helpers as uh                    # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else r"D:\BARC\Data\Tumor"
_HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = (sys.argv[2] if len(sys.argv) > 2
          else next((os.path.join(_HERE, d, "winner.joblib")
                     for d in ("study_run_3sse", "study_run_3sse_d2")
                     if os.path.isfile(os.path.join(_HERE, d,
                                                    "winner.joblib"))),
                    os.path.join(_HERE, "study_run_3sse", "winner.joblib")))
if not os.path.isfile(BUNDLE):
    sys.exit(f"bundle not found: {BUNDLE}\nUsage: python "
             f"verify_worker_refs.py [data_root] [bundle.joblib]")

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
app.setStyle("Fusion")
app.setStyleSheet(uh.STYLESHEET)
import gui                               # noqa: E402
import modeling                          # noqa: E402
from clinical_data import REFERENCE_MARKERS  # noqa: E402

bundle = modeling.load_bundle(BUNDLE)
print(f"bundle: {bundle.get('model_name')} paired={bundle.get('paired')}")

files, skipped = [], []
for dirpath, _dns, fns in os.walk(ROOT):
    for fn in sorted(fns):
        if not fn.lower().endswith((".txt", ".dat", ".csv")):
            continue
        if (any(m in fn.lower() for m in REFERENCE_MARKERS)
                or fn.lower() == "split_log.txt"):
            skipped.append(fn)
            continue
        files.append(os.path.join(dirpath, fn))
print(f"selection: {len(files)} spectra ({len(skipped)} non-sample skipped)")

# mirror _prediction_files: parent dir of the first file when files come
# from a multi-file selection; here ROOT is a folder
worker = gui.PredictWorker(bundle, files, ROOT, manual_ref_dir=None)
payload, failed = {}, []
worker.done.connect(lambda d: payload.update(d))
worker.failed.connect(lambda tb: failed.append(tb))
worker.run()                              # synchronous

if failed:
    print("WORKER FAILED:\n", failed[0])
    sys.exit(1)
skips = [l for l in payload["logs"] if l.startswith("SKIPPED ")]
reasons = [l for l in payload["logs"] if l.startswith("Margin references")]
print(f"predicted: {len(payload['rows'])} / {len(files)}")
print(f"skipped:   {len(skips)}")
print(f"auto refs built: {payload['n_auto_refs']}")
for l in reasons:
    print(" ", l)
pred = Counter(r[1] for r in payload["rows"])
print("prediction counts:", dict(pred))
for r in payload["rows"][:6]:
    print(f"  {r[0]} -> {r[1]} (p={r[2]:.3f})")
