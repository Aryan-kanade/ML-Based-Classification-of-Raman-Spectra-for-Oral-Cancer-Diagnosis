"""
verify_deploy_fix.py — 2026-09-06 acceptance check for the deploy-path
fixes (wn_calibrate applied at predict time, paired reference alignment).

Loads the deployed 3SSE bundle, walks the real clinical tree
(<root>/<class>/<patient>), builds each patient's normal reference with
the SAME code the GUI uses (paired.reference_vector + auto-discovery),
predicts every spectrum through modeling.predict_with_bundle, and prints
spectrum-level + patient-level confusion against the folder labels.

Expected after the fix: Normal patients are classified Normal again
(these are the model's own training patients — a functioning, even
overfit, model must get them right; before the fix EVERY spectrum came
out Tumor p>=0.94).

Run:  python verify_deploy_fix.py [data_root] [bundle.joblib]
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset as ds               # noqa: E402
import modeling                    # noqa: E402
import paired as paired_mod        # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else r"D:\BARC\Data"
BUNDLE = (sys.argv[2] if len(sys.argv) > 2
          else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "model_3SSE__PCA___XGBoost___Ensemble__"
                            "top-3_.joblib"))

bundle = modeling.load_bundle(BUNDLE)
print(f"bundle: {bundle.get('model_name')}  paired={bundle.get('paired')}")
print(f"prep wn_calibrate="
      f"{bundle['prep_params'].wn_calibrate if hasattr(bundle['prep_params'], 'wn_calibrate') else '?'}"
      f"  threshold={bundle.get('threshold')}  "
      f"calibrator={bundle.get('calibrator')}")
classes = list(bundle["classes"])
pos = classes[1] if classes[1].lower() in ("tumor", "b") else classes[0]
thr = bundle.get("threshold")

# ---- collect files per (class, patient) --------------------------------
tree: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
for cls in os.listdir(ROOT):
    cdir = os.path.join(ROOT, cls)
    if not os.path.isdir(cdir):
        continue
    for pat in os.listdir(cdir):
        pdir = os.path.join(cdir, pat)
        if not os.path.isdir(pdir):
            continue
        for dp, _dns, fns in os.walk(pdir):
            for fn in sorted(fns):
                if fn.lower().endswith((".txt", ".dat", ".csv")):
                    tree[cls][pat].append(os.path.join(dp, fn))

n_files = sum(len(v) for c in tree.values() for v in c.values())
print(f"data: {n_files} files, classes={sorted(tree)}, "
      f"patients Normal={len(tree.get('Normal', {}))} "
      f"Tumor={len(tree.get('Tumor', {}))}")

# ---- predict: reference per patient, every spectrum --------------------
spec_tp = spec_fp = spec_tn = spec_fn = 0
site_probs: dict[str, list[tuple[float, bool]]] = defaultdict(list)
ref_cache: dict[str, object] = {}
skip = 0
for pat in sorted(set(tree.get("Normal", {})) | set(tree.get("Tumor", {}))):
    normals = tree.get("Normal", {}).get(pat, [])
    tumors = tree.get("Tumor", {}).get(pat, [])
    if bundle.get("paired") and not normals:
        skip += len(tumors)
        continue
    if pat not in ref_cache and bundle.get("paired"):
        ref_cache[pat] = paired_mod.reference_vector(bundle, normals)
    ref = ref_cache.get(pat)
    for f, is_t in [(x, True) for x in tumors] + [(x, False) for x in normals]:
        wn, it = ds.load_spectrum(f)
        out = modeling.predict_with_bundle(bundle, wn, it, reference=ref)
        pred = out["prediction"]
        p_pos = out["probabilities"].get(pos, float("nan"))
        site_probs[pat].append((p_pos, is_t))
        is_pos_pred = (pred == pos)
        if is_t and is_pos_pred:
            spec_tp += 1
        elif is_t:
            spec_fn += 1
        elif is_pos_pred:
            spec_fp += 1
        else:
            spec_tn += 1

print(f"\nskipped (no normal reference): {skip} tumor spectra")
print("spectrum-level confusion (truth x pred):")
print(f"            pred {pos:<8} pred {next(c for c in classes if c != pos):<8}")
print(f"  true {pos:<8} {spec_tp:<14} {spec_fn:<14}")
print(f"  true {next(c for c in classes if c != pos):<8} {spec_fp:<14} {spec_tn:<14}")
sens = spec_tp / max(spec_tp + spec_fn, 1)
spec = spec_tn / max(spec_tn + spec_fp, 1)
print(f"  sens={sens:.3f}  spec={spec:.3f}  "
      f"acc={(spec_tp + spec_tn) / max(spec_tp + spec_fn + spec_fp + spec_tn, 1):.3f}")

# margin-model patient rollups: each patient contributes normal SITES and
# tumor SITES; "one class per patient" (the GUI banner rule) is not the
# right yardstick when the same patient has both site types
det = det_tot = 0        # tumor sites detected per patient (any site >= thr)
fa = fa_tot = 0          # normal sites flagged per patient (any site >= thr)
mean_t = []
for pat, sites in sorted(site_probs.items()):
    t = [p for p, is_t in sites if is_t]
    n = [p for p, is_t in sites if not is_t]
    cut = thr if thr is not None else 0.5
    if t:
        det_tot += 1
        det += any(p >= cut for p in t)
        mean_t.append(float(np.mean(t)))
    if n:
        fa_tot += 1
        fa += any(p >= cut for p in n)
print(f"\nmargin rollups (cut={0.5 if thr is None else thr:.3f} calibrated):")
print(f"  tumor-site detection:   {det}/{det_tot} patients with >=1 "
      f"tumor site flagged")
print(f"  normal-site false alarm:{fa}/{fa_tot} patients with >=1 "
      f"normal site flagged")
if mean_t:
    print(f"  mean P({pos}) on tumor sites: {np.mean(mean_t):.3f} "
          f"(min {np.min(mean_t):.3f}, max {np.max(mean_t):.3f})")
