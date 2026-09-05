"""
clinical_data.py — loader for the clinical Raman dataset layout:

    <root>/<class>/<patient folder>/<spectrum>.csv

e.g.  <root>/Normal/Patient_07/20250107_site07_n0_interpolated.csv

<root> is found per device by find_data_root() — no hardcoded paths.

The class label comes from the TOP-LEVEL folder (never the filename —
several filenames are malformed), and every patient folder yields a
subject key so splits can be grouped by patient (the dataset is PAIRED:
the same subject usually has both Normal and Tumor spectra; a split that
puts one subject on both sides leaks).

Automatic data hygiene (everything is reported, nothing is deleted from
disk):
  * white/black/dark/background reference spectra are excluded,
  * spectra whose TH*/NH* site token contradicts the class folder are
    dropped (mislabeled acquisitions),
  * byte-identical duplicate spectra are collapsed to one copy,
  * spectra that appear IDENTICALLY under two different classes are
    dropped entirely (impossible labels — pure leakage).

`patient_split()` gives a stratified, patient-grouped train/val/test
split (approx. 70/15/15) using StratifiedGroupKFold.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import dataset as ds
from preprocessing import spike_score

# spectra whose spike score exceeds this are flagged as low quality.
# The score distribution is instrument-specific; on the current dataset
# the median is ~60 and true cosmic-ray outliers sit above ~300 (p95).
SPIKE_FLAG_THRESHOLD = 300.0

# top-level folder name -> canonical class label (case-insensitive)
CLASS_SYNONYMS = {
    "normal": "Normal", "control": "Normal", "healthy": "Normal",
    "benign": "Normal",
    "tumor": "Tumor", "tumour": "Tumor", "cancer": "Tumor",
    "malignant": "Tumor",
}
# filename markers of instrument references (not patient tissue)
REFERENCE_MARKERS = ("white", "black", "dark", "bkg", "background",
                     "ref", "reference", "blank", "calib")
_NUM_RE = re.compile(r"\d+")


# --------------------------------------------------------------------------
# Layout detection
# --------------------------------------------------------------------------
def is_clinical_layout(folder: str) -> bool:
    """True if `folder` contains >=2 subfolders that themselves contain
    spectrum files one or two levels down (class/patient/file layout)."""
    if not os.path.isdir(folder):
        return False
    class_dirs = 0
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isdir(path):
            continue
        has_spectra = False
        for sub in [path] + [os.path.join(path, s)
                             for s in sorted(os.listdir(path))
                             if os.path.isdir(os.path.join(path, s))]:
            try:
                entries = os.listdir(sub)
            except OSError:
                continue
            if any(f.lower().endswith((".txt", ".dat", ".csv"))
                   for f in entries):
                has_spectra = True
                break
        if has_spectra:
            class_dirs += 1
            if class_dirs >= 2:
                return True
    return False


# --------------------------------------------------------------------------
# Portable dataset-root discovery (works on any device — no hand-edits)
# --------------------------------------------------------------------------
# per-DEVICE pointer: the home dir does not travel with a copied project
# folder, so each machine remembers its own dataset location here.
DATA_POINTER = os.path.join(os.path.expanduser("~"), ".raman_app_data_dir")


def find_data_root() -> str | None:
    """
    First existing clinical-layout folder, checked in order:
      1. RAMAN_DATA_DIR environment variable (explicit override),
      2. the ~/.raman_app_data_dir pointer (auto-written the first time
         a clinical folder is loaded in the GUI),
      3. Data/ next to this package, ~/Desktop/data/Data, ~/Data,
         raman_app/Data.
    Returns None when no candidate exists — callers fall back to --data /
    Browse.  Every candidate is validated with is_clinical_layout().
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))
    home = os.path.expanduser("~")
    cands: list[str] = []
    env = os.environ.get("RAMAN_DATA_DIR")
    if env:
        cands.append(env)
    try:
        with open(DATA_POINTER, encoding="utf-8") as fh:
            cands.append(fh.read().strip())
    except OSError:
        pass
    cands += [os.path.join(app_dir, "..", "Data"),
              os.path.join(home, "Desktop", "data", "Data"),
              os.path.join(home, "Data"),
              os.path.join(app_dir, "Data")]
    for c in cands:
        if c and is_clinical_layout(c):
            return os.path.normpath(c)
    return None


def remember_data_root(path: str):
    """Persist the chosen dataset root for THIS device (see DATA_POINTER)."""
    try:
        with open(DATA_POINTER, "w", encoding="utf-8") as fh:
            fh.write(path)
    except OSError:
        pass                     # best-effort: the candidates still work


def subject_key(patient_folder: str) -> str:
    """Normalize a patient folder name to a subject key.

    Patient_15 / TDOC015 / 'TDOC015 Spectra pro'  ->  'S15'
    (same subject measured in both naming generations).  Folders without
    a number keep their uppercased name.
    """
    m = _NUM_RE.search(patient_folder)
    return f"S{int(m.group())}" if m else patient_folder.strip().upper()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class ClinicalData:
    spectra: list                      # list[ds.Spectrum]
    groups: list[str]                  # subject key per spectrum
    grid: np.ndarray | None            # shared wavenumber grid, if identical
    report: dict = field(default_factory=dict)
    spike_scores: list[float] = field(default_factory=list)  # per spectrum
    flagged: list[bool] = field(default_factory=list)        # per spectrum


def _content_hash(wn: np.ndarray, it: np.ndarray) -> str:
    """Hash the numeric content (immune to line-ending/format differences)."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(wn, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(it, dtype=np.float64).tobytes())
    return h.hexdigest()


def load_clinical_dataset(root: str) -> ClinicalData:
    """
    Load <root>/<class>/<patient>/*.csv with reference exclusion,
    duplicate collapsing and cross-class-collision dropping.
    Returns ClinicalData(spectra, groups, grid, report).
    """
    if not os.path.isdir(root):
        raise ValueError(f"Not a folder: {root}")

    report: dict = {
        "root": os.path.abspath(root),
        "class_map": {},          # original folder -> class label
        "references_excluded": [],
        "duplicates_dropped": [],      # (kept_path, dropped_path)
        "cross_class_dropped": [],     # paths dropped for impossible labels
        "site_token_dropped": [],      # TH*/NH* token contradicts class folder
        "load_errors": [],
        "grid_identical": True,
    }

    # ---- 1. scan class/patient/file --------------------------------------
    # spectra are loaded AS-IS over their full axis (15.5-3862 cm-1 on
    # this instrument) — no edge trimming, every data point is kept
    # (user decision 2026-09-04)
    records = []                    # (class, subject, rel, path, wn, it)
    for cls_dir in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls_dir)
        if not os.path.isdir(cls_path):
            continue                                # e.g. split_log.txt
        label = CLASS_SYNONYMS.get(cls_dir.strip().lower(), cls_dir.strip())
        report["class_map"][cls_dir] = label
        for pat_dir in sorted(os.listdir(cls_path)):
            pat_path = os.path.join(cls_path, pat_dir)
            if not os.path.isdir(pat_path):
                continue
            for fn in sorted(os.listdir(pat_path)):
                if not fn.lower().endswith((".txt", ".dat", ".csv")):
                    continue
                rel = os.path.join(cls_dir, pat_dir, fn)
                if any(m in fn.lower() for m in REFERENCE_MARKERS):
                    report["references_excluded"].append(rel)
                    continue
                # site-token vs folder-class check (2026-09-05 audit): a
                # TH*/NH* site token contradicting its class folder is a
                # mislabeled acquisition (a tumor-site file inside Normal/
                # trains as Normal and no dedupe can catch a unique copy).
                # Only fires for unambiguous files (one clear token).
                if label in ("Normal", "Tumor"):
                    has_t = re.search(r"TH\d", fn) is not None
                    has_n = re.search(r"NH\d", fn) is not None
                    if ((label == "Normal" and has_t and not has_n)
                            or (label == "Tumor" and has_n and not has_t)):
                        report["site_token_dropped"].append(rel)
                        continue
                path = os.path.join(pat_path, fn)
                try:
                    wn, it = ds.load_spectrum(path)
                except Exception as exc:
                    report["load_errors"].append(f"{rel}: {exc}")
                    continue
                records.append((label, subject_key(pat_dir), rel, path,
                                wn, it))
    if not records:
        raise ValueError(
            f"No spectra found under {root!r} "
            "(expected <class>/<patient folder>/*.csv)"
        )
    report["files_scanned"] = len(records) + len(report["references_excluded"])

    # ---- 2. dedupe: identical content -------------------------------------
    # cross-class collisions make the label ambiguous -> drop every copy;
    # same-class duplicates -> keep the first (paths sorted above)
    by_hash: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_hash.setdefault(_content_hash(rec[4], rec[5]), []).append(i)

    keep = []
    for _, idxs in sorted(by_hash.items()):
        classes = {records[i][0] for i in idxs}
        if len(classes) > 1:
            report["cross_class_dropped"].extend(records[i][2] for i in idxs)
            continue
        keep.append(idxs[0])
        for i in idxs[1:]:
            report["duplicates_dropped"].append(
                (records[idxs[0]][2], records[i][2]))
    keep.sort()

    # ---- 3. shared-grid check ---------------------------------------------
    ref_wn = records[keep[0]][4]
    grid = np.asarray(ref_wn, dtype=float)
    if any(len(records[i][4]) != len(grid) or
           not np.allclose(records[i][4], grid, rtol=0, atol=1e-6)
           for i in keep):
        report["grid_identical"] = False
        grid = None        # consumer falls back to dataset.common_grid()

    # ---- 4. build spectra --------------------------------------------------
    spectra, groups, scores = [], [], []
    for i in keep:
        label, subj, rel, path, wn, it = records[i]
        spectra.append(ds.Spectrum(name=rel.replace(os.sep, "/"),
                                   path=path, wavenumbers=wn,
                                   intensities=it, label=label))
        groups.append(subj)
        scores.append(spike_score(it))
    flagged = [s >= SPIKE_FLAG_THRESHOLD for s in scores]
    report["spike_scores"] = {
        "median": float(np.median(scores)),
        "p95": float(np.percentile(scores, 95)),
        "max": float(np.max(scores)),
    }
    report["flagged_spectra"] = [spectra[i].name for i in range(len(spectra))
                                 if flagged[i]]

    # ---- 5. per-class / per-patient stats ----------------------------------
    cls_counts: dict[str, int] = {}
    pat_counts: dict[str, dict[str, int]] = {}
    for s in spectra:
        cls_counts[s.label] = cls_counts.get(s.label, 0) + 1
        pat_counts.setdefault(s.label, {})
    per_pat: dict[str, set[str]] = {}
    for s, g in zip(spectra, groups, strict=True):
        per_pat.setdefault(g, set()).add(s.label)
    for g, labels in per_pat.items():
        for l in labels:
            pat_counts[l][g] = pat_counts[l].get(g, 0) + 1
    report["class_counts"] = cls_counts
    report["patients_per_class"] = {c: len(v) for c, v in pat_counts.items()}
    report["patients_in_both_classes"] = sorted(
        g for g, ls in per_pat.items() if len(ls) > 1)
    report["n_spectra"] = len(spectra)
    report["n_subjects"] = len(per_pat)
    return ClinicalData(spectra, groups, grid, report, scores, flagged)


# --------------------------------------------------------------------------
# Patient-grouped split
# --------------------------------------------------------------------------
def patient_split(groups, labels, seed: int = 42):
    """
    Stratified, patient-GROUPED ~70/15/15 split -> (train_idx, val_idx,
    test_idx).  7 StratifiedGroupKFold folds: 5 train / 1 val / 1 test —
    no subject ever lands on two sides of the split.
    """
    groups = list(groups)
    n = len(groups)
    if n == 0:
        raise ValueError("No spectra to split.")
    n_groups = len(set(groups))
    if n_groups < 7:
        raise ValueError(
            f"Grouped split needs >= 7 distinct patients (found "
            f"{n_groups}) - too few subjects for a grouped 70/15/15 split."
        )
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    folds = [test for _, test in sgkf.split(np.zeros(n),
                                            np.asarray(labels), groups)]
    folds.sort(key=len, reverse=True)      # largest folds -> test, then val
    train_idx = np.sort(np.concatenate(folds[2:])).tolist()
    val_idx = np.sort(folds[1]).tolist()
    test_idx = np.sort(folds[0]).tolist()
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def format_report(report: dict) -> str:
    lines = [
        "Clinical dataset report",
        "=" * 60,
        f"Root: {report.get('root')}",
        f"Files scanned: {report.get('files_scanned', '?')}",
    ]
    for folder, label in report.get("class_map", {}).items():
        n = report.get("class_counts", {}).get(label, 0)
        p = report.get("patients_per_class", {}).get(label, 0)
        lines.append(f"  class {label!r:<8} (folder {folder!r}): "
                     f"{n} spectra / {p} patients")
    both = report.get("patients_in_both_classes", [])
    lines.append(f"Kept: {report.get('n_spectra')} spectra / "
                 f"{report.get('n_subjects')} subjects "
                 f"({len(both)} subjects have both classes)")
    # {..} inside an f-string must stay on ONE line — a newline
    # there is 3.12+-only syntax and broke the app on a Python 3.11 device.
    grid_txt = ("identical in every file" if report.get("grid_identical")
                else "NOT identical - interpolating to common grid")
    lines.append(f"Shared wavenumber grid: {grid_txt}")
    refs = report.get("references_excluded", [])
    lines.append(f"Reference spectra excluded: {len(refs)}")
    for r in refs:
        lines.append(f"  - {r}")
    dups = report.get("duplicates_dropped", [])
    lines.append(f"Duplicate copies dropped: {len(dups)}")
    for kept, drop in dups:
        lines.append(f"  - {drop}  (== {kept})")
    cross = report.get("cross_class_dropped", [])
    lines.append(f"Cross-class identical spectra dropped "
                 f"(impossible labels): {len(cross)}")
    for c in cross:
        lines.append(f"  - {c}")
    spikes = report.get("spike_scores", {})
    if spikes:
        lines.append(f"Spike score (cosmic-ray indicator): median "
                     f"{spikes['median']:.0f} · p95 {spikes['p95']:.0f} · "
                     f"max {spikes['max']:.0f}")
        fl = report.get("flagged_spectra", [])
        lines.append(f"Low-quality spectra flagged (spike score >= "
                     f"{SPIKE_FLAG_THRESHOLD:.0f}): {len(fl)}")
        for f in fl:
            lines.append(f"  - {f}")
    errs = report.get("load_errors", [])
    if errs:
        lines.append(f"Load errors: {len(errs)}")
        for e in errs:
            lines.append(f"  - {e}")
    return "\n".join(lines)


def write_report(path: str, report: dict):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(format_report(report) + "\n")
