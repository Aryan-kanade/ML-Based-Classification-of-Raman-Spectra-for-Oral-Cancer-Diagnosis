"""
dataset.py — loading Raman spectra, parsing class labels from filenames,
building a common wavenumber grid, and generating a demo dataset.

File format: 2 columns (wavenumber, intensity) separated by any whitespace,
one spectrum per file.  Class label is auto-detected from the filename via
a `C<number>` token, e.g.  P01_cAg_785_C8_3.txt  ->  class "C8".
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

CLASS_TOKEN_RE = re.compile(r"(?:^|[_.\-])(C\d+)(?=[_.\-]|$)")


@dataclass
class Spectrum:
    name: str                 # filename without extension
    path: str
    wavenumbers: np.ndarray   # ascending
    intensities: np.ndarray
    label: str = ""           # auto-detected class token, e.g. "C8"

    def __len__(self) -> int:
        return len(self.wavenumbers)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def parse_class_token(filename: str) -> str:
    """Extract the class token (e.g. 'C8') from a filename. '' if absent."""
    m = CLASS_TOKEN_RE.search(filename)
    return m.group(1) if m else ""


def load_spectrum(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Robustly load a 2-column spectrum file (skips headers/bad lines)."""
    wns, ins = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                wn, val = float(parts[0]), float(parts[1])
            except ValueError:
                continue  # header or comment line
            wns.append(wn)
            ins.append(val)
    if len(wns) < 10:
        raise ValueError(f"Could not parse a spectrum from: {path}")
    wn = np.asarray(wns, dtype=float)
    it = np.asarray(ins, dtype=float)
    order = np.argsort(wn)
    return wn[order], it[order]


def load_folder(folder: str) -> list[Spectrum]:
    """Load every .txt/.dat/.csv spectrum in a folder (non-recursive)."""
    spectra = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".txt", ".dat", ".csv")):
            continue
        path = os.path.join(folder, fn)
        try:
            wn, it = load_spectrum(path)
        except Exception as exc:  # skip unreadable files, caller can log
            print(f"[dataset] skipping {fn}: {exc}")
            continue
        spectra.append(Spectrum(
            name=os.path.splitext(fn)[0],
            path=path,
            wavenumbers=wn,
            intensities=it,
            label=parse_class_token(fn),
        ))
    return spectra


def common_grid(spectra: list[Spectrum], max_points: int = 2000) -> np.ndarray:
    """
    Build a common ascending wavenumber axis covering the intersection of
    all spectra ranges, with the median number of points.
    """
    if not spectra:
        raise ValueError("No spectra loaded.")
    lo = max(float(s.wavenumbers.min()) for s in spectra)
    hi = min(float(s.wavenumbers.max()) for s in spectra)
    if lo >= hi:
        raise ValueError(
            f"Spectral ranges do not overlap (common span {lo:.1f}..{hi:.1f} cm-1)."
        )
    n = int(np.median([len(s) for s in spectra]))
    n = int(np.clip(n, 50, max_points))
    return np.linspace(lo, hi, n)


def to_matrix(spectra: list[Spectrum], grid: np.ndarray,
              labels: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Interpolate every spectrum onto `grid` -> (X, y)."""
    X = np.empty((len(spectra), len(grid)), dtype=float)
    for i, s in enumerate(spectra):
        X[i] = np.interp(grid, s.wavenumbers, s.intensities)
    y = list(labels) if labels is not None else [s.label for s in spectra]
    return X, y


# --------------------------------------------------------------------------
# Demo dataset generator (derived from a real spectrum)
# --------------------------------------------------------------------------
# Spectral peak windows (cm-1) that get scaled per class: typical SERS
# biochemical contributors (lipids/proteins/nucleic acids).
_PEAK_CENTERS = np.array([480, 525, 640, 725, 830, 880, 1003, 1095,
                          1130, 1205, 1240, 1335, 1450, 1555, 1580, 1655])
_PEAK_WIDTH = 28.0


def _peak_mask(wn: np.ndarray) -> np.ndarray:
    m = np.zeros_like(wn, dtype=bool)
    for c in _PEAK_CENTERS:
        m |= np.abs(wn - c) <= _PEAK_WIDTH
    return m


def generate_demo_data(source_path: str, out_dir: str,
                       n_per_class: int = 15, seed: int = 42) -> str:
    """
    Create a synthetic 3-class dataset (C1 / C5 / C8) from one real spectrum:
    the baseline is kept, the Raman peak signal is scaled per class, plus
    noise, small wavenumber shifts and baseline tilt.  Files are named
    DEMO_S{i}_cAg_785_{class}_{rep}.txt so the loader auto-detects classes.

    Returns the output directory.
    """
    from preprocessing import als_baseline, savgol_smooth

    rng = np.random.default_rng(seed)
    wn, y = load_spectrum(source_path)
    y = savgol_smooth(y, window=15, poly=3)
    baseline = als_baseline(y, lam=1e5, p=0.01, niter=10)
    peaks = y - baseline
    pk = _peak_mask(wn)
    noise_sigma = 0.03 * float(np.std(peaks[pk])) if pk.any() else 0.01 * float(np.std(y))

    # class -> peak gain (separable by design so the demo shows good metrics)
    class_gains = {"C1": 0.45, "C5": 0.95, "C8": 1.55}

    os.makedirs(out_dir, exist_ok=True)
    for cls, gain in class_gains.items():
        for i in range(n_per_class):
            shift = rng.normal(0.0, 0.35)
            wns = wn + shift
            tilt = rng.normal(0.0, 0.01) * (wn - wn.min()) / max(float(np.ptp(wn)), 1.0)
            gain_i = gain * rng.normal(1.0, 0.06)
            synth = baseline.copy()
            synth[pk] += gain_i * peaks[pk]
            synth = synth + tilt * float(np.median(np.abs(baseline)))
            synth = synth + rng.normal(0.0, noise_sigma, size=wn.shape)
            rep = (i % 3) + 1
            fname = f"DEMO_S{i + 1:02d}_cAg_785_{cls}_{rep}.txt"
            with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fh:
                for w, v in zip(wns[::-1], synth[::-1], strict=True):  # descending, like source
                    fh.write(f"{w:.2f}\t{v:.1f}\n")
    return out_dir


def find_default_source_spectrum() -> str | None:
    """Look for a real spectrum .txt next to the app folder (parent dir)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand_dir in (os.path.dirname(here), here):
        if not os.path.isdir(cand_dir):
            continue
        for fn in sorted(os.listdir(cand_dir)):
            if fn.lower().endswith(".txt") and not fn.upper().startswith("DEMO"):
                return os.path.join(cand_dir, fn)
    return None
