"""
dataset.py — loading Raman spectra, parsing class labels from filenames,
building a common wavenumber grid.

File format: 2 columns (wavenumber, intensity) separated by any whitespace,
one spectrum per file.  Class label is auto-detected from the filename via
a `C<number>` token, e.g.  S07_tissue_C8.txt  ->  class "C8".
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
