"""
preprocessing.py — standard Raman spectral preprocessing pipeline.

Order (each step toggleable via PreprocessParams):
  0. region crop          (keep crop_min..crop_max cm-1, when wn is given)
  1. wavelet denoising      (soft threshold, universal threshold)
  2. Savitzky-Golay         (smoothing / optional derivative)
  3. ALS baseline removal   (Eilers & Boelens asymmetric least squares)
  4. normalization          (vector norm / SNV / none)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

try:
    import pywt  # PyWavelets
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

try:
    import pybaselines  # noqa: F401 (availability probe: arPLS)
    HAS_PYBASELINES = True
except ImportError:
    HAS_PYBASELINES = False


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass
class PreprocessParams:
    crop_min: float = 400.0        # keep wavenumbers >= this (0 = no crop)
    crop_max: float = 1800.0       # keep wavenumbers <= this (0 = no crop)
    despike: bool = False          # Whitaker-Hayes removal (measured:
                                   # excluding heavily spiked spectra
                                   # beat despiking on this dataset)
    despike_z: float = 7.0         # modified Z-score threshold
    despike_window: int = 5        # replacement moving-average width (odd)
    wavelet: bool = True            # enable wavelet denoising
    wavelet_name: str = "sym8"
    wavelet_level: int = 4
    sg_window: int = 11             # odd
    sg_poly: int = 3
    sg_deriv: int = 0               # 0/1/2
    baseline_method: str = "als"    # 'als' | 'arpls' (Baek 2015)
    als_lambda: float = 1e5         # smoothness
    als_p: float = 0.01             # asymmetry
    als_niter: int = 10
    norm: str = "vector"            # 'vector' | 'snv' | 'none'

    def validate(self) -> "PreprocessParams":
        p = PreprocessParams(**asdict(self))
        p.crop_min = max(0.0, float(p.crop_min))
        p.crop_max = max(0.0, float(p.crop_max))
        if p.crop_min and p.crop_max and p.crop_min >= p.crop_max:
            p.crop_min = p.crop_max = 0.0     # invalid range -> no crop
        p.despike_z = float(np.clip(p.despike_z, 3.0, 20.0))
        p.despike_window = int(max(3, p.despike_window))
        if p.despike_window % 2 == 0:
            p.despike_window += 1
        if p.baseline_method == "arpls" and not HAS_PYBASELINES:
            p.baseline_method = "als"          # graceful fallback
        p.sg_window = int(max(5, p.sg_window))
        if p.sg_window % 2 == 0:
            p.sg_window += 1
        p.sg_poly = int(np.clip(p.sg_poly, 1, p.sg_window - 2))
        p.sg_deriv = int(np.clip(p.sg_deriv, 0, 2))
        if not HAS_PYWT:
            p.wavelet = False
        return p


def despike(y: np.ndarray, z_thresh: float = 7.0, window: int = 5
            ) -> np.ndarray:
    """
    Whitaker & Hayes (2018) cosmic-ray removal: modified Z-scores of the
    first difference (robust: median/MAD) flag outlier points; flagged
    points are replaced by a local moving average.  Real Raman peaks are
    wide enough to survive; single-point spikes are not.
    """
    y = np.asarray(y, dtype=float)
    d = np.diff(y, prepend=y[0])
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    if mad == 0:
        return y
    z = 0.6745 * (d - med) / mad
    spikes = np.abs(z) > z_thresh
    if not spikes.any():
        return y
    out = y.copy()
    half = max(1, int(window) // 2)
    for i in np.where(spikes)[0]:
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        out[i] = np.mean(np.concatenate([y[lo:i], y[i + 1:hi]]))
    return out


def crop_mask(wn: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Boolean mask of the wavenumber range to KEEP (all True if no crop)."""
    p = params.validate()
    m = np.ones(len(wn), dtype=bool)
    if p.crop_min > 0:
        m &= wn >= p.crop_min
    if p.crop_max > 0:
        m &= wn <= p.crop_max
    # degenerate crop (nothing left) -> keep everything
    return m if m.any() else np.ones(len(wn), dtype=bool)


def spike_score(y: np.ndarray) -> float:
    """
    Cosmic-ray spike indicator: largest consecutive intensity jump relative
    to the median jump.  Clean spectra score ~5-30; spiked ones 60+.
    """
    d = np.abs(np.diff(np.asarray(y, dtype=float)))
    med = np.median(d)
    return float(d.max() / med) if med > 0 else 0.0


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------
def wavelet_denoise(y: np.ndarray, wavelet: str = "sym8",
                    level: int | None = None) -> np.ndarray:
    """Soft-threshold wavelet denoising with the universal threshold."""
    if not HAS_PYWT:
        return y
    n = len(y)
    lvl = pywt.dwt_max_level(n, wavelet) if level is None else level
    lvl = max(1, min(int(lvl), pywt.dwt_max_level(n, wavelet)))
    coeffs = pywt.wavedec(y, wavelet, level=lvl)
    # universal threshold from finest detail band (robust sigma estimate)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2.0 * np.log(n))
    den = [coeffs[0]] + [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
    out = pywt.waverec(den, wavelet)[:n]
    # degenerate inputs (e.g. constant zeros) can yield NaNs in the
    # reconstruction — never let them escape
    return np.nan_to_num(out)


def savgol_smooth(y: np.ndarray, window: int = 11, poly: int = 3,
                  deriv: int = 0) -> np.ndarray:
    w = int(max(5, window))
    if w % 2 == 0:
        w += 1
    if w >= len(y):
        w = len(y) - 1 if len(y) % 2 == 0 else len(y) - 2
        w = max(5, w)
    p = int(np.clip(poly, 1, w - 2))
    if p <= deriv:
        p = deriv + 1
    return savgol_filter(y, w, p, deriv=deriv)


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                 niter: int = 10) -> np.ndarray:
    """Asymmetric-least-squares baseline (Eilers & Boelens 2005)."""
    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L - 2, L), format="csc")
    DTD = lam * (D.T @ D)
    w = np.ones(L)
    z = y.copy()
    for _ in range(int(niter)):
        W = sparse.diags(w, 0, format="csc")
        Z = (W + DTD).tocsc()
        z = spsolve(Z, w * y)
        w_new = p * (y > z) + (1.0 - p) * (y <= z)
        if np.allclose(w_new, w):
            break
        w = w_new
    return z


def normalize(y: np.ndarray, method: str = "vector") -> np.ndarray:
    if method == "vector":
        nrm = np.linalg.norm(y)
        return y / nrm if nrm > 1e-12 else y
    if method == "snv":  # standard normal variate
        mu, sd = y.mean(), y.std()
        return (y - mu) / sd if sd > 1e-12 else y - mu
    return y  # 'none'


def pqn_normalize(y: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Probabilistic Quotient Normalization (Dieterle et al. 2006): divide
    the spectrum by the MEDIAN of its quotients against a reference
    spectrum.  Robust when most features are unchanged and only overall
    intensity (dilution / coupling) varies — typical for tissue spectra.
    The reference must be leakage-free (training-fold median, or the
    patient's own normal in paired mode).
    """
    y = np.asarray(y, dtype=float)
    ref = np.asarray(reference, dtype=float)
    mask = np.abs(ref) > 1e-12
    if not mask.any():
        return y
    quotients = y[mask] / ref[mask]
    scale = np.median(quotients)
    return y / scale if abs(scale) > 1e-12 else y


def arpls_baseline(y: np.ndarray, lam: float = 1e5,
                   niter: int = 10) -> np.ndarray:
    """arPLS baseline (Baek et al. 2015, Analyst) via pybaselines."""
    from pybaselines import Baseline
    b = Baseline()
    out = b.arpls(y, lam=lam, max_iter=int(niter) * 5)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def preprocess_spectrum(y: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Apply the full pipeline to one spectrum (on the common grid)."""
    p = params.validate()
    out = np.nan_to_num(np.asarray(y, dtype=float))   # guard odd inputs
    if p.despike:
        out = despike(out, p.despike_z, p.despike_window)
    if p.wavelet:
        out = wavelet_denoise(out, p.wavelet_name, p.wavelet_level)
    out = savgol_smooth(out, p.sg_window, p.sg_poly, p.sg_deriv)
    # baseline removal (skipped for derivative spectra — the derivative
    # already suppresses smooth baselines)
    if p.sg_deriv == 0:
        if p.baseline_method == "arpls":
            baseline = arpls_baseline(out, p.als_lambda, p.als_niter)
        else:
            baseline = als_baseline(out, p.als_lambda, p.als_p, p.als_niter)
        out = out - baseline
    out = normalize(out, p.norm)
    return np.nan_to_num(out)                         # never emit NaN


def preprocess_matrix(X: np.ndarray, params: PreprocessParams,
                      wn: np.ndarray | None = None) -> np.ndarray:
    """
    Apply the pipeline to a matrix of spectra.  When the wavenumber axis
    `wn` is given, the crop range from `params` is applied FIRST and the
    returned matrix has only the kept columns.
    """
    p = params.validate()
    X = np.asarray(X, dtype=float)
    if wn is not None:
        X = X[:, crop_mask(np.asarray(wn, dtype=float), p)]
    return np.vstack([preprocess_spectrum(row, p) for row in X])
