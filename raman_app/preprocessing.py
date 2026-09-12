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

import warnings
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
    crop_min: float = 500.0        # keep wavenumbers >= this (0 = no crop;
                                   # default = tuned working range)
    crop_max: float = 2000.0       # keep wavenumbers <= this (0 = no crop)
    despike: bool = False          # Whitaker-Hayes removal (measured:
                                   # excluding heavily spiked spectra
                                   # beat despiking on this dataset)
    despike_z: float = 7.0         # modified Z-score threshold
    despike_window: int = 5        # replacement moving-average width (odd)
    wavelet: bool = True            # enable wavelet denoising
    wavelet_name: str = "sym8"
    wavelet_level: int = 4
    wavelet_threshold: str = "universal"  # 'universal'|'bayes'|'sure'
    wavelet_mode: str = "soft"            # 'soft'|'hard'|'garrote'
    wavelet_cycle: int = 0                 # cycle-spin shifts (0 = off)
    sg_window: int = 11             # odd
    sg_poly: int = 3
    sg_deriv: int = 0               # 0/1/2
    detrend: bool = False           # subtract a linear trend first
    baseline_method: str = "als"    # 'als'|'arpls'|'iarpls'|'pspline'|'snip'
    als_lambda: float = 1e5         # smoothness
    als_p: float = 0.01             # asymmetry
    als_niter: int = 10
    norm: str = "vector"            # 'vector'|'snv'|'area'|'minmax'|'none'
    wn_calibrate: bool = False      # align spectra on the Phe-1003 peak

    @classmethod
    def saliva(cls) -> "PreprocessParams":
        """
        Saliva-SERS preset (Hanna 2024: saliva is the leading biofluid —
        93.6/94.0 sens/spec): wide crop so the thiocyanate 2100–2136 QC
        band is measured, despiking on (biofluids are spike-prone), fast
        SNIP baseline, SNV normalization.
        """
        return cls(crop_min=400.0, crop_max=2300.0, despike=True,
                   baseline_method="snip", norm="snv")

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
        p.wavelet_threshold = (p.wavelet_threshold.lower()
                               if p.wavelet_threshold.lower() in
                               ("universal", "bayes", "sure")
                               else "universal")
        p.wavelet_mode = (p.wavelet_mode.lower()
                          if p.wavelet_mode.lower() in
                          ("soft", "hard", "garrote") else "soft")
        p.wavelet_cycle = int(np.clip(p.wavelet_cycle, 0, 5))
        if p.baseline_method != "als" and not HAS_PYBASELINES:
            p.baseline_method = "als"          # graceful fallback
        p.norm = (p.norm.lower() if p.norm.lower() in
                  ("vector", "snv", "area", "minmax", "none") else "vector")
        p.sg_window = int(max(5, p.sg_window))
        if p.sg_window % 2 == 0:
            p.sg_window += 1
        p.sg_poly = int(np.clip(p.sg_poly, 1, p.sg_window - 2))
        p.sg_deriv = int(np.clip(p.sg_deriv, 0, 2))
        if not HAS_PYWT:
            p.wavelet = False
        return p


def _spike_runs(spikes: np.ndarray, max_run: int) -> np.ndarray:
    """Keep only flag RUNS of length <= max_run.  A cosmic-ray event is
    1-2 detector pixels (<= 3 consecutive difference outliers); a real
    peak's steep FLANKS raise |z| over long contiguous runs (46 points
    on a sigma=8pt peak at SNR 2000) and must not be rewritten."""
    out = np.zeros_like(spikes)
    i = 0
    n = len(spikes)
    while i < n:
        if spikes[i]:
            j = i
            while j < n and spikes[j]:
                j += 1
            if j - i <= max_run:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def despike(y: np.ndarray, z_thresh: float = 7.0, window: int = 5
            ) -> np.ndarray:
    """
    Whitaker & Hayes (2018) cosmic-ray removal: modified Z-scores of the
    first difference (robust: median/MAD) flag outlier points; flagged
    points are replaced by a local mean of UNFLAGGED neighbours only.

    Audit fixes (2026-09-12):
      * d[0] uses y[1] as the left neighbour (an index-0 spike used to be
        undetectable: ``prepend=y[0]`` forced d[0]=0);
      * mad == 0 falls back to the MEAN absolute deviation instead of
        silently returning the spike untouched (flat/quantized data);
      * only short flag runs (<= 3) are rewritten — longer runs are the
        flanks of real sharp peaks, not cosmic rays;
      * replacement means exclude every flagged index in the window, so
        a neighbouring spike can no longer re-inject ~25% of itself.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return y
    d = np.diff(y, prepend=y[1])                 # d[0] = y[0]-y[1]: a
    med = np.median(d)                           # spike AT index 0 is
    mad = np.median(np.abs(d - med))             # now detectable
    if mad <= 0:
        mad = float(np.mean(np.abs(d - med)))    # constant/quantized data
    if mad <= 0:
        return y                                 # truly flat: nothing to fix
    z = 0.6745 * (d - med) / mad
    spikes = _spike_runs(np.abs(z) > z_thresh, max_run=3)
    if not spikes.any():
        return y
    out = y.copy()
    flagged = np.where(spikes)[0]
    fset = set(flagged.tolist())
    half = max(1, int(window) // 2)
    for i in flagged:
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        clean = [y[j] for j in range(lo, hi) if j not in fset]
        if clean:                                # unflagged neighbours only
            out[i] = float(np.mean(clean))
    return out


def crop_mask(wn: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Boolean mask of the wavenumber range to KEEP (all True if no crop)."""
    p = params.validate()
    m = np.ones(len(wn), dtype=bool)
    if p.crop_min > 0:
        m &= wn >= p.crop_min
    if p.crop_max > 0:
        m &= wn <= p.crop_max
    # degenerate crop (nothing left) -> keep everything, but SAY so —
    # a crop range inside a data gap would otherwise silently no-op
    if not m.any():
        warnings.warn(
            f"crop [{p.crop_min}, {p.crop_max}] keeps nothing on this "
            "grid — cropping disabled (check the wavenumber range)",
            stacklevel=2)
        return np.ones(len(wn), dtype=bool)
    return m


def spike_score(y: np.ndarray) -> float:
    """
    Cosmic-ray spike indicator: largest consecutive intensity jump relative
    to the median jump.  The score scales with peak sharpness AND SNR — a
    clean high-SNR spectrum can score 100+ — so it is only a ranking
    statistic; the operational flag threshold is SPIKE_FLAG_THRESHOLD=300
    in clinical_data.py (calibrated on this instrument's distribution:
    clean median ~60, true outliers 300-1700).
    """
    d = np.abs(np.diff(np.asarray(y, dtype=float)))
    med = np.median(d)
    return float(d.max() / med) if med > 0 else 0.0


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------
def _mad_sigma(detail: np.ndarray) -> float:
    """Robust noise estimate from one detail-coefficient band (MAD)."""
    s = float(np.median(np.abs(detail)) / 0.6745)
    return s if s > 1e-12 else 1e-12


def _sure_threshold(d: np.ndarray, sigma: float) -> float:
    """
    SUREShrink (Donoho & Johnstone 1995): the soft-threshold minimizing
    Stein's Unbiased Risk Estimate,
        SURE(T) = n*sigma^2 + sum_i min(|w_i|,T)^2 - 2*sigma^2*#{|w_i|<=T},
    evaluated at the sorted |w| candidates (the minimizer sits on one).
    Per detail level — the adaptive rule of the internship report.
    """
    n = len(d)
    if n == 0:
        return 0.0
    a = np.sort(np.abs(d))
    cs = np.cumsum(a ** 2)
    k = np.arange(1, n + 1)                    # #{|w| <= a_k} for T = a_k
    sure = n * sigma ** 2 + cs + (n - k) * a ** 2 - 2.0 * sigma ** 2 * k
    best = int(np.argmin(sure))
    return float(a[best]) if sure[best] < n * sigma ** 2 else 0.0


def _bayes_threshold(d: np.ndarray, sigma: float) -> float:
    """
    BayesShrink (Chang et al. 2000): T = sigma^2 / sigma_x with the signal
    variance estimated per level as var(d) - sigma^2 (floored at a small
    fraction so flat noise bands still get shrunk hard).
    """
    var = float(np.mean(d ** 2))
    sig_x = max(var - sigma ** 2, sigma ** 2 / 100.0)
    t = sigma ** 2 / sig_x
    return float(min(t, np.max(np.abs(d)) if len(d) else t))


def _shrink(d: np.ndarray, thresh: float, mode: str) -> np.ndarray:
    """Soft / hard / non-negative-garrote shrinkage of detail coeffs."""
    a = np.abs(d)
    if mode == "hard":
        return np.where(a > thresh, d, 0.0)
    if mode == "garrote":      # Gao 1998: w*(1 - (T/w)^2), smoother than
        return np.where(       # hard, less bias than soft
            a > thresh, d * (1.0 - (thresh / np.maximum(a, 1e-12)) ** 2),
            0.0)
    return pywt.threshold(d, thresh, mode="soft")


def _denoise_once(y: np.ndarray, wavelet: str, lvl: int,
                  threshold: str, tmode: str) -> np.ndarray:
    coeffs = pywt.wavedec(y, wavelet, level=lvl)
    sigma0 = _mad_sigma(coeffs[-1])            # global noise, finest band
    den = [coeffs[0]]
    for d in coeffs[1:]:
        if threshold == "sure":
            t = _sure_threshold(d, _mad_sigma(d))
        elif threshold == "bayes":
            t = _bayes_threshold(d, sigma0)
        else:                                   # 'universal' (VisuShrink)
            t = sigma0 * np.sqrt(2.0 * np.log(len(y)))
        den.append(_shrink(d, t, tmode))
    return pywt.waverec(den, wavelet)[:len(y)]


def wavelet_denoise(y: np.ndarray, wavelet: str = "sym8",
                    level: int | None = None,
                    threshold: str = "universal", tmode: str = "soft",
                    cycle: int = 0) -> np.ndarray:
    """
    Wavelet denoising with selectable threshold rule
    ('universal' | 'bayes' | 'sure'), shrinkage mode ('soft' | 'hard' |
    'garrote') and optional cycle-spinning (translation-invariant
    averaging over +-`cycle` circular shifts; Coifman & Donoho).
    """
    if not HAS_PYWT:
        return y
    n = len(y)
    lvl = pywt.dwt_max_level(n, wavelet) if level is None else level
    lvl = max(1, min(int(lvl), pywt.dwt_max_level(n, wavelet)))
    if cycle <= 0:
        out = _denoise_once(y, wavelet, lvl, threshold, tmode)
    else:
        acc = np.zeros(n)
        for s in range(-cycle, cycle + 1):
            acc += np.roll(_denoise_once(np.roll(y, s), wavelet, lvl,
                                         threshold, tmode), -s)
        out = acc / (2 * cycle + 1)
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
    """Asymmetric-least-squares baseline (Eilers & Boelens 2005).

    The second-difference penalty uses SUPER-diagonal offsets [0, 1, 2]
    — Eilers' ``diff(speye(L), 2)`` orientation.  The previous
    sub-diagonal [0, -1, -2] form made rows 0-1 of D degenerate
    (e_0 and e_1 - 2 e_0), adding a spurious shrink-to-zero penalty on
    the left edge: the baseline collapsed toward 0 over the first
    ~10-20% of every spectrum (2026-09-12 audit; fixed — number
    changing for the default 'als' pipeline)."""
    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), format="csc")
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


def normalize(y: np.ndarray, method: str = "vector",
              wn: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if method == "vector":
        nrm = np.linalg.norm(y)
        return y / nrm if nrm > 1e-12 else y
    if method == "snv":  # standard normal variate
        mu, sd = y.mean(), y.std()
        return (y - mu) / sd if sd > 1e-12 else y - mu
    if method == "area":               # unit integrated area (report §3.1.4)
        # trapezoid ON the wavenumber axis when given — np.trapezoid(y)
        # silently assumes unit spacing and is wrong off-grid
        a = np.trapezoid(y, np.asarray(wn, dtype=float) if wn is not None
                         else None)
        return y / a if abs(a) > 1e-12 else y
    if method == "minmax":             # [0, 1] (report §3.1.4)
        lo, hi = y.min(), y.max()
        return (y - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(y)
    return y  # 'none'


def pqn_normalize(y: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Probabilistic Quotient Normalization (Dieterle et al. 2006): divide
    the spectrum by the MEDIAN of its quotients against a reference
    spectrum.  Robust when most features are unchanged and only overall
    intensity (dilution / coupling) varies — typical for tissue spectra.
    The reference must be leakage-free (training-fold median, or the
    patient's own normal in paired mode).

    Paired mode applies this AFTER baseline removal, so the reference is
    signed and crosses zero; quotients against |ref| ≈ 0 explode.  The
    weakest 10% of |reference| values are therefore masked out of the
    median (2026-09-12 audit) — the median is barely moved, the
    ill-conditioning is gone.
    """
    y = np.asarray(y, dtype=float)
    ref = np.asarray(reference, dtype=float)
    a = np.abs(ref)
    positive = a[a > 1e-12]
    floor = (float(np.quantile(positive, 0.10))
             if positive.size else 0.0)
    mask = a > max(floor, 1e-12)
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


def _pybase(y: np.ndarray, method: str, lam: float) -> np.ndarray:
    """Dispatch into pybaselines (iarpls / pspline_arpls / snip).

    'snip' has no lambda parameter of its own, so `lam` is mapped
    monotonically onto max_half_window (3.2 * lam^0.25, clipped to
    10-60) — otherwise a lambda sweep silently does nothing for snip."""
    from pybaselines import Baseline
    b = Baseline()
    if method == "iarpls":          # Ye et al. 2020 — beats arPLS on
        out = b.iarpls(y, lam=lam)  # steep fluorescent Raman decays
    elif method == "pspline":       # penalized-spline Whittaker hybrid
        out = b.pspline_arpls(y, lam=lam)
    else:                           # 'snip' — fast statistics of
        hw = int(np.clip(round(3.2 * lam ** 0.25), 10, 60))
        out = b.snip(y, decreasing=True, max_half_window=hw)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def detrend_linear(y: np.ndarray) -> np.ndarray:
    """Subtract a least-squares linear trend (broad slope/curvature of
    fluorescence tails that ALS variants can leave behind)."""
    x = np.arange(len(y), dtype=float)
    coef = np.polyfit(x, y, 1)
    return y - np.polyval(coef, x)


def phe1003_position(y: np.ndarray, wn: np.ndarray,
                     center: float = 1003.0, half: float = 15.0
                     ) -> float:
    """
    Sub-sample peak position of the phenylalanine ~1003 cm-1 ring-breathing
    band: parabola vertex through the window's tallest local maximum.
    Stable anchor for wavenumber-calibration drift checks (the
    literature's Si-520.7 substitute for tissue work).  The vertex fit
    (3 points around the argmax, analytic) replaced the intensity
    centroid, which was biased toward the window centre by up to most of
    a pixel and not idempotent (2026-09-12 audit).
    """
    wn = np.asarray(wn, dtype=float)
    m = (wn >= center - half) & (wn <= center + half)
    if m.sum() < 5:
        return center
    w, yy = wn[m], np.asarray(y, dtype=float)[m]
    i = int(np.argmax(yy))
    lo, hi = max(0, i - 2), min(len(yy), i + 3)
    xs, vs = w[lo:hi], yy[lo:hi]
    if len(xs) >= 3 and vs.max() > vs.min():
        # parabola vertex through the 3 points straddling the apex:
        # x* = x1 + h/2 * (y0 - y2)/(y0 - 2 y1 + y2), h = grid step
        try:
            j = min(max(i, lo + 1), hi - 2)
            x3, y3 = w[j - 1:j + 2], yy[j - 1:j + 2]
            den = y3[0] - 2 * y3[1] + y3[2]
            if den != 0:
                h = 0.5 * ((x3[1] - x3[0]) + (x3[2] - x3[1]))
                return float(x3[1] + h * (y3[0] - y3[2]) / den)
        except Exception:
            pass
        try:
            c = np.polyfit(xs - xs[lo], vs, 2)
            if c[0] != 0:
                return float(xs[lo] - c[1] / (2 * c[0]))
        except Exception:
            pass
    return float(w[i])


# Physical anchor for wavenumber calibration: the phenylalanine
# ring-breathing band sits at 1003.0 cm-1 (literature constant).  A FIXED
# anchor keeps the drift per-spectrum — no cohort statistics — so
# calibration is identical inside CV folds and at predict time.  (The
# previous cohort-median anchor leaked held-out rows into CV scoring and
# degenerated to a no-op when a single spectrum was preprocessed at
# prediction; fixed 2026-09-05.)
PHE1003_ANCHOR = 1003.0


def _phe1003_present(y: np.ndarray, wn: np.ndarray,
                     center: float = PHE1003_ANCHOR, half: float = 15.0
                     ) -> bool:
    """Prominence test: does the Phe-1003 window contain an actual peak?
    Noise is estimated from the flanking bands (±[half, half+80] cm-1),
    NOT from inside the window — a real peak's own flanks inflate the
    first-difference noise and mask the peak (2026-09-12 audit,
    regression-found).  Without this test, argmax of pure noise returns
    a pseudo-position and calibrate_wn shifts peak-free spectra by an
    arbitrary amount."""
    wn = np.asarray(wn, dtype=float)
    m = (wn >= center - half) & (wn <= center + half)
    lo_band = (wn >= center - half - 80.0) & (wn < center - half)
    hi_band = (wn > center + half) & (wn <= center + half + 80.0)
    yy = np.asarray(y, dtype=float)
    if m.sum() < 5:
        return False
    sides = np.concatenate([yy[lo_band], yy[hi_band]]) \
        if (lo_band.sum() + hi_band.sum()) >= 10 else yy[m]
    d = np.abs(np.diff(sides))
    noise = 1.4826 * np.median(d) if len(d) else 0.0
    base = np.median(sides)
    return bool(yy[m].max() - base > 5.0 * max(noise, 1e-12))


def estimate_wn_drift(X: np.ndarray, wn: np.ndarray) -> np.ndarray:
    """Per-spectrum drift (cm-1) of the Phe-1003 band vs the fixed
    physical anchor (leakage-free; identical at predict time).  Spectra
    WITHOUT a detectable Phe-1003 peak get drift 0 — no alignment is
    better than alignment to noise."""
    wn = np.asarray(wn, dtype=float)
    pos = np.array([phe1003_position(row, wn)
                    if _phe1003_present(row, wn) else PHE1003_ANCHOR
                    for row in X])
    return pos - PHE1003_ANCHOR


def calibrate_wn(X: np.ndarray, wn: np.ndarray) -> np.ndarray:
    """
    Resample every spectrum so its Phe-1003 band sits exactly on the
    fixed anchor position (icoshift-lite single-anchor alignment).
    Mostafapour 2023: wavenumber alignment mattered more than ANY
    intensity normalization.
    """
    X = np.asarray(X, dtype=float)
    wn = np.asarray(wn, dtype=float)
    drift = estimate_wn_drift(X, wn)
    out = np.empty_like(X)
    for i, row in enumerate(X):
        if abs(drift[i]) <= np.median(np.diff(wn)):   # sub-grid drift:
            out[i] = row                              # nothing to fix
        else:
            out[i] = np.interp(wn, wn - drift[i], row)
    return out


def align_to_grid(wavenumbers: np.ndarray, intensities: np.ndarray,
                  grid: np.ndarray,
                  params: PreprocessParams) -> np.ndarray:
    """Interpolate one raw spectrum onto `grid` and apply the training-time
    wavenumber calibration when the params request it.

    Deploy-time twin of the calibrate→crop order in `preprocess_matrix`:
    `predict_with_bundle` and `paired.reference_vector` must go through
    this so a bundle trained with wn_calibrate=True receives identically
    aligned features at prediction time.  (2026-09-06: the predict path
    used to skip calibration entirely, feeding such models shifted,
    out-of-distribution features — every prediction saturated to one
    class.)"""
    p = params.validate()
    order = np.argsort(wavenumbers)
    wn = np.asarray(wavenumbers)[order]
    it = np.asarray(intensities)[order]
    grid = np.asarray(grid, dtype=float)
    # coverage guard (mirrors predict_with_bundle): np.interp constant-
    # extrapolates edge values, silently padding a too-narrow file with
    # garbage instead of failing (2026-09-12 audit)
    if wn[0] > grid.min() + 1e-9 or wn[-1] < grid.max() - 1e-9:
        raise ValueError(
            f"spectrum axis [{wn[0]:.1f}, {wn[-1]:.1f}] does not span the "
            f"training grid [{grid.min():.1f}, {grid.max():.1f}] — the "
            "reference/model needs the full range")
    y = np.interp(grid, wn, it)
    if p.wn_calibrate:
        y = calibrate_wn(y.reshape(1, -1), grid)[0]
    return y


_STAGE_CACHE: dict = {}          # (spectrum-hash, stage-key) -> array
_STAGE_CACHE_CAP = 6000          # ~6000 cached spectra-stages is plenty


def _cached_stage(spec_hash: int, stage_key: tuple, fn):
    """A3 (2026-09-05): optimize sweeps / honest checks re-preprocess the
    same spectra dozens of times, differing only in baseline (λ/p/method)
    or normalization.  The despike→detrend→wavelet→SG PREFIX and the
    BASELINE itself are cached per (spectrum, stage params) so a sweep
    of N configs costs ~2-3 distinct prefixes/baselines, not N full
    pipelines."""
    if len(_STAGE_CACHE) > _STAGE_CACHE_CAP:
        _STAGE_CACHE.clear()
    key = (spec_hash, stage_key)
    val = _STAGE_CACHE.get(key)
    if val is None:
        val = fn()
        _STAGE_CACHE[key] = val
    return val


def clear_stage_cache():
    """Call when the underlying data changes (new folder / new matrix)."""
    _STAGE_CACHE.clear()


def preprocess_spectrum(y: np.ndarray, params: PreprocessParams,
                        wn: np.ndarray | None = None) -> np.ndarray:
    """Apply the full pipeline to one spectrum (on the common grid).
    `wn` (the spectrum's own cropped axis) is used by area normalization
    and may be omitted when the spacing is uniform."""
    p = params.validate()
    raw = np.nan_to_num(np.asarray(y, dtype=float))   # guard odd inputs
    h = hash(raw.tobytes())                # cheap: 16 KB per spectrum

    def _prefix():
        out = raw
        if p.despike:
            out = despike(out, p.despike_z, p.despike_window)
        if p.detrend:
            out = detrend_linear(out)
        if p.wavelet:
            out = wavelet_denoise(out, p.wavelet_name, p.wavelet_level,
                                  p.wavelet_threshold, p.wavelet_mode,
                                  p.wavelet_cycle)
        return savgol_smooth(out, p.sg_window, p.sg_poly, p.sg_deriv)

    prefix_key = (p.despike, p.despike_z, p.despike_window, p.detrend,
                  p.wavelet, p.wavelet_name, p.wavelet_level,
                  p.wavelet_threshold, p.wavelet_mode, p.wavelet_cycle,
                  p.sg_window, p.sg_poly, p.sg_deriv)
    out = _cached_stage(h, prefix_key, _prefix)
    # baseline removal (skipped for derivative spectra — the derivative
    # already suppresses smooth baselines)
    if p.sg_deriv == 0:

        def _baseline():
            if p.baseline_method == "arpls":
                return arpls_baseline(out, p.als_lambda, p.als_niter)
            if p.baseline_method in ("iarpls", "pspline", "snip"):
                return _pybase(out, p.baseline_method, p.als_lambda)
            return als_baseline(out, p.als_lambda, p.als_p, p.als_niter)

        out = out - _cached_stage(
            h, prefix_key + ("bl", p.baseline_method, p.als_lambda,
                             p.als_p, p.als_niter), _baseline)
    out = normalize(out, p.norm, wn)
    return np.nan_to_num(out)                         # never emit NaN


def preprocess_matrix(X: np.ndarray, params: PreprocessParams,
                      wn: np.ndarray | None = None) -> np.ndarray:
    """
    Apply the pipeline to a matrix of spectra.  When the wavenumber axis
    `wn` is given, the crop range from `params` is applied FIRST and the
    returned matrix has only the kept columns.  `wn_calibrate` aligns all
    spectra on their Phe-1003 band BEFORE cropping (cohort-level step).
    """
    p = params.validate()
    X = np.asarray(X, dtype=float)
    wnc = None
    if wn is not None:
        wn = np.asarray(wn, dtype=float)
        if p.wn_calibrate:
            X = calibrate_wn(X, wn)
        m = crop_mask(wn, p)
        wnc = wn[m]                    # per-spectrum axis after crop (for
        X = X[:, m]                    # area normalization)
    return np.vstack([preprocess_spectrum(row, p, wnc) for row in X])
