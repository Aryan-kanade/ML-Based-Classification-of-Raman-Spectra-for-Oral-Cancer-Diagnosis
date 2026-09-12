# Spike / Quality Report — Clinical Raman Dataset

**Date:** 2026-09-12
**Data source:** `D:\BARC\Data` (clinical layout: `Normal/<patient>/`, `Tumor/<patient>/`)
**Detector app:** `raman_app` (`clinical_data.py`, `preprocessing.py`)

---

## 1. Dataset summary

| Stage | Count |
|---|---|
| Files scanned | 339 |
| Reference spectra excluded (white/black/dark in filename) | 4 |
| Duplicate copies dropped (byte-identical within a class) | 18 |
| Cross-class identical spectra dropped | 0 |
| **Kept (rows in the Data-page table)** | **317** |
| **Flagged as spiked (score ≥ 300)** | **18** |
| Unflagged (✓ ok) | 299 |

Score distribution across kept spectra: **median ≈ 60 · p95 ≈ 318 · max = 1716**.

---

## 2. How a file gets flagged

At load time every spectrum gets a **spike score**
(`preprocessing.py:146`):

```
spike score = max |I[i+1] − I[i]|  ÷  median |I[i+1] − I[i]|
              (biggest single-point jump) ÷ (typical jump)
```

- **< 300 → ✓ ok** — smooth spectrum; the biggest jump is ordinary noise
- **≥ 300 → ⚠ spiked** — one point stands hundreds of times above the
  normal step size: a cosmic ray / hot pixel / interpolation artifact

A real Raman band is dozens of points wide; these artifacts rise and
fall within a single point (~1.9 cm⁻¹ grid step). The threshold sits in
the empty gap between ordinary noise (~60) and true outliers (300+);
see the calibration comment at `clinical_data.py:42`.

---

## 3. All 18 spiked files, with their spikes

Spike position = wavenumber of the single-point outlier; "L/R" = the
intensities immediately left/right of it (the spike towers over both).

### Normal class (8 files)

| # | File (under `Data\Normal\`) | Score | Spike (cm⁻¹) | Intensity | L / R neighbors |
|---|---|---:|---:|---:|---|
| 1 | `Patient_32\x32NH0_csv_interpolated.csv` | **1716** | 1980.8 | 4691 | 2155 / −2715 |
| 2 | `Patient_50\x50NH0_2_csv_interpolated.csv` | 1172 | 1592.2 | 523 | 77 / 143 |
| 3 | `TDOC 06\26112024TDOC006NH1_interpolated.csv` | 435 | 1498.2 | 3487 | 1082 / 1687 |
| 4 | `TDOC 10\29112024TDOOC0010NH0_interpolated.csv` | 944 | 1498.2 | 4009 | 1015 / 1823 |
| 5 | `TDOC 11\30112024TDOC11NH1_interpolated.csv` | 487 | 1498.2 | 4239 | 1166 / 1845 |
| 6 | `TDOC091 Spectra pro\18052026TDOC091NH0.csv` | 310* | 3078.5 | 507 | 145 / 0 |
| 7 | `TDOC091 Spectra pro\18052026TDOC091NH02.csv` | 348* | 3078.5 | 824 | 255 / −19 |
| 8 | `TDOC092 Spectra pro\25052026TDOC092NH0.csv` | 450* | 1592.2 | 543 | 162 / 169 |

### Tumor class (10 files)

| # | File (under `Data\Tumor\`) | Score | Spike (cm⁻¹) | Intensity | L / R neighbors |
|---|---|---:|---:|---:|---|
| 9 | `Patient_19\08012025TDOC19TH0_interpolated.csv` | 567 | 1498.2 | 3708 | 1024 / 1747 |
| 10 | `Patient_32\x32TH0_1_csv_interpolated.csv` | 373 | 1980.8 | 4744 | 2077 / −2680 |
| 11 | `TDOC 11\30112024TDOC11TH1_interpolated.csv` | 1395 | 1498.2 | 4865 | 1547 / 2424 |
| 12 | `TDOC017\02012025TDOC17TH0 (2)_interpolated.csv` | 740 | 1498.2 | 3286 | 946 / 1573 |
| 13 | `TDOC071 Spectra pro\02022026TDOC071TH02.csv` | 310* | 2532.6 | 1053 | 243 / 111 |
| 14 | `TDOC091 Spectra pro\18052026TDOC091TH01.csv` | 348* | 3078.5 | 473 | 193 / −27 |
| 15 | `TDOC091 Spectra pro\18052026TDOC091TH02.csv` | 415* | 3078.5 | 559 | 218 / 33 |
| 16 | `TDOC092 Spectra pro\25052026TDOC092TH0.csv` | 450* | 3015.0 | 605 | 339 / 9 |
| 17 | `TDOC092 Spectra pro\25052026TDOC092TH01.csv` | 358* | 2551.3 | 421 | 60 / 108 |
| 18 | `TDOC092 Spectra pro\25052026TDOC092TH02.csv` | 307* | 1592.2 | 568 | 165 / 147 |

\* See the Rayleigh-edge caveat in §5.

Additional secondary spikes (weaker, not the score driver): file 1 also
has single-point glitches at 1592.2 and 1679.1; files 3/4/5/9/11 have
milder spikes at ~926.5 and ~1456.3 riding on real broad bands;
file 16 also has 3796.5 and 3191.9.

---

## 4. The key finding: spikes are NOT random

The spike positions **recur at identical wavenumbers** across
independent files, patients, and measurement dates:

| Spike position | Files affected |
|---|---|
| **1498.2 cm⁻¹** | 6 (TDOC 06 N, TDOC 10 N, TDOC 11 N+T, Patient_19 T, TDOC017 T) |
| **3078.5 cm⁻¹** | 4 (TDOC091 N×2 + T×2) |
| **1592.2 cm⁻¹** | 4 (Patient_50 N, TDOC092 N+T×2… plus Patient_32's secondary) |
| **1980.8 cm⁻¹** | 2 (Patient_32 N + T — both with a −2700 dip beside the spike: interpolation scars) |
| 2532.6 / 2551.3 / 3015.0 | 1 each (TDOC071/092 "pro" files) |

Random cosmic rays land anywhere. Same-position artifacts across
different patients and days indicate **fixed detector hot pixels or
interpolation-stage artifacts** in the instrument pipeline.

Only **5 patients** account for most flags: TDOC091 (4 files),
TDOC092 (4), TDOC 11 (2), Patient_32 (2), plus TDOC 06/10, Patient_50,
Patient_19, TDOC017, TDOC071 with 1 each.

---

## 5. Caveats

1. **Rayleigh edge (files 6–8, 13–18, marked \*):** in the
   "Spectra pro" files the formula's largest jump technically lands on
   the steep Rayleigh/edge-filter roll-off at ~20–37 cm⁻¹ — a broad,
   legitimate feature spanning many points. That edge is what pushes
   the score over 300; the single-point artifacts listed in the tables
   (mid-spectrum, on a near-zero baseline) are the visible damage.
   Either way the flag is defensible — the files do contain
   single-point artifacts — but the score number alone overstates the
   cosmic-ray severity for these six.
2. **Patient_32** has strong **negative dips** (−2700) immediately
   beside its spikes — classic interpolation scarring around a hot
   pixel, not a physical signal.
3. Several 1498.2 cm⁻¹ spikes ride on top of a broad real hump
   (~600–1800 baseline) — the spike itself is still a clean
   single-point excursion.

---

## 6. What happens to these files in the app

- **Data page → Quality column**: each of the 18 rows shows
  ⚠ spiked (red); the other 299 show ✓ ok (green). The flag is
  recomputed on every load and lives in memory only.
- **Training**: the "Exclude flagged spectra" checkbox (ON by
  default, and only active when despiking is OFF) drops all 18 from
  training — the log line reports the count.
- **Despike option** (Whitaker & Hayes 2018, modified z > 7 → replace
  with local mean) attempts repair instead of exclusion, but it is OFF
  by default: measured on this dataset, heavily spiked spectra are not
  reliably recoverable and **exclusion scored better**.
- **Paired mode**: spiked spectra are excluded the same way
  (`paired_features(exclude=…)`); patients whose remaining spectra
  lose a class entirely drop out of pairing.

---

## 7. Files dropped before the table (for completeness)

Never appear as rows; only in the load report / `data_report.txt`:

- **References (4):** `TDOC034 …NH0white`, `…TH0_pro_black`,
  `…TH0_raw_black`, `…TH0white` (all `_interpolated.csv`)
- **Duplicates (18):** the `TDOC015–019` copies that exist both under
  `TDOC0XX` and `Patient_XX` folders (first copy kept, byte-identical
  SHA-1)

---

## 8. Recommendation

Since the artifact positions are **fixed** (§4), a small **hot-pixel
position mask** — interpolate across the handful of known-bad
wavenumbers (1498.2, 1592.2, 1980.8, 2532.6, 2551.3, 3015.0, 3078.5)
during preprocessing — would *clean* most of these spectra instead of
*dropping* them. That matters for paired mode: Patient_32, TDOC 11 and
TDOC091/092 currently lose flagged spectra in BOTH classes, shrinking
the paired dataset. Worth implementing if sample size becomes the
bottleneck; until then the default exclusion remains the safe choice.

---

*Generated from `D:\BARC\Data` (339 files, last full scan in
`.zcode/audit_cli_run/data_report.txt`) and per-file spike analysis;
detection logic: `raman_app/preprocessing.py:146` (scorer),
`raman_app/clinical_data.py:45` (threshold).*
