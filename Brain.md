# Brain.md — the project's memory

> **Read this before touching any code.** It contains everything a new
> session needs: architecture, data flow, every parameter default,
> hyperparameter grids, GUI inventory, real results, gotchas. Do NOT
> re-explore the whole tree.
>
> **Maintenance rule (mandatory):** after any non-trivial change, update
> the matching section below. Keep the "Last updated" stamp current.
> Keep it dense — tables and one-liners, no prose padding.

Last updated: 2026-09-07 (WORKSPACE CLEANUP for a portable copy — §32;
previously 2026-09-06: WHOLE-PROJECT DEEP AUDIT + REMEDIATION — ~190
findings, fixed in batches B0–B4, §20 lists the number-changing
semantics; previously 2026-09-05: WHOLE-PROJECT ERROR HANDLING, see gotcha #23;
2026-09-04: COMPLETE UI/UX modernization: Preprocess page
premium layout — live ①–⑥ step-chip strip, six step panels in a 2-col
× 3-row grid of two-column FORMS (full label words, 300px floors, smart
enabling), 6th panel = CHANGES & RESET (live modified-vs-defaults
markers + ↺ Reset), Preview/Optimize buttons full-width BELOW the grid,
full-width preview with empty state, NO splitter; design language
v2 (10.5pt CardHeader, 15pt PageTitle, StepChip/StepArrow QSS, radius
14/12, hover states, stronger shadows); nav shell (MDL2 sidebar icons
with fallback, checked accent bar, footer step dots, Ctrl+1–6);
Start page two-column with step chips; Data quality chips (✓/⚠
colored); Train controls numbered 1·Mode 2·Models 3·CV; Predict
pills/status moved into the Predictions card + margin-mode explainer
card; Result anchor-pill jump row (10 sections, `_scroll_to_card`);
diag panels show "computing…" placeholders at start (`_diag_running`);
window min 1100×700 + maximized remembered; default crop = tuned
500/2000 (PARAMS_VERSION 3), coif3/coif5 wavelets removed with sym8
fallback; project-wide UI/graph
consistency pass: one
graph design system in plotting.py — named semantic colors, normalized
fonts, shared `plot_count_bars`/`plot_sign_bars` helpers, styled seed
boxplot, cm⁻¹ units on band annotations, apply_style in reproduce_study,
HiDPI attributes; all inline graphs migrated off raw hex; fixed the
orphaned Locked-FINAL Result card (never showed), Predict got a
splitter, 3SSE Run button styled via primary property, banner/seq
labels via QSS (`BannerTitle`/`SeqCounter`/`SeqTop5`), harmonized table
caps, Data canvas floor + 26px rows; Preprocess page redesigned: 2-column params,
draggable splitter, preview grows per class + updates LIVE after every
tweak; fixed NameError `_cal`/`threshold` in render_result_page;
Predict page overhauled: two-column, big
resizable chart, triage+bar table shared with Result, live progress
pills; ALL diagnostics auto-run serially after every
training — no clicking, each in its own bordered card; Preprocess
preview shows one row PER CLASS —
Normal + Tumor side by side; each Train diagnostic graph in its own
bordered sub-card, square confusion-matrix canvas; loader keeps the
FULL 15.5–3862 axis — edge trim removed, every data point used;
2026-09-03: inline per-result panels, deep_test 18/18, default crop
15/3862, demo removed, portable data-root discovery)

---

## 1. What this project is

Desktop app + headless CLI that classifies **SERS Raman spectra (785 nm)
of oral-cancer tissue** into `Normal` / `Tumor`. Research triage support
for clinicians — **not** a diagnostic device; histopathology stays the
reference standard.

- **Real dataset**: `D:\BARC\Data\<Class>\<Patient>\<file>.csv` —
  341 files scanned → **317 spectra / 72 subjects kept** (68 with both
  classes = paired), after hygiene: 4 white references, 18 duplicate
  copies, 2 cross-class identical files dropped, **155 padded files
  edge-trimmed in the loader** (gotcha #18); 18 spectra spike-flagged
  (score ≥ 300). **Never commit patient data** (gitignored).
- Single-centre, no external validation. Confounders (site,
  keratinisation, tobacco) unrecorded.
- `D:\BARC\Documatation\` was deleted 2026-09-02 (user confirmed copies
  elsewhere; had held 8 reference PDFs incl. oral-cancer thesis,
  portable Raman design rules, biomedicines 11-01984, JCM 08-013131,
  Sahu 2016).

## 2. Environment & how to run

Windows 10/11, Git Bash shell, **Python 3.14.6** (system install, no
venv), PyQt5 binding (auto-detect PyQt5→PyQt6→PySide6 by import success
order, `qt_compat.py`; ~15 enum constants normalized Qt5/Qt6). Deps
pinned in `raman_app/requirements.txt`: numpy 2.5.2, scipy 1.18.1,
scikit-learn 1.9.0, pandas 3.0.5, matplotlib 3.11.1, PyWavelets 1.9.0,
joblib 1.5.3, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10,
pybaselines 1.2.1, shap 0.52.0, torch 2.13.0, PyQt5 5.15.11. Optional
deps (pybaselines/shap/torch/xgboost/lightgbm/catboost) all have
graceful fallbacks via module HAS_* flags. **GPU since 2026-09-04**:
torch is the **+cu130** build (RTX 3050 6 GB, driver 581.86/CUDA 13.0);
`modeling.torch_device()` = cuda unless `RAMAN_DEVICE=cpu`. 1D-CNN
fits on the GPU but stores fitted weights on CPU (portable bundles;
predict runs on CPU — batches are tiny). ViT scripts auto-pick CUDA.
3SSE loky workers set RAMAN_DEVICE=cpu (`_pin_threads`) — N child
processes would mean N CUDA contexts (~300 MB VRAM each). Boosters
(XGBoost/CatBoost/LightGBM) are CPU **by default** and need explicit
`RAMAN_DEVICE=gpu` (§16c gotcha: auto-GPU boosters OOM'd loky
children; they lose at n≈300 anyway). **tabpfn==6.0.6** (TabPFN v2.5
foundation model entry; HF weights auto-download once, no account —
do NOT upgrade to 8.x, it demands a browser login). GPU CNN folds are
not bit-identical to
CPU (CUDA atomics) — seeds still fix splits, weights may differ
slightly.

```bash
cd raman_app
python main.py                                  # GUI (or run_app.bat)
python reproduce_study.py --data D:/BARC/Data --out study_run_X        # full real study
python reproduce_study.py --data D:/BARC/Data --mode paired --out study_run_paired
python test_all.py                              # 40 unit tests
QT_QPA_PLATFORM=offscreen python gui_test.py    # GUI walk (writes result_report.txt in-tree!)
QT_QPA_PLATFORM=offscreen python deep_test.py   # 18 adversarial scenarios
```

Tests hardcode `C:\Windows\Fonts` — Linux CI would need patching (no CI
exists). `matplotlib.use("Agg")` for all headless figure writing.

## 3. File map (raman_app/, 12.2k lines)

| File | ~Lines | Role |
|---|---|---|
| `gui.py` | 6261 | 6-page MainWindow, 5 QThread workers, all slots. See §14. |
| `modeling.py` | 1346 | Model registry + nested grouped CV + bundles + importance + bootstrap/McNemar. §7-8, 12. |
| `test_all.py` | 1054 | 47 unit tests incl. regressions for the 2026-08-30 fixes, new models, data-root discovery. |
| `deep_test.py` | 716 | 18 adversarial GUI/CLI scenarios; `wait_analysis()` helper. |
| `vit_train.py` / `vit_model.py` / `vit_test.py` | 551/207/144 | Secondary ViT track (torch). §15. |
| `plotting.py` | 439 | rcParams theme, `MatplotlibCanvas` (lazy Qt-binding pin), plot helpers. |
| `ui_helpers.py` | 387 | QSS stylesheet, `pill()`, tooltips, HOW_TO/METRIC_HELP, settings I/O. |
| `study_stats.py` | 375 | Friedman+Nemenyi, LOPO, seed/noise checks, BH-FDR band stats. §10. |
| `clinical_data.py` | 324 | `load_clinical_dataset()` → `ClinicalData` + hygiene + report. §5. |
| `reproduce_study.py` | 285 | One-shot headless study (works on real data since 2026-08-30). |
| `clinical.py` | 259 | TRIPOD layer: Platt, DeLong, operating points, triage, PPV/NPV, DCA. §9. |
| `preprocessing.py` | 254 | `PreprocessParams` + spectrum pipeline. §6. |
| `biochemistry.py` | 250 | BANDS/RATIOS tables, NMF, keratin flags, plausibility. §11. |
| `optimize.py` | 228 | Preprocessing auto-tune. §15. |
| `dataset.py` | 148 | Flat loader, `common_grid`, `to_matrix`. §5. |
| `paired.py` | 160 | Within-patient deviation features. §4. |
| `gui_test.py` / `main.py` / `qt_compat.py` | small | Harnesses / entry points. |

Root: `AGENTS.md` (pointer here), `.gitignore` —
excludes `Data/`, `Documatation/`, `_external/`,
`renders/`, `study_run*/`, `*.joblib`, `study_manifest.json`,
`result_report.*`, `settings.json`, `session.log`,
`vit_outputs/`, `__pycache__/`, `*.zip`.

## 4. Data flow (memorize this)

```
raw csv → load (clinical layout | flat)        clinical_data.py / dataset.py
        → hygiene (refs, dedupe, cross-class, spike flags)      §5
        → common_grid (range intersection, median #points, ≤2000) + to_matrix
        → EITHER standard:  preprocess_matrix(X_raw)            preprocessing.py §6
          OR     paired:    paired_features(X_raw, y, g, FULL grid)  paired.py
                             (preprocesses internally — takes RAW X!)
        → evaluate_models (nested grouped CV, 15 rows)          modeling.py §7-8
        → winner (pooled mean macro-F1) → clinical layer        clinical.py §9
        → bundle (.joblib) → predict_with_bundle                modeling.py §12
        → reports (txt/html), freeze_study manifest             gui.py §14
```

**Paired mode** (scientifically strongest): per patient with both
classes, reference = mean of their preprocessed Normal spectra; every
kept spectrum becomes `spectrum − own_normal_reference` (optionally PQN
against the reference first). Removes ~61% between-patient variance.
Unpaired patients excluded (counted in `n_unpaired_excluded`).
Prediction-time reference: `paired.reference_vector(bundle, paths)` or
GUI `PredictWorker._auto_reference` (`<root>/Normal/<patient>/`).

## 5. Loaders & data hygiene

**dataset.py**
- `Spectrum` dataclass: `name` (filename sans ext), `path`,
  `wavenumbers` (ascending), `intensities`, `label=""`.
- Class token: `re r"(?:^|[_.\-])(C\d+)(?=[_.\-]|$)"` → e.g.
  `S07_tissue_C8.txt` → `"C8"`; absent → `""`.
- `load_spectrum`: comma→space, whitespace split, skips bad lines,
  **ValueError if <10 points**, sorts ascending.
- `load_folder`: non-recursive, `.txt/.dat/.csv` sorted.
- `common_grid(spectra, max_points=2000)`: lo=max of mins, hi=min of
  maxs (intersection; raises if empty), n = median length clipped
  [50, 2000] → `np.linspace(lo, hi, n)`.
- `to_matrix`: `np.interp(grid, s.wavenumbers, s.intensities)` per row.

**clinical_data.py**
- Layout `<root>/<class>/<patient>/*.csv`; label from top-level folder
  only (filenames unreliable).
- **Portable root discovery** (works on any device — no hand-edits):
  `find_data_root()` checks RAMAN_DATA_DIR env var →
  `~/.raman_app_data_dir` pointer (auto-written by the GUI the first
  time a clinical folder is loaded; home doesn't travel with a copied
  project) → `raman_app/../Data` → `~/Desktop/data/Data` → `~/Data` →
  `raman_app/Data`; every candidate validated with
  `is_clinical_layout`. Used by GUI startup fallback and the `--data`
  defaults of sequential.py / reproduce_study.py.
- `CLASS_SYNONYMS` (lowercase): normal/control/healthy/benign→`Normal`;
  tumor/tumour/cancer/malignant→`Tumor`; else passthrough `strip()`ed.
- `REFERENCE_MARKERS` (lowercase substrings of filename): white, black,
  dark, bkg, background, ref, reference, blank, calib.
- `subject_key(folder)`: first number in folder name → `S{int}` (so
  `Patient_15`, `TDOC123`, `TDOC123 Spectra pro` → `S123`); no digit →
  `folder.strip().upper()`.
- Duplicates: SHA-1 over float64 raw bytes of (wn, it) — immune to
  line endings. Same-class dup → keep first; hash in >1 class → **all
  copies dropped** (`cross_class_dropped`, pure leakage).
- **Site-token check (2026-09-06 audit)**: `TH\d` token inside
  `Normal/` (or `NH\d` inside `Tumor/`) contradicts the folder label →
  dropped as `site_token_dropped` (unambiguous files only). Real-data
  effect: ONE patient's `TH01` spectrum — a UNIQUE mislabeled tumor
  spectrum in Normal (no byte-twin, so SHA-1 could never catch it) —
  dropped; ANOTHER patient's `TH0` Normal copy dropped by token and its
  genuine Tumor copy now KEPT (the cross-class rule used to destroy
  BOTH). n stays 317 but healthier (Normal 142 / Tumor 175).
- `SPIKE_FLAG_THRESHOLD = 300.0` (median spike score of this dataset
  ~60; clean ≈5–30). `spike_score(y) = max|diff| / median|diff|`.
- `is_clinical_layout`: ≥2 subfolders containing spectra one OR two
  levels down.
- Shared grid only if all spectra match first spectrum exactly
  (`allclose` atol 1e-6); else `grid=None` → consumer falls back to
  `common_grid`.
- Report keys: `root, class_map, references_excluded,
  duplicates_dropped, cross_class_dropped, site_token_dropped,
  load_errors, grid_identical, files_scanned, spike_scores{median,p95,max}, flagged_spectra,
  class_counts, patients_per_class, patients_in_both_classes,
  n_spectra, n_subjects`.
- `patient_split(groups, labels, seed=42)`: StratifiedGroupKFold
  n_splits=7, folds sorted by size desc → fold0=test, fold1=val,
  rest=train (≈70/15/15); needs ≥7 subjects else ValueError.

## 6. Preprocessing (preprocessing.py)

**`PreprocessParams` fields + defaults** (validated copy, never mutated):

| Field | Default | Meaning / clamp in `validate()` |
|---|---|---|
| `crop_min` / `crop_max` | 500 / 2000 (tuned working range; since 2026-09-04) | 0 disables bound; min≥max ⇒ both 0 (no crop) |
| `despike` | **False** (exclusion beat despiking here) | Whitaker–Hayes |
| `despike_z` | 7.0 | clip [3, 20] |
| `despike_window` | 5 | ≥3, forced odd |
| `wavelet` | True | False if PyWavelets missing |
| `wavelet_name` | "sym8" | — |
| `wavelet_level` | 4 | capped at `dwt_max_level` |
| `sg_window` | 11 | ≥5, forced odd, shrunk if ≥ len(y) |
| `sg_poly` | 3 | clip [1, w−2], raised above deriv |
| `sg_deriv` | 0 (0/1/2) | deriv≠0 **skips baseline step** |
| `baseline_method` | "als" ("als"\|"arpls") | arpls falls back to als w/o pybaselines |
| `als_lambda` | 1e5 | GUI enters log10 exponent |
| `als_p` | 0.01 | asymmetry |
| `als_niter` | 10 | arpls gets ×5 max_iter |
| `norm` | "vector" ("vector"\|"snv"\|"none") | — |

**`preprocess_spectrum` order** (per spectrum; no cross-sample fit ⇒
leakage-free): `nan_to_num` → ① despike (modified Z `0.6745·(d−med)/MAD`
of first difference; spikes `|z|>despike_z` replaced by neighborhood
mean) → ② wavelet denoise (`sym8` lvl 4, universal threshold
`σ√(2ln n)`, σ = median|finest detail|/0.6745, soft threshold, approx
untouched) → ③ Savitzky-Golay (`savgol_filter(w, p, deriv)`) → ④
baseline: **skipped when `sg_deriv≠0`**; arPLS via
`pybaselines.Baseline().arpls(lam, max_iter=50)` else ALS (Eilers:
`sparse` 2nd-difference penalty, weights `p·(y>z)+(1−p)·(y≤z)`, 10 iters)
→ ⑤ normalize: `vector` = y/‖y‖; `snv` = (y−mean)/std; `none`.
`pqn_normalize(y, ref)`: quotient spectrum y/ref (|ref|>1e-12), scale =
median of quotients, y/scale (Dieterle 2006; ref must be leakage-free).
`crop_mask(wn, p)`: boolean KEEP mask; degenerate ⇒ all True.
`preprocess_matrix(X, p, wn)`: crops columns FIRST, then per-row
pipeline.

## 7. Cross-validation methodology (modeling.py)

- **`evaluate_models(X, y, model_names=None, k_folds=5, seed=42,
  progress_cb=None, groups=None, repeats=1, wavenumbers=None)` →
  (results, winner)**. Requires ≥2 classes, min class ≥2; k =
  clip(k_folds, 2, min(10, min_class, n_groups)).
- **Outer**: repeats × StratifiedGroupKFold(k, shuffle, rseed=seed+rep·100)
  (StratifiedKFold without groups); splits pooled in list order
  [rep0 f0..fk, rep1 f0..fk, …].
- **Inner per fold**: `inner_k = min(3, min_class_tr, n_groups_tr)`;
  `_tune_hyperparams` = exhaustive `GridSearchCV(scoring="f1_macro",
  n_jobs=-1)` with grouped 3-fold CV; returns estimator refit on the
  full training fold with best params (NOT gs.best_estimator_).
- **Binary threshold**: candidates = 99 quantiles (1–99 %) of pooled
  inner-OOF scores (`cross_val_predict(..., method="predict_proba")`);
  best by macro-F1, Youden-J tiebreak; kept per fold **only if it beats
  0.5** on inner OOF; final `res.threshold` = median of kept values.
- **OOF probabilities averaged across repeats** (`oof_sum/oof_cnt`,
  fix 2026-08-30); confusion matrices pool raw counts across repeats.
- **Winner tiebreak key**: `(macro_f1, macro_sens, macro_spec)` max.
  Final refit on ALL data with the most frequently chosen hyperparams
  (`param_counter.most_common(1)`).
- `fit_maybe_grouped`: passes `groups=` to `.fit` only if the estimator
  signature accepts it (StackedEnsemble yes, sklearn no) — via
  `inspect.signature`.
- **`CalibratedSVC` calibrates GROUP-AWARE since 2026-09-06** —
  `fit(X, y, groups=None)`; with groups it precomputes
  StratifiedGroupKFold index splits for `CalibratedClassifierCV`
  (same-patient spectra never straddle a calibration fold; the old
  per-spectrum split leaked patients into every downstream
  probability). PCA+SVM CV numbers may shift slightly vs older runs.
- **`evaluate_pipeline` (honest nested check, GUI-only)**: per outer
  fold, preprocessing re-chosen by inner `optimize.optimize_preprocessing`
  over a compact 3-config grid: {crop 400–1800, deriv 0, vector},
  {crop 400–1800, deriv 1, vector}, {no crop, deriv 0, vector}. Returns
  `{mean_f1, std_f1, fold_f1s, fold_choices[{fold, preprocess, model,
  f1}]}`.
- **`learning_curve_by_groups(X, ye, groups, est, k=5, seed=42,
  fractions=(0.25, 0.5, 0.75, 1.0))`**: shuffles unique groups, takes
  `max(2, round(n·frac))` patients, inner grouped CV macro-F1 →
  (n_patients, means, stds).
- Diagnostics wiring: Friedman on per-fold scores averaged across
  repeats; `bootstrap_ci` patient-level when groups given (both fixed
  2026-08-30); exact McNemar.

## 8. Model registry — full hyperparameter grids (24 rows; 2026-09-05:
+1D-CNN ensemble (5 seeds, synth 0.3), +TabPFN foundation model — §16c)

All share `random_state=42` where applicable. "PCA" pipelines =
StandardScaler → PCA(n_components=0.95).

| Model | Estimator | Grid |
|---|---|---|
| PCA + SVM (RBF) | → CalibratedSVC | C∈[1,10,100]; gamma∈["scale",0.01] |
| PCA + LDA | → LinearDiscriminantAnalysis | [{lsqr, shrinkage∈[None,"auto"]},{svd}] |
| PCA + Logistic Regression | balanced, max_iter 5000 | C∈[0.1,1,10] |
| PCA + KNN | KNeighbors | n_neighbors∈[3,5,7]; weights∈[uniform,distance] |
| PCA + Gaussian NB | GaussianNB | var_smoothing∈[1e-9,1e-7] |
| Random Forest | 250 trees, balanced, n_jobs=-1 | max_depth∈[None,12]; min_samples_leaf∈[1,3] |
| Extra Trees | 250 trees, balanced, n_jobs=-1 | max_depth∈[None,12]; min_samples_leaf∈[1,3] |
| Hist Gradient Boosting | PCA→HGB(lr 0.1, balanced) | max_iter∈[100,300] |
| PLS-DA | PLSDAClassifier | n_components∈[2,3,5,8] |
| PCA + MLP | hidden (64,), max_iter 1500 | alpha∈[1e-3,1e-1] |
| Isolation Forest (OvR) | IsolationForestOvR | n_estimators∈[100,300] |
| XGBoost (if installed) | 200 trees, depth 4, hist | max_depth∈[3,5]; lr∈[0.05,0.2] |
| LightGBM (if installed) | 200 trees, depth 4, balanced, verbosity −1 | max_depth∈[3,5]; lr∈[0.05,0.2] |
| CatBoost (if installed) | 200 iters, depth 4, auto_class_weights, verbose 0, no file writes | depth∈[3,5]; lr∈[0.05,0.2] |
| 1D-CNN (if torch) | `CNN1DClassifier` (see below) | **empty** — one seeded fit per fold |
| Peak bands + RF | `needs_wn`: 16 band means (width 30) → RF 250 | max_depth∈[None,8] |
| Ensemble (top-3) | resolved post-hoc | — |

- **`CNN1DClassifier`** (modeling.py, sklearn-compatible torch wrapper,
  BaseEstimator/ClassifierMixin): Conv1d(1→32,k7)+BN+ReLU+MaxPool →
  Conv1d(32→64,k5)+BN+ReLU+AdaptiveAvgPool → Dropout(0.2) → Linear
  (~25k params; trains on `torch_device()` = GPU when present, weights
  stored on CPU). Class-weighted CE, Adam 1e-3, 40 epochs, batch
  32, stratified 15 % val split, early stop patience 6 with best-weight
  restore; `torch.manual_seed(seed)`; X auto-shaped (n,1,L);
  predict_proba = batched softmax in eval/no_grad. Pickles into bundles.
- **PLSDAClassifier**: PLSRegression on one-hot Y; predict_proba =
  softmax of PLS scores (pseudo-probabilities — monotone per class).
- **Ensemble (top-3)**: VotingClassifier(soft) over the 3 best
  error-free models' refit pipelines, same `run_cv`.
- **Stacked (top-3)**: StackedEnsemble — grouped inner-OOF base
  probabilities (n × 3·n_classes blocks) → LogisticRegression(max_iter
  2000) meta-learner; accepts `groups` in fit.
- `BAND_NAMES`: 17 named bands 782–1655 for labeling importance peaks.

## 9. Clinical evaluation layer (clinical.py)

- `TRIAGE`: NEGATIVE / INDETERMINATE / POSITIVE → actions "routine
  follow-up" / "re-sample or biopsy" / "biopsy / refer".
- `PREVALENCE_SCENARIOS`: dental screening 0.05, lesion clinic 0.30,
  biopsy queue 0.60 (drive PPV/NPV table).
- `operating_points(y, p, sens_target=.90, spec_target=.90)` from
  `roc_curve`: rule-out = highest threshold with tpr≥target; rule-in =
  highest threshold with fpr≤1−spec_target; None when unreachable.
- `triage(p, lo, hi, fallback=0.5)`: NaN→INDETERMINATE; single
  threshold fallback if points invalid; else p≥hi→POS, p≤lo→NEG.
- `ppv_npv(sens, spec, prev)`: Bayes; NaN-safe.
- `fit_platt/apply_platt`: LogisticRegression(C=1e6) on logit(p) →
  (a,b); `p' = σ(a·logit(p)+b)`; None if <10 rows/1 class.
- `calibration_bins(y, p, n_bins=8)`: equal-count CONTIGUOUS blocks of
  the sorted p (`np.array_split(argsort(p), n_bins)` — since 2026-09-06;
  the old interleaved `order[b::n]` put every bin across the full
  range, flattening ECE/reliability to the global mean). rows
  (mean_p, observed, n).
- `decision_curve(y, p, t=0.05..0.95 step .05)`: NB = TP/n − FP/n·t/(1−t);
  treat-all = prev − (1−prev)·t/(1−t); returns (thr, nb_model, nb_all).
- `wilson_ci(k, n)`, `delong_auc_ci(y, p)` (tie-corrected midranks,
  returns auc, se, lo, hi).
- `triage_arithmetic(y, tiers, prev)`: per-1000 {biopsies_all,
  biopsies_triage, cancers_caught, cancers_in_indeterminate,
  cancers_missed_negative, reduction_pct, caught_or_flagged_pct}.

## 10. Diagnostics (study_stats.py + modeling helpers)

- `NEMENYI_Q` (α=.05, Demšar 2006): k2..10 = 1.960, 2.349, 2.569,
  2.728, 2.850, 2.936, 2.998, 3.045, 3.081; beyond: 3.081+0.02(k−10).
  `friedman_nemenyi(scores)`: model→per-fold F1 lists; ≥3 models =
  friedmanchisquare, exactly 2 = Wilcoxon fallback; avg ranks (1=best,
  ties averaged); CD = q·√(k(k+1)/6n); returns {ok, chi2, p,
  significant, cd, n_blocks, avg_ranks, best, vs_best}.
- `bh_fdr(pvals)`: BH adjusted (monotone, clipped).
- `lopo_evaluate(X, y, groups, est, {}, classes, seed)`: fixed params,
  per-patient rows `(patient, n_spectra, accuracy, mean_p_pos,
  dominant_true_class)`; returns + {n_patients, f1, cm, auc, accs,
  sens, spec}; skips patients leaving <2 classes.
- `seed_stability(..., seeds=(0..4), k=5)` → list of macro-F1.
- `noise_robustness(..., levels=(0, .01, .02, .05))`: Gaussian noise at
  INFERENCE only: `Xte + N(0,1)·per-row-std·level`, shared rng →
  [(level, f1)].
- `per_patient_rollup(y, groups, oof, classes)`: argmax rows sorted
  worst-first, `hard_flag = acc<0.5`.
- `band_stats_paired(X, y, groups, wn)`: per band, per-patient
  (patient,class)-mean areas → delta tumor−normal → Wilcoxon signed-rank
  (needs ≥5 deltas) → BH-FDR; rows `(center, molecule, assignment,
  direction, median_delta, p_fdr, significant)`; 2 classes only.
- `region_importance(X, y, groups, wn)` (Gini): RF 400 trees,
  uniform_filter1d smooth 21, top-5 bands by mass, min gap 21; bands =
  (center, width, share). `region_importance_shap`: RF 300 + TreeExplainer
  (`check_additivity=False`), signed mean; bands = (center, share, name,
  sign) — name from nearest BAND_NAMES within 25 cm⁻¹; +pushes toward
  sorted-classes[1].

## 11. Biochemistry reference (biochemistry.py)

**BANDS** — (center, halfwidth, molecule, assignment, direction: +1 up
in tumor / −1 down / 0 unspecific):

| cm⁻¹ | ±hw | Molecule | Assignment | Dir |
|---|---|---|---|---|
| 785 | 8 | nucleic acids | DNA/RNA phosphate backbone (O-P-O) | +1 |
| 815 | 8 | collagen | C-C stretch, proline-rich | −1 |
| 854 | 8 | collagen | proline/hydroxyproline ring | −1 |
| 938 | 10 | keratin | C-C backbone (keratin marker) | −1 |
| 1003 | 8 | protein | phenylalanine ring breathing | +1 |
| 1090 | 10 | nucleic acids | PO2− stretch | +1 |
| 1240 | 12 | collagen/protein | amide III | −1 |
| 1335 | 10 | nucleic acids | purine bases (A, G) | +1 |
| 1445 | 10 | lipids/protein | CH2/CH3 deformation | 0 |
| 1554 | 10 | protein | tryptophan ring | +1 |
| 1578 | 8 | nucleic acids | purine bases (A, G) | +1 |
| 1655 | 12 | protein/lipid | amide I (α-helix) / C=C | +1 |

- **RATIOS**: nucleic/protein=785/1003 (proliferation);
  collagen/protein=854/1003 (stroma); lipid/protein=1445/1003 (membrane
  turnover); amide I/III=1655/1240 (secondary structure).
- `band_area`: trapezoid over center±hw (NaN if <2 pts).
- `ratio_table(wn, X, y, groups)` → (class_rows, paired_rows
  (key, n_patients, mean per-patient Δ tumor−normal)).
- `nmf_components(X, wn, k=5, seed)`: clip ≥0, NMF(init="nndsvda",
  max_iter 600); labels auto-named by argmax z-scored band areas →
  (H, labels, W). `component_deltas(W, y, groups)`: paired Δ rows.
- `keratin_index` = area(938±10)/area(1003±8); `keratin_flags(wn, X,
  z_thresh=2.5)`: robust MAD·1.4826 outlier rule → (median, mad,
  n_flagged, n_total).
- `plausibility(bands, tol=25)`: nearest BANDS center; verdicts
  agree/opposite/unknown-sign/unassigned; returns (rows, (agree, known)).

## 12. Bundles (modeling.py)

`save_bundle(path, winner, wavenumbers, prep_params, dataset_name,
paired, **extra)` → joblib dict: `model_name, pipeline, classes,
threshold, wavenumbers, prep_params, macro, dataset_name, paired[,
calibrator, op_points, region_bands]`. When a binary calibrator exists
the stored **threshold is re-mapped through `apply_platt`** (predict-time
space). `predict_with_bundle`: interpolate onto stored grid → stored
preprocessing → optional paired reference subtraction → Platt →
threshold; tolerates legacy uncropped bundles (feature-length guessing).

## 13. Real study results (2026-08-30, seed 42, 5-fold grouped ×3)

> **STALE on two counts now** (loader/crop change gotcha #18 AND the
> 2026-09-06 site-token hygiene change) — re-run before citing.

Hygiene per run: 317 spectra / 72 patients → 18 spike-flagged excluded
(despike off). **v1 artifacts (study_run_standard/paired/3sse) deleted
2026-09-02** — all stale-pipeline; §13/§15 numbers below predate the
trimmed-loader pipeline (see gotcha #18). v2 runs
`raman_app/study_run_paired_v2/` (winner Extra Trees 0.766/0.827) and
`study_run_standard_v2/` — **artifact folders DELETED 2026-09-07 (§32);
the numbers in this file are the record — re-run reproduce_study to
regenerate them**.

| Mode | Winner | macro-F1 (95% CI) | AUC (DeLong) | LOPO |
|---|---|---|---|---|
| Standard | Peak bands + RF | 0.594 (0.54–0.65) | 0.610 (0.55–0.68) | F1 0.589 |
| **Paired** | **Extra Trees** | **0.702 (0.66–0.76)** | **0.788 (0.74–0.84)** | F1 0.703, AUC 0.792 (64 pat.) |

Paired: rule-out p≤0.35 (sens 91%), rule-in p≥0.67 (spec 91%), Brier
0.190→0.185. Literature (spectrum-level splits, looser): ~0.89/0.85.
**No locked one-shot holdout headlessly; no external validation.**

## 14. GUI reference (gui.py)

**Shell**: `TAB_START..TAB_RESULT = range(6)`; SidebarNav (215px,
checkable), PageStack with 170 ms fade, footer back/next
(`NEXT_LABELS`), pages wrapped in QScrollArea. `SUITE_VERSION = 5`
gates saved model-checkbox restore (bumped when the model list
changes). Train page = everyday flow first (mode → models → CV →
Start training → progress → Save) with the 3SSE card BELOW it
(**collapsible** via `b_3sse_collapse`, auto-expands when its search
starts). Mode dropdown uses **UserRole roles** (`mode_kind()` /
`set_mode_kind()`: standard/paired/paired-pqn/seq-standard/seq-paired)
— never rely on combo indices. Model chooser = **2-column** grid
(3 columns forced a 520px card minimum → inner side scrollbar; 2 cols
+ shorter labels keep the whole page ≤ ~860px min — no h-scroll) +
presets (`_apply_model_preset`: all/none/classical/fast) + live
"N of M models selected" counter (`_models_changed`, red < 3).
**Train layout (2026-09-04, user request): NO result tabs** — right
column stacks every card in one page scroll: Result banner →
"Confusion matrix · ROC · diagnostics" (buttons + inline panels) →
Model comparison → Per-class metrics → Biochemistry; `_reveal` scrolls
the page (`stack.widget(TAB_TRAIN).ensureWidgetVisible`). deep_test
`s_train_cards` pins this (no QTabWidget, cm/roc side-by-side, panels
collapse).
Diagnostic buttons live in `self._diag_buttons` (8), grouped under
section labels, **disabled until a winner exists** (enabled in
`on_train_done`, disabled when a training starts).
**`restore_last_3sse()`** runs at the end of `__init__`: if
`study_run_3sse/winner.json` exists it rebuilds the winner ModelResult
(folder deleted 2026-09-02 → restore now silently skips; drop a new
3SSE run there to re-enable)
(cm/oof/y_true are PERSISTED via `sequential.persist_run` — `_dumpable`
keeps ndarrays as JSON lists, plus threshold/calibrator) + the
validated singles, loads the fitted chain from winner.joblib, and
calls `on_train_done` — banner/tables/plots/Save/diagnostics/chain
flow are never blank after a restart; a fresh training overwrites.
`modeling.load_bundle` registers `__main__`-pickled CLI classes
(SequentialChain, _SpectralSlice) so CLI-saved bundles unpickle
anywhere. deep_test harness: file dialogs patched to ("", "") and
`JOBLIB_MULTIPROCESSING=0` (threading backend — loky pools + QThreads
are the Windows access-violation race). A
bottom **Activity-log dock** (QPlainTextEdit, 300 lines, fed by
`log()`, status-bar toggle button) shows errors in-app; `closeEvent`
asks before stopping running workers (No aborts the close; 3SSE keeps
its checkpoint).
Menu: File = Open folder (Ctrl+O),
Reload (F5), Load model (Ctrl+L), Save model (Ctrl+S), **Reset session
(keep training)…**, Exit (Ctrl+Q); Help = How to (F1), Metrics, About.
**`reset_session()`** (that menu item): guards running workers (same 6
as closeEvent) + confirm dialog → clears Data/Preprocess/Predict/Result
(state attrs, widgets, plots, `settings["folder"]`; prep params →
`PreprocessParams()` defaults; Train-page data-coupled widgets →
initial: b_train disabled, "Load data first", flag/replicate chk
disabled) and KEEPS everything training produced (winner, results,
bundle, `_region_bands`, `_op_points`, `_lc_data/_lc_wn`,
`_locked/_lopo/_seed/_noise/_friedman/_band_stats/_honest`,
`_seq_payload`) — Predict works on new data immediately after a reset. Logging: `log()` → stdout +
`session.log` (rotating 1 MB ×2).

**Start** (reworked 2026-09-04, fills the window — no bottom void):
columns row takes stretch 1 (old trailing `addStretch` deleted);
left = hero (stretch 1, vertically centered content + 〰 `HeroGlyph`
— QSS existed unused) → step chips → hint → "How to use this app"
card (stretch 2, reuses `uh.HOW_TO` = F1 text); right = progress +
quick-actions cards (stretch 1 each, status lines spread with
interleaved stretches, buttons `addWidget(b, 1)` grow, min 46 px).
Hero pills = live counts; quick actions = Load data folder, Predict
with saved model. gui_test `[0]` asserts stretch(0)==1 + walkthrough
label present.

**Data**: `folder_edit` + Browse/Reload; `count_label`; `table` 7 cols
(File, Patient, Class — double-click editable, Points, WN min/max,
Quality "spiked"/"ok"; `cellChanged`→`on_cell_changed` writes
`self.labels[row]`, guarded by `_loading_table`); Plot selected (≤10) / **View file…** (per-file browser dialog:
scrollable file list + Prev/Next + arrow keys; RAW PANEL ALWAYS
FULL-RANGE with outside-crop regions shaded — viewing is decoupled
from the crop spins; after-panel = cropped pipeline; SPIKE-FLAGGED
marker; `view_file_dialog`) / Plot class means → `data_canvas`.

**Preprocess**: crop spins (0–5000 step 50, def 500/2000 = tuned
working range;
saved params restore is gated by PARAMS_VERSION so default changes take
effect); despike
chk + z spin (3–20 def 7); wavelet chk + combo (sym4/6/8, db4/6 —
coif3/5 removed 2026-09-04; unknown saved names fall back to sym8)
+ level 1–8 def 4; SG window 5–51 def 11, poly 1–5 def 3,
deriv combo 0/1/2; baseline combo als/arpls + λ exponent 2–9 def 5,
p 0.001–0.5 def .01, iters 5–30 def 10; norm combo vector/snv/none.
Preview (**one row per class** since 2026-09-04 — Normal AND Tumor each
get a raw + preprocessed pair; a Data-page selection wins for its class,
others fall back to first-of-class; processed curve on twin y-axis —
raw ~1e3 vs processed ~1e-2 after vector-norm, shared axis squashed it
flat; x ASCENDING low→high, unlike all other plots which keep Raman
high→low) → `prep_canvas`; **Optimize preprocessing** →
`OptimizeWorker` → `_apply_params(best)` + `save_best_params`.
**Page layout (2026-09-04, user request + polish)**: param card is ONE
vertical flow in pipeline order; EVERY stage sits in a bordered
`QFrame#DiagPanel` mini-panel built by the local `panel(title, *rows)`
helper (header + one compact widget line per row) — REGION (keep range),
DENOISING (one line per stage: despike chk + Z spin · wavelet chk +
combo + level spin), SMOOTHING (win/poly/deriv), BASELINE
(method/λ10^/p/iter), NORMALIZATION (method) — all STACKED full-width
(three side-by-side forced a 728px-min band → horizontal scrollbar;
stacking cut the page minimum width 865 → 447px). deriv combo texts are
display-only ("0/1/2") — read/written by INDEX. Preview/Optimize
buttons equal-width (stretch 1) under them, `optimize_status` below;
preview panel titles use the file BASENAME (full relpath overflowed);
legend fontsize 8. params | preview sit in a **QSplitter**
(stretch 1:3, starts ~520/660 — the chart gets the room); the preview
canvas `setMinimumHeight(400)` in the builder, raised per class count at
preview time (`100·max(5, 2.3n+0.8)`); **live preview**: every param
widget is
connected to `_schedule_preview` → 350 ms single-shot `_preview_timer`
→ redraw (bursts from `_apply_params` collapse into one redraw).

**Train**: `combo_mode` = Standard / Margin (vs own normal) / Margin+PQN
(idx ≥1 ⇒ paired, needs groups); `model_checks` 14 checkboxes from
`ALL_MODEL_NAMES` + All/None; `spin_folds` 2–10 def 5; `spin_seed`
0–9999 def 42; `chk_repeat` (×3), `chk_exclude_flagged` (def checked,
only if flags exist), `chk_avg_replicates` (per patient×class means).
Result card: stat banner + `compare_table` (★ winner, green ≥.90 /
amber ≥.70 / red). **Right column (2026-09-04, latest): pinned Result
banner (stats + `chain_flow`) above FOUR stacked cards** —
"Confusion matrix · ROC · diagnostics" (buttons + `diag_stack`),
"Model comparison" (`compare_table`), per-class table, "Biochemistry";
all visible in ONE page scroll (tabs removed on user request — nothing
behind a click); left Controls card NOT scroll-wrapped (slimmed to fit
instead). Buttons → `run_learning_curve`,
`run_region_importance`, `run_honest_check`, `run_locked_eval`,
`run_lopo`, `run_seed_stability`, `run_noise_check`,
`run_biochemistry`. **Since 2026-09-03 the "Confusion matrix · ROC ·
diagnostics" card has NO pre-placed charts**: results render INLINE,
each in its OWN bordered sub-card (`QFrame#DiagPanel`, `_diag_panel(key,
title, size)` → frame + canvas + caption; keys cm/roc after training,
lc/regions/lopo/seeds/noise per click, honest/locked text-only) —
nothing overwrites, honest+locked show here (Result page stays as the
report view), retraining drops stale panels (`on_train_done`). Panel
headers are COLLAPSIBLE QToolButtons (click ▾/▸ — chart folds away,
caption verdict stays; `container=` param lets `_draw_winner_plots`
put cm (4.6×3.6) + roc (5.4×3.6) side by side in `_winner_row` at the
top of the stack, 46:54 stretch); `_reveal(frame)` switches the tab +
`ensureWidgetVisible` so each finished result scrolls into view
itself. Square
charts (confusion matrix) use a taller canvas ratio than
wide ones (9×3.6 in). Panel
canvases draw **synchronously** (`draw()`, not `draw_idle()`) so no
idle timer can fire on a deleted canvas (was RuntimeError + native
abort; fixed 2026-09-03 together with the run_lopo
`n_test_patients`→`n_patients` KeyError — a silent qFatal until
then). **Since 2026-09-04 the whole battery runs AUTOMATICALLY after
every training** (`_on_train_done_then_diags` → `run_all_diagnostics`
→ serial `_diag_queue`, fast→slow: regions, learning curve, seeds,
noise, locked (`auto=True` skips its confirm dialog), LOPO, honest; a
"Run all diagnostics" button does the same for an existing winner).
Chain invariants: `_run_async.finish/fail` + `on_honest_done/failed`
call `_pop_diag_queue()`; each finished worker is `wait()`ed before
the next starts; `run_honest_check` only accepts a real
QAbstractButton as sender (the chain's sender() is the worker object —
both were silent native aborts); `start_training` refuses to overlap
the chain (gotcha #16); restore/3SSE paths do NOT auto-run.
Biochemistry card (`bio_canvas`, `bio_comp_canvas`,
`bio_table` 6 cols); per-class table (Class, n, sens, spec, prec, F1).

**Predict** (since 2026-09-04: **two-column layout** — setup cards left
(2/5): model card with `model_path_edit`/Load…/`model_info` and
`ref_row_widget` INSIDE it (paired bundles only); input card
`spec_path_edit` **editable** (paste a folder path or `;`-joined
files; `textChanged` arms Predict) or folder/file browse + full-width
primary Predict button; results right (3/5): `pred_pills` (n spectra +
per-class counts, positive class red via `uh.repolish`), `predict_status`
(empty state → live "Predicting… N%" from worker progress → done
summary), card "3 — Predictions" = `pred_canvas` **8×4.5 in, holder
minHeight 300, zoom/pan toolbar** (live first-spectrum plot during the
run, `_draw_prediction_overview` after: all traces + mean + paired
reference | class-count bars) over `pred_table` **4 cols** (File,
Prediction, Triage, P as progress bar — filled by
**`_fill_prediction_table`**, shared with Result's `r_pred_table`;
helpers `_n_classes`/`_threshold_now`/`_positive_class` (bundle
fallback → triage works in predict-only sessions)). Data flow
unchanged: `_prediction_files()` = **recursive `os.walk`** — any
nesting depth
(<root>/<class>/<patient>/<anything>/*.csv), skips REFERENCE_MARKERS
+ split_log.txt (junk files fail per-file in the worker, logged);
paired auto-refs are **file-anchored**: `_auto_reference` walks UP
from each predicted file (≤6 levels) to find the clinical tree (an
ancestor with a `Normal*` dir), patient = relpath component under
the class dir — so ANY input works (whole tree, class folder, single
patient folder, mixed `;` file lists, any depth); ref spectra
collected recursively; Predict → `PredictWorker`; on done the app
still jumps to Result (Predict stays populated for return visits).

**Result** (top→bottom): hero banner + verdict + `LikelihoodMeter`;
model card `r_stats`{sens,spec,f1,auc} + `r_model_note` (AUC DeLong CI +
patient-bootstrap F1 CI); `r_ppv_table` (3 prevalence scenarios);
`r_triage_table`; `r_locked_card` (70/15/15 one-shot FINAL);
`r_deep_card` (LOPO/seeds/noise/Friedman); `r_pp_table` per-patient OOF
(worst-first, red <50%); `r_local_card` "Explain this prediction"
(`run_local_explain`, surrogate SHAP) + `r_local_canvas`;
`r_pred_table` | `dist_canvas`; `r_spec_canvas` (traces + mean +
reference + amber bands); `r_pat_table` patient verdicts; validation
2×2 (`r_cm/roc/cal/dca_canvas`); `r_bio_table`; reading card; buttons
Freeze study / Save figures / Save report (txt) / Save report (HTML).

**Key state attrs**: `spectra, labels, groups, spike_flags, grid,
X_raw, _data_key, _proc_cache, results, winner, bundle, _lc_data
(X,yy,gg), _lc_wn, _pred_rows/_pred_probs/_pred_spectra/_pred_wn,
_pred_reference, _patient_rows, _region_bands, _op_points,
_locked_result, _lopo_result, _seed_result, _noise_result,
_friedman_result, _local_bands, _band_stats, _paired_mode,
_honest_result, _honest_btn` + workers (`worker, _opt_worker,
_pred_worker, _honest_worker, _analysis_worker`).

**Workers** (QThread, signal/slot, never touch widgets in run()):
Train, Optimize, Pipeline (honest), Predict, **FuncWorker** (generic).
**`_run_async(label, fn, on_done)`**: off-thread fn, disables sender
button, failure dialog + session.log — used by the 8 analysis buttons.
`closeEvent`: wait(3000) → terminate → wait(500) for all 5 workers,
then save settings.

**freeze_study manifest** (`study_manifest.json`, gitignored — patient
IDs): created, app, data_folder, n_spectra, n_patients, paired_mode,
preprocessing (asdict), code_hashes (SHA-1 every .py), data_hashes,
winner{model, macro_f1, sens, spec}, nested_honest_f1?, predictions[].

**Report writers**: txt sections = header, dataset, preprocessing,
winner, comparison, predictions+triage, plain reading, literature table
(Han 2022 / 2025 OSCC meta / Purohit 2026 / VELscope / toluidine blue),
PPV/NPV by setting, TRIPOD+AI reporting, deep evaluation, per-patient
top-worst-5, FDR band stats. HTML = printable, embedded CSS + 6 base64
PNGs @150 dpi, predictions ≤300 rows.

**settings.json** (live UI state — drifts per session; not canonical):
params (mirrors §6 defaults), folds, seed, models (gated by
SUITE_VERSION), geometry hex, folder, last_model, predict_folder.
Startup: saved `folder` if still on disk, else `find_data_root()` (§5)
— a copied project auto-finds its data on any device.

## 15. 3SSE — sequential architecture search (sequential.py)

Evaluates **every** ordered chain over the model registry:
17 singles + 17·16 pairs + 17·16·15 triples = **4,369 architectures**.
Chain semantics (probabilities only): `X → A → OOF P1 → [X+P1] → B →
OOF P2 → [X+P1+P2] → C → final OOF`. Inter-layer predictions are
grouped `cross_val_predict` OOF (patient-grouped, no leakage).
"Ensemble (top-3)" is candidate #17, resolved post-singles from the
singles ranking (voting over the top-3 defaults) — nested ensembles
are possible and documented.

- **QUICK screening**: single-level OOF stacking, k=3 folds (default),
  default hyperparameters (identical for all 4,369 → fair level
  comparison). Efficiency: layer-1 OOF shared; **pair-major** loop
  (one feature matrix per pair, all 15 third models evaluated while
  hot, O(1) memory); pairs processed best-first; **early-abandon**
  bounds the MEAN PER-FOLD F1 (`metrics["f1_mean"]` — the ranking key
  for triples since 2026-09-06; pooled-OOF F1 was never bounded by the
  fold-mean so pruning on it was unsound); patient truth = DOMINANT
  class (was first-row — arbitrary on paired data); default beam
  `prune_pairs=50` forfeits GLOBAL top-N exactness (exact within beam);
  parallel across pairs (joblib, 1 thread inside);
  float32; JSONL checkpoint + `--resume`; `--prune-pairs K` beam option.
- **FULL validation**: top-20 per level through nested grouped CV —
  OOF features regenerated inside each outer training fold only
  (`validate_arch`); winner picked by (F1, sens, spec) from validated
  results only (`pick_overall`) — 1-model can legitimately win.
- **`SequentialChain`** (sklearn protocol, `ClassifierMixin` first!):
  fit = per-layer grouped OOF stacking + full-data fits; predict_proba
  chains, appending each layer's proba columns. Drop-in `pipeline` for
  the existing bundle/Platt/threshold/predict stack.
  **Model-specific features (spec §9)**: "Peak bands + RF" needs the
  wn-aligned axis, so at chained positions (≥1) it is wrapped with
  `_SpectralSlice` (spectral columns only); ensemble voters likewise.
- CLI: `python sequential.py [--data ROOT] --mode paired --out
  study_run_3sse [--k 3 --top 20 --jobs -2 --resume --models ...]`;
  ROOT defaults to `find_data_root()` (§5), missing data → clean
  exit 2; writes screening.jsonl, report.txt, run_meta.json,
  winner.joblib.
  `prepare_dataset()` mirrors reproduce_study (GUI parity).
- GUI: Train-page combo entries 3/4 = "3SSE search (standard/paired)";
  `SeqSearchWorker` (existing QThread pattern) runs screening +
  `validate_top` + `finalize_winner`; live progress `n/4369 · best ·
  ETA`; `SeqResultsDialog` = 4 sortable tabs (Single/2-Model/3-Model/
  Overall Winner, `ArchTableModel` + proxy — 4,080 rows sort in ms) +
  "Save winner as model bundle" (goes through `save_bundle` +
  `_set_bundle`, so the whole Predict/clinical stack works on it).
  **First-class training outcome**: on completion `on_seq_done` wraps
  the winner as a real `ModelResult` (per_class values must be
  `(mean, std)` TUPLES keyed by class NAME — consumers index `[0]`)
  and calls `on_train_done`, so banner/compare-table/per-class/plots/
  Save/Result-page/Predict all work on the chain; `chain_flow`
  (ChainFlowWidget, custom-painted Spectrum→Model→P→…→Verdict
  diagram) shows in the banner. Train page has a 3SSE status card
  (visible for 3SSE modes): live search-space counts from checked
  models, big n/total counter, current best + F1 + ETA (structured
  `search_progress` signal), phase switch to "nested validation".
  "View saved 3SSE results…" opens the last run from
  `study_run_3sse/` (screening.jsonl + validated.json + winner.json;
  report.txt fallback) — save-button disabled for loaded runs (no
  in-memory chain; winner.joblib already exists). The card is ALWAYS
  visible and carries the primary **"Run 3SSE architecture search
  now"** button (`run_3sse_now`): guards data + busy state, auto-picks
  paired (groups present) vs standard 3SSE mode in the dropdown, then
  calls `start_training()`. Card extras: **Fast screening** toggle
  (k=2 + skips `sequential.SLOW_MODELS` = 1D-CNN/CatBoost/XGBoost, CNN
  8 epochs), rough **time estimate** (archs·k/240 + 60·0.4 min,
  calibrated on the measured full run), **Cancel** (cooperative
  `cancel_check` → `SearchCancelled`; Run hides/Cancel shows while
  running), **live top-5 leaderboard** (`leaderboard_cb` at batch
  boundaries), **checkpoint resume** (worker passes
  `study_run_3sse/archs.jsonl`, deletes on success, keeps on
  cancel/crash — Run resumes; replay filters to the current model
  subset; the 7.2 MB archs.jsonl from the completed 2026-09-06 run
  was DELETED 2026-09-07 §32 — new searches start fresh, the
  winner/validated restore path never reads it). After validation the worker auto-runs **significance**:
  exact McNemar (winner OOF vs best-single OOF, identical folds) +
  3-seed stability; shown in the winner tab ("SIGNIFICANCE (auto)").
  Dialog: **"Export this tab to CSV…"** (active ranking tab →
  utf-8-sig CSV). NOTE: ad-hoc driver scripts that call
  `sequential.search` must live IN raman_app (loky main-module
  pickling from a $TEMP __main__ segfaults; main.py is unaffected).
- Tests (in test_all.py): space counts/order/no-repeats, search smoke,
  chain + joblib roundtrip for 1/2/3 layers, OOF-leakage probe
  (patient-unique offsets: in-sample ≈1, grouped OOF ≪ 1).

## 16. Secondary tracks

**optimize.py** — GRID (8 base + 3 arPLS): no-crop/vector;
crop400-1800 × {deriv0 vector, deriv0 snv, deriv1 vector, deriv1 snv,
deriv2 vector}; crop800-1800 deriv1; λ=1e6 deriv0; arPLS × {deriv0
vector, deriv1 vector, deriv0 snv}. Scored by best of {PCA+SVM,
RF 300, PCA+LogReg} macro-F1 under grouped CV; spike-flagged excluded
unless `--keep-flagged`. `BEST_PARAMS_PATH =
vit_outputs/preprocess_best.json` (`{preprocess, score_macro_f1,
model}`) — vit_train `--preset best` consumes it.

**ViT track** (torch, CLI only, not in GUI suite):
- `ViTConfig`: seq_len 512, patch 32 (→16 tokens), dim 128, depth 4,
  heads 4, mlp_ratio 2.0, dropout 0.1; Conv1d patch embed + CLS token +
  learned pos-embed; pre-norm blocks; head on CLS.
- `vit_train.py` args: epochs 25, patience 10 (val macro-F1), batch 16,
  lr 1e-4, wd 1e-4, test/val 0.15/0.15, seed 42, `--cv K` grouped k-fold
  (inner grouped val), `--preset best`. Clinical layout → patient_split
  (grouped); flat → stratified spectrum split. Augmentation: Gaussian
  σ=5% std, roll ±2, 50% mixup β(0.2,0.2). AdamW + warmup(≤5)+cosine,
  class-weighted CE. Outputs: vit_model.pt (+curves, confusion PNGs).
  Checkpoint stores splits as filename lists + standardization stats.
- `vit_test.py`: re-evaluates the exact checkpoint test files; prints
  accuracy + report; writes vit_test_report.txt + confusion PNG.

### 3SSE real-data results (2026-08-30, paired mode, seed 42)

All 4,369 architectures screened (k=3, 3,462 pruned by the sound
early-abandon bound — 79 % of triples never finished their folds);
top-20 per level nested-validated (5-fold grouped). Artifacts in
`raman_app/study_run_3sse/` (screening.jsonl, report.txt,
winner.joblib).

| Level | Best architecture | F1 (nested) | AUC |
|---|---|---|---|
| Single | Extra Trees | 0.702 | 0.789 |
| 2-Model | Ensemble (top-3) → Extra Trees | 0.697 | 0.787 |
| **3-Model** | **PCA+GNB → PCA+SVM(RBF) → Extra Trees** | **0.725** | **0.796** |

**OVERALL WINNER: the 3-model chain** (F1 0.725 / sens 0.726 / spec
0.726 / AUC 0.796) — beats the paired baseline (Extra Trees 0.702 /
0.788) by +0.023 F1, +0.008 AUC. **BUT McNemar on the identical outer
folds: b=16, c=10, exact p=0.327 — the improvement is NOT
statistically significant.** Honest verdict: sequential chaining gives
a modest point-estimate gain, unproven at this sample size (n=287
spectra / 64 patients). Known limitation: winner chosen from the top-20
per level on the same seed; no seed-stability rerun yet. Screened
with default hyperparameters (identical for all 4,369 — fair).

## 16. Gotchas & invariants (read before editing)

1. `load_clinical_dataset()` returns a **`ClinicalData` dataclass**
   (spectra/groups/grid/report/spike_scores/flagged) — NOT a tuple.
2. `paired.paired_features` takes **RAW X + FULL grid**; preprocesses
   internally once. Never pass preprocessed X (regression test pins).
3. **GUI parity**: reproduce_study must mirror the GUI data path —
   `common_grid`/`to_matrix`, exclude spike-flagged when
   `despike=False`, same label stripping.
4. Bundle `threshold` lives in **Platt-calibrated** space.
5. `fold_f1` order = [rep0 f0..k, rep1 f0..k, …]; block statistics must
   reshape `(repeats, k)` and average over repeats.
6. `bootstrap_ci(..., groups=)` resamples groups, not rows.
7. Tests pressing async buttons need `wait_analysis(win)` /
   `worker.wait()` + `processEvents` — results land in queued callbacks.
8. `gui_test.py`/`deep_test.py` **overwrite `result_report.txt/.html`
   in the source tree** (known leftover).
9. PLS-DA probabilities are softmax-of-scores (pseudo-probs). Friedman
    q_alpha hand-tabulated. `band_stats_paired` returns only FDR p.
10. Duplicated helpers (consolidate someday): `_encode` ×3, PCA builders
    ×6 + optimize copy, rank-with-ties ×2; optimize uses
    random_state=0 vs suite 42.
11. Library modules `print` instead of logging (dataset, optimize,
    vit_train) — GUI mirrors prints into session.log.
12. sklearn "delayed/Parallel" UserWarnings in paired runs + joblib/
    loky shutdown warnings in deep_test — benign, every run.
13. `sg_deriv≠0` **skips baseline correction** by design (derivative
    suppresses smooth baselines).
14. qt_compat binding = import-success order (PyQt5 first), no QT_API
    env pinning; plotting.MatplotlibCanvas pins the binding for
    matplotlib at canvas creation.
15. **ModelResult consumers expect `per_class[class_name][metric]` as
    `(mean, std)` tuples** — building a winner by hand must match this
    (3SSE integration hit exactly this).
16. **Concurrent loky pools from several QThreads are a native-crash
    race on Windows** (access violation in joblib retrieval; heavier
    child imports like torch/Qt make it likelier). Never start Train +
    Optimize + Honest workers simultaneously — deep_test's workers
    scenario runs them serially.
17. deep_test `make_gui()` patches `uh.load_settings` to `{}` —
    settings.json last-folder restore would silently hand "no data"
    scenarios real data. Test harnesses must isolate from developer
    settings.
18. Dataset = TWO file populations on ONE shared grid (all 341 CSVs:
    identical 2271-row axis 15.5→3862 cm⁻¹): `Patient_xx/*_interpolated.csv`
    = linear OR constant fill outside measured ~504–2001 (per-file,
    NOT class-linked; 155 files); `TDOCxxx Spectra pro/raw/*.csv` =
    genuinely measured full-range (180). **Since 2026-09-04 the loader
    keeps every file AS-IS over its full axis (user decision: no data
    point may be dropped)** — `trim_edge_padding` was REMOVED; the 155
    interpolated files again contribute their padded outer regions
    (known tradeoff, chosen for completeness). Common grid = full
    15.5–3862 (~2000 pts after median-clip); default crop 500/2000
    keeps the tuned window (780 pts). **All §13/§15 results predate
    these loader/crop changes**
    (padded grid + crop 400/1800) — re-run the study.
    Session settings: despike on, arPLS (see #20/§14; crop 500/2000 is
    now the default).
19. **Unhandled Python exceptions inside Qt slots qFatal-abort the
    process silently on PyQt ≥5.5** (exit code 127, no traceback) —
    hit via a ValueError in `view_file_dialog`'s `show()` closure.
    **RESOLVED 2026-09-05: `MainWindow._install_excepthook()`** installs
    a `sys.excepthook` (log + status bar + ONE dialog per session,
    restored in closeEvent) — PyQt routes slot exceptions to a custom
    hook instead of aborting; gui_test [8] probes it with a raising
    timer callback. Debug offscreen with `faulthandler.enable()` +
    marker prints remains valid for NATIVE crashes.
20. Tuned session preprocessing (2026-09-01, tuned on a representative normal spectrum,
    metric = residual-hump 500–900 / %<-0.02): crop 500/2000,
    despike ON z=7, **arPLS λ=1e5** (hump 0.054 vs ALS 0.116 — ALS
    leaves a big baseline hump on these steep SERS decays), rest
    default. despike ON also recovers the 18 spike-flagged spectra
    into training (paired n 287 → ~305).
21. **3SSE worker RAM budget** (2026-09-05): loky children each import
    torch+boosters (~1.5-2 GB); CPUs-1 workers OOM'd a 16 GB laptop with
    ~7.7 GB free (CatBoostError 'bad allocation' /
    TerminatedWorkerError at the pairs loop). Fix: `_safe_jobs`
    (sequential.py) sizes workers by AVAILABLE RAM via ctypes
    GlobalMemoryStatusEx — 2 GB/worker, ceiling 4 — and `as_env_device`
    caps worker thread pools (CatBoost thread_count=1, XGB/LGBM/RF/ET
    n_jobs=1 when RAMAN_DEVICE=cpu). Verified: 156 archs incl.
    CatBoost/XGBoost on the real paired data in 8.5 min, 3 workers, no
    OOM.
23. **Whole-project error handling (2026-09-05, user request after a
    Model Lab "bad allocation" killed the run)**:
    (a) `RAMAN_DEVICE=cpu` is now setdefault-ed at GUI startup AND in
    `sequential.search()` — the `as_env_device` thread caps
    (CatBoost thread_count=1/CPU, XGB/LGBM/RF n_jobs=1) apply to
    PARENT-process fits too (the uncapped GUI-process CatBoost fit was
    the actual crash; set the env to "gpu" to opt out);
    (b) 3SSE screening is ERROR-TOLERANT: workers return
    `(key, None, None, err)` instead of raising; failed archs are
    recorded as `{"arch","level","error"}` JSONL rows (resume treats
    them done, never retried; rankings/cutoffs/winner filter them out);
    pairs whose level-1 OOF failed record "level-1 features
    unavailable"; the parent Ensemble OOF is guarded too — one broken
    model costs one architecture, never the run
    (`test_search_survives_broken_model` pins it);
    (c) Model Lab degrades gracefully: validate/finalize failure of the
    tuned chain → falls back to the best SINGLE model (tune=False);
    (d) all 6 QThread workers wrap run() → failed.emit (buttons always
    re-enable); (e) deep_test patches `remember_data_root` +
    `save_settings` process-wide — temp clinical trees can no longer
    poison ~/.raman_app_data_dir/settings.json (a leftover 32-spectrum
    toy dataset was silently restored as "last session" 2026-09-05).
22. **torch + full MainWindow in ONE test process natively aborts**
    (exit 127, no traceback; 2026-09-05): the stage3 test's
    CNN-fit/grad-cam followed by an UN-isolated `gui.MainWindow()`
    (loads the real 317-spectrum dataset) dies on this machine — even
    with RAMAN_DEVICE=cpu; each half passes alone. Same native-race
    family as #16/#17 (heavy imports + Qt). test_all reaches 47 PASSes
    and dies at stage3; stage3+stage4 need splitting or isolation.
21. **Never hardcode device paths** (D:\BARC\Data, C:\Users\…): use
    `clinical_data.find_data_root()` (env → pointer → known spots, §5).
22. **`QToolBar { background: transparent }` made matplotlib 3.11
    toolbar icons (zoom/pan/save…) WHITE-on-white** — mpl's
    `_IconEngine._is_dark_mode()` reads the toolbar palette background
    and transparent polishes to black (value 0 < 128 → paint icons
    white). QSS now pins `background: #ffffff`; gui_test `[0b]`
    asserts the palette stays light. Any future dark toolbar styling
    must re-check icon contrast.
    sequential's 3SSE checkpoint parent dir is auto-created before
    append; missing `--data` exits 2 with a hint, not a traceback.
22. **No newline inside an f-string's `{..}`** — that is Python 3.12+
    syntax (PEP 701); on the dev machine (3.14) it runs, but the module
    SyntaxErrors on older Pythons (copied project to a 3.11 device
    2026-09-04, hand-edit needed in `format_report`). Keep the
    expression on one line or hoist it into a parenthesized variable.
`test_all.test_fstring_device_portability` source-scans every .py
24. **Deploy-side preprocessing must go through `preprocessing.align_to_grid`**
    (2026-09-06, CRITICAL fix). `preprocess_matrix` (training) applies
    `calibrate_wn` BEFORE crop when `wn_calibrate=True`, but
    `predict_with_bundle`/`paired.reference_vector` used to skip it —
    bundles trained with calibration were fed shifted, out-of-distribution
    features at predict time and the deployed 3SSE winner called EVERY
    spectrum Tumor p≈0.94–0.97 (normal patients included). The Phe-1003
    anchor was already fixed/predict-time-safe (§20 fix 6); the predict
    path just never invoked it. `align_to_grid` (interp + calibrate) is
    the single shared entry point now; `test_paired_deploy_parity_wn_calibrate`
    pins train/deploy FEATURE parity + end-to-end probability parity
    (with a non-vacuous negative control). Any new predict path must use
    it, never raw `np.interp + crop + preprocess_spectrum`.
    so it cannot come back.
23. **GPU acceleration semantics (2026-09-04)** — works on ANY CUDA
    GPU, detection never by card name: `modeling.gpu_ok()` (cached;
    torch-CUDA check, XGBoost cuda-canary fallback) + `boost_device()`;
    the device name in `device_report()` comes from the driver.
    Registry: XGBoost `device=boost_device()`, CatBoost
    `task_type=GPU-if-ok` (LightGBM: no GPU in standard Windows wheels
    — skipped). `RAMAN_DEVICE=cpu` is the universal kill-switch,
    checked on EVERY gpu_ok() call (env wins over the cache).
    3SSE: estimators are built in the parent (GPU params baked in),
    pickled into loky workers — `sequential.wrap()` re-points them via
    `modeling.as_env_device()`; `_tune_hyperparams` runs GPU-model
    GridSearchCV with `n_jobs=1` (n loky children × ~300 MB CUDA
    contexts would swamp VRAM; grids are 2×2 so serial is free).
    torch: 1D-CNN + ViT train under AMP (autocast+GradScaler) with
    TF32 gated by `get_device_capability() >= (8,0)`; vit_train gained
    `--device {auto,cuda,cpu}` (honors RAMAN_DEVICE) + pin_memory.
    XGB GPU boosters predict from CPU numpy by design — the "Falling
    back to prediction using DMatrix" warning is filtered in modeling.
    GUI logs `device_report()` at train start; reproduce_study prints
    it. Verified live: nvidia-smi 34–70% utilization during training
    on the RTX 3050 6GB laptop card.

## 16b. Speed & Accuracy program (2026-09-05) — scorecard
Baseline before the program (bench.py, seed 42, real data, paired):
Extra Trees F1 0.774 / AUC 0.826 · screening 500 archs (10 models) 12.5
min · baselines 174 s. Every change below measured via `bench.py`
(fixed seed/data; `--full` adds nested validation + LOPO).

SPEED (all landed):
- A1 3SSE engine: 2-fold screening ladder (`screen_folds` in board;
  nested validate_top re-ranks at full fidelity), beam prune_pairs=50
  DEFAULT (CLI+GUI), WARM cutoff seeded from pairs F1 −0.10, third
  candidates tried best-single-F1-first, joblib memmap (max_nbytes=100)
  shares X once, singles pooled. Checkpoint now records ALL levels with
  their OOF arrays — resume replay = 0.8 s vs 23.4 s full (29×, smoke).
- A2 CNN epochs 40→25. Lazy threshold-CV REJECTED (accuracy-coupled:
  threshold feeds OOF predictions).
- A3 preprocessing stage cache (preprocessing.py `_STAGE_CACHE` keyed
  by spectrum hash + stage params): sweeps pay ~2-3 distinct
  prefixes/baselines instead of N full pipelines.
- A4 parallel refits REJECTED with evidence: LOPO refits ET with
  n_jobs=-1 internally — parallel loops would oversubscribe.
- A5 cross-session OOF cache: deferred (low ROI after A1 checkpointing).

ACCURACY (every lever A/B-gated on the real data):
- B1 `sequential.tune_chain`: greedy per-layer grid tuning on each
  layer's real stacking context (grouped inner CV, reuses
  _tune_hyperparams); `finalize_winner(tune=True)` nested-validates
  tuned vs untuned and deploys tuned when not worse (params returned).
- B2 paired sweep (optimize.optimize_paired + GRID_PAIRED, 15 configs,
  62 s — deviation features had NEVER been swept): **defaults
  CONFIRMED optimal** — ALS λ 1e6 within noise (0.768 vs 0.761);
  deriv-1 HURTS paired (0.683) though it helped standard; despike-ON
  recovery of the 18 spiked spectra DROPS F1 to 0.731 — exclusion
  re-confirmed; crop 500/2000 ≫ 400/1800 (0.761 vs 0.697).
- B3 registry model "Spectral + band features" (SUITE_VERSION 7, 22
  models): FeatureUnion[PCA(spectrum) ‖ 16 band intensities] + ET;
  wn-aligned in chains via WN_ALIGNED_MODELS.
- B4 `AveragedChain`: deployable winner = 3-seed averaged chain
  (finalize_winner). Patient-level F1 is a bench headline
  (per-patient mean-P verdict).
- B5 isotonic-by-ECE, B6 augmentation-for-trees: scoped out
  (calibrator format is deploy-coupled; CNN-only aug already default).

C1 **Model Lab** (Train page, ⚡ button): one click = paired sweep →
apply best params → 3SSE beam search (checked models) → validate_top →
tuned, seed-averaged AveragedChain → panel report + bundle saved to
study_run_3sse/winner.joblib. FuncWorker gained a progress signal.

bench.py = the program's scorecard (bench/latest.json); run before/
after any future change.

## 16c. Speed & Accuracy program 2 (2026-09-05, evening) — deep-research batch

Research basis: wall-clock map + accuracy-lever map + SOTA review
(TabPFN v2 Nature-2025, RamanBench-2026, Farnesi-2025, Jeng-2019).

SPEED (landed, result-preserving unless noted):
- **validate_top parallelized** (`_worker_validate` + joblib
  `_safe_jobs`, generator-order results re-assembled into screening-
  rank order): the ~25-min SERIAL nested-validation phase now uses the
  same ≤4-worker pool as screening. Test: parallel == serial (1e-12).
- **`modeling._tune_inner`** replaces GridSearchCV + the separate
  threshold `cross_val_predict` in `run_cv`: ONE inner pass per
  (param combo × inner fold) collects pooled OOF probabilities —
  hyperparams AND binary threshold come from the same pooled OOF
  (~4 fewer fits per fold per model; StackedEnsemble threshold now
  from its own nearly-free `meta_oof_` = 3 logistic CVs instead of 3
  full stacked refits). SEMANTIC CHANGE (flagged): hyperparam scoring
  is pooled-OOF macro-F1, not mean-of-inner-folds.
  Backend rule (new invariant): **threads, never loky, for inner-CV
  fits** — `_NOT_THREAD_SAFE(name)` (XGBoost/CatBoost/LightGBM/
  1D-CNN/TabPFN) runs serial-in-process; everything else
  `Parallel(prefer="threads", 4)`. Reason below.
- **noise_robustness fit-once**: folds fit once, re-predicted per
  noise level with pre-drawn level-major noise → BIT-identical curve,
  20→5 fits. Test asserts equality vs the old loop.
- **LOPO stays serial by default** (`jobs=1` param kept): the §16b-A4
  measured rejection (tree winners' internal n_jobs=-1 oversubscribe)
  still stands; the machinery exists for future opt-in.

ACCURACY (wired, A/B-gated where the data exists):
- **"1D-CNN ensemble (5 seeds)" registry entry** (24 models now):
  `CNNEnsemble` (Lakshminarayanan 2017 deep ensemble) finally
  registered, WITH `synth=0.3` — within-class `lorentzian_synthesize`
  blends of the TRAINING split only (val split stays real so
  early-stop stays honest). Plain 1D-CNN stays the controlled variant.
- **TTA at deploy**: `predict_with_bundle` uses `predict_proba_tta`
  when the final estimator has it (implemented-since-long, never
  called).
- **optimize grids extended** (+4 standard, +4 paired): wn_calibrate
  (Phe-1003 alignment — the project's own literature note), wavelet
  bayes/garrote+cycle-spin (−23% RMSE measured), als_p=0.001; CLI
  gained `--mode paired` (GRID_PAIRED was GUI-Model-Lab-only).
- **Patient-level metrics first-class**: `study_stats.
  patient_level_metrics` (per-patient mean-P, dominant-true-class)
  surfaces in the Train banner line and reproduce_study summary —
  the field's reported level (Jeng 2019 ~+6pp, Farnesi 2025).
- **TabPFN (foundation model)** registry entry: PCA(0.95) →
  TabPFNClassifier (v2.5 via **tabpfn==6.0.6** — last release whose
  HF weights need NO account; 8.x demands a Prior Labs browser login,
  do NOT upgrade blindly; weights auto-download once; research
  license). Zero tuning; runs on the torch GPU.
- **despike-ON-for-paired REVERTED before landing**: §16b-B2 already
  measured it DROPS paired F1 (0.731 vs 0.766) — exclusion
  re-confirmed; the plan's B5 was wrong and the measurement won.

GOTCHA (new, hard-won): **full-registry training crashed twice**
(CatBoost import-time CUDA probe OOM → native abort; then loky
children each loading torch-CUDA DLLs → OpenBLAS alloc failures).
Root fixes: (1) `gpu_ok()` now requires explicit **RAMAN_DEVICE=gpu**
— boosters are CPU by default (they lose at n≈300 anyway); torch
(1D-CNN/ViT/TabPFN) stays GPU-by-default via `torch_device()`.
(2) `_tune_inner` thread backend (above). (3) `sequential.
_child_cpu_only()` hides the GPU from loky children (search wrapper +
validate_top) so no child ever probes CUDA at import.
**Windows-vista fail-fast gotcha (same day)**: constructing
MainWindow WITHOUT `app.setStyle("Fusion") + setStyleSheet(
uh.STYLESHEET)` now fail-fasts (0xC0000409, silent, bash exit 127) —
the default windowsvista style + this QSS-heavy UI broke after the
tabpfn pip churn. run_app/deep_test/gui_test style first (safe);
test_all's stage3 GUI block now does too. Debug red herring: the
crash survives tabpfn stubs, CPU-only torch and stubbed pages —
purely style-engine.
**pip dependency casualty**: installing tabpfn silently DOWNGRADED
pandas 3.0.5 → 2.3.3 (restored; re-verify numpy/sklearn pins after
ANY tabpfn install). tabpfn 6.0.6 imports a sklearn-1.9-removed
private API — use **6.4.1** exactly.

SKIP list (researched, rejected — do not re-litigate without new
evidence): MoE (no small-n evidence, gating adds variance) · GAN/
diffusion augmentation (cost >> gain at n≈300, only evidence is
non-grouped CV) · learnable baseline networks (variance source) ·
FDA machinery (≈PCA parity) · cuML/WSL2 GPU for classical stack
(no Windows build; tiny-n loses anyway) · TabICL (large-n tool) ·
isotonic/stacked calibration (Platt is the small-n standard) ·
snapshot ensembles (cyclic LR not worth tuning).

## 17. Testing map

| Suite | What it proves | Runtime |
|---|---|---|
| `test_all.py` (56) | loader hygiene (incl. site-token), preprocessing, grouped+repeated CV/ensembles, bundle roundtrip, optimize, ViT forward/checkpoint, clinical stats, biochem, FDR/Friedman + regressions (clinical reproduce path, paired single-preprocess, OOF repeat pooling, patient bootstrap, data-root discovery, device-portability scan; 2026-09-06: CONTIGUOUS calibration-bins regression, auc_power formula regression) | ~2 min |
| `deep_test.py` (19) | blank-GUI guards, one-class train, predict edge paths, legacy bundles, clinical auto refs, locked eval (+`_lc_data_key` tag in hand-built scenarios), HTML report, deep diagnostics, reset-session, train-tabs layout, CLI roundtrip (cwd-safe since 2026-09-06) | ~60 s |
| `gui_test.py` (8 steps) | 6-page walk: load→preprocess→train→save→predict→result→report (report asserted in the TEMP APP_DIR since 2026-09-06, not in-tree) | ~20 s |

CI COMMITTED since 2026-09-06 (`.github/workflows/ci.yml`: windows,
py3.14, offscreen, ruff==0.16.5 pinned, HF-weights cache for TabPFN,
90-min timeout, pip cache). Optional-dep honesty line printed by
`test_all.main()` when torch/shap/pybaselines/lightgbm/catboost/
xgboost/tabpfn are missing.
**2026-09-03 (resolved):** deep_test used to segfault (exit 139)
mid-suite — root cause: pending matplotlib `draw_idle` timers firing
on canvases of CLOSED scenario windows (`RuntimeError: wrapped C/C++
object … has been deleted` → native abort). Fixes: (a) `check()` now
flushes the event loop after every scenario, (b) the new inline Train
panels draw synchronously (`draw()`). Full suite now runs **18/18,
exit 0** on this machine.

## 18. Git state & open items

Repo `D:\BARC`, branch `main`, init 2026-08-30; per-commit history:
`git log --oneline`. Patient data & artifacts gitignored (§3).

**Open (prioritized):**
0. **History PURGED 2026-09-06** (`git filter-repo` removed the
   initially-committed patient spectrum; pre-rewrite backup bundle at
   `D:\BARC-pre-purge.bundle`; residual ID STRINGS may remain in old
   history blobs — run the broad text filter before publishing).
1. ~~`--locked` one-shot holdout for reproduce_study~~ DONE (A1).
2. ViT vs classical comparison in one harness; ViT predict not in GUI.
3. gui.py 7.7k monolith (split per page); txt/HTML report duplication.
4. ~~No CI~~ COMMITTED 2026-09-06 (needs its first green run).
5. Winner selected on the reporting CV (nested eval is opt-in, GUI-only);
   no external cohort; confounders unrecorded.
6. 3SSE winner: run seed-stability (5 seeds) on the winning chain +
   baseline before claiming the +0.023 F1 gain (McNemar p=0.327 so
   far); screening used default hyperparameters only.
7. §13 numbers stale on TWO counts (gotcha #18 + site-token hygiene) —
   full re-run needed; bench re-baseline too.
8. Known-remaining audit lows deferred deliberately (see §20 "not
   fixed" list).
9. 3SSE winner's honest nested F1 ≈ 0.56 vs optimistic 0.839 (§21):
   deployed as-is per user decision (both numbers shown on card +
   welcome banner); re-run Honest check after any retrain so the saved
   `nested_honest_f1` matches the saved winner. GUI patient banner
   verdict for MARGIN bundles still mixes a patient's normal+tumor
   sites into one mean (§21 verification note) — margin-mode patient
   rollup semantics deserve their own pass someday.

## 19. Research roadmap changelog (see raman_app/RESEARCH.md)

**Stage 0+1 (2026-09-04, after the deep study of the Bidipta Rana
internship report + ~120 sources):**
- `biochemistry.REPORT_BANDS`: the report's Table 4.1 (19 significant
  wavenumbers + assignments) as a literature reference for agreement
  checks.
- Wavelet denoising grew up (`preprocessing.wavelet_denoise`):
  threshold rules **universal|bayes|sure** (BayesShrink Chang 2000,
  SUREShrink Donoho-Johnstone per-level), shrinkage **soft|hard|garrote**
  (Gao 1998), **cycle-spinning** (±N shifts, translation-invariant).
  Measured on synthetic Lorentzians: bayes/sure cut RMSE ~23% below the
  old fixed universal rule. coif5 back in the GUI combo (report's
  wavelet). Params: `wavelet_threshold/wavelet_mode/wavelet_cycle`
  (defaults preserve old behavior — no PARAMS_VERSION bump needed).
- Baselines +: **iarpls / pspline-arpls / snip** via installed
  pybaselines (`_pybase`); norms +: **area / minmax**; **detrend**
  (linear) step; **Phe-1003 wavenumber calibration**
  (`estimate_wn_drift`/`calibrate_wn`/`wn_calibrate` param —
  icoshift-style single anchor; literature: axis alignment beat every
  intensity normalization).
- `modeling.TTestSelect` (Welch t + Cohen's d, p<0.05 & |d|>0.1, the
  report's filter — binary, fallback=all, leakage-safe inside CV
  pipelines) and `modeling.PLSScores` (PLS LV scores as features;
  binary uses 1-D codes so 5 components work on 2 classes, multiclass
  one-hot).
- **`reproduce_report.py`**: the report's exact pipeline (crop 700–1800,
  ASLS, vector norm, coif5/SURE/soft, t-test filter, PCA(5)/PLS(5), its
  7 models with its tuned XGB params) run BOTH spectrum-level 5-fold
  (report protocol) AND patient-grouped 5×3 + Friedman (honest
  protocol); writes `report_replication.md`. **Result on this cohort:
  ALL models land at F1 0.52–0.55, Friedman p=0.44 — the report's
  PLS+XGB 0.733 does not survive grouped CV here; PLS+XGB is still
  best under protocol A (0.549) — direction consistent, magnitude not.**
- GUI Preprocess page: new combos/spins (threshold rule, shrinkage
  mode, cycle shifts, iarpls/pspline/snip, area/minmax, detrend,
  Phe-1003 align) wired through read_params/_apply_params; pipeline
  chip shows the rule/mode.
- tests: `test_adaptive_wavelet_and_new_steps`,
  `test_ttest_select_pls_scores_report_replication` — suite 49/49,
  gui_test PASS.

**Stage 2 (2026-09-04, model arsenal):**
- Registry 17→**21** (SUITE_VERSION 5→6): **Sparse PLS-DA**
  (loading-based top-k selection, Lê Cao 2008), **PLS + XGBoost**
  (the report's winner), **PCA + XGBoost**, **t-test filter + XGBoost**
  (report's Welch/Cohen-d filter inside the pipeline — leakage-proof).
  All flow into 3SSE automatically (SLOW_MODELS untouched).
- `modeling.PLSScores`: PLS LV scores as features (binary → 1-D codes
  so 5 components work on 2 classes; multiclass → one-hot).
- 1D-CNN gains ViT-parity augmentation in-fit: noise 5%, roll ±2,
  mixup β(0.2,0.2), SpecAugment band-mask (CNN1DClassifier(augment=True)
  — changes CNN CV numbers vs pre-stage runs).
- Deep uncertainty: `CNN1DClassifier.predict_proba_mc` (MC-dropout),
  `CNNEnsemble` (n-seed mean + `predict_proba_std` disagreement);
  `vit_train.mc_dropout_predict` + `gradient_saliency` for the ViT;
  ViT augmentation gains SpecAugment masking.
- `modeling.lorentzian_synthesize`: physics-safe synthetic spectra
  (convex blends + smooth gain/tilt perturbations; fold-internal only).
- PR curves: `modeling.pr_points` + `plotting.plot_pr`; the Result
  page and the diagnostics ROC panel are now 1×2 ROC+PR figures;
  AUPRC logged.

**Stage 3 (2026-09-04, uncertainty & explainability):**
- Conformal prediction (hand-rolled split-conformal LAC, no MAPIE):
  `clinical.conformal_q/conformal_sets/conformal_metrics`; Predict and
  Result tables have a **"Set 90%"** column (two classes = ambiguous,
  "∅ abstain" = INDETERMINATE); coverage/abstain logged after training.
- `modeling.permutation_auc_p`: patient-level label-shuffle null
  (50 perms × 3-fold) → empirical p for the winner's OOF AUC; new
  "Permutation AUC" diagnostics button (classical winners; CNN too
  slow). Real synthetic signal → p≤0.07; pure noise → p>0.1 (tested).
- Calibration quality: `clinical.ece` + `clinical.isotonic_compare`
  (ECE raw/Platt/isotonic logged; Platt stays applied at n<500).
- Deep/PLS explainability: `CNN1DClassifier.grad_cam` (1-D Grad-CAM,
  forward+full-backward hooks on the last Conv1d — grad_OUT not
  grad_in), `modeling.pls_vip` (per-component SSY VIP formula),
  `modeling.winner_importance` dispatcher, ViT `gradient_saliency`.
- `biochemistry.agreement_report`: top-k importance vs literature
  BANDS + REPORT_BANDS (match fractions + rank correlation); new "Band
  agreement" diagnostics button (also in the auto battery) logs
  matches per wavenumber.
- Patient verdicts upgraded to **probability-mean** (Jeng 2019:
  mean P(positive) over ALL the patient's spectra, ≥0.5 rule;
  majority-vote counts still displayed) — `_aggregate_patients` now
  groups by index so `self._pred_probs` aligns.
- tests: `test_stage3_conformal_permutation_explainability` — 51/51,
  gui_test PASS (pred_table now 5 columns).

**Stage 4 (2026-09-04, clinical/strategic):**
- `clinical.auc_power` (Hanley-McNeil variance): detectable AUC at 80%
  power + n-per-group for a target — used in the model card and
  validate_external.
- `modeling.export_model_card`: TRIPOD+AI-flavored card (data,
  preprocessing, protocol, F1/AUC/AUPRC + power, limitations) written
  automatically as `<model>_card.md` on every Save best model.
- Saliva mode: `PreprocessParams.saliva()` preset (crop 400–2300 so the
  thiocyanate band is measured, despike, SNIP, SNV — "🧪 Saliva preset"
  button on the Preprocess page); `bio.SALIVA_BANDS`,
  `bio.thiocyanate_index` (2100–2136 QC/discriminative marker).
- `bio.BANDS` now includes the carotenoid pair (1155/1520, fall in
  cancer) — the classic oral-cancer narrative was missing.
- `bio.biochemical_shift`: class-mean band deltas + Welch p (top-3
  logged after every data load); `bio.dataset_qc` (median SNR, spike
  proxy, thiocyanate availability) logged per data state.
- `modeling.covariate_fusion_cv` + "Fuse covariates (CSV)" diagnostics
  button: patient-keyed CSV → logistic meta-model [p, covariates] vs
  [p] under grouped CV; ΔAUC logged (Hanna 2024 open gap).
- `validate_external.py`: external-cohort CLI for any saved bundle —
  metrics + AUC power + band-profile PSI drift verdict vs the internal
  folder.
- Predict page: **"⏺ Live folder"** toggle — QFileSystemWatcher on the
  spectra folder, 800 ms debounce, auto re-classify (real-time
  screening station; inference sub-100 ms).
- Threshold/prevalence sweep: already covered by the static PPV/NPV ×3
  prevalences + DCA panels (interactive slider deliberately not added).
- tests: `test_stage4_power_card_fusion_saliva_external` — **52/52**,
  gui_test PASS.

**Research-stage totals**: 21-model registry, adaptive denoising,
reproduce_report head-to-head, conformal triage, permutation AUC,
Grad-CAM/VIP/saliency + band agreement, patient probability-mean,
covariate fusion, external validation, model cards, saliva mode, live
prediction. RESEARCH.md holds the study + sources.

**ROADMAP v3 batch 1 (2026-09-05, see raman_app/ROADMAP.md):**
- **A1** locked one-shot FINAL EXAM in reproduce_study (--no-locked to
  skip): patient 70/15/15 split, winner refit on train patients, one
  shot on unseen test + DeLong CI. Real-data mini run: 45 spectra /
  10 patients, F1 0.505, AUC 0.504 (CI 0.32–0.68) — honest and noisy
  at this n (run the full config for the real number).
- **B1** gui_test redirects APP_DIR to a temp dir (reports/logs out of
  the tree). **B2** `.github/workflows/ci.yml` (windows, offscreen,
  test_all + gui_test + ruff, log artifacts) + pyproject ruff config;
  repo now lint-clean. **B3** LICENSE (MIT, research-only note),
  CITATION.cff. Ruff pass also fixed a LATENT BUG: sequential CLI's
  `main` referenced `_dumpable` outside its function → hoisted to
  module level.
- **J1** `modeling.label_error_report` (confident-learning flags from
  winner OOF: p_self<0.10/p_other>0.90 = likely, 0.25/0.75 = review)
  + "Label errors (OOF)" Train-page button: logs flags AND selects the
  rows in the Data table for double-click relabel. Synthetic test:
  8/8 planted mislabels caught, 0 false.
- **E3** TTA: `CNN1DClassifier.predict_proba_tta` + `vit_train.
  tta_predict` (noise+roll+mask averaged inference).
- **E7** `modeling.band_stability`: per-profile top-k Jaccard +
  consensus stable-band mask (which bands are REAL across folds).
- **J2** `modeling.phantom_cohort`: ground-truth synthetic OSCC cohort
  (planted 785/1003/1090/1335 up, 1155/1520 down in tumor; patient
  random effects, baseline humps, spikes) — CI smoke + explainability
  ground truth; test asserts planted bands dominate a diff profile and
  PCA+LDA scores F1>0.9 on it.
- **A2** no new code needed: 3SSE significance_of already computes
  5-seed stability + McNemar (persisted since commit e108167); the
  open item is an execution run, not code.
- **B7+C4** live mode is now truly INCREMENTAL: `_live_fire` classifies
  only mtime-new files via `predict_with_bundle`, appends rows, and
  shows measured ms/spectrum — no more folder-wide re-runs.

---

## 20. 2026-09-06 WHOLE-PROJECT DEEP AUDIT + REMEDIATION (B0–B4)

Three-agent adversarial audit + line verification: **~190 findings**
(16 CRITICAL / 35 HIGH / ~67 MED / ~72 LOW). Commits `011bb58`,
`2267d71`, `ea00383`-family, `432cbd3`, `434bcb9` + audit-B1 commit.

### Number-changing semantic fixes (re-run anything older)
1. `_tune_hyperparams` grouped branch had a NameError → **B1 chain
   tuning NEVER RAN in paired mode**; fixed + refits best params on the
   full fold. `tune_chain` now wraps wn-aligned layers with
   `_SpectralSlice` (band-feature winners aborted tuning before).
2. `calibration_bins` interleaved → contiguous; **every ECE /
   reliability / Platt-vs-isotonic number ever printed was invalid**
   (flattened to |mean p − prev|).
3. `auc_power` divided by √2 → power claims were ~1.4× optimistic
   (model cards + validate_external).
4. `CalibratedSVC` calibration now patient-grouped (precomputed SGKF
   splits) — PCA+SVM probabilities change.
5. 3SSE screening: triple ranking/cutoff on `f1_mean` (mean per-fold
   F1) — the abandon bound's true objective; patient truth = dominant
   class; `validate_arch` no longer falls back to in-sample features
   (degenerate-inner-fold archs now FAIL and are recorded as errors).
6. `wn_calibrate` anchors on the PHYSICAL Phe-1003 position (constant)
   instead of the cohort median — leakage-free and predict-time-real.
7. bench `patient_level_f1` = per-class-conditioned patient verdicts
   (old version averaged a paired patient's Normal+Tumor probabilities
   against the first row's label — reorder noise).
8. `lorentzian_synthesize` documented honestly: convex same-class
   blends + gain/tilt, NO Lorentzian fitting (never was).

### Data hygiene
`site_token_dropped` (§5) — the site-token TH-in-Normal mislabel was
live in training. Patient-spectrum history PURGED 2026-09-06
(filter-repo); ID strings scrubbed from README/docstrings/tooltips/
tests (a light re-scrub of Brain.md §5 examples ran before the first
GitHub push, 2026-09-06).

### GUI fixes
on_train_failed re-enables ALL buttons; `_pop_diag_queue` in
try/finally; SeqSearchWorker import-try restructure; start_training
guards opt/pred/seq/lab workers; honest-check guards the analysis
chain; locked-eval dataset-consistency guard (`_lc_data_key` tag —
deep_test hand-built scenarios must set it); label-error review maps
OOF rows via `_train_row_map` (None in averaged/paired modes — logs
filenames instead) and accumulates row selections; Model Lab: PQN
winner propagates (mode switches to Margin+PQN), own out-dir
`study_run_lab/` (no longer clobbers study_run_3sse), ⚡→⏹ cooperative
cancel; live mode refuses paired bundles without a reference;
`predict_with_bundle` rejects axes that don't span the stored grid
(interp would fabricate); reset_session clears `_train_row_map`.

### Reports/packaging truth
"patient-grouped CV" statements conditional on groups actually being
present (txt + HTML + literature table + TRIPOD section); model card
records the REAL folds/repeats/grouping + observed-AUC power label;
freeze_study/report record TRAINING-time params (`_params_at_train`) +
tz-aware timestamp + per-file guarded hashing; reproduce_report
Friedman now uses PATIENT blocks (repeats were correlated blocks);
README/ROADMAP truth pass; vit_model imports without torch;
install.bat = pinned requirements.txt; main.py faulthandler +
startup_error.log + visible failure box; session.log handler guarded;
CI committed (§17).

### Not fixed (deliberate, LOW + scoped-out)
`_STAGE_CACHE` cross-thread benign race; env setdefault BLAS pinning
after LOPO; 3SSE parent RAMAN_DEVICE=cpu session leak; OOF
repeat-averaging feeding DeLong (documented methodology tradeoff);
PLS-DA pseudo-probs; padded-interpolated regions (user decision);
conformal exchangeability caveat; GUI-thread freezes on heavy
report/freeze operations; Qt6 enum stragglers outside hot paths;
deep_test tempdir leakage; single-group SGKF crashes (raise cleanly
now but no friendly dialog). Gotcha numbering has legacy duplicates
(#21/#22/#23 reused) — NOT renumbered to keep old references stable;
new entries continue at #24.

## 21. 2026-09-06 DEPLOY-PATH HOTFIX — "3SSE calls every Normal patient Tumor"

User report: the saved 3SSE winner (PCA+XGBoost → Ensemble top-3, paired,
card `model_3SSE__PCA___XGBoost___Ensemble__top-3__card.md`) predicted
Tumor p≈0.94–0.97 for EVERY file incl. all NH (session.log 11:54/11:56).
Root cause was NOT statistical overfitting alone — the deploy path was
broken. Four fixes, no retraining (bundle unchanged):

1. **wn_calibrate train/deploy mismatch (root cause)**: training ran
   `calibrate_wn` (Phe-1003) before crop; `predict_with_bundle` +
   `paired.reference_vector` never did → all deploy features OOD →
   saturated single-class output. Fixed via new shared
   `preprocessing.align_to_grid` used by both call sites (gotcha #24).
2. **Silent absolute-spectrum predictions in margin mode**:
   PredictWorker now SKIPS files whose patient has no Normal reference
   (log line "SKIPPED …" + message box); never feeds a margin model an
   absolute spectrum silently.
3. **Patient verdict rule**: `_aggregate_patients` cut at hardcoded 0.5
   on CALIBRATED probabilities (≡ raw 0.41 — far more liberal than the
   validated 0.663-raw/0.7505-calibrated operating point). Now uses the
   bundle's stored calibrated threshold (fallback 0.5 only without one;
   the pmax-based fallback branch keeps legacy 0.5).
4. **Double-Platt in `_threshold_now`**: loaded bundles' threshold is
   already calibrated (`save_bundle` maps it) but was Platt-mapped again
   for display (0.7505 → 0.826). Winner branch still maps (its
   threshold is raw). `_op_points_now` audited: correct as-is.

Tests (59/59 pass, ruff clean): `test_paired_deploy_parity_wn_calibrate`
(feature + probability parity, negative control proves non-vacuous),
`test_paired_predict_skips_without_reference`,
`test_patient_verdict_uses_bundle_threshold` (also pins no-double-Platt).

Honesty upgrades (user decision "show both numbers"): model card gains
per-class sens/spec/F1 + the honest (nested) macro-F1 next to the
optimistic one; `save_bundle` persists `per_class` + `nested_honest_f1`;
welcome banner shows nested F1 when the Honest check ran. The EXISTING
3SSE card got a hand-written "Honest performance context" + "Deploy-path
fix verification" addendum (its per-class CV numbers were never
persisted; bundle now stores them for future saves).

Verification (`raman_app/verify_deploy_fix.py`, dev tree 341 files,
5 tumor spectra correctly skipped for missing reference):
spectrum-level sens 0.749 / spec 0.941 / acc 0.836 (was: everything
Tumor, spec≈0); margin rollup at 0.750-calibrated cut: 58/74 patients
with ≥1 tumor site flagged, 5/75 with a normal site flagged. Residual
imperfection = the model's honest quality (nested F1 ≈ 0.564, McNemar
p=0.629 vs paired PCA+SVM), not the deploy path. NOTE: the verification
includes hygiene-excluded files (spike-flagged etc.), and the GUI
banner's one-class-per-patient verdict is a poor yardstick for margin
data (patients have BOTH site types) — open item 9.

## 22. 2026-09-06 (evening) — 3SSE diagnostics crash + margin auto-reference never worked

User report after the §21 fixes: (a) no 3SSE learning-curve graph —
learning curve / seed stability / noise robustness / locked eval / LOPO
ALL failed; (b) predict now said "None of the files could be predicted:
… no patient-specific NORMAL reference was found" (my §21 guard doing
its job — but for EVERY file).

### (a) sklearn clone contract (5 diagnostics dead)
All five diagnostics `clone(winner.pipeline)`; `SequentialChain` /
`AveragedChain.__init__` did `self.estimators = list(estimators)` —
sklearn requires __init__ to store params VERBATIM (clone's post-init
identity check fails on the copied list → "Cannot clone object
AveragedChain"). Fixed: verbatim storage, `get_params` returns the same
object. `test_3sse_chain_cloneable` pins clone + learning_curve on a
chain. NOTE: diagnostics fit clones WITHOUT groups (ungrouped chain-OOF
inside) — pre-existing behavior, unchanged deliberately.

### (b) auto-reference NEVER ran in the GUI — the "" manual ref
`start_predict`: `manual_ref = ref_path_edit.text().strip()` — an EMPTY
edit stays `""`, and PredictWorker enabled per-patient auto references
only `if manual_ref_dir is None`. `""` disabled BOTH paths →
`_auto_reference` never called → ref=None for every file. Combined with
§21's discovery: EVERY past GUI paired-mode prediction ran on ABSOLUTE
spectra (proof: "Margin mode: built N…" never appears in any
session.log; yesterday's all-Tumor p≈0.95 garbage). Fixes:
`text.strip() or None` in start_predict + worker checks `not
self.manual_ref_dir` (falsy, belt-and-braces).

### Hardening shipped with it
- `_auto_reference` no longer swallows exceptions (`except: pass` →
  per-file reason dict; run log prints "Margin references: N unresolved
  — first: <file>: <reason>"): no-tree / no Normal/<patient> folder /
  reference-build exception are all visible now.
- Manual reference = tree ROOT (user hit this 11:59: "No reference
  spectra could be loaded") → actionable ValueError ("…looks like a
  clinical tree root — pick the patient's NORMAL folder or clear it").
- `test_auto_reference_real_layout`: real layout through the real
  PredictWorker — paired patients predict with auto refs (n_auto_refs
  counted), tumor-only patients skip with logged reason, "" behaves as
  None, tree-root manual ref fails with the actionable message.

### Verification (real tree, saved Extra Trees winner, Tumor-only
selection — the exact 13:32 failing run): 180/185 predicted, 73 auto
references built, 5 skipped with "no Normal/Patient_{21,28,34} folder",
tumor-site probabilities 0.53–0.62 (healthy, unsaturated).
`verify_worker_refs.py` replays it headless. Suite 61/61, ruff clean.

## 23. 2026-09-06 (night) — wave 3: save crash, winner CSV, view-saved, batched predict, cancelable diagnostics

User reports after wave 2 (app NOT yet restarted in between — several
symptoms were pre-fix code): 3SSE bundle save crash, winner-tab CSV
export refused, "View saved 3SSE" found nothing, prediction "too much
slower", and every click answered "Another analysis is still running".

1. **save_bundle SimpleNamespace crash** (wave-2 regression): bare
   `winner.per_class` — the 3SSE dialog (gui:7243) and Model Lab
   (gui:5580, sequential:1465) build namespace winners without it.
   Fixed: `getattr(winner, "per_class", None)`
   (`test_save_bundle_namespace_winner`).
2. **Winner-tab CSV export**: `_export_csv` refused the Overall Winner
   tab; now exports winner + nested-best-per-level rows + significance
   note in the ranking-table format (`test_3sse_dialog_winner_tab_csv`).
3. **View saved 3SSE found nothing**: GUI searches never persisted
   screening.jsonl/validated.json/winner.json (CLI-only), and Model Lab
   writes study_run_lab. Fixed: `MainWindow._persist_3sse_payload`
   (numpy/tuple/NaN-sanitized; chain object excluded) on every GUI
   search/Model-Lab completion; `_latest_3sse_run_dir()` picks the most
   recent of study_run_3sse/study_run_lab by artifact mtime;
   view_saved_3sse degrades gracefully to winner-only and the message
   names both dirs. NOTE: the CURRENT study_run_3sse (significance.json
   + winner.joblib only) still yields "no run" until the next GUI
   search — restore_last_3sse deliberately still reads only
   study_run_3sse (Lab provenance protection, 2026-09-05).
   (`test_3sse_persist_and_latest_dir`.)
4. **Prediction 20 min → 16 s**: profiling showed the Extra Trees
   winner's Ensemble contains a TabPFN voter costing ≈6.5 s PER
   predict_proba CALL on CPU (references/preprocess were already
   negligible: 14-20 ms/patient, <1 ms/spectrum warm). New
   `modeling.predict_with_bundle_many` batches the whole folder into
   ONE call per feature-length group (PredictWorker: read → batch →
   assemble phases). Measured batch-vs-single equivalence: max |Δp|
   ≈ 8e-8 (float noise); replay of the 185-file Tumor run: identical
   predictions, 180 predicted / 5 skipped / 73 refs, 16 s wall.
   (`test_batched_bundle_prediction_equivalence`, incl. per-row error
   isolation for bad axes.)
5. **Cancelable diagnostics**: the wave-2 clone fix made the
   auto-diagnostic chain (…→LOPO→honest) genuinely run — LOPO on an
   AveragedChain winner is hours and the busy box was information-only.
   Now: `_analysis_busy_box` names the job + elapsed and offers Cancel;
   `cancel_check` (cooperative, boundary-granular) added to
   learning_curve_by_groups / seed_stability / noise_robustness /
   lopo_evaluate (raises "diagnostics cancelled by user"; `_run_async`
   treats it as a quiet cancel, chain keeps draining);
   `_diag_cancel.clear()` on every chain/_run_async start (stale
   cancels can't kill new jobs); `_honest_worker` cleared in
   done+failed (`test_diagnostics_cancel_check`). Restart required for
   all of this to load.

Suite 66/66, ruff clean. GOTCHA: restart the app before retesting —
three waves of fixes today are on disk but an old process keeps old
code.

## 24. 2026-09-06 (late) — wave 4: view-saved KeyError('f1') + auto-diagnostics too slow on chain winners

1. **KeyError('f1') viewing a saved 3SSE run** (gui traceback at
   SeqResultsDialog._rows): the error-tolerant search records FAILED
   archs as {"arch","level","error"} with NO metrics; my wave-3
   persistence wrote them to screening.jsonl and the reload gave every
   record a metrics key (possibly empty), so the ranking sort crashed.
   Fixed at three layers: _rows only ranks entries whose metrics dict
   has a numeric f1 (.get defaults everywhere incl. the row formatter);
   view_saved carries the "error" through instead of fabricating
   metrics; _persist_3sse_payload writes the error field. Winner pills
   / summary lines / winner-tab CSV all use .get(NaN) now — a partial
   winner.json can no longer crash the dialog.
2. **"Training and graph showing slow"**: post-clone-fix the auto
   diagnostics battery genuinely runs; on CHAIN winners every refit
   multiplies the inner fits (~45 pipeline fits per refit → learning
   curve ≈ 900 fits, LOPO potentially hours; plain winners: whole
   battery ≈ 2 min, measured 16:49 session). Fixes: (a) Train-page
   checkbox "Auto-run after training" (default ON) — untick to skip the
   post-training battery entirely; (b) `_is_chain_winner()` guard in
   run_all_diagnostics: for Averaged/Sequential chain winners the AUTO
   chain runs only regions + band agreement + honest (winner-
   independent ~20 s) and logs which heavy items were skipped (manual
   buttons + busy-box Cancel remain); the manual "Run all diagnostics"
   button still runs the FULL battery for any winner.
3. Test-isolation lessons: (a) every Qt-widget test must set Fusion +
   STYLESHEET before constructing (vista+QSS fail-fast — the dialog
   test segfaulted 127/139 without it); (b) isolated MainWindow tests
   must clear `win.winner` — the startup restore now finds
   study_run_3sse/winner.json (persisted by wave 3) and installs a real
   winner (broke the no-winner branch of a threshold test).

Suite 67/67, ruff clean. Side-proof in test output: the startup restore
loaded the persisted 3SSE winner from the user's 16:49 session — the
wave-3 persistence works end-to-end.

## 25. 2026-09-06 (final) — wave 5: proactive audit (3 Explore agents) fixes

User asked to hunt remaining issues before calling it solved. Findings
+ fixes (70/70 tests, ruff clean; old-bundle verify numbers UNCHANGED
after the predict refactor — sens 0.749/spec 0.941/acc 0.836):

1. **Band agreement dead since days** (live TypeError in every
   diagnostics run): `modeling.winner_importance` called
   `region_importance_shap(X, y, wn)` — wn bound to the groups slot.
   Fixed: `(X, y, None, wn)`. `test_winner_importance_runs` pins it.
2. **PQN deploy parity (same class as the wn_calibrate bug)**:
   Margin+PQN trains on `pqn_normalize(s, ref) − ref`; every deploy
   path computed `s − ref` and NO bundle flag recorded PQN. Fixed:
   `save_bundle(pqn=)` stores the flag; new shared
   `modeling._apply_reference` (length check + optional
   `pqn_normalize` + subtract) used by BOTH predict paths; flag passed
   at Model Lab (`payload["use_pqn"]`), GUI saves (`self._pqn_mode`
   captured when mode_kind == "paired-pqn"), seq CLI, reproduce_study;
   startup restore reads it. `test_pqn_deploy_parity` proves deploy ==
   training construction for flagged bundles AND that unflagged
   bundles differ (non-vacuous).
3. **Saved-run viewer None-metrics crashes** (NaN→null after JSON):
   new `SeqResultsDialog._num` (isinstance numeric else NaN) used in
   pills / nested max / winner text / CSV; view_saved's per-line try
   now also covers the level/arch access (malformed jsonl lines skip).
4. **Persistence/restore provenance**: GUI '3SSE search' now also
   saves study_run_3sse/winner.joblib (restarts used to pair the NEW
   winner.json with the PREVIOUS run's stale pipeline silently);
   restore verifies bundle model_name == winner arch (skip + log on
   mismatch), warns visibly when winner.joblib is missing/unloadable
   and disables Save (metrics-only restore used to pickle
   pipeline=None); `_persist_3sse_payload` CLEARS stale
   validated/winner/significance files for absent sections and logs
   failures (no more silent except); restore uses persisted `classes`
   (3-class reshape crash) and bails on null f1 BEFORE formatting;
   sequential.py `persist_run` actually writes validated.json/
   winner.json now (dead code before — CLI runs left the previous
   run's artifacts behind).
5. **Log-less failures**: data_report.txt write failure and
   optimize.save_best_params failure now log a line each.

GOTCHA: PQN bundles saved BEFORE this wave have no `pqn` flag — if a
Margin+PQN winner was saved earlier, re-save it (or retrain) so the
flag travels; prediction of an old PQN bundle silently skips PQN.

## 26. 2026-09-06 (wave 6) — regression: persist_run ndarray TypeError failed COMPLETED searches

session.log 18:02:28: "3SSE search failed: TypeError: Object of type
ndarray is not JSON serializable (oof_proba in validated metrics)".
Wave-5's persist_run started writing validated.json/winner.json, but
the GUI worker's (and CLI's) validated/winner metrics carry
oof_proba/cm/y_true as ndarrays — `_num` handled np scalars and
lists, NOT ndarrays. The worker's guard caught only OSError, so a
PERSISTENCE error aborted the search AFTER screening+validation+
finalize+significance had all finished (and skipped the checkpoint
cleanup + done.emit).

Fixes: (1) `_num` converts ndarray → tolist; (2) SeqSearchWorker's
persist guard broadened to Exception + FILE_LOG.exception —
persistence NEVER fails a completed run; (3) CLI main() wraps
persist_run the same way (report/meta/winner.joblib still written).
`test_persist_run_handles_ndarrays` pins oof/cm/y_true + NaN→null.
71/71, ruff clean. LESSON (same as _auto_reference/_persist): any
post-result persistence step must degrade with a log line, never
propagate — a finished computation's value must not be hostage to its
bookkeeping.

## 27. 2026-09-06 (wave 7) — chain-winner diagnostics UX: frozen "computing…" graphs

User report: Seed stability + Noise graphs "not showing after waiting
too long" on the Extra Trees → PCA+SVM chain winner. Log proof: on the
previous PLAIN winner both completed in seconds (18:12); the chain
guard correctly skipped them in the auto-chain (18:18:32 log line);
the user then started Seed stability MANUALLY — 25 chain refits
(≈900 inner pipeline fits) with ZERO feedback: panel stuck on
"computing…", Cancel unreachable (busy box only appears when clicking
ANOTHER analysis button).

Fixes: (1) `progress` callbacks in seed_stability (per seed),
noise_robustness (per fold), learning_curve_by_groups (per fraction),
lopo_evaluate (per patient — also in the serial branch); GUI runners
pass `self._emit_progress` (worker-thread → FuncWorker.progress →
`_analysis_progress` on the GUI thread: "Seed stability — seed 2/5:
F1 0.61" in train_status + statusBar). (2) Always-visible-while-
running "⏹ Cancel analysis" button in the diagnostics row (shown at
_run_async start, hidden in finish/fail; sets _diag_cancel). (3)
`_confirm_heavy_diag`: manual clicks of LC/seeds/noise/LOPO on a
chain winner get a cost heads-up (Yes/No; auto-chain behavior
unchanged). 72/72, ruff clean.

## 28. 2026-09-06 (wave 8) — junk file in the manual reference folder killed the run

session.log 21:21:27: reference folder = Data\Normal (class folder) →
ref_files=[split_log.txt] (matches .txt, unparseable) →
paired.reference_vector raised on the FIRST bad file → whole run died
as generic "Prediction failed". The wave-3 tree-root guard only fired
when the folder had ZERO matching files.

Fixes: (1) reference_vector is junk-tolerant — per-file
ValueError/OSError skip; raises only when NOTHING loads, naming the
skipped files + reasons (also hardens _auto_reference against stray
junk in patient folders); (2) the worker's manual-ref branch wraps the
call and raises an actionable message ("CLEAR the reference field to
auto-find per-patient references… or pick the folder that DIRECTLY
contains one patient's NORMAL spectra"); (3) on_predict_failed now
surfaces the exception line in the dialog instead of pointing at
session.log blindly. `test_reference_vector_junk_tolerance` pins all
three. 73/73, ruff clean.

## 29. 2026-09-06 (wave 9) — manual normal-reference input REMOVED

User decision (after being shown that paired mode itself is what makes
the models work): remove ONLY the manual reference-folder input; per-
patient AUTO references from the clinical tree are the one path.
Rationale: the manual field only ever produced mis-picked folders
(tree roots, class folders, junk .txt) that failed runs — three
separate user-facing errors in one day.

Removed: Predict-page reference row (label/edit/Browse +
_set_bundle visibility), browse_reference, start_predict manual_ref
logic, PredictWorker.manual_ref_dir + the whole manual branch (tree-
root trap, junk guidance). Live mode now honestly says it is not
available for margin models (the manual input was its only reference
source). KEPT: paired.py, _auto_reference, reference_vector (junk-
tolerant), bundle["paired"], Model Lab, sequential paired modes —
all trained models unchanged. deep_test updated (new "Clinical tree
required" dialog title); junk-in-patient-folder tolerance pinned in
test_auto_reference_real_layout. 73/73 + 19/19 deep checks + ruff.

## 30. 2026-09-06 (wave 10) — full-project automated test pass

User-requested complete test audit. Inventory (2 Explore agents): 73
unit + 19 deep + 9 gui checks existed; gaps = 6 modeling/preprocessing
helpers, settings persistence (always stubbed), report writers,
plotting helpers, ui_helpers widgets, live mode, 3SSE GUI completion,
diag queue/cancel flow, failure callbacks, freeze/figures real paths,
TrainWorker success path. Added 13 tests (test_all 73→86) +
`_qt_app_styled`/`_isolated_main_window`/`_stub_native_dialogs_once`
harness helpers + `_flat_folder_td`.

Bugs found & fixed:
1. plotting.plot_sign_bars: %-style fmt inside f-string
   {v:{fmt}} → ValueError on EVERY call (dead code, never run before)
   — both orientations; now `fmt % v`.
2. gui._draw_winner_plots crashed on cm=None (metrics-only winners,
   reachable via restore) → guard + log-skip.
3. gui.on_seq_done left oof_proba as a raw list → downstream plot
   crash → np.asarray (restore already converted; parity restored).
4. deep_test dialog stub returned ("", "") for getExistingDirectory
   (single-string API) → truthy tuple → EVERY figure export in deep
   scenarios silently failed ("expected str... not tuple") — stub
   returns "" now.
5. Harness lesson (not product): an UNREFERENCED QApplication gets
   GC'd → later widget construction goes qFatal (exit 127/139) —
   _qt_app_styled result must be HELD; native modal dialogs in
   offscreen runs corrupt subsequent window construction → stubbed
   once per process (deep_test pattern).

Remaining (documented, not fixed): TrainWorker's QThread success path
natively crashes deterministically in test processes (gotcha #16/#22
family — even fresh subprocesses; guard/error paths ARE covered by
deep_test) → the real compute chain is tested synchronously instead;
sequential.py CLI main() has no end-to-end test (runtime); no line-
coverage tool installed (functional coverage 136/151 public functions
= 90%); chain_factory remains dead code (now pinned by nothing —
candidate for deletion).

Final: 86/86 unit + gui_test PASSED + 19/19 deep + ruff clean.

## 31. 2026-09-06/07 (wave 11) — the 5 recommended improvements, implemented

1. **CI coverage floor**: `coverage==7.16.0` installed in CI; unit step
   now `coverage run --source=. test_all.py` + `coverage report
   --include=<15 app/library modules> --fail-under=60`. Measured
   locally: 65% library line coverage (63% overall incl. scripts).
   Rule: raise the floor as coverage improves, never lower it.
2. **deep_test.py in CI**: new step after gui_test (same offscreen env
   + HF cache). 19 scenario checks now gate every push/PR.
3. **QThread-training crash ROOT-CAUSED** (bisection ladder, each step
   3x): bare QThread+sklearn STABLE → +project imports STABLE →
   +evaluate_models STABLE → +MainWindow WITHOUT startup restore
   STABLE (3/3) → WITH restore CRASHES. The startup 3SSE restore
   (unpickling the saved AveragedChain during __init__) poisons later
   QThread training in synthetic drivers — native 0xC0000005. NOT
   loky (JOBLIB env was a 1-run fluke; env runs later crashed 3/3),
   NOT RF n_jobs, NOT BLAS threads, NOT the Qt event pump (crashes
   with pure sleep too). Interactive app unaffected (user trained all
   day with restores between). Mitigations: JOBLIB_MULTIPROCESSING=0
   set at gui import (honest comment: gotcha-#16 hardening, NOT a fix
   for this); test now exercises the REAL QThread chain in a fresh
   subprocess with a clean APP_DIR (restore skipped) —
   `test_trainworker_success_path` covers sync + QThread paths.
4. **Dead code**: `sequential.chain_factory` deleted (zero refs).
   `plot_sign_bars` KEPT as a pinned plotting utility (no caller
   invented — UI scope creep).
5. **`test_sequential_cli_smoke`**: sequential.py CLI end-to-end on
   `_make_clinical_tree` synthetic data (--models subset, --top 1,
   --skip-validation → seconds; asserts exit 0 + screening.jsonl/
   report.txt/run_meta.json; note: --skip-validation skips finalize,
   so no winner.joblib by design).

Suite now 87/87 + GUI PASSED + 19/19 deep + ruff clean + coverage
65%/floor 60.

## 32. 2026-09-07 — workspace cleanup (portable-copy ready)

User goal: slim D:\BARC to exactly what the project needs to RUN on
another device. Reclaimed ~12.6 MB (34 → ~21 MB). App behavior is
unchanged — everything deleted is either auto-regenerated or was
never read at runtime.

DELETED:
- Caches/scratch: `__pycache__/`, `.ruff_cache/`, `_i.txt`,
  `report_replication.md`, empty `study_run/` dir,
  `study_standard_v2.log` (0 B), `session.log.1` (session.log itself
  was locked by a running app instance — delete after close; the app
  recreates it anyway), `settings.json` (local UI state, regenerates
  with defaults — but the running app rewrites it on close; delete
  again before copying).
- Stale artifacts (numbers live here §13/§16): `study_run_paired_v2/`
  (2.5 MB incl. old winner.joblib), `study_run_standard_v2/`,
  `data_report.txt` (rewritten per data load), `study_manifest.json`
  (patient IDs — privacy, regenerated on demand),
  `vit_outputs/vit_test_confusion_matrix.png` + `vit_test_report.txt`
  (kept `preprocess_best.json` — vit_train --preset best reads it),
  `study_run_3sse/archs.jsonl` (7.2 MB resume checkpoint, §15),
  `Data/*/split_log.txt` (test junk, skipped by all loaders).
- Git-tracked junk (one commit, history preserved): `err2.log` (empty,
  accidental), `walkthrough-raman-data-pipeline.html` (generated),
  the 4 `model_*_card.md` files (unstaged deletions finalized — the
  §21 card addendum text survives in git history).

KEPT (runtime-required): all source, `Data/` (auto-found by
find_data_root), `study_run_3sse/` winner.json + winner.joblib +
validated.json + screening.jsonl (the on-disk set; startup
restore_last_3sse + view-saved read them — deleting = blank Train
page + Save disabled), `bench/latest.json`
(scorecard), `.git/`, `.github/` CI, docs.

Verification: import smoke test passed; 3SSE winner files present;
git clean after commit.
