# Brain.md — the project's memory

> **Read this before touching any code.** It contains everything a new
> session needs: architecture, data flow, every parameter default,
> hyperparameter grids, GUI inventory, real results, gotchas. Do NOT
> re-explore the whole tree.
>
> **Maintenance rule (mandatory):** after any non-trivial change, update
> the matching section below. Keep the "Last updated" stamp current.
> Keep it dense — tables and one-liners, no prose padding.

Last updated: 2026-08-30 (expanded to full reference detail; git `876b624` code state)

---

## 1. What this project is

Desktop app + headless CLI that classifies **SERS Raman spectra (785 nm)
of oral-cancer tissue** into `Normal` / `Tumor` (legacy demo mode: 3
classes C1/C5/C8). Research triage support for clinicians — **not** a
diagnostic device; histopathology stays the reference standard.

- **Real dataset**: `D:\BARC\Data\<Class>\<Patient>\<file>.csv` —
  341 files scanned → **317 spectra / 72 subjects kept** (68 with both
  classes = paired), after hygiene: 4 white references, 18 duplicate
  copies, 2 cross-class identical files dropped; 18 spectra spike-flagged
  (score ≥ 300). **Never commit patient data** (gitignored).
- **Demo data**: synthetic, from seed spectrum `P01_cAg_785_C8_3.txt`
  (repo root) via `dataset.generate_demo_data`. All metrics ≈1.000 on
  it — **never report demo numbers as results**.
- Single-centre, no external validation. Confounders (site,
  keratinisation, tobacco) unrecorded.
- `D:\BARC\Documatation\` (typo'd name) holds 8 reference PDFs: oral
  cancer thesis (1-70), portable Raman spectrometer design rules, Raman
  tongue cancer (biomedicines 11-01984), Raman for oncology, wavelet
  denoising, JCM 08-013131, Sahu 2016 in-vivo subsite classification,
  plus Raman.docx.

## 2. Environment & how to run

Windows 10/11, Git Bash shell, **Python 3.14.6** (system install, no
venv), PyQt5 binding (auto-detect PyQt5→PyQt6→PySide6 by import success
order, `qt_compat.py`; ~15 enum constants normalized Qt5/Qt6). Deps
pinned in `raman_app/requirements.txt`: numpy 2.5.2, scipy 1.18.1,
scikit-learn 1.9.0, pandas 3.0.5, matplotlib 3.11.1, PyWavelets 1.9.0,
joblib 1.5.3, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10,
pybaselines 1.2.1, shap 0.52.0, torch 2.13.0, PyQt5 5.15.11. Optional
deps (pybaselines/shap/torch/xgboost/lightgbm/catboost) all have
graceful fallbacks via module HAS_* flags.

```bash
cd raman_app
python main.py                                  # GUI (or run_app.bat)
python main.py --smoke                          # headless smoke test
python reproduce_study.py --demo --mini         # 30 s synthetic check
python reproduce_study.py --data D:/BARC/Data --out study_run_X        # full real study
python reproduce_study.py --data D:/BARC/Data --mode paired --out study_run_paired
python test_all.py                              # 40 unit tests
QT_QPA_PLATFORM=offscreen python gui_test.py    # GUI walk (writes result_report.txt in-tree!)
QT_QPA_PLATFORM=offscreen python deep_test.py   # 17 adversarial scenarios
```

Tests hardcode `C:\Windows\Fonts` — Linux CI would need patching (no CI
exists). `matplotlib.use("Agg")` for all headless figure writing.

## 3. File map (raman_app/, 12.2k lines)

| File | ~Lines | Role |
|---|---|---|
| `gui.py` | 5121 | 6-page MainWindow, 5 QThread workers, all slots. See §14. |
| `modeling.py` | 1346 | Model registry + nested grouped CV + bundles + importance + bootstrap/McNemar. §7-8, 12. |
| `test_all.py` | 912 | 41 unit tests incl. 4 regressions for the 2026-08-30 fixes + new-models test. |
| `deep_test.py` | 644 | 17 adversarial GUI/CLI scenarios; `wait_analysis()` helper. |
| `vit_train.py` / `vit_model.py` / `vit_test.py` | 558/207/143 | Secondary ViT track (torch). §15. |
| `plotting.py` | 439 | rcParams theme, `MatplotlibCanvas` (lazy Qt-binding pin), plot helpers. |
| `ui_helpers.py` | 391 | QSS stylesheet, `pill()`, tooltips, HOW_TO/METRIC_HELP, settings I/O. |
| `study_stats.py` | 375 | Friedman+Nemenyi, LOPO, seed/noise checks, BH-FDR band stats. §10. |
| `clinical_data.py` | 324 | `load_clinical_dataset()` → `ClinicalData` + hygiene + report. §5. |
| `reproduce_study.py` | 282 | One-shot headless study (works on real data since 2026-08-30). |
| `clinical.py` | 259 | TRIPOD layer: Platt, DeLong, operating points, triage, PPV/NPV, DCA. §9. |
| `preprocessing.py` | 254 | `PreprocessParams` + spectrum pipeline. §6. |
| `biochemistry.py` | 250 | BANDS/RATIOS tables, NMF, keratin flags, plausibility. §11. |
| `optimize.py` | 228 | Preprocessing auto-tune. §15. |
| `dataset.py` | 182 | Flat loader, `common_grid`, `to_matrix`, demo generator. §5. |
| `paired.py` | 160 | Within-patient deviation features. §4. |
| `smoke_test.py` / `gui_test.py` / `main.py` / `qt_compat.py` | small | Harnesses / entry points. |

Root: `AGENTS.md` (ponytail rules + pointer here), `P01_…txt` (demo
seed), `.gitignore` — excludes `Data/`, `Documatation/`, `_external/`,
`renders/`, `study_run*/`, `*.joblib`, `study_manifest.json`,
`result_report.*`, `settings.json`, `session.log`, `demo_data/`,
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
  `P01_cAg_785_C8_3.txt` → `"C8"`; absent → `""`.
- `load_spectrum`: comma→space, whitespace split, skips bad lines,
  **ValueError if <10 points**, sorts ascending.
- `load_folder`: non-recursive, `.txt/.dat/.csv` sorted.
- `common_grid(spectra, max_points=2000)`: lo=max of mins, hi=min of
  maxs (intersection; raises if empty), n = median length clipped
  [50, 2000] → `np.linspace(lo, hi, n)`.
- `to_matrix`: `np.interp(grid, s.wavenumbers, s.intensities)` per row.
- Demo generator: 16 peak centers
  `[480,525,640,725,830,880,1003,1095,1130,1205,1240,1335,1450,1555,1580,1655]`
  cm⁻¹, width 28; 3 classes with peak gains C1=0.45 / C5=0.95 / C8=1.55,
  15 spectra/class; noise: wavenumber shift N(0,0.35), baseline tilt
  N(0,0.01), gain jitter N(1,0.06), additive N(0, 0.03·std(peaks));
  files written descending-wavenumber, named
  `DEMO_S{i:02d}_cAg_785_{cls}_{rep}.txt`.

**clinical_data.py**
- Layout `<root>/<class>/<patient>/*.csv`; label from top-level folder
  only (filenames unreliable).
- `CLASS_SYNONYMS` (lowercase): normal/control/healthy/benign→`Normal`;
  tumor/tumour/cancer/malignant→`Tumor`; else passthrough `strip()`ed.
- `REFERENCE_MARKERS` (lowercase substrings of filename): white, black,
  dark, bkg, background, ref, reference, blank, calib.
- `subject_key(folder)`: first number in folder name → `S{int}` (so
  `Patient_15`, `TDOC015`, `TDOC015 Spectra pro` → `S15`); no digit →
  `folder.strip().upper()`.
- Duplicates: SHA-1 over float64 raw bytes of (wn, it) — immune to
  line endings. Same-class dup → keep first; hash in >1 class → **all
  copies dropped** (`cross_class_dropped`, pure leakage).
- `SPIKE_FLAG_THRESHOLD = 300.0` (median spike score of this dataset
  ~60; clean ≈5–30). `spike_score(y) = max|diff| / median|diff|`.
- `is_clinical_layout`: ≥2 subfolders containing spectra one OR two
  levels down.
- Shared grid only if all spectra match first spectrum exactly
  (`allclose` atol 1e-6); else `grid=None` → consumer falls back to
  `common_grid`.
- Report keys: `root, class_map, references_excluded,
  duplicates_dropped, cross_class_dropped, load_errors, grid_identical,
  files_scanned, spike_scores{median,p95,max}, flagged_spectra,
  class_counts, patients_per_class, patients_in_both_classes,
  n_spectra, n_subjects`.
- `patient_split(groups, labels, seed=42)`: StratifiedGroupKFold
  n_splits=7, folds sorted by size desc → fold0=test, fold1=val,
  rest=train (≈70/15/15); needs ≥7 subjects else ValueError.

## 6. Preprocessing (preprocessing.py)

**`PreprocessParams` fields + defaults** (validated copy, never mutated):

| Field | Default | Meaning / clamp in `validate()` |
|---|---|---|
| `crop_min` / `crop_max` | 400 / 1800 | 0 disables bound; min≥max ⇒ both 0 (no crop) |
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

## 8. Model registry — full hyperparameter grids (17 rows)

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
  (~25k params, CPU). Class-weighted CE, Adam 1e-3, 40 epochs, batch
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
- `calibration_bins(y, p, n_bins=8)`: equal-count bins via interleaved
  `order[b::n_bins]`; rows (mean_p, observed, n).
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

Hygiene per run: 317 spectra / 72 patients → 18 spike-flagged excluded
(despike off). Artifacts (summary.txt, 3 PNGs, winner.joblib,
run_meta.json, data_report.txt) in `raman_app/study_run_standard/` and
`study_run_paired/` (gitignored).

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
gates saved model-checkbox restore (bumped when the model list changes).
Menu: File = Open folder (Ctrl+O),
Reload (F5), Load model (Ctrl+L), Save model (Ctrl+S), Exit (Ctrl+Q);
Help = How to (F1), Metrics, About. Logging: `log()` → stdout +
`session.log` (rotating 1 MB ×2).

**Start**: 3 hero pills (live counts), progress card, quick actions —
Load data, Generate demo, **One-click demo** (`quick_start`: demo →
load → Train → `start_training`), Predict with saved model.

**Data**: `folder_edit` + Browse/Reload; `count_label`; `table` 7 cols
(File, Patient, Class — double-click editable, Points, WN min/max,
Quality "spiked"/"ok"; `cellChanged`→`on_cell_changed` writes
`self.labels[row]`, guarded by `_loading_table`); Plot selected (≤10) /
Plot class means → `data_canvas`.

**Preprocess**: crop spins (0–5000 step 50, def 400/1800); despike
chk + z spin (3–20 def 7); wavelet chk + combo (sym4/6/8, db4/6,
coif3/5) + level 1–8 def 4; SG window 5–51 def 11, poly 1–5 def 3,
deriv combo 0/1/2; baseline combo als/arpls + λ exponent 2–9 def 5,
p 0.001–0.5 def .01, iters 5–30 def 10; norm combo vector/snv/none.
Preview (2-panel) → `prep_canvas`; **Optimize preprocessing** →
`OptimizeWorker` → `_apply_params(best)` + `save_best_params`.

**Train**: `combo_mode` = Standard / Margin (vs own normal) / Margin+PQN
(idx ≥1 ⇒ paired, needs groups); `model_checks` 14 checkboxes from
`ALL_MODEL_NAMES` + All/None; `spin_folds` 2–10 def 5; `spin_seed`
0–9999 def 42; `chk_repeat` (×3), `chk_exclude_flagged` (def checked,
only if flags exist), `chk_avg_replicates` (per patient×class means).
Result card: stat banner + `compare_table` (★ winner, green ≥.90 /
amber ≥.70 / red). Buttons → `run_learning_curve`,
`run_region_importance`, `run_honest_check`, `run_locked_eval`,
`run_lopo`, `run_seed_stability`, `run_noise_check`,
`run_biochemistry`. Canvases `cm_canvas` + `roc_canvas`; biochemistry
card (`bio_canvas`, `bio_comp_canvas`, `bio_table` 6 cols); per-class
table (Class, n, sens, spec, prec, F1).

**Predict**: model card (`model_path_edit`, Load…, `model_info`,
`ref_row_widget` visible only for paired bundles); input card
(`spec_path_edit`, folder or `;`-joined files; `_prediction_files()`
walks `<root>/<patient>/` and `<root>/<class>/<patient>/`, skips
REFERENCE_MARKERS + split_log.txt); Predict → `PredictWorker`;
`pred_canvas` + `pred_table` (File, Prediction, P).

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

## 15. Secondary tracks

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

## 16. Gotchas & invariants (read before editing)

1. `load_clinical_dataset()` returns a **`ClinicalData` dataclass**
   (spectra/groups/grid/report/spike_scores/flagged) — NOT a tuple.
2. `paired.paired_features` takes **RAW X + FULL grid**; preprocesses
   internally once. Never pass preprocessed X (regression test pins).
3. **GUI parity**: reproduce_study must mirror the GUI data path —
   `common_grid`/`to_matrix`, exclude spike-flagged when
   `despike=False`, same label stripping.
4. Bundle `threshold` lives in **Platt-calibrated** space.
5. Demo data saturates all metrics (1.000) — McNemar/Friedman degenerate.
6. `fold_f1` order = [rep0 f0..k, rep1 f0..k, …]; block statistics must
   reshape `(repeats, k)` and average over repeats.
7. `bootstrap_ci(..., groups=)` resamples groups, not rows.
8. Tests pressing async buttons need `wait_analysis(win)` /
   `worker.wait()` + `processEvents` — results land in queued callbacks.
9. `gui_test.py`/`deep_test.py` **overwrite `result_report.txt/.html`
   in the source tree** (known leftover).
10. PLS-DA probabilities are softmax-of-scores (pseudo-probs). Friedman
    q_alpha hand-tabulated. `band_stats_paired` returns only FDR p.
11. Duplicated helpers (consolidate someday): `_encode` ×3, PCA builders
    ×6 + optimize copy, peak centers ×2 (dataset.py width 28 vs
    modeling.py width 30), rank-with-ties ×2; optimize uses
    random_state=0 vs suite 42.
12. Library modules `print` instead of logging (dataset, optimize,
    vit_train) — GUI mirrors prints into session.log.
13. sklearn "delayed/Parallel" UserWarnings in paired runs + joblib/
    loky shutdown warnings in deep_test — benign, every run.
14. `sg_deriv≠0` **skips baseline correction** by design (derivative
    suppresses smooth baselines).
15. qt_compat binding = import-success order (PyQt5 first), no QT_API
    env pinning; plotting.MatplotlibCanvas pins the binding for
    matplotlib at canvas creation.

## 17. Testing map

| Suite | What it proves | Runtime |
|---|---|---|
| `test_all.py` (40) | loader hygiene, preprocessing, grouped+repeated CV/ensembles, bundle roundtrip, optimize, ViT forward/checkpoint, clinical stats, biochem, FDR/Friedman + 4 regressions (clinical reproduce path, paired single-preprocess, OOF repeat pooling, patient bootstrap) | ~40 s |
| `deep_test.py` (17) | blank-GUI guards, one-class train, predict edge paths, legacy bundles, clinical auto refs, locked eval, HTML report, deep diagnostics, CLI demo roundtrip | ~60 s |
| `gui_test.py` (7) | 6-page walk: load→preprocess→train→save→predict→result→report | ~20 s |
| `smoke_test.py` | headless e2e on demo data | ~15 s |

No pytest config, no coverage, no CI; plain-assert custom collectors.

## 18. Git state & open items

Repo `D:\BARC`, branch `main`, init 2026-08-30; per-commit history:
`git log --oneline`. Patient data & artifacts gitignored (§3).

**Open (prioritized):**
1. `--locked` one-shot holdout for reproduce_study (port GUI
   `run_locked_eval`, ~30 lines) — honest final-exam number for §13.
2. ViT vs classical comparison in one harness; ViT predict not in GUI.
3. gui.py 5.1k monolith (split per page); txt/HTML report duplication.
4. No CI/packaging/LICENSE; tests write artifacts in-tree; Windows font
   paths.
5. Winner selected on the reporting CV (nested eval is opt-in, GUI-only);
   no external cohort; confounders unrecorded.
