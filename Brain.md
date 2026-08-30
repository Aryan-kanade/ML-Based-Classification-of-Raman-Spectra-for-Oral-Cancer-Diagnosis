# Brain.md — the project's memory

> **Read this before touching any code.** It contains everything a new
> session needs: architecture, data flow, methodology, real results,
> gotchas, and a changelog. Do NOT re-explore the whole tree.
>
> **Maintenance rule (mandatory):** after any non-trivial change, update
> the matching section below. Keep the "Last updated" stamp current.
> Keep it dense — no prose padding.

Last updated: 2026-08-30 (after the "real clinical study" overhaul, git `876b624`)

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
- **Demo data**: synthetic, generated from seed spectrum
  `P01_cAg_785_C8_3.txt` (repo root) by `dataset.generate_demo_data`.
  All metrics ≈1.000 on it — **never report demo numbers as results**.
- Single-centre, no external validation. Confounders (site,
  keratinisation, tobacco) unrecorded.

## 2. Environment & how to run

Windows 10/11, Git Bash shell, **Python 3.14.6**, PyQt5 binding
(auto-detects PyQt6/PySide6 via `qt_compat.py`). Deps pinned in
`raman_app/requirements.txt` (numpy 2.5.2, scikit-learn 1.9.0,
matplotlib 3.11.1, xgboost 3.4.1, pybaselines 1.2.1, shap 0.52.0,
torch 2.13.0 — optional: pybaselines/shap/torch/xgboost all have
graceful fallbacks).

```bash
cd raman_app
python main.py                        # GUI
python main.py --smoke                # headless smoke test
python reproduce_study.py --demo --mini   # 30 s synthetic pipeline check
python reproduce_study.py --data D:/BARC/Data --out study_run_X        # full real study
python reproduce_study.py --data D:/BARC/Data --mode paired --out study_run_paired
python test_all.py                    # 40 unit tests (plain asserts, pytest-compatible)
QT_QPA_PLATFORM=offscreen python gui_test.py    # scripted GUI walk (writes result_report.txt in source tree!)
QT_QPA_PLATFORM=offscreen python deep_test.py   # 17 adversarial scenarios
```

Tests hardcode `C:\Windows\Fonts` — Linux CI would need patching (no CI
exists). `matplotlib.use("Agg")` for all headless figure writing.

## 3. File map (raman_app/, 12.2k lines total)

| File | ~Lines | Role |
|---|---|---|
| `gui.py` | 5121 | The monolith: 6-page MainWindow (~30 state attrs), 5 QThread workers, all slots. See §8. |
| `modeling.py` | 1189 | Model registry (`model_specs`), nested grouped CV (`evaluate_models`), bundles, importance, bootstrap/McNemar. See §5-6. |
| `test_all.py` | 887 | 40 unit tests incl. leakage-specific ones + 4 regressions for the 2026-08-30 fixes. |
| `deep_test.py` | 644 | 16+1 adversarial GUI/CLI scenarios; `wait_analysis()` helper for async buttons. |
| `vit_train.py` / `vit_model.py` / `vit_test.py` | 558/207/143 | Secondary deep-learning track (1D ViT, torch). Not in the GUI suite. |
| `plotting.py` | 439 | Matplotlib theme, `MatplotlibCanvas` (pins Qt binding), plot helpers. Backend-agnostic. |
| `ui_helpers.py` | 391 | Stylesheet, dialogs, settings load/save (`settings.json`), help texts. |
| `study_stats.py` | 375 | Friedman+Nemenyi (hand-tabulated q_alpha), LOPO, seed stability, noise robustness, BH-FDR band stats. |
| `clinical_data.py` | 324 | `load_clinical_dataset()` → `ClinicalData` dataclass + hygiene + `write_report`. |
| `reproduce_study.py` | 282 | One-shot headless study. **Now works on real data** (was broken until 2026-08-30). |
| `clinical.py` | 259 | TRIPOD layer: Platt, DeLong AUC CI, Wilson, operating points, triage, PPV/NPV, calibration, DCA. |
| `preprocessing.py` | 254 | `PreprocessParams` dataclass + `preprocess_spectrum/matrix`. Pipeline order in §4. |
| `biochemistry.py` | 250 | Band table (literature), ratio table, NMF components, keratin flags, plausibility. |
| `optimize.py` | 228 | Preprocessing auto-tune (grouped CV over crop/deriv/norm grid). |
| `dataset.py` | 182 | Flat loader (`C<number>` filename tokens), `common_grid`, `to_matrix`, demo generator. |
| `paired.py` | 160 | Within-patient deviation features — the strongest mode. See §4. |
| `smoke_test.py` / `gui_test.py` / `main.py` / `qt_compat.py` | small | Entry points / harnesses. |

Root: `AGENTS.md` (ponytail dev rules), `P01_…txt` (demo seed),
`Documatation/` (8 literature PDFs, typo'd name), `renders/`,
`_external/` (ponytail clone — unrelated), `.gitignore` (privacy +
artifacts; `study_run*/`, `Data/`, `study_manifest.json`, `*.joblib`,
`result_report.*`, `settings.json`, `session.log` all excluded).

## 4. Data flow (memorize this)

```
raw csv → load (clinical layout | flat)      clinical_data.py / dataset.py
        → hygiene (refs, dedupe, cross-class, spike flags)
        → common_grid (intersection range, median #points, ≤2000) + to_matrix
        → EITHER standard:  preprocess_matrix(X_raw)          preprocessing.py
          OR     paired:    paired_features(X_raw, y, g, FULL grid)  paired.py
                             (preprocesses internally — takes RAW X!)
        → evaluate_models (nested grouped CV, 15 rows)         modeling.py
        → winner (pooled mean macro-F1) → clinical layer       clinical.py
        → bundle (.joblib) → predict_with_bundle               modeling.py
        → reports (txt/html), freeze_study manifest
```

**Preprocessing order** (`preprocess_spectrum`, per-spectrum, no
cross-sample fit → leakage-free): optional Whitaker-Hayes despike
(**default OFF** — measured pilot) → wavelet soft-threshold →
Savitzky-Golay → ALS/arPLS baseline (pybaselines if present) →
vector/SNV/PQN normalise. `preprocess_matrix` crops FIRST via
`crop_mask` (default 400–1800 cm⁻¹).

**Paired mode** (the scientifically strongest): for each patient with
both classes, reference = mean of their preprocessed Normal spectra;
every kept spectrum becomes `spectrum − own_normal_reference`
(optionally PQN against the reference first). Removes the ~61%
between-patient variance. Patients without a normal reference are
excluded (counted). Prediction-time reference: `paired.reference_vector`
or GUI `PredictWorker._auto_reference` (`<root>/Normal/<patient>/`).

## 5. Cross-validation methodology (modeling.py)

- **Outer**: `repeats` (3) × `StratifiedGroupKFold(n_splits=5)` (patient-
  grouped; plain `StratifiedKFold` without groups). Different shuffle
  per repeat (`seed + rep*100`); folds pooled.
- **Inner** (per outer fold): `GridSearchCV` with grouped 3-fold over
  the hyperparameter grid (`_tune_hyperparams`); binary F1 threshold
  tuned on pooled inner-OOF and kept only if it beats 0.5. The outer
  test fold never touches its own tuning.
- **OOF probabilities are averaged across repeats** (fix 2026-08-30;
  previously only the last repeat was kept) and feed: operating points,
  Platt, AUC/DeLong, bootstrap CI. Confusion matrices pool raw counts.
- **Winner** = best pooled mean macro-F1 (sens/spec tiebreak), refit on
  all data with most-chosen hyperparams → that refit pipeline is the
  bundle (deployment model, no test set by design).
- `evaluate_models(...)` extras: "Ensemble (top-3)" soft voting +
  "Stacked (top-3)" logistic meta-learner compete under the same CV;
  `evaluate_pipeline` = honest nested variant (preprocessing re-chosen
  per fold — GUI "honest check" button only, not in reproduce_study).
- `ModelResult` dataclass carries `oof_proba`, `y_true_encoded`,
  `groups` (per-row patient ids — added 2026-08-30 for patient-level
  bootstrap), `thresholds`, `fold_f1` ([rep0_f0, rep0_f1, …, rep1_f0…]),
  `cm`, `pipeline`, `error`.
- Diagnostics: LOPO (`study_stats.lopo_evaluate`, k = n_patients), seed
  stability (5 seeds), noise robustness, Friedman+Nemenyi **on per-fold
  scores averaged across repeats** (fix 2026-08-30), `bootstrap_ci`
  (patient-level when `groups` passed — fix 2026-08-30), exact McNemar,
  BH-FDR for band stats, Wilson/DeLong CIs, decision-curve analysis.

## 6. Model suite & bundles

Registry `model_specs()` — 13 base + 2 ensembles (15 comparison rows):
PCA+SVM(RBF), PCA+LDA, PCA+LogReg, PCA+KNN, PCA+GNB, RF, ExtraTrees,
HistGradientBoosting, PLS-DA, PCA+MLP, IsolationForest-OvR, XGBoost
(if installed), Peak-bands+RF, Ensemble(top-3), Stacked(top-3).
`RANDOM_STATE = 42`.

**Bundle** (joblib dict, `save_bundle`/`predict_with_bundle`):
`model_name, pipeline, classes, threshold (Platt-calibrated space!),
wavenumbers (grid), prep_params, macro, dataset_name, paired[,
calibrator, op_points]`. Predict-time: interpolate onto stored grid →
re-apply stored preprocessing → optional reference subtraction → Platt →
threshold. Legacy uncropped bundles still accepted (feature-length
guessing).

## 7. Real study results (2026-08-30, seed 42, 5-fold grouped ×3)

Hygiene per run: 317 spectra / 72 patients → 18 spike-flagged excluded
(despike off). Full tables + PNGs + winner.joblib in
`raman_app/study_run_standard/` and `raman_app/study_run_paired/`
(gitignored).

| Mode | Winner | macro-F1 (95% CI) | AUC (DeLong) | LOPO |
|---|---|---|---|---|
| Standard | Peak bands + RF | 0.594 (0.54–0.65) | 0.610 (0.55–0.68) | F1 0.589 |
| **Paired** | **Extra Trees** | **0.702 (0.66–0.76)** | **0.788 (0.74–0.84)** | F1 0.703, AUC 0.792 (64 pat.) |

Paired operating points: rule-out p≤0.35 (sens 91%), rule-in p≥0.67
(spec 91%); Brier 0.190→0.185 (Platt). Literature benchmark (spectrum-
level splits, looser): ~0.89/0.85. **No locked one-shot holdout has been
run headlessly** (GUI button exists) and no external validation exists.

## 8. GUI structure (gui.py)

- **6 pages** (TAB_* constants): Start, Data, Preprocess, Train, Predict,
  Result. `MainWindow.__init__` ~30 state attrs: `spectra, labels, groups,
  spike_flags, grid, X_raw, _lc_data (X,yy,gg), _lc_wn, winner, results,
  bundle, _region_bands, _band_stats, _locked_result, _lopo_result…`.
- **Workers** (all QThread, signal/slot, never touch widgets in run()):
  `TrainWorker`, `OptimizeWorker`, `PipelineWorker` (honest eval),
  `PredictWorker` (file loop + auto per-patient refs), `FuncWorker`
  (generic fn runner). **`_run_async(label, fn, on_done)`** runs any
  callable off-thread, disables the sender button, shows a failure
  dialog — used by the 8 analysis buttons (learning curve, region
  importance, biochemistry, locked eval, LOPO, seed stability, noise,
  local explain).
- `render_result_page` (~600 lines) is the largest method. Report writers
  `save_result_report` (txt) + `save_result_report_html` (self-contained,
  base64 PNGs). `freeze_study` writes `study_manifest.json` (code+data
  SHA1s, patient rows — **gitignored, contains patient IDs**).
- Persistence: `settings.json` (UI only: params, folds, seed, model
  checkboxes gated by `SUITE_VERSION=4`, geometry hex, folders).
  `session.log` via rotating RotatingFileHandler (1 MB × 2).
- `closeEvent` waits 3 s then terminates all 5 workers (fix 2026-08-30 —
  prevents QThread-destroyed-while-running crash on exit).

## 9. Gotchas & invariants (read before editing)

1. `load_clinical_dataset()` returns a **`ClinicalData` dataclass**
   (fields: spectra/groups/grid/report/spike_scores/flagged) — NOT a
   tuple. Two call sites once unpacked it as a tuple → crash.
2. `paired.paired_features` takes **RAW X + FULL grid**; it preprocesses
   internally exactly once. Never pass preprocessed X (double-processing
   bug, fixed 2026-08-30; regression test pins this).
3. **GUI parity**: reproduce_study must mirror the GUI data path —
   `common_grid`/`to_matrix`, exclude spike-flagged spectra when
   `despike=False`, same label stripping.
4. `winner.threshold` in the bundle lives in **Platt-calibrated**
   probability space (save_bundle re-applies the calibrator to it).
5. Demo data saturates every metric (1.000) — McNemar/Friedman become
   degenerate there. Only judge on real data.
6. `fold_f1` ordering is [rep0 fold0..k, rep1 fold0..k, …] — anything
   block-based must reshape `(repeats, k)` and average over repeats.
7. `bootstrap_ci(y, pred, groups=…)` resamples **groups**; without
   groups, rows. Single-group input ⇒ zero-width CI (pinned by test).
8. Tests that press async buttons must `wait_analysis(win)` /
   `worker.wait()` + `processEvents` — results land in queued callbacks.
9. `gui_test.py` **overwrites `result_report.txt` in the source tree**
   (known leftover; deep_test does the same for .html).
10. PLS-DA "probabilities" are softmax-of-scores (pseudo-probs).
    Friedman q_alpha is hand-tabulated. `band_stats_paired` returns only
    the FDR-adjusted p (raw p discarded).
11. Duplicated helpers to consolidate someday: `_encode` ×3
    (modeling/study_stats/gui), PCA pipeline builders ×6 + an optimize
    copy, peak-band centers ×2, rank-with-ties ×2. `optimize.py` uses
    `random_state=0` vs the suite's 42 (inconsistent but harmless).
12. Library modules still `print` instead of logging (dataset.py,
    optimize.py, vit_train.py) — the GUI mirrors prints into session.log.
13. sklearn "delayed should be used with Parallel" UserWarnings during
    paired runs, and joblib/loky shutdown warnings in deep_test — both
    benign, seen on every run.

## 10. Testing map

| Suite | What it proves | Runtime |
|---|---|---|
| `test_all.py` (40) | loader hygiene, preprocessing, grouped+repeated CV/ensembles, bundle roundtrip, optimize, ViT forward/checkpoint, clinical stats, biochem, FDR/Friedman, + 4 regressions (clinical reproduce path, paired single-preprocess, OOF repeat pooling, patient bootstrap) | ~40 s |
| `deep_test.py` (17) | blank-GUI guards, one-class train, predict edge paths, legacy bundles, clinical-tree auto refs, locked eval, HTML report, deep diagnostics, CLI demo roundtrip | ~60 s |
| `gui_test.py` (7 steps) | whole 6-page walk: load→preprocess→train→save→predict→result→report | ~20 s |
| `smoke_test.py` | headless end-to-end on demo data (45 spectra) | ~15 s |

No pytest config, no coverage, no CI. All suites are plain-assert with
custom collectors (`main()` at bottom).

## 11. Git / repo state

Repo at `D:\BARC` (init 2026-08-30), branch `main`. Per-commit history:
`git log --oneline`. Patient data & run artifacts gitignored — see §3.

## 12. Open items (prioritized)

1. **Locked one-shot holdout not headless** — `--locked` flag for
   reproduce_study (port GUI `run_locked_eval` logic, ~30 lines). Would
   put an honest final-exam number next to §7.
2. ViT track never compared with the classical suite in one harness; no
   GUI predict integration for ViT.
3. gui.py still a 5.1k-line monolith (per-page controllers would be the
   split); txt/HTML report builders duplicate content logic.
4. No CI, no packaging (.exe), no LICENSE; tests write artifacts into
   the source tree; Windows-only font paths.
5. Sci-rigor ceilings: winner selected on the same CV that reports
   (nested `evaluate_pipeline` exists but is opt-in, GUI-only); no
   external cohort; confounders unrecorded.
