# Raman Oral-Cancer Classifier

A desktop application (Python + Qt) that classifies Raman spectra
(e.g. SERS oral-cancer spectra, 785 nm) with **maximum sensitivity,
specificity and F1 score**.

It loads a folder of 2-column `.txt` spectra, auto-detects the class of
each file from its name (`S07_tissue_C8.txt` → class **C8**),
preprocesses them with standard Raman pipelines, cross-validates a suite
of machine-learning models, and automatically keeps the model with the
best **macro-F1 / sensitivity / specificity**.

## Clinical data layout (`Data/<class>/<patient>/<spectrum>.csv`)

The app also understands the **clinical dataset layout** directly — point
it at a folder like `D:\BARC\Data`:

```
Data/
├── Normal/                 <- class label = this folder's name
│   ├── Patient_07/         <- all spectra of one subject stay together
│   │   ├── 20250107_site07_n0_interpolated.csv
│   │   └── ...
│   └── Subject058 Spectra pro/
│       └── 20250804_site58_t0.csv
└── Tumor/
    └── Patient_07/ ...
```

Loading is automatic (GUI *Folder…* button or `python vit_train.py` with no
`--data`). `clinical_data.py` then:

* **labels** every spectrum from the top-level folder (Normal/Tumor;
  Control/Healthy/Benign → Normal, Cancer/Malignant → Tumor) — never
  from the filename (some are malformed),
* derives a **patient key** from the folder name (`Patient_07`,
  `Subject007`, `Subject058 Spectra pro` → subject *S7/S58*) — the dataset is
  **paired** (the same subject usually has Normal *and* Tumor spectra),
  so every split and every CV fold is **grouped by patient**: one
  subject never appears on both sides of a split,
* **excludes instrument references** (filenames containing
  white/black/dark/bkg/background/ref/blank/calib),
* **drops mislabeled acquisitions** — a `TH*` (tumor-site) token inside
  `Normal/` or an `NH*` (normal-site) token inside `Tumor/` contradicts
  the folder label and is excluded (impossible labels),
* **collapses exact duplicate spectra** (e.g. one subject exported twice
  under two folder names),
* **drops spectra that appear identically under two different classes**
  (impossible labels — pure leakage),
* verifies whether all files share one wavenumber grid (they do: 2271
  points, 15.5–3862 cm⁻¹) and skips re-interpolation when they do,
* writes a full **data hygiene report** (`data_report.txt` /
  `vit_outputs/data_report.txt`) listing everything excluded and why.

On the current dataset: 341 files scanned → 317 spectra / 72 subjects
kept (4 references excluded, 18 duplicate copies dropped, 2
site-token-mislabeled acquisitions dropped — tumor-site files that sat
inside the `Normal/` folder).

## Quick start

```bat
run_app.bat               :: double-click launcher (no console window)
python main.py            # same, from a terminal
```

## Workflow in the GUI

1. **Data** — press *Folder…* and pick the folder with your spectra
   (2-column `.txt`/`.dat`/`.csv` files: wavenumber TAB intensity).
   The class of each file is parsed from the `C<number>` token in the
   filename; you can edit any label by double-clicking the *Class* column.
2. **Preprocess** — tune wavelet denoising, Savitzky–Golay smoothing
   (+ optional derivative), ALS baseline correction and normalization;
   preview the effect on the selected spectrum (the processed panel also
   shows the raw trace as a faded reference).
3. **Train && Evaluate** — choose models, set CV folds, press
   *Start training*. The comparison table shows mean ± std of
   sensitivity / specificity / F1 (macro) from stratified k-fold CV;
   the best model is bolded, with its pooled confusion matrix, per-class
   metrics and ROC curve (binary case). *Save best model…* writes a
   `.joblib` bundle that contains the model **and** the preprocessing
   settings.
4. **Predict** — load a saved model, then browse a **folder** of new
   spectra or pick individual `.txt` files, press *Predict*. You get the
   predicted class, class probabilities
   (sorted bar chart with the winner highlighted) and an annotated plot
   of the first spectrum. (Binary models use the F1-optimal threshold
   found during cross-validation.)

Every plot has an interactive toolbar — zoom, pan, and save a
high-resolution PNG of the figure.

## How sensitivity / specificity / F1 are maximized

* **Leakage-free nested evaluation** — inside every CV fold,
   hyperparameters are selected with an inner grid search on the training
   split only.
* **Binary threshold optimization** — for 2-class problems the decision
   threshold is tuned on an inner holdout to maximize F1 (no optimistic
   bias), instead of the default 0.5.
* **Auto-selection** — the model with the best cross-validated macro-F1
   wins (sensitivity, then specificity as tie-breakers) and is refit on
   all data.
* Per-class **sensitivity, specificity, precision and F1** are reported
  with mean ± std across folds, plus the pooled confusion matrix.

## Models compared (24-entry registry; optional deps degrade gracefully)

PCA+SVM (RBF, grouped-sigmoid calibrated) · PCA+LDA · PCA+Logistic
Regression · PCA+KNN · PCA+Gaussian Naive Bayes · Random Forest ·
Extra Trees · Hist Gradient Boosting · PLS-DA · Sparse PLS-DA ·
PCA+MLP · Isolation Forest (one-vs-rest) · Peak bands + RF ·
Spectral + band features · PLS + XGBoost · PCA + XGBoost ·
t-test filter + XGBoost · XGBoost/LightGBM/CatBoost (if installed) ·
1D-CNN and its 5-seed ensemble (torch) · TabPFN foundation model ·
Ensemble (top-3) · Stacked (top-3)

The **Isolation Forest** entry uses one anomaly detector per class — a
spectrum is assigned to the class whose detector finds it least
anomalous.

## Squeezing the best out of the data

* **Region cropping** (`Keep range`, default 500–2000 cm⁻¹) — the
  fingerprint region carries the signal; outside it is mostly noise
  (measured: +0.06–0.12 macro-F1 on the clinical set).
* **Optimize preprocessing** (Preprocess page, or `python optimize.py`) —
  tries crop / derivative / normalization combinations under
  patient-grouped CV and applies the winner. The result is saved to
  `vit_outputs/preprocess_best.json` and picked up by `vit_train.py`.
* **Quality flagging** — spectra with cosmic-ray spikes are flagged in
  the Data table (Quality column) and excluded from training by default.
* **Repeat CV ×3** (Train page) — three fold shuffles pooled, so the
  winning model is chosen on stabler numbers.
* **Ensemble (top-3)** — soft voting over the three best models, added
  to the comparison automatically.
* **Learning curve** (Train page) — macro-F1 at 25/50/75/100% of the
  patients with the winning model: shows whether collecting more
  patients is worth it.
* **ViT training** uses augmentation (noise + shift), cosine LR with
  warmup, and early stopping on validation macro-F1.

## Vision Transformer (deep-learning baseline)

Besides the classical-ML suite, the project includes a compact
**spectral Vision Transformer** (pure PyTorch, no torchvision/timm).
Each preprocessed spectrum is resampled to 512 points, split into 16
patches of 32, linearly embedded, and classified by a small transformer
encoder (dim 128, depth 4, 4 heads, CLS-token head).

```bat
python vit_train.py --data D:\data\my_spectra --epochs 25
python vit_test.py               :: re-evaluate the held-out test set
```

`vit_train.py` prints per-epoch train/val loss & accuracy, then the test
accuracy and a full classification report, and saves into `vit_outputs/`:
the checkpoint (`vit_model.pt`), the loss/accuracy curves
(`vit_training_curves.png`) and the test confusion matrix
(`vit_confusion_matrix.png`). `vit_test.py` reproduces the evaluation
exactly (the split filenames, wavenumber grid and preprocessing settings
are stored in the checkpoint).

## Files

| file | purpose |
|---|---|
| `main.py` | entry point (launches the GUI) |
| `gui.py` | Qt main window (PyQt5/PyQt6/PySide6 auto-detected) |
| `dataset.py` | spectrum loading, class parsing, common grid |
| `clinical_data.py` | clinical layout loader: folder labels, patient groups, dedupe, hygiene report |
| `optimize.py` | preprocessing auto-tune (grouped CV grid search) |
| `preprocessing.py` | wavelet denoise · Savitzky–Golay · ALS baseline · normalization |
| `modeling.py` | model suite, nested CV, metrics, threshold tuning, persistence, stacked ensemble |
| `clinical.py` | medical evaluation: operating points, triage tiers, PPV/NPV at prevalence, calibration (Platt), decision-curve, Wilson/DeLong CIs |
| `biochemistry.py` | literature band table, band-ratio markers, paired deltas, SHAP plausibility check, NMF unmixing, keratin guard |
| `reproduce_study.py` | one-shot headless reproduction of the full study (bundle + summary + figures) |
| `reproduce_report.py` | head-to-head replication of the internship report's protocol vs patient-grouped CV |
| `sequential.py` | 3SSE sequential architecture search (singles/pairs/triples chains) |
| `paired.py` | within-patient deviation features (paired mode) |
| `study_stats.py` | Friedman/Nemenyi, LOPO, seed/noise stability, FDR band stats |
| `bench.py` | speed & accuracy scorecard (writes bench/latest.json) |
| `validate_external.py` | external-cohort CLI for any saved bundle |
| `plotting.py` | matplotlib helpers (spectra, confusion matrix, ROC, calibration, DCA) |
| `ui_helpers.py` | QSS design system, pills/tooltips, settings I/O |
| `qt_compat.py` | PyQt5/PyQt6/PySide6 binding auto-detection |
| `vit_model.py` | spectral Vision Transformer model + checkpoint I/O |
| `vit_train.py` | train the ViT on a spectra folder (curves + confusion matrix) |
| `vit_test.py` | re-evaluate a trained ViT checkpoint on its test set |
| `test_all.py` / `gui_test.py` / `deep_test.py` | 97 unit tests · GUI walk · 19 adversarial scenarios |

## Testing

```bat
python test_all.py    :: 97 unit/regression tests (synthetic data, ~2 min)
python gui_test.py    :: headless end-to-end GUI flow (offscreen, ~20 s)
python deep_test.py   :: adversarial sweep: error paths, workers, CLI (~60 s)
```

## Compute devices (CPU / GPU)

The app runs CPU-only out of the box and uses the GPU where one is
genuinely available — controlled by the `RAMAN_DEVICE` environment
variable (default **auto**; the GUI Start page shows a live
"Compute device" card, both CLIs print a device summary):

| Mode | torch models (1D-CNN, 5-seed CNN ensemble, ViT, TabPFN) | XGBoost / CatBoost | LightGBM + scikit-learn |
|---|---|---|---|
| `auto` (default) | **CUDA** when a CUDA context initializes (a real forward pass is probed at startup) | CPU (measured: at n≈300 CPU wins and GPU boosters OOM'd worker processes) | CPU |
| `gpu` | CUDA | **GPU** (parent process only) | CPU (no genuine GPU backend) |
| `cpu` | forced CPU | forced CPU | CPU |

`gpu` is **strict**: if no usable CUDA backend exists, the app fails
loudly at startup / CLI entry instead of silently running CPU.
LightGBM pip wheels contain no GPU build (source build with
`-DUSE_GPU=1` required) — honestly reported as CPU. Bundle weights
are stored on CPU, so saved-model predictions are **identical**
whether the training ran on GPU or CPU (verified MAX |Δp| = 0).
Full details + benchmarks: `GPU_ACCELERATION.md`,
`GPU_ACCELERATION_AUDIT.md`, `GPU_STRESS_TEST_AUDIT.md` (repo root).

## Performance reporting policy

The number the GUI and reports show as performance is the **nested
honest estimate** (preprocessing re-chosen inside every CV fold,
pooled confusion matrix) — not the selection-CV numbers, which are
labeled as a "selection ranking". Supplementary scientific metrics
(balanced accuracy, MCC, PR-AUC, Brier, ECE, explicit TP/TN/FP/FN,
threshold-stability stats with an instability warning) are shown
separately, with PATIENT-level and SPECTRUM-level results explicitly
labeled and never mixed.

## Validation summary

Five audit rounds + a formula audit + a GPU implementation audit + a
long-running GPU soak test (all reports at the repo root: `MASTER_
AUDIT_REPORT.md`, `FINAL_RELEASE_AUDIT.md`, `METRIC_IMPROVEMENT_
IMPLEMENTATION.md`, `GPU_*.md`). Final state: **97/97 unit · 19/19
deep · GUI suite · ruff clean · 25/25 registry (Extra Trees F1
0.760) · zero patient-level leakage (fingerprint probes at chance)
· persistence bit-exact across processes · all scientific baselines
bit-identical after every change · GPU soak 15/15 PASS (20× train,
100× predictions, memory plateau, 0 CUDA errors).**

## Scientific limitations

* **Single-centre dataset, no external cohort validation** — every
  number is internal cross-validation; a second site is required
  before any clinical claim.
* The headline numbers are the honest nested estimates (e.g. honest
  macro-F1 ≈ 0.56 on standard data); selection-CV numbers are higher
  and are deliberately not reported as performance.
* **Threshold instability**: per-fold tuned thresholds can vary
  widely (e.g. 0.01–0.88 for one model); the app displays the full
  spread and a warning rather than hiding it.
* **Patient-level on paired winners**: patients contribute both
  Normal and Tumor sites; the dominant-class patient rollup is a
  documented approximation (shown with labels, not hidden).
* **GIGO boundary**: physically odd but finite-signal inputs can
  still receive confident predictions; NaN/no-signal/degenerate
  inputs are rejected with clear errors on every path.
* CNN folds trained on GPU are not bit-identical to CPU folds (CUDA
  atomics); seeds fix the splits, and saved-bundle predictions are
  device-independent.

## Reproducing the study

One command reproduces the whole pipeline headlessly with a fixed seed
(bundle, comparison table, clinical summary, figures, run metadata):

```bat
python reproduce_study.py                      :: clinical data at Data\
python reproduce_study.py --mode paired-pqn    :: margin mode + PQN
```

### Real clinical dataset results (317 spectra / 72 subjects, 2026-08-30)

> **STALE — re-run before citing.** These numbers predate the
> 2026-09-04 loader change (padded edges kept, crop 500/2000) and the
> 2026-09-05 hygiene change (site-token drops). The v2 run artifacts
> were removed 2026-09-07; `Brain.md` §13 is the surviving record.

5-fold patient-grouped CV ×3, seed 42, 18 quality-flagged spectra
excluded; CIs are patient-level.

| Mode | Winner | macro-F1 (95% CI) | AUC (DeLong 95% CI) | LOPO |
|---|---|---|---|---|
| Standard | Peak bands + RF | 0.594 (0.54–0.65) | 0.610 (0.55–0.68) | F1 0.589 |
| Paired (within-patient deviation) | Extra Trees | 0.702 (0.66–0.76) | 0.788 (0.74–0.84) | F1 0.703, AUC 0.792 |

Paired referencing — each spectrum as its deviation from the same
patient's normal reference — is the clinically realistic margin scenario
and clearly outperforms pooled classification here. Literature SERS
meta-analyses report ~0.89/0.85 sens/spec, but on spectrum-level splits;
these patient-grouped numbers are the stricter standard and the honest
current state of this single-centre dataset.

`test_all.py` covers the clinical loader (dedupe / cross-class /
references / patient grouping), cropping + spike scoring, peak-band
features, grouped+repeated CV with the ensemble, the model-bundle
roundtrip, `optimize.py`, and the ViT (forward pass, checkpoint
roundtrip). `gui_test.py` walks the whole six-page flow: load →
preprocess → train → save → predict (with reference-file skipping) →
Result page → report.

## Methods & references

Each core method follows the primary literature:

* **Despiking** — Whitaker & Hayes (2018), *A simple algorithm for
  despiking Raman spectra*, Analytica Chimica Acta (modified Z-score +
  moving-average repair). Measured honestly on this dataset: recovering
  the heavily spiked spectra via despiking scored WORSE (macro-F1 0.52)
  than excluding them (0.62) — the flagged spectra are broadly corrupted,
  not single-point cosmic rays — so exclusion remains the default and the
  despiker is an opt-in tool.
* **Baseline** — Eilers & Boelens (2005) ALS; **arPLS**: Baek et al.
  (2015), Analyst (via `pybaselines`).
* **Normalization** — vector / SNV; **PQN**: Dieterle et al. (2006),
  *Probabilistic quotient normalization*, Analytical Chemistry
  (reference = the patient's own normal in paired mode → leakage-free).
* **Paired-reference classification** — motivated by the variance
  decomposition of this dataset (~61% between-patient variance); models
  tumor-margin assessment with a same-patient normal reference.
* **Explainability** — Gini region importance and signed spectral SHAP
  (Contreras et al. 2024, Analytical Chemistry pattern) mapped to named
  biochemical bands.
* **Augmentation (ViT)** — Gaussian noise, wavenumber shift, and mixup
  (Zhang et al. 2018).
* **Evaluation rigor** — patient-grouped stratified CV (repeated),
  pooled-threshold tuning, bootstrap CIs, exact McNemar tests, nested
  pipeline evaluation; published oral-cancer Raman meta-analyses report
  ~89–90% pooled sensitivity/specificity but mostly under spectrum-level
  splits — our grouped numbers are the stricter standard.
* **Clinical evaluation** (TRIPOD+AI / PROBAST+AI aligned) — rule-out /
  rule-in operating points on the pooled out-of-fold ROC (sens ≥ 0.90 /
  spec ≥ 0.90), three-tier triage (NEGATIVE / INDETERMINATE / POSITIVE)
  with per-tier actions, PPV/NPV at plausible prevalence (Patton, JADA:
  PPV collapses at low prevalence), reliability diagram and decision
  curve (net benefit). Literature anchors: Han 2022 (Frontiers in
  Oncology meta, 0.89/0.84), 2025 OSCC meta (0.89/0.91), Purohit 2026
  (0.90/0.89); clinical context: VELscope ~0.84/~0.45, toluidine blue
  ~0.63/~0.83.
* **Biochemistry** — literature band panel (785/1090/1335/1578 nucleic
  acids; 1003 phenylalanine; 1554 tryptophan; 854/1240 collagen; 1445
  CH2; 1655 amide I) with expected tumor directions (Sharma 2021; Faur
  2022: OSCC shows raised nucleic-acid + protein content, collagen
  loss); band-ratio markers with paired patient deltas (each patient
  their own control); NMF non-negative unmixing into auto-labelled
  biochemical components; a plausibility check of the model's SHAP
  bands against the literature; and a keratin site-effect guard
  (keratinisation varies by anatomical site — Matthies 2021, Hanna
  2024).

## Notes

* Python 3.11–3.14 (the pinned dependency set needs ≥3.11; the source
  avoids 3.12-only f-string syntax, enforced by a source-scanning
  test). On the newest Python versions PyQt5 wheels may be missing —
  the app falls back to PyQt6/PySide6 automatically.
