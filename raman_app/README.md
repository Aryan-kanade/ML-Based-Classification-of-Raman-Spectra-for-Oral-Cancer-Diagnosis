# Raman Oral-Cancer Classifier

A desktop application (Python + Qt) that classifies Raman spectra
(e.g. SERS oral-cancer spectra, 785 nm) with **maximum sensitivity,
specificity and F1 score**.

It loads a folder of 2-column `.txt` spectra, auto-detects the class of
each file from its name (`P01_cAg_785_C8_3.txt` → class **C8**),
preprocesses them with standard Raman pipelines, cross-validates a suite
of machine-learning models, and automatically keeps the model with the
best **macro-F1 / sensitivity / specificity**.

## Clinical data layout (`Data/<class>/<patient>/<spectrum>.csv`)

The app also understands the **clinical dataset layout** directly — point
it at a folder like `D:\BARC\Data`:

```
Data/
├── Normal/                 <- class label = this folder's name
│   ├── Patient_15/         <- all spectra of one subject stay together
│   │   ├── 17122024TDOC15NH0_interpolated.csv
│   │   └── ...
│   └── TDOC058 Spectra pro/
│       └── 04082025TDOC058TH0.csv
└── Tumor/
    └── Patient_15/ ...
```

Loading is automatic (GUI *Folder…* button or `python vit_train.py` with no
`--data`). `clinical_data.py` then:

* **labels** every spectrum from the top-level folder (Normal/Tumor;
  Control/Healthy/Benign → Normal, Cancer/Malignant → Tumor) — never
  from the filename (some are malformed),
* derives a **patient key** from the folder name (`Patient_15`,
  `TDOC015`, `TDOC058 Spectra pro` → subject *S15/S58*) — the dataset is
  **paired** (the same subject usually has Normal *and* Tumor spectra),
  so every split and every CV fold is **grouped by patient**: one
  subject never appears on both sides of a split,
* **excludes instrument references** (filenames containing
  white/black/dark/bkg/background/ref/blank/calib),
* **collapses exact duplicate spectra** (e.g. `Patient_15–19` content is
  byte-identical to `TDOC015–019`),
* **drops spectra that appear identically under two different classes**
  (impossible labels — pure leakage),
* verifies whether all files share one wavenumber grid (they do: 2271
  points, 15.5–3862 cm⁻¹) and skips re-interpolation when they do,
* writes a full **data hygiene report** (`data_report.txt` /
  `vit_outputs/data_report.txt`) listing everything excluded and why.

On the current dataset: 341 files scanned → 317 spectra / 72 subjects
kept (4 references excluded, 18 duplicate copies dropped, 2 cross-class
identical files dropped).

## Quick start

```bat
run_app.bat               :: double-click launcher (no console window)
python main.py            # same, from a terminal
python main.py --smoke    # headless end-to-end test
```

## Workflow in the GUI

1. **Data** — press *Folder…* and pick the folder with your spectra
   (2-column `.txt`/`.dat`/`.csv` files: wavenumber TAB intensity).
   The class of each file is parsed from the `C<number>` token in the
   filename; you can edit any label by double-clicking the *Class* column.
   Only one real spectrum right now? Press *Generate demo data* on the
   Start page — it derives a synthetic 3-class dataset (C1/C5/C8) from it
   for testing.
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

## Models compared (12 + XGBoost if installed)

PCA+SVM (RBF) · PCA+LDA · PCA+Logistic Regression · PCA+KNN ·
PCA+Gaussian Naive Bayes · Random Forest · Extra Trees ·
Hist Gradient Boosting · PLS-DA · PCA+MLP (neural net) ·
Isolation Forest (one-vs-rest) · XGBoost (if installed)

The **Isolation Forest** entry uses one anomaly detector per class — a
spectrum is assigned to the class whose detector finds it least
anomalous.

## Squeezing the best out of the data

* **Region cropping** (`Keep range`, default 400–1800 cm⁻¹) — the
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
python vit_train.py              :: train (demo data by default, 10 epochs)
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
| `main.py` | entry point (`--smoke` runs the headless test) |
| `gui.py` | Qt main window (PyQt5/PyQt6/PySide6 auto-detected) |
| `dataset.py` | spectrum loading, class parsing, demo-data generator |
| `clinical_data.py` | clinical layout loader: folder labels, patient groups, dedupe, hygiene report |
| `optimize.py` | preprocessing auto-tune (grouped CV grid search) |
| `preprocessing.py` | wavelet denoise · Savitzky–Golay · ALS baseline · normalization |
| `modeling.py` | model suite, nested CV, metrics, threshold tuning, persistence, stacked ensemble |
| `clinical.py` | medical evaluation: operating points, triage tiers, PPV/NPV at prevalence, calibration (Platt), decision-curve, Wilson/DeLong CIs |
| `biochemistry.py` | literature band table, band-ratio markers, paired deltas, SHAP plausibility check, NMF unmixing, keratin guard |
| `reproduce_study.py` | one-shot headless reproduction of the full study (bundle + summary + figures) |
| `plotting.py` | matplotlib helpers (spectra, confusion matrix, ROC, calibration, DCA) |
| `vit_model.py` | spectral Vision Transformer model + checkpoint I/O |
| `vit_train.py` | train the ViT on a spectra folder (curves + confusion matrix) |
| `vit_test.py` | re-evaluate a trained ViT checkpoint on its test set |
| `smoke_test.py` | headless end-to-end verification |

## Testing

```bat
python test_all.py    :: fast unit/regression tests (synthetic data, ~30s)
python gui_test.py    :: headless end-to-end GUI flow (offscreen)
python smoke_test.py  :: headless pipeline check on demo data
python deep_test.py   :: adversarial sweep: error paths, workers, CLI
```

## Reproducing the study

One command reproduces the whole pipeline headlessly with a fixed seed
(bundle, comparison table, clinical summary, figures, run metadata):

```bat
python reproduce_study.py                      :: clinical data at Data\
python reproduce_study.py --demo --mini        :: quick synthetic check
python reproduce_study.py --mode paired-pqn    :: margin mode + PQN
```

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

* The generated `demo_data/` is **synthetic** (derived from one real
  spectrum) — use it to test the workflow, not for scientific claims.
* Replace it with your real labeled folder whenever it is available; no
  code changes are needed.
* Python 3.9+ recommended. On very new Python versions PyQt5 wheels may
  be missing — the app falls back to PyQt6/PySide6 automatically.
