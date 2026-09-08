# CURRENT FORMULA AUDIT — Raman Spectroscopy ML Application

Extraction date: 2026-09-08 · Tree: commit `4c01fc7` + working changes
(§36–§40 of Brain.md). **Factual extraction only — no judgment, no
changes.** Every formula below was read from the current code;
`file:line` references are approximate to the working tree.

---

## 1. Accuracy

**Formula:** `trace(cm) / cm.sum()`
**Where:** computed explicitly only in `modeling.evaluate_pipeline`
(as `"acc"` of the pooled honest CM, modeling.py ~1985) and in
`sequential._metrics_from_oof` (`"acc": np.trace(cm)/max(cm.sum(),1)`,
sequential.py:234). The main `run_cv`/`class_metrics_from_cm` path
does **not** compute accuracy at all.
**Inputs:** pooled confusion matrix over outer-fold test rows.
**Aggregation:** one pooled CM → one accuracy number.

## 2. Precision (per class i, from CM)

**Formula:** `prec_i = TP_i / (TP_i + FP_i)` where `TP_i = cm[i,i]`,
`FP_i = cm[:,i].sum() - cm[i,i]`; `0.0` when denominator is 0.
**Code:** `modeling.class_metrics_from_cm` (modeling.py:1171–1185):
```python
tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
prec = tp / (tp + fp) if (tp + fp) else 0.0
```
**Inputs:** a confusion matrix (per fold in run_cv; pooled in
evaluate_pipeline / screening).

## 3. Recall

**Formula:** identical to sensitivity (below) — the code computes one
quantity, `sens`, used as both recall and sensitivity. No separate
"recall" implementation exists.

## 4. Sensitivity (per class)

**Formula:** `sens_i = TP_i / (TP_i + FN_i)`, `FN_i = cm[i].sum() -
cm[i,i]`; `0.0` on empty denominator.
**Code:** same function (modeling.py:1180).

## 5. Specificity (per class)

**Formula:** `spec_i = TN_i / (TN_i + FP_i)` with
`TN_i = total - TP_i - FN_i - FP_i`; `0.0` on empty denominator.
**Code:** modeling.py:1179.

## 6. F1 (per class)

**Formula:** `f1_i = 2·prec_i·sens_i / (prec_i + sens_i)`; `0.0` when
`prec_i + sens_i == 0`.
**Code:** modeling.py:1183. Additionally, fold-level macro-F1 is
computed by sklearn: `f1_score(yte, pred, average="macro")`
(modeling.py:1666) and by `f1_score(y_all, argmax(P))` in
`_tune_inner` (modeling.py:1346).

## 7. Confusion matrix

**Construction:** `sklearn.metrics.confusion_matrix(y_true, pred,
labels=range(len(classes)))` — i.e. **rows = actual (encoded 0..C-1),
columns = predicted**, labels ordered by `classes` (see §8).
**Sites:**
- `run_cv` per outer fold, summed into `cm_total` (modeling.py:1663-64)
- `evaluate_pipeline` pools test rows across folds into ONE cm
- `sequential._metrics_from_oof` (screening/3SSE, sequential.py:228)
**Binary layout:**
```text
cm = [[TN, FP],      # actual class 0 (Normal)
      [FN, TP]]      # actual class 1 (Tumor)
```
(The independent Round-1/3 audits reproduced these CMs bit-exactly
under that convention.)

## 8. Class mapping

**Source of truth:** `classes = sorted(set(y))` (modeling.py:1500).
Encoding: `ye = _encode(y, classes)` → index of the label in the
sorted list. With `{'Normal','Tumor'}`:
```text
0 = 'Normal'   (negative)
1 = 'Tumor'    (positive; pos_idx = 1, modeling.py:1515)
```
`predict_proba` columns follow the estimator's `classes_` = the same
sorted order passed via encoded labels. GUI positive class:
`_positive_class()` returns `sorted(classes)[1]`-equivalent from the
bundle's `classes` list. Screenings (`sequential`) encode the same
way (`search`/`validate_arch`).

## 9. Probability

**Source per family:**
- All sklearn models: `estimator.predict_proba(X)` (pipeline step
  `clf`), columns in sorted-class order → `proba[:, 1] = p(Tumor)`.
- PCA+SVM: `CalibratedSVC` wrapper — inner `SVC(probability=False)`
  wrapped in `CalibratedClassifierCV` (group-aware since 2026-09-06)
  → probabilities from sigmoid calibration, not Platt-on-SVC-directly.
- PLS-DA: `predict_proba = softmax(PLS regression scores)`
  (pseudo-probabilities, monotone per class) — modeling.py:192-210.
- 1D-CNN / CNN-ensemble (torch): batched `softmax` in `eval()/no_grad`;
  optional TTA (`predict_proba_tta`) averages noise/roll/mask views;
  CNN-ensemble additionally averages 5 seeds (mean) and reports
  `predict_proba_std` disagreement.
- TabPFN: library `predict_proba` (foundation model).
- 3SSE chains: each layer's `predict_proba` output columns are
  **appended** to the feature vector for the next layer
  (`SequentialChain.predict_proba` chains `[X | P1 | P2 ...]`).
- Boosters at screening: `XGB/LGBM/CatBoost predict_proba`.
- AUC-only fallback: when a model has no `predict_proba`,
  `decision_function` scores are used for the honest-check AUC
  (evaluate_pipeline; plain SVC case).
**Post-processing:** Platt calibration `p' = sigmoid(a·logit(p)+b)`
(clinical.py:205-211) applied at predict time when the bundle carries
`calibrator`; stored thresholds are remapped into calibrated space at
save time (modeling.py:2099-2105).

## 10. Threshold

**Selection:** per outer fold, `_tune_inner` collects POOLED
inner-OOF probabilities of the winning hyperparameter combo; binary
threshold = argmax of macro-F1 over **99 quantile candidates
(1%..99%) of the scores**, Youden-J as tie-break
(`best_f1_threshold`, modeling.py:1188-1205). The tuned threshold is
**kept only if it beats 0.5** on the pooled inner OOF (macro-F1
comparison, modeling.py:1356-1358 / 1647-1650), else `None` (argmax
0.5 rule). StackedEnsemble: same procedure on its own `meta_oof_`.
**Global threshold stored:** `res.threshold = median(valid per-fold
thresholds)` (modeling.py:1676-78) — `None` if no fold kept one.
**Prediction rule:** `pred = positive if p_positive >= threshold else
negative` (modeling.py:1657-58; per-fold value during CV, stored
median at deploy). Threshold is tuned on **training-fold inner OOF
only** — never on outer test folds.
**Deploy:** `predict_with_bundle` returns probabilities + threshold;
GUI triage uses rule-out/rule-in operating points
(`clinical.operating_points`: highest threshold with tpr ≥ target /
highest with fpr ≤ 1−target, from `roc_curve` on winner OOF).

## 11. Cross-validation

**Structure** (`evaluate_models`, modeling.py:1475+):
- `k = clip(k_folds, 2, min(10, min_class))`, then
  `clip(k, 2, n_distinct_groups)` when groups present.
- Outer: `repeats × StratifiedGroupKFold(k, shuffle=True,
  random_state=seed + repeat·100)` (plain StratifiedKFold without
  groups). Group key = patient subject id (`S{first number in patient
  folder}` via `clinical_data.subject_key`).
- Inner (hyperparams + threshold): `inner_k = min(3, min_class_tr,
  n_groups_tr)` grouped as well; ONE pass (`_tune_inner`) scores each
  param combo by **pooled inner-OOF macro-F1** (argmax prediction)
  and tunes the threshold on the winner's pooled scores.
- Default seed 42 (GUI spin), folds 5, repeats 1 (GUI checkbox ×3).
**OOF:** `oof_sum[te] += proba; oof[seen] = oof_sum/oof_cnt` —
probabilities **averaged across repeats** (modeling.py:1670-72).
**Honest check** (`evaluate_pipeline`, modeling.py:1898+): outer
grouped k folds; inside EACH fold `optimize.optimize_preprocessing`
re-chooses preprocessing on training patients only (compact 3-config
grid), best of {PCA+SVM, RF, PCA+LogReg} by inner grouped 3-fold
macro-F1; the fold's chosen pipeline is scored on the held-out
patients. Pooled predictions → sens/spec/acc/auc/cm (§1-5).

## 12. Final metric aggregation (TWO coexisting conventions)

**(a) Selection CV (`run_cv` → ModelResult.macro/per_class):**
- per fold: `class_metrics_from_cm(fold_cm)` → per-class
  sens/spec/prec/f1
- per class: **mean and std across folds** of each metric
  (`_aggregate`, modeling.py:2058-71)
- `macro[k] = mean over classes` of the per-class fold-means;
  `macro std = std over classes` of those means.
  ⇒ displayed "sens/spec/F1" = **mean-of-folds, then mean-over-
  classes** (NOT a pooled-CM metric). Table header says exactly:
  "Mean ± std across CV folds".
- std: `np.std` → **ddof=0** everywhere (modeling.py:2065,2070).
**(b) Pooled (honest check, 3SSE screening, CLI comparisons):**
metrics computed ONCE from a single pooled CM (§1-7).
The GUI banner/Result/reports display (a)'s selection numbers ONLY as
a PRELIMINARY tag; the number shown as performance is the honest
check (b) (2026-09-08 §36 display policy).

## 13. Counting

**One training/evaluation sample = one SPECTRUM (one file).**
Pipeline counts (real data, current loader):
```text
341 files on disk (class/patient/*.csv)
loader: 339 scanned − 4 white/dark references − 18 same-class byte
  duplicates (SHA-1 over float64 of parsed wn,int) − 2 site-token
  files = 317 kept spectra / 72 subjects (Normal 142 / Tumor 175)
spike-flagged: 18 of 317 (excluded from training when despike=off)
standard training n = 299 (317 − 18 flagged)
paired training: 68 patients with both classes → 287 deviation rows
  (4 one-class patients excluded, counted n_unpaired_excluded)
```
Grouping key: first number in the patient folder name
(`Patient_15`, `TDOC15` → `S15`); class label = top-level folder only
(`CLASS_SYNONYMS` normalized). Train/val/test = CV folds of PATIENTS
(no patient crosses folds — audited 0 overlap). Locked eval:
`patient_split` = StratifiedGroupKFold(7) → fold0 test / fold1 val /
rest train.

## 14. Preprocessing (exact sequence, `preprocess_spectrum`)

Per spectrum, in order (all parameters from `PreprocessParams`):
1. `nan_to_num`
2. **despike** (optional, default OFF): Whitaker-Hayes modified
   Z-score of first difference: `z = 0.6745·(d−median(d))/MAD(d)`,
   `|z| > 7` → replace point by mean of neighbors (window ±2).
3. **wavelet denoise** (default ON, sym8 level 4, universal
   threshold): MAD noise estimate from finest band
   `σ = median(|d|)/0.6745`; per-band threshold
   `T = σ·sqrt(2·ln n)` (universal) | BayesShrink | SUREShrink;
   shrinkage soft/hard/garrote
   `w·(1−(T/w)²)`; optional cycle-spinning = mean over ±N shifts.
4. **Savitzky-Golay** `savgol_filter(window=11, poly=3, deriv∈{0,1,2})`.
5. **baseline** (skipped when deriv≠0): ALS
   (Eilers 2005: solve `(W+λ·DᵀD)z = W·y`, weights
   `p·(y>z)+(1−p)·(y≤z)`, λ=1e5 p=0.01, ≤10 iters) or arPLS /
   iarpls / pspline / snip via pybaselines; then `y −= baseline`;
   optional detrend (linear) before baseline.
6. **normalize** (preprocessing.py:284): `vector` = `y/‖y‖₂` (if
   ‖y‖>1e-12, else unchanged); `snv` = `(y−mean)/std`;
   `area` = `y/trapezoid(y)`; `minmax` = `(y−min)/(max−min)`.
7. Order of operations on matrices: **crop columns first**
   (`crop_mask`: keep `crop_min ≤ wn ≤ crop_max`, default 500–2000),
   then 1-6 per row. Optional Phe-1003 wavenumber calibration
   (`align_to_grid`: interp + `calibrate_wn`) BEFORE crop.
**Paired feature** (paired.py:75-83): `ref = mean(preprocessed
Normal spectra of the patient)`; feature = `preprocessed_spectrum −
ref` (optionally `pqn_normalize(s, ref)` first: `s / median(s[ref≠0]/
ref[ref≠0])`).
**Fit location:** steps 1-7 are per-spectrum (no cross-sample fit);
scalers/PCA/feature-selection live INSIDE each pipeline → fitted on
training folds only.

## 15. PCA

`PCA(n_components=0.95, random_state=42)` inside a Pipeline AFTER
`StandardScaler()` (z-score per feature) — fit on the training fold
by sklearn's pipeline mechanics (never globally). Components chosen
by **explained-variance ratio ≥ 95% cumulative**. Variants:
`PCA(n_components=5/10)` (PCA+XGB), `PLSRegression` components for
PLS paths, `TTestSelect` (Welch t + Cohen's d≥0.1, p<0.05) as an
in-pipeline feature selector.

## 16. Model registry (actual constructors + grids)

All `random_state=42` where applicable; grids tuned by pooled
inner-OOF macro-F1 (§10). "PCA" pipelines = StandardScaler →
PCA(0.95) → clf.

| Model | Fixed params | Grid |
|---|---|---|
| PCA+SVM(RBF) | CalibratedSVC | C∈[1,10,100]; gamma∈['scale',0.01] |
| PCA+LDA | LDA | solver lsqr+shrinkage[None,'auto'] / svd |
| PCA+LogReg | balanced, max_iter 5000 | C∈[0.1,1,10] |
| PCA+KNN | KNN | k∈[3,5,7]; weights∈[uniform,distance] |
| PCA+GNB | GaussianNB | var_smoothing∈[1e-9,1e-7] |
| Random Forest | 250 trees, balanced, n_jobs=-1 | depth[None,12]×leaf[1,3] |
| Extra Trees | 250 trees, balanced | depth[None,12]×leaf[1,3] |
| PCA+HGB | lr 0.1, balanced | max_iter∈[100,300] |
| PLS-DA | softmax(PLS scores) | n_components∈[2,3,5,8] |
| Sparse PLS-DA | loading-based top-k | n_select∈[30,100] |
| PCA+MLP | hidden (64,), iter 1500 | alpha∈[1e-3,1e-1] |
| IsolationForest OvR | per-class IF | n_estimators∈[100,300] |
| XGBoost / LGBM / CatBoost | 200 rounds/depth 4 | depth[3,5]×lr[0.05,0.2] |
| 1D-CNN (+5-seed ens) | torch, 40 ep, Adam 1e-3 | (no grid) |
| TabPFN | tabpfn 6.0.6 | (no grid) |
| Peak bands + RF | 16 band means (width 30) → RF 250 | depth[None,8] |
| Spectral+band (B3) | FeatureUnion[PCA ‖ 16 bands]+ET | — |
| Ensemble top-3 | VotingClassifier soft | — |
| Stacked top-3 | LR meta over grouped inner-OOF probs | — |

## 17. Sequential / 3SSE combination math

- Chain: `X → A → OOF-P1 → [X‖P1] → B → OOF-P2 → [X‖P1‖P2] → C`.
  Inter-layer probabilities = **patient-grouped cross_val_predict
  OOF** of the previous layer (leakage-free); each layer's own fit is
  on its full input. Final prediction chains fitted layers, appending
  each layer's probability columns.
- Screening score: `_metrics_from_oof` — pooled-CM macro metrics +
  `f1_mean` (mean per-fold F1) for triples; early-abandon bounds
  `f1_mean`. Default hyperparameters for all candidates (identical →
  fair level comparison).
- Winner: nested grouped validation of top-20 per level
  (`validate_arch` regenerates OOF features inside each outer
  training fold only); picked by (F1, sens, spec). AveragedChain =
  3-seed averaged chain. Candidate counts: N singles + N(N−1) pairs
  + beam-limited triples (default beam 50 pairs → ≤50·(N−2)).

## 18. Aggregation inventory (means/medians)

- fold-mean of per-class metrics → macro mean-over-classes (§12a)
- median of per-fold thresholds (§10)
- median of patient Normal spectra → paired reference (§14)
- median of PQN quotients (§14)
- mean of repeats' OOF probabilities (§11)
- mean over ±cycle wavelet shifts; mean of 5 CNN seeds
- mean per-patient probability → patient-level verdict
  (`_aggregate_patients`: mean P, cut at bundle threshold)
- `patient_level_metrics`: per-patient mean-P argmax → F1/AUC

## 19. Statistical calculations

- **Bootstrap CI** (macro-F1, 1000 resamples, seed 42): resamples
  **groups** (patients) when provided; percentile 2.5/97.5
  (modeling.py:1842+).
- **Wilson CI** (proportion k/n, z=1.96): the Wilson score interval
  formula (clinical.py:328-337).
- **DeLong AUC CI**: tie-corrected midranks, Hanley-McNeil variance
  (clinical.py:339+).
- **McNemar exact**: binomial on discordant pairs b, c.
- **Friedman + Nemenyi**: `friedmanchisquare` on per-fold F1;
  CD = qα·sqrt(k(k+1)/6n), q table k=2..10 hand-tabulated.
- **BH-FDR** for band statistics; **Wilcoxon signed-rank** for
  paired band deltas; **one-way ANOVA** share for between-patient
  variance (`paired.variance_analysis`).
- **PPV/NPV (Bayes)**: `ppv = sens·prev / (sens·prev + (1−spec)·
  (1−prev))`, `npv = spec·(1−prev)/((1−sens)·prev + spec·(1−prev))`
  (clinical.py:106-120), at prevalences 0.05/0.30/0.60.
- **Platt**: `LogisticRegression(C=1e6)` on `logit(p)` → (a, b).
- std: numpy default **ddof=0** in all project-computed stds
  (`_aggregate`, learning curve, seed stability).

## 20. Rounding

Calculation is full float64 throughout. Rounding exists ONLY at
display/persistence: f-strings (`:.3f` metrics, `:.0%` percentages,
`.3f` probabilities) in GUI/tables/reports; `round(x, 4)` for
manifest `nested_honest_f1`; CSV exports write formatted strings.
No rounded value re-enters any computation.

## 21. GUI numbers (current, post-§36 honest-display policy)

- Train banner SENS/SPEC/F1 + Result `r_stats`: **nested honest
  estimate** (`_honest_display_numbers`: live `_honest_result` →
  saved-bundle `nested_honest_f1` → selection-CV tagged PRELIMINARY).
- AUC: honest pooled AUC when present, else winner-OOF ROC AUC.
- Comparison table: selection-CV fold-mean ± std (labeled "SELECTION
  ranking — not the reported performance").
- Predictions: `predict_with_bundle(_many)` probabilities (+ Platt),
  triage NEGATIVE/INDETERMINATE/POSITIVE from operating points
  p≤lo / between / p≥hi.
- PPV/NPV table: Bayes at 3 prevalences (honest sens/spec when
  available).

## 22. CLI / reports

- `reproduce_study.py` / `sequential.py` CLI: the SAME
  `evaluate_models`/`search` code paths (no separate metric math);
  winner text uses fold-mean macro sens/spec/F1; locked mode adds a
  one-shot 70/15/15 patient-split evaluation.
- txt report: honest sens/spec/F1 + protocol row + PRELIMINARY tag;
  HTML report: same numbers + warn box when honest missing; PPV/NPV
  table only when honest sens/spec finite.
- `winner.json/joblib`, `screening.jsonl`, `validated.json`: pooled
  screening metrics (+ `metrics_fair` = Train-page-identical
  evaluate_models re-runs for singles since §34).

## 23. Duplicate implementations (same metric, multiple sites)

| Metric | Sites | Same formula? |
|---|---|---|
| sens/spec/prec/f1 | `modeling.class_metrics_from_cm` (single source) | yes — callers: run_cv, evaluate_pipeline, screening, validate_external |
| macro-F1 | sklearn `f1_score(average='macro')` (fold/scoring) vs CM-derived (tables) | **two coexisting conventions**: fold-mean (selection) vs pooled (honest/screening) — labeled, documented (§12) |
| accuracy | evaluate_pipeline + `_metrics_from_oof` only | same pooled formula |
| AUC | `roc_points` (winner plots), `roc_auc_score` (screening/external), `delong_auc_ci` | equivalent trapezoid/midrank forms |
| counting | loader report vs GUI Data table vs report header | same loader counters |
| spike score / band area / keratin index | preprocessing / biochemistry single implementations | single |

**NOT DETERMINED / AMBIGUOUS:** none — every displayed number was
traced to one of the implementations above. (ViT track
`vit_train/vit_test` computes its own accuracy/F1 from its own
checkpoint splits — separate secondary track, not wired into the
main metrics.)
