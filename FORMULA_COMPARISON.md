# FORMULA COMPARISON — Raman ML Application

Stage-2 metric audit (2026-09-08), analysis only. Baseline =
`CURRENT_FORMULA_AUDIT.md`. Evidence: `audit_metrics_independent.py`
(ALL PASS: synthetic battery exact; real-data project-vs-independent
equal on every overlapping metric), plus audit rounds 1–4.

Classifications: **KEEP / REPLACE / SUPPLEMENT / REMOVE / KEEP WITH
LIMITATION**. Rule applied: a replacement requires a mathematical,
statistical, ML or scientific reason — never a higher score.

## Master comparison table

| Calculation | Current formula / method | Alternative considered | Recommendation | Reason | Risk |
|---|---|---|---|---|---|
| Sensitivity (per class) | TP/(TP+FN) | — | **KEEP** | standard, independently verified bit-exact | none |
| Specificity (per class) | TN/(TN+FP), TN=total−TP−FN−FP | — | **KEEP** | standard, verified | none |
| Precision (per class) | TP/(TP+FP) | PPV naming, weighted prec | **KEEP** (rename as PPV in reports) | same formula as PPV; zero-denominator → 0.0 documented | none |
| Recall | = sensitivity (one impl.) | separate impl. | **KEEP** (report as "Sensitivity" clinically) | identical math; clinical naming | none |
| F1 (per class) | 2PR/(P+R) | weighted/micro F1 | **KEEP macro-F1** for selection | imbalance (142/175) + both errors matter; micro≈accuracy | none |
| Accuracy | (TP+TN)/N, pooled only | remove | **KEEP** (supplementary) | familiar; harmless when shown with balanced acc | misleading alone under imbalance |
| Balanced accuracy | not computed | (sens+spec)/2 | **SUPPLEMENT** | 55/45 imbalance mild but real; equals mean of the two already-reported numbers | near-redundant with sens+spec pair |
| MCC | not computed | (TP·TN−FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | **SUPPLEMENT** | one-number summary of the whole CM; robust to imbalance; verified 0.5025 on the TP8/TN7/FP2/FN3 reference | high variance at n≈300 (honest-check MCC 0.13); must be labeled |
| Confusion matrix | [[TN,FP],[FN,TP]], rows=actual, labels=sorted classes | — | **KEEP** (verified convention) | reproduced bit-exact in audits | none |
| TP/TN/FP/FN + N reporting | CM shown, counts not spelled out | explicit counts | **SUPPLEMENT** (report TP/TN/FP/FN/N in every final eval) | N=TP+TN+FP+FN must be checkable | none |
| ROC-AUC | pooled OOF proba (col 1 = Tumor); winner plot trapezoid; screening roc_auc_score; DeLong CI | per-fold AUC | **KEEP pooled** | threshold-independent; single estimate with CI; grouping respected via folds | spectrum-level DeLong ignores clustering (see CI row) |
| PR-AUC / AP | AUPRC computed in ROC panels (stage 2) but not in headline | average precision | **SUPPLEMENT** to reported set | sensitivity + FP control matter at 5–60% prevalence; AP .85 vs AUC .82 on GNB | prevalence-dependent interpretation |
| Threshold selection | inner-OOF pooled probs; 99 quantile candidates; max macro-F1; Youden-J tiebreak; keep only if beats 0.5; median per-fold for deploy | A fixed 0.5; B pure Youden; D balanced-acc max; E cost-sensitive; F clinical constraint | **KEEP WITH LIMITATION** (see threshold analysis below) | leakage-free (inner OOF only); F1 objective matches selection; 0.5-guard prevents unstable tuning | measured instability (GNB folds 0.010–0.884) MUST be reported |
| Threshold stability reporting | per-fold list stored (`thresholds`), not surfaced | report mean/median/SD/min/max/IQR | **SUPPLEMENT** (display the stored spread) | data already exists; instability is a real finding | none |
| Fold aggregation — selection | mean of per-fold per-class metric → mean over classes (± std ddof=0) | pooled only | **KEEP for selection** | fold-mean rewards stability across patient groups; labeled "Mean ± std across CV folds" | none |
| Fold aggregation — final performance | ONE pooled CM → metrics once | fold-mean | **KEEP pooled for final/reporting** | every spectrum equally weighted; matches deploy reality; duality already display-separated (§36) | none |
| Patient-level rollup | per-patient MEAN probability ≥ bundle (calibrated) threshold | median; majority vote; recalibrated patient model | **KEEP mean-P** (Jeng 2019 standard); SUPPLEMENT majority-vote count display (already shown) | mean-P uses all evidence, already threshold-consistent; switching to median/vote = unproven benefit | none |
| Primary reporting unit | spectrum-level primary; patient shown in banner/diagnostics | patient primary | **RECLASSIFY: patient-level PRIMARY for clinical reporting, spectrum-level secondary** | the clinical decision unit is the patient; no code change implied (both computed) — a reporting-policy change | patient n=68 → wider CIs; must show both |
| SD across folds | np.std ddof=0 | ddof=1 | **KEEP WITH LIMITATION** | descriptive spread of 5 fold values; ddof choice changes nothing material at k=5; must be LABELED "SD across folds", never "SE" | slight downward bias (0.8× at k=5) |
| Bootstrap CI (macro-F1) | 1000 group(patient)-level resamples, percentile 2.5/97.5 | BCa | **KEEP** (percentile, clustered) | cluster bootstrap is the correct unit; BCa adds little at B≈1000 with smooth statistic | percentile slight undercoverage |
| Wilson CI (props) | Wilson score, z=1.96 | — | **KEEP** | never-degenerate; verified contains point estimate | none |
| DeLong AUC CI | midrank Hanley-McNeil on spectrum rows | cluster-robust SE | **KEEP WITH LIMITATION** | mathematically correct (verified); spectrum-level SE understates width on clustered data → prefer patient bootstrap CI for headline AUC | anti-conservative CI |
| PPV/NPV | Bayes at prev 0.05/0.30/0.60 | dataset-prevalence PPV | **KEEP scenario-based** | sens/spec travel; cohort prevalence (55%) is a design artifact, not a population | misread as measured PPV if unlabeled |
| PLS-DA probability | softmax(PLS scores) | Platt/isotonic on top | **KEEP WITH LIMITATION + LABEL "pseudo-probability"** | monotone score→rank correct for AUC/threshold search; NOT calibrated — never present as calibrated p | misinterpretation |
| Calibration reporting | ECE + isotonic-vs-Platt logged; calibration plots | Brier; slope/intercept | **SUPPLEMENT Brier** for the deployed bundle (measured: GNB .221, RF .190) | one-number calibration quality; trivially computable from OOF | none |
| Platt scaling | LogisticRegression(C=1e6) on logit(p) | isotonic default | **KEEP** (n<500 rule already in place) | small-n standard; isotonic overfits at n≈300 | none |
| PCA | StandardScaler→PCA(0.95) inside pipeline | tune n_components in inner CV; fixed k; no PCA | **KEEP** (leakage-safe, verified fit-inside-fold); component tuning = optional SUPPLEMENT | 0.95 EV is the field default; tuning adds variance to selection | none |
| Preprocessing: vector norm | y/‖y‖₂ | — | **KEEP** | literature standard for SERS; scale-invariance verified (×10 identical) | none |
| Preprocessing: SNV | (y−μ)/σ | — | **KEEP** | standard | none |
| Preprocessing: area norm | y/∫y (trapz) | — | **KEEP** (used by report replication) | matches internship-report protocol | none |
| Preprocessing: minmax | (y−min)/(max−min) | — | **KEEP** | optional preset only | amplifies noise — acceptable as opt-in |
| Despike | Whitaker-Hayes modified Z (0.6745·(d−med)/MAD, \|z\|>7) | — | **KEEP** | canonical cosmic-ray method | none |
| Wavelet | MAD σ + universal/bayes/sure threshold, soft/hard/garrote, cycle-spin | — | **KEEP** | literature-backed, measured −23% RMSE for bayes/sure | none |
| Savitzky-Golay | savgol(11,3,deriv) | — | **KEEP** | standard | none |
| Baseline ALS/arPLS | Eilers 2nd-diff penalty / pybaselines | — | **KEEP** | standard; tuned λ documented | none |
| PQN | y / median(y/ref) | — | **KEEP** | Dieterle 2006 verbatim | none |
| Model-selection objective | pooled inner-OOF **macro-F1** | balanced acc; MCC; AUC; multi-objective | **KEEP macro-F1** (analysis below) | alignment with class-balanced clinical goal; stable, threshold-consistent | none |
| GIGO guards / zero-signal checks | raw + post-preprocess validation | — | **KEEP** (audits 2–4) | prevents silent confident garbage | over-strict rejection — thresholds set 6+ orders below real signal |

## Threshold strategy analysis (A–F)

| Strategy | Leakage risk | Stability | Clinical meaning | Verdict |
|---|---|---|---|---|
| A. fixed 0.5 | none | perfect | ignores prevalence/imbalance | baseline only |
| B. pure Youden J | low (inner OOF) | moderate | maximizes both-error sum | already the tie-break |
| C. F1-max (current) | **none — inner OOF only, never outer test** | **measured UNSTABLE for probability-miscalibrated models** (GNB folds 0.010/0.884/0.058/0.096/0.021) | matches the selection objective | KEEP + report the spread |
| D. balanced-acc max | low | similar instability | symmetric errors | redundant with C here (macro-F1 ≈ balanced on 2 classes) |
| E. cost-sensitive | low | depends on costs | the clinically right frame once costs are agreed | future work — needs cost elicitation |
| F. clinical constraint (sens≥90% rule-out) | none — already computed SEPARATELY as operating points | high | triage semantics | KEEP as the deployed triage layer (complements C) |

**Threshold recommendation:** keep C for model selection/threshold
formation; keep F (operating points) for clinical deployment; ALWAYS
publish the per-fold threshold list (currently stored but not shown).

## Fold-aggregation decision matrix

| Use | Aggregation | Justification |
|---|---|---|
| Model selection | fold-mean ± SD | penalizes group instability |
| Final performance | pooled CM | equal spectrum weighting = deploy reality |
| Clinical reporting | pooled spectrum-level + patient-level rollup | decision unit |
| GUI | honest pooled (§36 policy) | single non-misleading number |
| Publication | pooled + both CIs (cluster bootstrap) | standard TRIPOD-style |

## Patient-aggregation comparison

Mean-P (current) vs median-P vs majority vote vs patient-recalibrated:
mean-P is the literature standard (Jeng 2019), uses the calibrated
probability scale consistently with the stored threshold, and the app
already displays vote counts alongside. No change recommended;
patient-level recalibration needs more patients than 68 to be stable.

## Real-data CURRENT vs SUPPLEMENT (same patients, folds, predictions —
formula effect isolated; from audit_metrics_independent.py run)

| Metric | GNB (paired) | RF (paired) | Honest check |
|---|---:|---:|---:|
| Sensitivity (macro) | 0.7692 | 0.7334 | 0.5659 |
| Specificity (macro) | 0.7692 | 0.7334 | 0.5659 |
| F1 (macro) | 0.7684 | 0.7323 | 0.5638 |
| Accuracy *(supplement)* | 0.7735 | 0.7352 | 0.5719 |
| Balanced acc *(supplement)* | 0.7709 | 0.7365 | 0.5659 |
| MCC *(supplement)* | 0.5414 | 0.4701 | 0.1321 |
| ROC-AUC *(supplement headline)* | 0.8168 | 0.8065 | — |
| PR-AP *(supplement)* | 0.8475 | 0.8456 | — |
| Brier *(supplement)* | 0.2207 | 0.1896 | — |

No existing metric was replaced, so no metric "decreased" — the
supplements are computed on identical predictions.
