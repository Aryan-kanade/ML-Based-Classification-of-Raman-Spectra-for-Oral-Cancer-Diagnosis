# FORMULA RECOMMENDATIONS — Raman ML Application

Stage-2 verdict (2026-09-08). Analysis only — **no production code was
modified**. Basis: `CURRENT_FORMULA_AUDIT.md`,
`audit_metrics_independent.py` (ALL PASS),
`FORMULA_COMPARISON.md`, audit rounds 1–4.

Priority order honored: CORRECTNESS → VALIDITY → REPRODUCIBILITY →
CLINICAL INTERPRETABILITY → PERFORMANCE.

---

## KEEP (unchanged — verified correct)

1. **All core formulas**: sensitivity TP/(TP+FN), specificity
   TN/(TN+FP), precision TP/(TP+FP), F1 2PR/(P+R), per-class from CM;
   confusion matrix `[[TN,FP],[FN,TP]]`, rows=actual, classes sorted
   (0=Normal, 1=Tumor). *Evidence: independent reimplementation
   matches project output exactly on real data; CMs reproduced
   bit-exact across 4 audit rounds.*
2. **Macro-F1 as the model-selection objective** — imbalance-aware,
   symmetric in the two errors, consistent with the tuned threshold's
   objective.
3. **The aggregation duality** — fold-mean ± SD for *selection*
   (labeled "Mean ± std across CV folds"), ONE pooled CM for *final
   performance* (deploy reality). Already display-separated by §36.
4. **Threshold pipeline** — inner-OOF-only tuning (zero outer-test
   leakage, audited), F1-max with Youden tiebreak, keep-if-beats-0.5,
   median-of-folds for deployment.
5. **Probabilities** — `predict_proba` with sorted-class columns;
   Platt for n<500; operating-points triage layer (sens≥90%
   rule-out / spec≥90% rule-in) as the clinical decision rule.
6. **Statistics** — cluster (patient-level) percentile bootstrap for
   F1 CI; Wilson score intervals; DeLong AUC CI; exact McNemar;
   Friedman+Nemenyi; BH-FDR; PPV/NPV Bayes at scenario prevalences.
7. **All preprocessing formulas** — vector/SNV/area/minmax norms,
   Whitaker-Hayes despike, wavelet (universal/bayes/sure, soft/hard/
   garrote, cycle-spin), Savitzky-Golay, ALS/arPLS baselines, PQN,
   crop-first ordering, per-spectrum (leakage-free) fitting.
8. **PCA(0.95) inside pipelines** — fitted on training folds only
   (audited); never global.
9. **Paired feature** `spectrum − mean(patient's preprocessed
   normals)` (+optional PQN) and patient rollup by **mean-P at the
   bundle's calibrated threshold**.

## REPLACE

**None.** No current formula is mathematically incorrect,
statistically inappropriate, or scientifically misleading *as
implemented and labeled*. (Candidate replacements were all evaluated
in FORMULA_COMPARISON.md and rejected for cause.)

## SUPPLEMENT (add; all computed on identical predictions — values
already measured in audit_metrics_independent.py)

| # | Addition | Where | Measured value (GNB / RF / honest) |
|---|---|---|---|
| 1 | **Balanced accuracy** = (sens+spec)/2 | final-eval tables | .7709 / .7365 / .5659 |
| 2 | **MCC** = (TP·TN−FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN)), 0 on zero-denominator | final-eval tables | .5414 / .4701 / .1321 |
| 3 | **PR average precision** (step-wise, pooled OOF) | final-eval tables | .8475 / .8456 / — |
| 4 | **Brier score** of the deployed bundle's OOF probabilities | model card / report | .2207 / .1896 / — |
| 5 | **Explicit TP/TN/FP/FN/N row** (N=TP+TN+FP+FN check) in every final evaluation + reports | tables | — |
| 6 | **Per-fold threshold list + SD/min/max/IQR** displayed next to the median threshold (data already stored in `ModelResult.thresholds`) | Train page / report | GNB [0.010, 0.884, 0.058, 0.096, 0.021] — max/min ratio ≈ 88 |
| 7 | **Patient-level PRIMARY framing for clinical reporting** (spectrum secondary) — both already computed; this is a reporting-order policy, not a code change | report/abstract | — |
| 8 | Label PLS-DA outputs **"pseudo-probability"** wherever shown | tables/docs | — |

## REMOVE

**None.** No metric is misleading given current labels; accuracy
stays (supplementary) because it is universally understood and shown
alongside balanced accuracy.

## KEEP WITH LIMITATION (documented, not hidden)

1. **Threshold instability** — measured per-fold spread is large for
   models whose raw probabilities are miscalibrated (GNB 0.010–0.884;
   RF kept tuned thresholds in only 2/5 folds). The deployed median
   (GNB 0.058) is therefore seed-sensitive. Report the spread
   (Supplement 6); the honest check + triage operating points are the
   stable decision layer.
2. **SD across folds uses ddof=0** — descriptive spread of k=5 fold
   values (≈0.8× the sample SD). Keep, but ALWAYS label "SD across
   folds", never "standard error".
3. **DeLong CI is spectrum-level** — mathematically correct but
   anti-conservative on clustered data; use the patient-level
   bootstrap CI as the headline AUC interval.
4. **PLS-DA softmax pseudo-probabilities** — fine for ranking/
   thresholds; never present as calibrated (Supplement 8 label).
5. **Percentile (not BCa) bootstrap** — adequate at B=1000 with a
   smooth statistic; revisit only if CIs drive decisions at the
   margin.
6. **GIGO boundary** — finite-signal physically-odd spectra can
   receive confident predictions (documented rounds 3–4; deploy spike
   flagging recommended someday).

## LIMITATIONS OF THIS AUDIT

- MCC/balanced-accuracy/PR-AP values are point estimates on one seed;
  their own CIs (bootstrap) were not computed for the supplements.
- Cost-sensitive thresholds (strategy E) require clinical cost
  elicitation that does not exist yet — flagged as future work, not
  evaluable now.
- No external cohort: every recommendation is internal-validation
  scoped.

---

## FINAL ANSWER — the exact metric system for a serious
scientific/clinical Raman study (this project)

**Model selection:** maximize **pooled inner-OOF macro-F1**
`mean_c 2·prec_c·sens_c/(prec_c+sens_c)` under patient-grouped nested
CV — KEEP current. Tie-break: macro sensitivity, then specificity
(current).

**Final evaluation (one number each, from ONE pooled out-of-fold
confusion matrix, patient-grouped nested CV):**

```text
TP, TN, FP, FN, N  with N = TP+TN+FP+FN
Sensitivity = TP/(TP+FN)            [Tumor detection rate]
Specificity = TN/(TN+FP)            [Normal detection rate]
PPV         = TP/(TP+FP)
F1 (macro)  = mean_c 2·prec_c·sens_c/(prec_c+sens_c)
Balanced acc = (Sensitivity + Specificity)/2
MCC = (TP·TN − FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN))
Accuracy = (TP+TN)/N                [supplementary]
```

**Rank metrics (pooled OOF probabilities, positive=Tumor):**

```text
ROC-AUC  (trapezoid/midrank)   + patient-cluster bootstrap 95% CI
PR-AP    (step-wise average precision)
```

**Calibration:** Brier score + calibration curve (Platt-applied
probabilities); ECE already logged.

**Uncertainty:** 95% cluster (patient-level) percentile bootstrap
(1000 resamples) for every headline metric; Wilson intervals for any
binomial proportion; exact McNemar for model comparisons.

**Threshold:** tuned on inner-OOF macro-F1 (current), deployed via
clinical operating points (rule-out sens≥90%, rule-in spec≥90%);
per-fold threshold spread always reported.

**Clinical reporting:** PPV/NPV at scenario prevalences (0.05 / 0.30
/ 0.60) via Bayes; **patient-level results PRIMARY** (per-patient
mean probability at the calibrated threshold), spectrum-level
secondary.

**Rounding:** full float64 computation; rounding display-only.

*Every formula above that the project already implements is verified
(bit-exact vs independent reimplementation); the supplements are
specified exactly and measured on the current predictions.*
