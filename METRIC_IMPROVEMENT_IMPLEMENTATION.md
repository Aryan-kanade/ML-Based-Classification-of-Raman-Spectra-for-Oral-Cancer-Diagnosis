# METRIC IMPROVEMENT IMPLEMENTATION — Raman ML Application

Date: 2026-09-08 · Base: the Round-4 release-gate tree (commit
`4c01fc7` + working changes). **Reporting supplements only — no
existing formula, prediction, probability, threshold, confusion
matrix, fold, seed, dataset, or model was changed.**

## Changes implemented

1. **`modeling.py` — new functions** (added after
   `class_metrics_from_cm`; nothing existing touched):
   - `balanced_accuracy_from_cm` = mean per-class recall
     (= (sens+spec)/2 binary)
   - `mcc_from_cm` = (TP·TN−FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN)),
     0.0 on zero denominator, NaN for non-binary (documented)
   - `brier_score` = mean(p_i − y_i)² over finite rows — computed on
     PROBABILITIES, never hard labels
   - PR-AUC: **reuses** existing `pr_points(...)[2]`
     (average_precision_score on p(Tumor); positive class = Tumor)
   - `threshold_stats(thresholds)` → per-fold values + min/max/mean/
     median/SD(ddof=0)/IQR/max-min spread ratio/n_kept; `unstable`
     flag computed from the data (>10× spread OR SD>0.2)
   - `patient_level_evaluation` — per-patient MEAN probability
     (aggregation UNCHANGED) at the EXISTING deployed threshold →
     patient CM → sens/spec/PPV/F1/accuracy/balanced-accuracy/MCC/
     ROC-AUC/PR-AUC/Brier + TP/TN/FP/FN/N; patient truth = dominant
     class (same rule as the existing rollup)
2. **`evaluate_pipeline` (honest check)** return dict GAINS
   `balanced_accuracy, mcc, brier, pr_auc` from the SAME pooled
   rows/probabilities — existing keys byte-identical.
3. **`sequential._metrics_from_oof`** gains additive keys
   (`bal_acc, mcc, brier, pr_auc`) on its pooled OOF; old
   screening.jsonl rows simply lack them (viewer is `.get`-safe).
4. **GUI (additive only)**: new Result-page card "Supplementary
   scientific metrics" — SPECTRUM LEVEL block (honest: bacc, MCC,
   PR-AUC, Brier, ECE + explicit TP/TN/FP/FN/N with the
   N=TN+FP+FN+TP check) and PATIENT LEVEL block (labeled, never
   mixed); threshold-stability block (per-fold list + stats +
   ⚠ WARNING when the calculated flag fires); PLS-DA winners show
   "Brier n/a — PSEUDO-probabilities".
5. **Reports**: txt + HTML gain the same sections with explicit
   `SPECTRUM LEVEL` / `PATIENT LEVEL` labels; `freeze_study` manifest
   gains a machine-readable `honest_metrics` block
   (`evaluation_unit`, `balanced_accuracy`, `mcc`, `roc_auc`,
   `pr_auc`, `brier_score`, `tp/tn/fp/fn/n`, `threshold_{min,max,
   mean,median,sd,iqr}`) — additive keys; historical results not
   overwritten.
6. **Tests**: `test_supplementary_metrics_formulas` (exact ground
   truth TP=8/TN=7/FP=2/FN=3 → acc .75, bacc .7525252, MCC .5025189;
   perfect/all-positive/all-negative/zero-denominator/empty/one-class
   safety; Brier-on-probabilities; threshold_stats incl. the
   instability flag; patient-level eval incl. dominant-tie truth;
   honest-check new keys + bacc==mean(sens,spec)) and
   `test_supplement_report_and_manifest` (card labels, report
   writers, invariance of surrounding content).

## Changes deliberately NOT implemented

- Model selection stays **macro-F1** (not MCC/balanced-accuracy/AUC).
- Threshold selection algorithm untouched (only its stability is now
  visible); no threshold was changed anywhere.
- No ddof change; fold SD stays ddof=0, now always labeled "SD",
  never "standard error".
- No new calibration methodology: ECE is the EXISTING
  `clinical.ece` — documentation added: **10 equal-count (quantile)
  CONTIGUOUS bins, mass-weighted |predicted−observed|; empty bins
  impossible by construction (equal-count blocking of finite
  probabilities)**; calibration is measured on OOF/held-out
  probabilities only, never fitted on outer test data.
- No replacement of any existing metric; no removals.

## Existing formulas preserved (verified)

Accuracy, precision, sensitivity/recall, specificity, F1 (per-class
and macro), confusion matrix [[TN,FP],[FN,TP]] with classes sorted
(0=Normal, 1=Tumor, positive=Tumor), probabilities
(`predict_proba`, Platt), CV structure/folds/seeds, threshold
pipeline, PPV/NPV Bayes, Wilson/DeLong/bootstrap CIs, PCA(0.95)
in-fold, all preprocessing. **Regression proof below.**

## New formulas (exact)

```text
BalancedAccuracy = mean_c( TP_c / (TP_c + FN_c) )
MCC(binary) = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
PR-AUC      = Σ_n (R_n − R_{n−1}) · P_n        (average precision, p(Tumor))
Brier       = (1/N) Σ (p_i − y_i)²             (y=1 Tumor; probabilities only)
ECE         = Σ_bins (n_b/N)·|mean_p_b − obs_b| (10 equal-count contiguous bins)
```

## Patient-level metrics (NEW card, semantics documented)

Per-patient probability = MEAN of their spectrum probabilities
(unchanged aggregation); patient positive when mean-P ≥ the deployed
(calibrated) threshold; patient truth = DOMINANT class.  **Known
limitation surfaced honestly**: on PAIRED winners, patients carry
both site types, ties resolve to Normal, and a spectrum-level tuned
threshold does not necessarily transfer to mean-P (measured on the
GNB paired winner: sens .77 / spec .15 / MCC −.10) — this mirrors the
pre-existing open item about margin-mode patient rollup semantics
(Brain §18 item 9) and is now VISIBLE instead of hidden. Standard
mode + threshold-consistent winners behave as expected (test ground
truth passes).

## Spectrum-level metrics

Honest nested estimate (existing keys unchanged) + supplements:
bacc/MCC/PR-AUC/Brier/ECE + explicit TP/TN/FP/FN/N with the
N = TN+FP+FN+TP invariant check (displays "N-CHECK FAILED!" and
never silently continues).

## Threshold stability (now visible, computed — never hardcoded)

Real GNB paired winner: per-fold [0.010, 0.884, 0.058, 0.096, 0.021]
→ median .0578, SD .362, spread 91.4× → **unstable flag = True** →
warning displayed in GUI + reports. RF: [—, —, 0.543, 0.570, —] →
stable. The warning criterion (>10× or SD>0.2) is computed from the
data; the GNB numbers above are measurements, not constants.

## Calibration

Brier (new) + ECE (existing, now documented) + existing calibration
curves and Platt/isotonic comparison unchanged. PLS-DA probabilities
labeled pseudo-probabilities; Brier reported n/a for PLS-DA winners.

## Tests

- New: 2 tests (formula battery incl. TP8/TN7/FP2/FN3 exact,
  degenerate safety, patient-level, threshold stats, report/manifest)
- Full suite after implementation: **96/96 unit + 19/19 deep + GUI
  PASSED + ruff clean** (all exit 0; `.zcode/i5_*.log`)

## Regression comparison / Scientific invariance (the success
criterion)

Rerun on real data, same seeds/folds (`.zcode` invariance run):

| Quantity | Baseline | After | Verdict |
|---|---|---|---|
| RF paired pooled CM | [[88,39],[36,124]] | identical | ✅ |
| RF per-fold thresholds | [—,—,.543,.570,—] | identical | ✅ |
| RF deployed threshold | 0.5566 | 0.5566 | ✅ |
| GNB fold-mean sens/F1 | .769/.768 | .7692/.7684 | ✅ |
| GNB pooled CM | [[95,32],[33,127]] | identical | ✅ |
| Honest existing keys | .566/.566/.572/.564 | identical | ✅ |
| Extra Trees k=3 F1 | 0.760 | 0.760 | ✅ |
| Dataset counts / folds / seeds | — | untouched | ✅ |

(The single "FAIL" during verification was a rounding bug in my own
comparison constants — actual values were bit-identical; corrected
comparison records PASS. No STOP-worthy change existed.)
Supplement values on identical predictions: honest bacc .5659,
MCC .1321, Brier .2539, PR-AUC .6392; GNB spectrum-level bacc .7709,
MCC .5414, PR-AP .8475, Brier .2207 (matches the pre-implementation
independent audit values).

## Remaining limitations

1. Patient-level evaluation on PAIRED winners inherits the known
   both-sites-per-patient rollup ambiguity (dominant-class truth) —
   surfaced with labels; a dedicated margin-mode patient semantics
   pass remains future work.
2. MCC on n≈300 honest predictions has wide uncertainty (0.13 here)
   — reported with that caveat, not as a verdict.
3. Spectrum-level DeLong CI remains anti-conservative under
   clustering (patient bootstrap remains the headline CI).
4. Threshold instability itself is unchanged BY DESIGN (visibility
   only); fixing it (e.g. probability calibration before threshold
   search) is a separate scientific experiment.

## Final table

| Metric/Area | Before | After | Changed prediction? | Changed formula? |
|---|---|---|---|---|
| Accuracy | pooled formula | same + shown with bacc | NO | NO |
| Sensitivity | TP/(TP+FN) | unchanged | NO | NO |
| Specificity | TN/(TN+FP) | unchanged | NO | NO |
| Precision | TP/(TP+FP) | unchanged | NO | NO |
| F1 | macro 2PR/(P+R) | unchanged (still selection objective) | NO | NO |
| Balanced Accuracy | N/A | mean per-class recall | NO | NEW |
| MCC | N/A | binary formula, zero-den safe | NO | NEW |
| ROC-AUC | pooled OOF p(Tumor) | unchanged (+documented positive class) | NO | NO |
| PR-AUC | in panels only | reported (avg precision, p(Tumor)) | NO | NEW (reuse of existing AP) |
| Brier | N/A | mean(p−y)² on probabilities | NO | NEW |
| Threshold | inner-OOF F1-max/median | unchanged + stability stats/warning | NO | NO |
| Confusion Matrix | [[TN,FP],[FN,TP]] | unchanged + explicit counts & N-check | NO | NO |
| Patient level | banner rollup only | full labeled metric card | NO | NEW (unchanged aggregation) |
| SD | ddof=0 | unchanged, labeled "SD" | NO | NO |

**Verdict: PASS** — existing formulas unchanged, existing
predictions/probabilities/thresholds/confusion matrices unchanged
(bit-identical), new formulas independently tested, patient/spectrum
levels clearly separated, threshold instability visible, no leakage
introduced.
