# Calculation Audit — Full Project Review & Remediation

**Date:** 2026-09-12
**Scope:** every calculation-bearing module — preprocessing, dataset,
paired, modeling, sequential, study_stats, clinical, biochemistry,
optimize, vit_train (≈4,500 lines of math/statistics)
**Method:** three parallel adversarial audits with numeric spot-checks
against canonical references (scipy closed forms, planted-signal
recovery, real-data verification), followed by line-verified fixes and
17 new regression tests.
**Status:** ALL findings fixed or explicitly guarded. Suite
118/118 (101 prior + 17 audit tests), ruff clean.

---

## 1. Executive summary

| Severity | Found | Fixed |
|---|---|---|
| BUG (wrong numbers) | 5 | 5 |
| RISK (methodological) | 13 | 13 |
| IMPROVE | 18 | 18 |

Two bugs were **number-changing and study-invalidating**:

1. **ALS baseline left-edge collapse** — the DEFAULT baseline method
   produced a spurious upward bowl across ~500–800 cm⁻¹ of every
   preprocessed spectrum. **Every default-preprocessed result (models,
   bundles, studies, plots) must be re-run.**
2. **3SSE level-3 screening metric garbage** — `f1_mean` for all
   triples was computed against an all-zeros label vector (ceiling
   0.497, correlation 0.031 with the true pooled F1 in the shipped
   `screening.jsonl`). All "best 3-model chain" conclusions from
   screening are invalid; the deployed level-2 winner (Random Forest →
   Extra Trees) was honestly validated and is unaffected.
   **Level-3 screening must be re-run.**

---

## 2. BUGs (all fixed)

### B1. ALS baseline second-difference matrix — `preprocessing.py:270`
`diags([1,-2,1], [0,-1,-2])` (sub-diagonal) made rows 0–1 of D
degenerate, adding a shrink-to-zero penalty on the LEFT edge.
Verified: linear baseline 50→150 recovered as b[0]=0.02 (buggy) vs
50.01 (fixed); on real Patient_15 data the buggy baseline was 0.5
counts at 500 cm⁻¹ where arPLS gives 1637.
**Fix:** offsets `[0, 1, 2]` (canonical Eilers orientation).
**NUMBER-CHANGING (default pipeline).** Test: `test_audit_als_left_edge`.

### B2. 3SSE triple label re-encoding — `sequential.py:291`
`_eval_last_layer` ran `np.searchsorted(string_classes, int_labels)`;
NumPy cast the ints to '0'/'1' strings which sort before every
alphabetic label → truth vector all-zeros → per-fold f1_mean garbage →
triple ranking, pruning and abandon cutoffs operated on a meaningless
scale (real artifact evidence: 859/1042 triples with f1_mean < 0.5
while pooled F1 > 0.6).
**Fix:** `ye = np.asarray(y, dtype=int)` (y is pre-encoded by design).
**NUMBER-CHANGING (level-3 screening).** Test:
`test_audit_sequential_f1_mean_matches_pooled`.

### B3. Brier score on non-probability margins — `modeling.py` (`evaluate_pipeline`)
Plain SVC (the optimizer's frequent pick) exposes only
`decision_function`; the pooled "brier" was computed on unbounded
margins (observed 8.64 — Brier is bounded [0,1]).
**Fix:** per-fold probability tracking; Brier = NaN unless every fold
contributed true probabilities (AUC/PR-AUC stay valid on margins —
rank-based). Test: guard verified; suite regression.

### B4. Nemenyi q-table — `study_stats.py`
k=7–10 entries contradicted Demšar 2006 Table 5(a) (verified against
`scipy.stats.studentized_range.ppf(0.95,k,inf)/√2`): 2.936/2.998/3.045/
3.081 → corrected to 2.949/3.031/3.102/3.164. The old values were
anti-conservative (0.4–2.6% too-small CD). The k>10 linear
extrapolation understated CD by ~7% at k=24 (the model registry has 24
models).
**Fix:** corrected table + scipy-based `_nemenyi_q(k)`; conservative
fallback slope if scipy is missing. Test: `test_audit_nemenyi_q_values`.

### B5. TTestSelect silent no-op with list labels — `modeling.py`
`y == classes[0]` on a Python list evaluates to scalar False → the
Welch filter ran on garbage, every p NaN, fallback silently kept ALL
features. Latent (current callers pass arrays) but silent when hit.
**Fix:** `y = np.asarray(y)` + optional `fdr=True` BH correction.
Test: `test_audit_ttestselect_list_y_and_fdr`.

---

## 3. RISKs (all fixed)

| # | Where | Issue → Fix |
|---|---|---|
| R1 | `paired.py` | Normal rows' reference included the row itself: deviations shrunk to exactly (k−1)/k (0.0 for k=1) and mismatched deploy time. → **Leave-one-out reference** for Normal rows; k=1 zero-rows kept but counted (`n_self_ref_rows`). NUMBER-CHANGING (paired mode). |
| R2 | `preprocessing.py` despike | Neighbouring spike re-injected ~25% of itself one point away (replacement read the raw array). → replacement means exclude ALL flagged indices. |
| R3 | `preprocessing.py` despike | mad==0 early-return silently kept the spike. → mean-absolute-deviation fallback. |
| R4 | `preprocessing.py` despike | Index-0 spike undetectable (`prepend=y[0]` forced d[0]=0). → `prepend=y[1]`. |
| R5 | `preprocessing.py` despike | Steep flanks of sharp real peaks flagged & rewritten (σ=8pt peak at SNR 2000 → 46 points). → run-length filter: only flag-runs ≤3 are rewritten (cosmic rays are 1–2 px). |
| R6 | `preprocessing.py` calibrate_wn | No peak-presence check: argmax of noise applied ±15 cm⁻¹ shifts. → prominence test (window max > 5×robust local noise) before shifting; flat/noisy spectra get drift 0. |
| R7 | `preprocessing.py` align_to_grid | `np.interp` constant-extrapolated a too-narrow reference file silently. → coverage guard mirroring `predict_with_bundle`. |
| R8 | `preprocessing.py` pqn | Applied to signed (baseline-removed) spectra; quotients near ref-zeros explode. → weakest 10% of \|ref\| masked from the median (robust, consistent train/deploy). |
| R9 | `sequential.py` warm cutoff | Seeded from pairs' POOLED F1 while triples prune on f1_mean — cross-scale pruning could drop valid triples. → warm seed removed; the exact abandon bound guarantees top-N within the beam. |
| R10 | `modeling.py` `_fit_proba` | Inner-loop fits without groups → CalibratedSVC calibrated across patient boundaries. → `fit_maybe_grouped(..., groups[tr])`. |
| R11 | `modeling.py` `_aggregate` | macro "± std" was the between-CLASS spread, reading like a CV error bar. → macro f1 ± = std of fold F1s (population ddof); per-class spread stays in per_class. |
| R12 | `sequential.py` finalize_winner | Tuned chain tuned on ALL rows then "nested-validated" on the same rows; deployed threshold/calibrator came from that optimistic OOF. → threshold/calibrator always derive from the UNTUNED OOF; `tuned_metrics` flagged `selection_biased`. |
| R13 | `modeling.py` permutation_auc_p | Permuted each patient's FIRST-ROW label — invalid for paired (mixed-label) patients. → mixed-label patients excluded & counted (`n_mixed_excluded`). |
| R14 | `clinical.py` isotonic_compare | ECEs fit and scored on the SAME probabilities (isotonic interpolates its own points → near-zero ECE by construction). → both Platt and isotonic are cross-fitted (k-fold fit/transform, pooled ECE). |
| R15 | `optimize.py`/`vit_train.py` | Winner's selection-CV max quoted as performance. → labeled "selection-CV (optimistic upper bound)" in `preprocess_best.json`, CLI and vit_train output. |
| R16 | `optimize.py`/`vit_train.py` | Flagged-spectra policy mismatched between selection (excluded) and use (kept when despike on). → `exclude_flagged` persisted in the preset; vit_train honors it (legacy presets keep the old rule). |
| R17 | patient truth ties | Dominant-class tie-break to the lowest code was silent. → tie patients counted everywhere (`patient_level_evaluation`, `_metrics_from_oof` rollup). |

## 4. IMPROVEMENTS (all done)

| Where | Change |
|---|---|
| `spike_score` docstring | thresholds corrected (score scales with SNR; operational flag = 300) |
| `phe1003_position` | parabola-vertex apex (was a biased, non-idempotent centroid) |
| `normalize('area')` | trapezoid on the wavenumber axis (was unit-spacing) |
| snip wrapper | lam → max_half_window mapping (λ sweeps were no-ops) |
| `crop_mask` | degenerate crop now WARNS instead of silently keeping everything |
| `paired_features` | case-insensitive/trimmed label matching; unlabeled drops counted |
| `best_f1_threshold` | exact midpoint enumeration over unique scores (was a 99-quantile subsample) |
| `band_stats_paired` | returns BOTH p_raw and p_fdr (raw used to be overwritten) |
| `biochemical_shift` | BH-FDR across bands + pseudo-replication caveat documented; rows carry (…, p_raw, p_fdr) |
| `band_ratios` / `keratin_index` | near-zero/negative denominator guard (NaN, not explosions) |
| `dataset_qc` SNR | first-difference noise ÷ √2 (was 41% high → SNR 29% low) |
| triage boundary | rule-out now strictly `p < thr` (boundary is a positive call under the `p ≥ thr` ROC convention) |
| 3SSE leaderboard | displays the metric it sorts by |
| `dataset.common_grid` | clip quirk fixed; short cohorts warn instead of fabricating resolution |
| vit_train | spiked-exclusion reason printed; preset policy honored |
| gui band report | consumes the 8-field rows; FDR label |

---

## 5. Verified correct (no action)

Independently reproduced against canonical references / planted
signals / real data — left untouched:

- **Statistics:** Wilson CI (bit-identical to closed form), DeLong AUC
  CI with midrank ties (equals sklearn to 1e-12), BH-FDR (statsmodels
  convention), Friedman χ² (canonical formula + tie-averaged ranks),
  Wilcoxon usage (scipy defaults, all-zero guard), McNemar exact
  (matches `scipy.stats.binomtest`), Hanley–McNeil power (past √2 fix
  intact), conformal quantile (⌈(n+1)(1−α)⌉), ECE (count-weighted,
  contiguous bins — past fix intact), PPV/NPV Bayes, decision-curve
  net benefit, triage per-1000 arithmetic.
- **Modeling:** nested StratifiedGroupKFold protocol (0 patient leaks
  across outer/inner folds, verified), OOF pooling across repeats
  (row-aligned means), `class_metrics_from_cm` orientation,
  `mcc_from_cm`, bootstrap_ci (group-resampling intact, percentile
  method), Platt fit-on-OOF/apply-at-deploy separation, threshold↔Platt
  decision-equivalence in bundles, ensemble soft-voting order,
  `fit_maybe_grouped`, grouped GridSearchCV + full-fold refit.
- **Sequential:** probability chaining implementation (OOF-proba
  feature concatenation, identical across workers/validate/deploy),
  AveragedChain seed-averaging semantics, abandon-bound arithmetic,
  significance_of McNemar pairing, prepare_dataset parity with
  reproduce_study.
- **Preprocessing:** PQN scale-invariance (exact), SNV/vector/minmax,
  crop bounds inclusivity, wavelet stack (SURE vs brute force,
  BayesShrink, non-negative garrote, MAD σ from cD1), savgol guards,
  pybaselines dispatch, pipeline ORDER (calibrate→crop→despike→detrend
  →wavelet→SG→baseline→normalize; single normalization pass),
  stage-cache keying, `common_grid` intersection, `to_matrix`
  interpolation (linear function reproduced to 4.5e-13).
- **Paired deploy parity:** `reference_vector` bit-identical to
  training-time reference construction in all four tested regimes.
- **Biochemistry:** band table assignments vs standard literature
  (1003/1090/1335/1445/1655/815/854/1155/1520/2120 all correct);
  LOPO aggregation; operating-point directionality; plotting numerics
  (Youden J, CM percentages, axis conventions).

---

## 6. What must be re-run after these fixes

| Fix | Invalidate |
|---|---|
| ALS offsets (default baseline) | **Every default-preprocessed model, bundle, study, figure** — the feature matrix itself changed (correctly). Re-run standard-mode training, reproduce_study, optimize sweeps, saved bundles. |
| sequential:291 encoding | **3SSE level-3 (triple) screening results** — old `study_run_3sse/screening.jsonl` triples' f1_mean/ranking/pruning; the deployed level-2 winner stands. |
| paired LOO reference | **Paired / paired-pqn mode numbers** (mildly — Normal-class features change). |
| despike overhaul | Any run with despike ON (better spike removal now). |
| macro ± std, Nemenyi CD, biochemical FDR, SNR | Displayed statistics change meaning/value — re-generate reports. |

Everything else is guard-rail work that only affects degenerate inputs.

## 7. Test coverage added

17 regression tests (`test_audit_*` in `raman_app/test_all.py`), one
per fixed area: ALS edge, despike guardrails (×4 sub-checks), wn-
calibrate guard+apex, align guard+area norm, paired LOO+labels, PQN
signed, TTestSelect list-y+FDR, threshold exactness, patient ties +
macro std, permutation mixed-exclusion, sequential f1_mean validity,
Nemenyi values, band p_raw/p_fdr, triage boundary, cross-fitted
isotonic, SNR √2, bioshift FDR + ratio guards. Updated: PQN offset
test (exact vs approximate split), paired-features manual LOO
reference, phantom-cohort assertion now on the FDR p, optimize/honest
tests now plant PEAK-based class signals (a constant DC offset is
legitimately absorbed by the corrected ALS baseline), permutation
tests cover both mixed-exclusion and fully-paired within-patient
fallback.

**Suite: 118/118 passed. ruff: clean.**
