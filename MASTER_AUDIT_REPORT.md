# MASTER EXECUTION AUDIT REPORT — Raman Spectroscopy ML Application

Audit 2 ("11/10" round) — 2026-09-08, follows Audit 1 (§37 of Brain.md).
Baseline: commit `4c01fc7` + 4 uncommitted working files (the same tree
was audited end-to-end; nothing was destroyed). Python 3.14.6, Windows,
pinned deps. All evidence captured live this session in
`.zcode/audit_*.log` / `final_*.log`.

**Method: execution-first.** Every claim below was produced by running
the real code on the real dataset (or purpose-built adversarial data)
and, for all scientific numbers, recomputing independently with
numpy/sklearn — never via the app's own helpers. Prior-audit claims
were re-verified, not reused.

---

## Verdict: ✅ PASS (with 2 LOW conditions)

The application is genuinely correct end-to-end. The full model
registry executes; metrics reproduce bit-exactly under independent
recomputation; there is no patient-level or preprocessing leakage; the
deploy path is feature-identical to training; persistence is bit-exact
across processes; the CLI and the direct training path produce
identical numbers. **Two real defects were found and fixed during this
audit** (BUG-1, BUG-2 — both input-hardening, neither affected any
scientific result computed from the real dataset), plus one LOW
observation accepted as-is (A-1).

---

## 1. Test suites (re-executed this round)

| Suite | Result | Evidence |
|---|---|---|
| `test_all.py` | **94/94 PASS** (93 prior + 1 new regression) | `final_test_all.log`, EXIT:0 |
| `deep_test.py` (adversarial GUI) | **19/19 PASS** | `final_deep.log`, EXIT:0 |
| `gui_test.py` | **GUI TEST PASSED** | `final_gui.log`, EXIT:0 |
| `ruff check .` | clean | inline |

## 2. Full model registry — every model executed (R1)

`evaluate_models(ALL_MODEL_NAMES)` on REAL paired data (287×780, 64
patients), k=3, seed 42: **25/25 models trained + predicted, 0
failures** (incl. XGBoost, LightGBM, CatBoost, 1D-CNN ×2, TabPFN,
ensembles). Winner: Extra Trees F1 0.760. Probabilities well-formed for
every model. (`audit_full_registry.log`)

## 3. Scientific attack tests (R2) — through the real training path

| Trap | Result | Verdict |
|---|---|---|
| Perfect separation | F1 = 1.000 | PASS — real signal is learned |
| Pure-noise features | F1 = 0.451 | PASS — chance, no leakage |
| Shuffled labels (real features) | F1 = 0.474 | PASS — chance |
| Constant spectra | F1 = 0.328, no crash (warnings only) | PASS |
| One class only | clean `ValueError: Need at least 2 distinct classes` | PASS |
| Tiny (3 / 2 patients) | loud "All models failed", no corruption | PASS (msg quality LOW note) |

## 4. Adversarial data files (R3) — 12 malformed cases vs real loaders

Empty / garbage-text / one-column / too-few-points → **clean per-file
skip with logged reason** (single loader: `ValueError`; folder loader:
skip + print; clinical loader: counted in `load_errors`). NaN rows,
Inf, unsorted axis, 3-column, extreme values, Unicode filename → load
correctly (axis auto-sorted; quality handled downstream by spike
flags / dedupe). `.log` extension filtered. **Zero crashes, zero
silent poison of neighboring files.**

## 5. Leakage (re-verified independently)

- Fold rosters (real groups, StratifiedGroupKFold k=5): patient
  overlap **[0,0,0,0,0]**
- Fingerprint trap (60 synthetic patients × 5 seeds): mean OOF
  accuracy **0.488 ≈ chance** — patient-unique offsets unlearnable
- Project leakage tests re-executed live: all PASS

## 6. Metric correctness (re-verified)

Independent recompute from raw `y_true_encoded` + `oof_proba` +
per-fold thresholds (same splitter): confusion matrices reproduce
**bit-exactly** in both modes (incl. threshold-bearing winners);
honest-check pooled sens/spec/acc match to all printed digits;
app "macro" = documented mean-of-folds aggregation. Probability
invariants hold: range [0,1] ✓, rows sum to 1 ✓, shape n×classes ✓,
CM total = n ✓.

## 7. Threshold & boundary semantics

Bundles store the median of per-fold tuned thresholds (fold list may
contain `None` = that fold kept 0.5) — predictions use `p ≥ thr`;
`threshold=None` bundles use 0.5/argmax. Verified by exact CM
replication. Vector-norm scale invariance: ×10 intensity → identical
prediction (fp noise only).

## 8. Persistence, restart, deployment parity (R4/R5/R7)

- Train → save → **fresh process** → load → re-predict the same 8
  files: **MAX |Δp| = 0.000e+00**
- Train vs deploy feature matrices for the same files:
  **MAX ABS DIFF = 0.000e+00**
- Axis-guard verified: a spectrum not spanning the training grid is
  REFUSED ("interpolation would fabricate…") — no silent extrapolation
- **CLI ↔ direct-path parity: EXACT** — `reproduce_study.py --mini
  --seed 42 --folds 5` reports PCA+LDA F1 0.510 / sens 0.516,
  identical to the direct `evaluate_models` run

## 9. BUGS FOUND AND FIXED THIS ROUND

### BUG-1 (HIGH) — non-finite / degenerate spectra were predicted with MAXIMAL confidence
- **Evidence:** `predict_with_bundle(all-NaN spectrum)` →
  `Tumor p=1.0000`; all-zero input → `Tumor p=1.0000` (silent garbage;
  `normalize()` keeps zeros finite, so no downstream error fired)
- **Root cause:** no input validation at the deploy boundary
- **Fix:** `modeling.predict_with_bundle` and
  `predict_with_bundle_many` now reject non-finite input (with a
  count) and zero-signal features with actionable `ValueError`s;
  `paired.reference_vector` SKIPS non-finite reference files with a
  logged reason (a poisoned reference previously contaminated every
  prediction built from it)
- **Regression:** `test_predict_rejects_nonfinite_and_degenerate`
  (single + batched + reference paths); all 5 deploy-path neighbor
  tests re-pass. One existing test used a `zeros()` DUMMY spectrum
  purely to exercise crop-length mechanics — its input was updated to
  `ones()` (assertions unchanged; all-zero input is now correctly
  rejected BY DESIGN and that rejection is itself pinned by the new
  regression test)
- **Scientific impact:** none on real data (0 files contain NaN) —
  this was a deploy-safety hole, now closed

### BUG-2 (LOW) — `plot_count_bars` crashed on NaN counts
- **Evidence:** `plot_count_bars(ax, ["a","b"], [nan, 2.0])` →
  `ValueError: cannot convert float NaN to integer`
- **Reachability:** callers pass integer counts — defensive gap only
- **Fix:** sanitize non-finite counts to 0; re-verified no-crash;
  ruff clean

### A-1 (LOW, accepted) — loader `files_scanned` = 339 vs 341 files on disk
All 24 drops are semantically explained (4 white-refs + 18 duplicates
+ 2 site-token files; arithmetic 339−4−18=317 exact); the 2-file scan
count gap has zero training impact. Unexplained count semantics only.

## 10. Resources, reproducibility, performance

- Memory loop (4× train→predict in one process): preprocessing stage
  cache **flat at 598 entries** (bounded, deterministic); cycles
  stable. (Win32 working-set API could not be read from this host —
  precise RSS NOT TESTED, honestly stated.)
- Reproducibility: RF trained in two separate processes → identical
  confusion matrices; full CLI rerun reproduces prior numbers exactly
- Perf (real data): load 0.4 s · paired prep 1.7 s · 1-model nested
  train 1.2 s · predict 0.02 ms/spectrum · full 25-model registry
  ~16 min · 3SSE 15-arch CLI run (screen+validate+finalize) exit 0

## 11. Fake / hardcode / dead-code (re-scanned)

Pattern scan clean (Qt placeholders, torch `model.eval()`, optional-dep
stubs only). AST dead-scan: only framework overrides and
runner-invoked tests. Security: no secrets; `joblib.load` of
user-picked bundles = documented local trust boundary.

## 12. NOT TESTED (explicit)

1. Interactive human-visibility GUI inspection (covered by offscreen
   click automation: 19 deep + 8 gui steps)
2. Precise working-set RSS growth (Win32 API unreadable from this
   host; stage-cache boundedness used as proxy)
3. External-cohort generalization (no second dataset exists)
4. GPU-timing variance (single machine)

## 13. Final scorecard (evidence-weighted)

```
Data Integrity:              98   (independent scan exact; A-1 count note)
GUI Functionality:           95   (27 automated scenarios; no human-eye pass)
Backend Integration:        100   (every path executed; CLI/GUI identical)
Preprocessing:              100   (bit-exact train/deploy parity)
Leakage Prevention:         100   (0 overlap; fingerprint trap ≈ chance)
Model Implementation:       100   (25/25 registry executed)
Cross Validation:           100   (rosters + OOF verified)
Metric Correctness:         100   (bit-exact independent recompute)
Threshold Correctness:      100   (per-fold semantics replicated)
Model Combinations:         100   (3SSE chain math exact; real runs)
Paired Mode:                100   (real runs + reference traps)
Sequential/3SSE:            100   (fresh CLI run + artifact consistency)
Persistence:                100   (cross-process 0.0)
Deployment Parity:          100   (feature diff 0.0)
Error Handling:              95   (fail-safe everywhere; 2 msg-quality notes)
Async/Workers:               95   (19 scenarios; queue gating tested)
Security:                    95   (clean; local trust boundary documented)
Performance:                 90   (measured where meaningful; RSS proxy)
Reproducibility:            100   (bit-identical across processes)
Test Coverage:               95   (94 unit + 19 + 8 + CI floor 60%)

OVERALL VERIFIED QUALITY:    97/100
```

## 14. Final verdict

- End-to-end works: **YES (executed, not inferred)**
- Buttons that only look functional: **NONE FOUND**
- Disconnected backends: **NONE FOUND**
- Patient/data leakage: **NONE (proven 3 independent ways)**
- Metrics truthful: **YES (bit-exact recompute)**
- Training = deployment: **YES (0.0 difference)**
- Hardcoded/fake results: **NONE**
- Fixed this round: **BUG-1 (deploy input validation), BUG-2 (plot
  NaN)** — both regression-covered; full suite green after

**# ✅ PASS**

---

# ROUND 3 — FINAL RELEASE / REGRESSION / SCIENTIFIC INTEGRITY AUDIT

Baseline for this round: same tree as Round 2 + Round-2 fixes. Every
Round-1/2 claim was re-verified with fresh execution.

## Round-2 fix verification (attacked, not trusted)

**BUG-1 exhaustive battery — 20 input classes × single/batch/reference
paths:** every NaN placement (one/multi/all/begin/end/in-crop/out-of-
crop/mixed), ±inf, mixed finite/inf/NaN, huge-finite, all-zeros,
subnormal scales, one-point spectra.

- All 13 non-finite/inf cases: **rejected on every path** ✅
- **3 residuals of the same bug class found and FIXED:**
  1. one-nonzero-point (one-hot) spectrum → was p(Tumor)=1.0 CONFIDENT
  2. subnormal-scale inputs (×1e-300 / 1e-300-constant) → were
     p=1.0 CONFIDENT (float-underflow in PCA after normalize()'s
     small-norm passthrough)
  3. an all-zero REFERENCE file warped the paired reference mean
     (max shift 0.057) instead of being skipped
- Fix: layered guards — raw-input sparsity/scale check
  (`_reject_raw_degenerate`, BEFORE preprocessing: the wavelet stage
  smears single spikes and hides them) + preprocessed finite/scale
  check (`_reject_degenerate`) in BOTH deploy paths;
  `reference_vector` now skips no-signal files with a reason. Check
  order: NaN → axis coverage → raw signal (restores the documented
  per-row axis error for mixed batches). Regression test extended to
  pin all cases incl. zero-reference non-warping; 7/7 deploy-path
  neighbor tests pass.
- Final battery verdict: **PASS** — 16 must-reject classes rejected on
  both paths; valid/constant/spiked-but-real inputs still accepted.

**BUG-2 battery — 13 cases** (ints, zeros, empty, NaN, ±inf, mixed,
floats, negative, 1e9, single/missing category, empty labels): all
handled; real caller path (prediction overview) renders. **PASS.**

## Valid-data scientific equivalence (critical)

Re-ran the exact Round-1/2 baseline configurations against recorded
numbers:
- paired 5-fold seed 42: RF confusion matrix `[[88,39],[36,124]]`
  **identical**; GNB fold-mean sens/F1 0.769/0.768 **identical**
- standard 5-fold seed 42: all three models identical (0.516/0.510,
  0.524/0.521, 0.599/0.596)
- honest check k=5 seed 42: 0.566/0.566/0.572/0.564 **identical**
- full registry re-run: **25/25 OK, winner Extra Trees F1 0.760 =
  Round-2 baseline exactly** (200 s)

**The fixes changed NOTHING for valid data.** Only invalid input
behavior changed (silent confident garbage → actionable rejection).

## Other round-3 evidence

- **Git diff audit:** 6 modified files + 1 new report; zero
  debug/temp/secret/hardcode additions; every hunk maps to intended
  work (honest-number display §36, audit fixes, tests, Brain.md).
- **3-process persistence:** processes B and C re-predicted the 8
  stored reference files with **MAX |Δp| = 0.000e+00** each.
- **GUI stale-state attack (real PredictWorker):** valid →
  NaN-file-mixed → valid: the NaN file produced NO row (no fake
  probability, no false success), valid files predicted normally, and
  the third run reproduced the first **exactly**.
- **Leakage probe (fresh construction, fresh seeds 11–44):** mean OOF
  accuracy 0.485 ≈ chance. Fold rosters: 0 patient overlap.
- **Clean-state suite:** 94/94 unit + 19/19 deep + GUI PASSED + ruff
  clean, all exit 0.

## Remaining limitations (LOW / INFO — documented, not hidden)

1. **GIGO on finite-signal OOD inputs:** a physically impossible but
   finite-signal input (e.g. negative-constant spectrum, or a real
   spectrum with one giant spike) can still receive a confident
   probability. Guards reject non-finite/degenerate/sparse inputs;
   rejecting every "physically odd but finite" spectrum would risk
   false rejections of legitimate data. Deploy-time spike-flagging
   (reusing `spike_score`) is a recommended future enhancement.
2. Tiny-dataset failure message is a raw traceback blob ("All models
   failed") — loud and safe, but not user-friendly.
3. A-1 (Round 1): loader `files_scanned` 339 vs 341 on disk — count
   semantics only; drops fully explained, zero training impact.
4. Precise Windows working-set RSS NOT MEASURED RELIABLY (bounded
   stage-cache used as proxy).
5. Human visual GUI inspection NOT TESTED (automated offscreen click
   coverage only).
6. External clinical cohort / GPU variance: NOT TESTED (absent).

## Round-3 scorecard (weighted per audit spec)

```
Scientific correctness      20/20  baselines identical; metrics bit-exact
Data integrity              15/15  loader arithmetic exact; A-1 = INFO
Leakage prevention          15/15  0 overlap; 2 independent probes ≈ chance
ML/model correctness        15/15  25/25 twice, winner identical
GUI/backend integration      9/10  automated only; no human-eye pass
Persistence/deployment      10/10  3×process 0.0; parity 0.0
Error handling               4/5   GIGO limitation (documented above)
Security                     3/3   clean; trust boundary documented
Performance/stability        3/4   RSS proxy only (not reliably measured)
Reproducibility              3/3   bit-identical across processes AND runs
TOTAL                       97/100
```

## # 🟡 RELEASE READY WITH LIMITATIONS

All critical paths pass with execution evidence; no unresolved
HIGH/MEDIUM defect; Round-2 regressions pass and were extended; valid
scientific results are provably unaltered (bit-exact baselines). The
verdict is 🟡 rather than 🟢 solely because of the LOW-risk limitations
listed above — chiefly the finite-signal GIGO boundary, the
unmeasured-RSS caveat, and the absence of human visual / external-
cohort validation. Evidence: `.zcode/r3_*.log`, this report, Brain.md
§37–39.

---

# ROUND 4 — FINAL RELEASE GATE

Date: 2026-09-08 (evening) · Commit base: `4c01fc7` (working tree)
· Python 3.14.6 / Windows · No commit or push performed (per gate rule).

## Scope

Pre-commit verification only: current diff, Round-3 fixes, invalid
input, scientific invariance, registry, leakage, metrics, persistence,
parity, GUI smoke, plotting, CLI, reproducibility, performance.

## Git Diff Audit

`git status/diff --stat/--check`: **7 modified files + 1 untracked
report, +717/−142** (the gate's assumed "5 files +221/−37" was
incorrect — corrected here). `diff --check` clean. Pattern scan
(secrets/credentials/debuggers/TODO/print): zero problematic hits (one
Brain.md documentation sentence matched "secret"). Every hunk maps to
intended work: honest-number display (§36), audit fixes, tests,
report, Brain.md.

## Round-3 Fix Verification (+2 gate fixes)

Guard order verified in code (RAW→validate→preprocess→POST-validate→
model, both deploy paths) and entry-point enumeration showed **one
unguarded path** — fixed:
- **GATE-FIX 1 (LOW):** `validate_external.py` evaluated NaN/no-signal
  spectra directly, silently distorting its printed metrics → now
  skips junk spectra with a printed reason; executed against a cohort
  containing NaN + zero files (skipped with reasons, valid evaluated,
  exit 0).
- **GATE-FIX 2 (LOW):** length-mismatched arrays (full axis + empty
  intensities) crashed with **IndexError** at the sort step, before
  any guard → clean ValueError now, placed BEFORE the indexing, both
  paths; pinned in the regression test.

## Invalid Input Battery

10 must-reject classes (NaN one/multi/all, ±inf, mixed, zeros,
one-point, near-zero, subnormal) × single + batch paths: **all
rejected with actionable messages**. Empty/mismatched/short-axis:
controlled ValueErrors. Zero reference file: skipped, reference
unwarped (allclose to good-only). No path yields a prediction for
invalid/no-signal input.

## Valid Scientific Invariance (exact)

- Paired RF confusion matrix **[[88,39],[36,124]] identical**
- GNB fold-mean **0.769/0.768 identical**; independent recompute of
  its CM from raw OOF + threshold: **equal**
- Honest check **0.566/0.566/0.572/0.564 identical**
- Full registry: **25/25 OK, Extra Trees F1 0.760 identical** (205 s);
  probability invariants verified for every model

## Leakage / Persistence / Parity / GUI / Plotting / CLI

Leakage probe (fresh seeds 7/77): **0.460 ≈ chance**. Persistence
A→B→C: **MAX |Δp| = 0.000e+00** both fresh processes. GUI smoke:
gui_test 8-step PASS + PredictWorker valid→NaN→valid → broken file
yields **no row**, recovery **identical**. Plotting 10-case battery
PASS. CLI parity fresh run: PCA+LDA **0.510 / sens 0.516 identical**
to direct path. Reproducibility: every baseline reproduced exactly
this session (twice for GNB/RF).

## Performance

load 0.5 s · paired train 1.3 s · predict 0.021 ms/spectrum — no
regression vs recorded values. **RSS: NOT RELIABLY MEASURED** (stated,
not invented).

## Remaining Limitations (unchanged, non-blocking)

Finite-signal physically-odd inputs can still be confident (GIGO
domain boundary — deliberately NOT "solved" by rejecting arbitrary
finite spectra) · tiny-data failure message is a traceback blob (LOW)
· loader scan-count note A-1 (INFO) · no human visual GUI inspection
(NOT TESTED) · no external cohort (NOT TESTED).

## Final Test Matrix

| Category | Result | Evidence |
|---|---|---|
| Git diff | PASS | +717/−142, 7+1 files, --check clean, 0 secrets |
| Round-3 fixes | PASS | code order verified; +2 gate fixes (validate_external, IndexError) |
| Invalid spectrum | PASS | 10 classes × 2 paths rejected |
| Reference guard | PASS | zero file skipped, reference unwarped |
| Valid-data invariance | PASS | all 4 baselines exact |
| Model registry | PASS | 25/25, ET 0.760, prob invariants |
| Leakage | PASS | probe 0.460 ≈ chance |
| Metrics | PASS | independent recompute equal |
| Persistence | PASS | A/B/C 0.000e+00 |
| Deployment parity | PASS | guards on every path; CLI = direct |
| GUI | PASS | gui_test + stale-state sequence |
| Plotting | PASS | 10 cases |
| CLI parity | PASS | 0.510/0.516 identical |
| Reproducibility | PASS | baselines exact, repeated |
| Performance | PASS | no regression |
| Full suite | PASS | 94/94 + 19/19 + GUI + ruff, exit 0 |

## Final Score

```
Scientific correctness      20/20   invariance exact, metrics reproduced
Data integrity              15/15   arithmetic exact; A-1 = INFO
Leakage prevention          15/15   2 fresh probes ≈ chance this gate
ML/model correctness        15/15   25/25 with probability invariants
GUI/backend integration      9/10   automated only (human-eye NOT TESTED)
Persistence/deployment      10/10   A/B/C exact; no unguarded path left
Error handling               4/5    GIGO boundary documented
Security                     3/3    diff clean; trust boundary documented
Performance/stability        3/4    RSS not reliably measurable
Reproducibility              3/3    exact across processes & days
TOTAL                       97/100
```

## Final Verdict

**🟡 RELEASE READY WITH LIMITATIONS** — safe for its intended
research/demo scope; the limitations above are documented and
non-blocking. The repository is **left uncommitted** with this
recommended commit message:

```
Honest-number display + 4-audit-round hardening; release-gated (97/100)

- Show ONLY the nested honest estimate as performance (banner, Result
  page, txt/HTML reports); honest check auto-runs first; selection CV
  demoted to a labeled ranking; compare-evaluate_pipeline pooled
  sens/spec/acc/auc/cm
- Deploy input hardening (audits 2-4): NaN/Inf, no-signal, sparse,
  subnormal and length-mismatched spectra rejected with actionable
  errors on every prediction path (single, batched, references,
  external validation CLI); zero reference files skipped
- plot_count_bars NaN-safe; validate_external skips junk spectra
- Tests: 94/94 + 19/19 deep + GUI + ruff; valid-data results proven
  bit-identical to pre-fix baselines (ET 0.760, GNB 0.769/0.768,
  honest 0.566/0.572); MASTER_AUDIT_REPORT.md rounds 1-4
```
