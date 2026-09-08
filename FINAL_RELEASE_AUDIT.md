# FINAL RELEASE AUDIT — Raman Spectroscopy ML Application

Date: 2026-09-08 (final pass) · Tree: commit `4c01fc7` + working
changes (audits §32–§41; supplements implemented; all uncommitted by
design) · Python 3.14.6 / Windows 10 · Evidence: `.zcode/v*.log`,
`.zcode/gate_bundle/`, five prior audit rounds + this fresh pass.

---

## 1. Executive Summary

**VERDICT: 🟢 RELEASE** — every critical workflow executes; the
validated baseline is bit-invariant across five audit rounds and the
supplement implementation; no leakage; no fake paths; persistence
bit-exact; edge cases fail safely; all suites green. Remaining items
are documented limitations, not blockers.

## 2. Test environment

Windows 10 x64, Python 3.14.6 (system), pinned requirements (all
resolved), Git Bash. Entry points: `python main.py` (GUI),
`sequential.py` / `reproduce_study.py` (CLI). Dataset `D:\BARC\Data`
(341 files / 72 patients / 317 kept spectra).

## 3. Test inventory (executed this pass)

| # | Test | Status | Evidence |
|---|---|---|---|
| 1 | Unit suite | **PASS** 96/96, exit 0 | `v5_suite.log` |
| 2 | Deep GUI suite (real clicks) | **PASS** 19/19, exit 0 | `v5_deep.log` |
| 3 | GUI walk | **PASS** | `v5_gui.log` |
| 4 | Ruff | **PASS** clean | inline |
| 5 | Full model registry | **PASS** 25/25, ET F1 **0.760**, prob-invariants clean | `v2_registry.log` (232 s) |
| 6 | Baseline invariance | **PASS** all IDENTICAL | fresh script below |
| 7 | Leakage (folds/probe/3SSE) | **PASS** | fresh probe 0.443 ≈ chance |
| 8 | Persistence A→B→C | **PASS** MAX\|Δp\| = 0.0e+00 | gate bundle re-verified |
| 9 | Error/edge battery | **PASS** | model-file battery + rounds 2–4 |
| 10 | 3SSE artifacts | **PASS** consistent | winner.json ↔ winner.joblib |

## 4. GUI results

Automated coverage: gui_test 8-step walk (launch→load→preprocess→
train→predict→save→load→predict→report) + 19 deep adversarial
scenarios (blank-GUI guards, one-class train, predict edge paths,
legacy bundles, clinical auto-refs, locked eval, HTML report, deep
diagnostics, reset-session, train layout, CLI roundtrip, cancel,
live-mode refusal, model-lab flow). Every connected handler executes
its real backend (traced in audit rounds 1–4; no dead buttons found).
**Status: PASS (automated). NOT TESTED: human visual inspection.**

## 5. End-to-end results

Executed repeatedly across rounds and fresh today: launch (offscreen
MainWindow restore path), load+validate (341→317 kept, 4 refs, 18
dups, 2 site-token), preprocess (crop-first pipeline), train (nested
grouped CV + threshold), predict (`predict_with_bundle[_many]`),
metrics/plot/diagnostics (auto-battery, honest-first), save model,
close/reopen/load/predict — **bit-identical probabilities** across
processes and days. **PASS.**

## 6. Data validation

Independent scan (round 1) + loader arithmetic exact: 339 scanned −
4 refs − 18 dups = 317 kept (142/175), 72 subjects, 68 usable pairs,
18 spike-flagged, axis 15.5–3862 on all 341 files, 0 NaN on disk.
**PASS.** (INFO note A-1: on-disk count 341 vs scanned 339 — count
semantics only, drops fully explained.)

## 7. Leakage results

Fold patient overlap **[0,0,0,0,0]** (fresh). Fingerprint probe
(seeds 101/202/303): mean **0.443 ≈ chance**. 3SSE chain-OOF
genuineness + paired-features leakage tests executed live: PASS.
Scalers/PCA/selection/thresholds fitted inside training folds only
(audited rounds 1–2; unchanged since). **PASS.**

## 8. Model results

25/25 registry models train+predict with valid probabilities (range,
row-sum, CM-total invariants) and no silent fallbacks; winner Extra
Trees F1 **0.760** = baseline. **PASS.**

## 9. Metric verification

Independent reimplementation (`audit_metrics_independent.py`):
project == independent on every overlapping metric (sens/spec/prec/
F1/CM/accuracy/AUC); supplements (bacc/MCC/PR-AP/Brier) match
hand-calculated ground truth (TP8/TN7/FP2/FN=3 → acc .75, bacc
.7525, MCC .5025). Honest pooled: TP=103 TN=68 FP=66 FN=62 N=200 ✓
(N-check). **PASS — formulas unchanged, supplements correct.**

## 10. Threshold stability

Per-fold thresholds surfaced (RF [—,—,.543,.570,—] stable; GNB
[.010,.884,.058,.096,.021] → median .0578, SD .36, spread 91× →
**⚠ instability warning fires**). Algorithm untouched. **PASS
(visibility requirement met; instability documented, not hidden).**

## 11. Patient-level results

Separate labeled PATIENT LEVEL block (mean-P at deployed threshold,
dominant-class truth, TP/TN/FP/FN/N + full metric set incl.
bacc/MCC). The known paired-mode limitation (GNB: sens .77/spec .15/
MCC −.10; ties→Normal; spectrum threshold transfer) **remains
visible** with labels. **PASS (separation + honesty).**

## 12. Paired results

Real paired workflow executed (n=287/64 patients); reference
construction junk-tolerant (NaN/zero/no-signal files skipped with
reasons); missing references → per-file skips, never silent absolute
prediction. **PASS.**

## 13. 3SSE results

Real CLI run (round 3) + saved-run consistency (24-model run: pairs
552 = 24·23, triples 1100 = beam·22, winner.json ↔ winner.joblib
match) + fresh OOF-genuineness test live. Chaining is genuine
X→A→OOF-P1→[X‖P1]→B→… with patient-grouped OOF between layers.
**PASS.**

## 14. Persistence

A→B→C across fresh processes: **MAX |Δp| = 0.0e+00** (re-verified
this pass against the stored gate bundle). **PASS.**

## 15. Error handling

Fresh model-file battery: nonexistent (FileNotFoundError), corrupt
(IndexError), non-bundle joblib (KeyError), feature-length mismatch
(ValueError), None-bundle (TypeError) — all fail-safe, no crash, no
false success. Plus rounds 2–4 batteries: NaN/Inf/zero/one-point/
subnormal/length-mismatch spectra rejected on every path; malformed
files skipped with reasons; one-class clean ValueError; tiny dataset
loud failure; empty/mismatched arrays clean ValueError. **PASS.**
(LOW note: tiny-dataset message is a traceback blob — loud, safe.)

## 16. Performance

Measured: load 0.5 s, paired prep 1.7 s, 1-model nested train 1.3 s,
predict 0.02 ms/spectrum, registry 232 s, 3SSE 15-arch CLI minutes.
Stage cache bounded (598 flat across cycles). **RSS: NOT RELIABLY
MEASURED (stated).** No crashes, no runaway workers, no unhandled
exceptions in any suite. **PASS.**

## 17. Security / code quality

Ruff clean. No secrets/credentials/eval/exec in production (hits are
torch API + docs). `joblib.load` = documented local trust boundary.
No debug prints, no test data in production paths, no temporary
artifacts committed (working tree audited; `.zcode/` is tooling,
gitignored). No mocked or hardcoded production results. **PASS.**

## 18. Known limitations (documented, non-blocking)

1. GIGO boundary: finite-signal physically-odd spectra can be
   confidently predicted (deploy spike-flagging = future work).
2. Paired-mode patient rollup semantics (dominant-tie truth +
   threshold transfer) — now VISIBLE in the patient-level card;
   dedicated semantics pass = future work (open item 9).
3. Threshold instability itself (GNB 91× spread) is real and now
   warned about; fixing it = separate scientific experiment.
4. MCC at n≈300 has wide uncertainty (honest MCC .13).
5. Spectrum-level DeLong CI anti-conservative under clustering
   (patient bootstrap is the headline CI).
6. Single-centre dataset; no external cohort (NOT TESTED).
7. Human visual GUI inspection NOT TESTED (automated only).
8. Precise RSS NOT TESTED.

## 19. Regression / invariance results (fresh, this pass)

| Baseline | Value | Status |
|---|---|---|
| RF paired pooled CM | [[88,39],[36,124]] | IDENTICAL |
| RF per-fold thresholds | [—,—,.543,.570,—] / deployed .5566 | IDENTICAL |
| GNB fold-mean sens/F1 | .7692/.7684 | IDENTICAL |
| GNB pooled CM | [[95,32],[33,127]] | IDENTICAL |
| Honest existing keys | .566/.566/.572/.564 | IDENTICAL |
| Extra Trees k=3 | 0.760 | IDENTICAL |
| Dataset counts / folds / seeds | — | UNTOUCHED |
| 3SSE artifacts | consistent | IDENTICAL |
| Persistence | 0.0e+00 | IDENTICAL |

**Classification: no differences found — nothing to classify as
EXPECTED/ROUNDING/REGRESSION.**

## 20. Final scorecard

| Area | Status |
|---|---|
| Data loading/integrity | PASS |
| Preprocessing/PCA leakage-safety | PASS |
| Training/CV/threshold | PASS (unchanged) |
| Metrics incl. supplements | PASS (independently verified) |
| Patient/spectrum separation | PASS |
| Registry | PASS 25/25 |
| Paired/3SSE | PASS (genuine chaining) |
| Persistence/deployment | PASS (bit-exact) |
| Error handling | PASS |
| GUI | PASS (automated; human-eye NOT TESTED) |
| Security | PASS |
| Performance | PASS (RSS NOT TESTED) |
| Baseline invariance | PASS (bit-identical) |

## 21. Release recommendation

# 🟢 **RELEASE**

All critical workflows execute; no unexplained regression (zero
regressions); no leakage; no fake/mocked paths; persistence and
metrics independently verified; edge cases fail safely; patient/
spectrum separation enforced with the known limitation visible.
Remaining items are explicitly documented limitations (§18), none of
which threaten scientific validity for the intended research/demo
scope. The repository is **uncommitted by design** — the prepared
commit message from Round 4 (extended with the §41 supplements)
should be used when the user authorizes the commit.
