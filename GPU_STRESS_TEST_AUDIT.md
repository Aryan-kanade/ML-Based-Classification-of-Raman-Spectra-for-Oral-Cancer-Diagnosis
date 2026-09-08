# GPU STRESS TEST AUDIT — Raman Spectroscopy ML Application

Date: 2026-09-08 (final stress validation) · Tree: `f3be4d0` + GPU
layer (§43, uncommitted). **Validation only — ZERO production-code
changes were made during this pass.** All evidence in
`.zcode/gpu_soak/`.

## 1. Environment

Windows 11 (10.0.26200) · Python 3.14.6 · torch 2.13.0+cu130 (CUDA
13.0, driver 581.86) · **NVIDIA GeForce RTX 3050 6GB Laptop GPU**
(6144 MiB, device 0) · xgboost 3.4.1 · catboost 1.2.10 · lightgbm
4.7.0. Runtime probe before testing: forward pass on `cuda:0`;
memory before: alloc 8.1 MiB / reserved 22 MiB / peak 8.1 MiB.

## 2. Baseline before stress

One normal GPU CNN train+predict on real data succeeded; baseline
probabilities saved (soak bundle); all probabilities finite, in
[0,1], rows sum to 1; probe device `cuda:0` (no CPU fallback).

## 3. 20× CNN training soak — **PASS**

Real dataset (n=299, standard mode), existing configuration (k=2,
seed 42, unchanged architecture/hyperparameters). **20/20 runs OK,
0 exceptions, 0 CUDA errors**; every run's predictions valid.

| Run | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Duration (s) | 2.35 | ~2.1 | ~2.1 | ~2.1 | 2.45 |
| Alloc (MiB) | 16 | 16 | 16 | 16 | 16 |

Stats: mean 2.10 / median 2.07 / min 1.72 / max 2.59 s. First run
includes CUDA warm-up; no degradation (first 2.35 vs last 2.45 s —
within noise).

## 4. GPU memory leak test — **PASS (stable plateau)**

Allocated memory **flat at 16 MiB** across all 20 runs (first-5 =
last-5 = 16 MiB). Reserved/cached grew to 48 MiB then plateaued —
**normal PyTorch caching-allocator behavior, not a leak** (allocated
is the live-tensor metric and it never grew). No OOM at any point.
After ALL soak stages: alloc 16 / reserved 48 MiB.

## 5. 100× prediction soak — **PASS**

100 single-spectrum predictions (real spectra) + 20 batch-5
predictions: **0 NaN, 0 Inf, 0 invalid labels, 0 CUDA errors, 0
memory runaway**. Single: mean 4.8 ms (first 9.95 — warm-up — last
4.69; no degradation). Batch-5: mean 33.6 ms (first 32.2, last
30.4). **First vs last prediction bit-identical.**

## 6. Model load/predict soak — **PASS**

20× load→predict→drop cycles on the saved CNN bundle: 9–13 ms per
cycle, stable. GPU-process vs forced-CPU-process predictions on the
same bundle: **MAX |Δp| = 0.0, label agreement 100%** (CPU-stored
weights by design — the architecture guarantees it, and it held).

## 7. 3SSE stress — **PASS**

3 complete real CLI 3SSE runs with the GPU CNN layer
(`--models "1D-CNN,PCA + Gaussian Naive Bayes"`, k=2, top=1):
**all exit 0**, device line `CNN=cuda (torch probe: cuda:0)` in
every run, artifacts written, no worker crashes, no VRAM runaway.
Grouped-OOF genuineness test passes after stress. Loky workers
remained CPU-pinned by design (unchanged, as required).

## 8. Paired stress — **PASS**

3 complete paired-mode runs through the normal evaluate path
(including the GPU CNN, 5-fold, seed 42): CNN OK every run, **CNN
F1 identical across runs (0.307 — same-seed deterministic)**,
winner GNB 0.768 identical, alloc 16 MiB stable. Reference
construction / pairing / thresholds untouched.

## 9. GUI long-run — **PASS**

Real offscreen MainWindow, real dataset (317 spectra), **5×
train(GPU CNN)→save→load→predict cycles in ONE process** — all 5
completed, exit code **0** (clean teardown), CUDA alive at the end
(alloc 32 MiB). Every cycle: winner valid, bundle round-trip OK,
prediction valid. Cycles 2–5 CNN F1 identical (0.323).
*Harness artifact, documented honestly:* cycle-1's recorded F1
(0.829) was a STALE pre-training (restored 3SSE) winner read by my
wait-loop's early break between worker-finish and the queued
on_train_done callback — a known queued-callback race in TEST
HARNESSSES (gotcha #7), not an app defect; the training itself ran
(following cycle drained its diagnostics battery). Two earlier
harness attempts failed (instant exit 127 / empty log) for the same
class of reason — an unreferenced QApplication (gotcha #30) —
diagnosed and fixed in the harness only.

## 10. GUI responsiveness — **PASS (automated)**

During training the event loop processed continuously (132–1484
processEvents iterations per cycle with live status/progress text
updates observed in the log: "Nested honest evaluation…", "Seed
stability…", "Locked FINAL evaluation…" etc.); controls never
deadlocked; the app remained usable across all cycles. **NOT TESTED:
human visual repaint** (offscreen).

## 11. CUDA context stability — **PASS**

Full-text scan of every soak log for `CUDA error / illegal memory
access / CUBLAS / CUDNN / out of memory / device-side assert /
context reset`: **CLEAN — zero hits**. `torch.cuda.is_initialized()`
true throughout; no silent exception swallowing observed.

## 12. CPU fallback detection — **PASS**

`RAMAN_DEVICE=auto` used cuda:0 from first probe to the END of the
soak (probe re-verified after all stages). `=cpu` forced CPU.
`=gpu` with no visible device still fails LOUDLY (exit-path
verified again post-soak). No GPU-capable torch operation ran on CPU
unexpectedly (probe device asserted each check).

## 13. OOM test — **PASS (no OOM)**

Realistic pressure workloads executed: 20× CNN training, 120
prediction operations, 3 3SSE runs, 3 paired runs, 20 model loads,
5 GUI train cycles — **no OOM occurred anywhere** (peak alloc
≤ 32 MiB of 6144 MiB).

## 14. Shutdown — **PASS**

Python process count identical before/after the stage-6 subprocess
work (4 ambient system processes, unchanged) — **no orphaned
application workers**; all CLI runs exited 0; GUI closed cleanly
(exit 0 in the final soak).

## 15. Restart — **PASS**

Fresh subprocess after the long session: CUDA re-initialized,
device detection correct (`cuda:0`), one GPU prediction returned
(p=0.5591, matches the soak bundle), no stale CUDA/model state.

## 16. Performance degradation — **NONE**

Training: mean 2.10 s stable across runs 1→20 (first 2.35 incl.
warm-up, last 2.45 — noise). Predictions: 9.95 ms → 4.69 ms
(IMPROVED after warm-up, then flat). No progressive increase in any
metric.

## 17. Memory trend

| Stage | Alloc | Reserved | Peak |
|---|---|---|---|
| before | 8 MiB | 22 MiB | 8 MiB |
| after 20 trains | 16 MiB | 48 MiB | 16 MiB |
| after 100 preds + loads | 16 MiB | 48 MiB | 16 MiB |
| after GUI 5 cycles | 32 MiB | — | 32 MiB |

Classification: **stable plateau** (reserved growth = caching
allocator; allocated never grows after run 1).

## 18. Correctness after stress — **PASS**

Post-stress: RF paired CM `[[88,39],[36,124]]` + thresholds,
GNB .7692/.7684 — **IDENTICAL to baseline**; registry winner
Extra Trees **0.760** identical; soak-bundle predictions still
valid and identical; leakage probe, 3SSE-genuineness and deploy
guards all PASS after stress.

## 19. Regression suite — **PASS**

**97/97 unit + 19/19 deep + GUI TEST PASSED + ruff clean** (exit 0
each) + **registry 25/25** (311 s) — all executed AFTER the stress
workload.

## 20. Failures encountered (complete list, nothing hidden)

1. *Harness:* S5 first attempt — 3 of 5 GUI cycles skipped by the
   app's own overlap guard (my wait-loop didn't wait for the
   analysis worker) + exit-time segfault in the offscreen test
   process (documented gotcha #22 family; interactive app
   unaffected). Fixed in the harness; final run clean (exit 0).
2. *Harness:* S5 attempts 2–3 — instant exit 127 (unreferenced
   QApplication GC → qFatal, gotcha #30). Fixed in the harness.
3. *App:* **none.** Zero CUDA errors, zero crashes, zero scientific
   changes across the entire soak.

## Known limitations (unchanged)

Boosters GPU = parent-only behind strict mode; LightGBM CPU wheel;
CNN-fold float caveat (GPU vs CPU training); human visual repaint /
external cohort not testable here; RSS not reliably measured on
this host.

## Final scorecard (§20 of the spec)

| Area | Status |
|---|---|
| CUDA context stability | PASS |
| GPU memory stability | PASS (16 MiB plateau) |
| CNN 20× training | PASS (20/20) |
| 100× prediction | PASS (0 invalid) |
| Model load/predict | PASS (0.0 Δ) |
| 3SSE repeated runs | PASS (3/3 exit 0) |
| Paired repeated runs | PASS (deterministic) |
| GUI responsiveness | PASS (automated; visual repaint NOT TESTED) |
| Worker stability | PASS |
| CPU/GPU fallback | PASS |
| Shutdown | PASS (no orphans) |
| Restart | PASS |
| Performance degradation | PASS (none) |
| Correctness after stress | PASS (baselines identical) |
| Regression suite | PASS (97/97 + 19/19 + GUI + ruff + 25/25) |

# 🟢 GPU STRESS TEST PASS

All success criteria of §21 met. No production code was modified at
any point during this pass.
