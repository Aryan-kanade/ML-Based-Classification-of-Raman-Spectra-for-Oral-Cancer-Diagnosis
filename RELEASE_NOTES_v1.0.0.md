# Raman Oral-Cancer Classifier — v1.0.0 Release Notes

**Release title:** v1.0.0 — Scientifically validated Raman spectroscopy
ML suite with real GPU acceleration

## Highlights

- **Desktop Raman ML pipeline** (Python + Qt): clinical dataset loader
  with full hygiene (patient grouping, dedupe, site-token and
  reference exclusion), literature-backed preprocessing (Whitaker-Hayes
  despiking, wavelet denoising, ALS/arPLS baseline, PQN), and a
  **25-model registry** (classical ML → XGBoost/LightGBM/CatBoost →
  1D-CNN + 5-seed ensemble → TabPFN foundation model → ensembles).
- **Paired (margin) mode** — each spectrum as its deviation from the
  same patient's normal reference (removes ~61% between-patient
  variance; the clinically realistic scenario), with leakage-free
  per-patient references and PQN variant.
- **3SSE sequential architecture search** — every ordered
  single/pair/triple chain over the registry with patient-grouped OOF
  chaining, beam screening, nested validation and deployable winner
  bundles.
- **Honest performance reporting** — the GUI and reports show the
  nested honest estimate (preprocessing re-chosen inside every fold);
  selection-CV numbers are labeled as a ranking, never reported as
  performance. Supplementary scientific metrics: balanced accuracy,
  MCC, PR-AUC, Brier, ECE, explicit TP/TN/FP/FN, threshold-stability
  stats + instability warning, PATIENT-level vs SPECTRUM-level
  results clearly separated.
- **Real GPU acceleration, runtime-proven** — torch models (1D-CNN,
  ensemble, ViT, TabPFN) on CUDA by default with a strict
  RAMAN_DEVICE = auto/gpu/cpu mode layer (gpu fails loudly, never
  silently); XGBoost/CatBoost GPU in explicit mode; honest CPU status
  for LightGBM/scikit-learn. Verified by a real forward-pass probe,
  not library capability claims.
- **Clinical evaluation layer** (TRIPOD+AI aligned): rule-out/rule-in
  operating points, three-tier triage, PPV/NPV at scenario
  prevalences, Platt calibration, decision curves, per-patient
  verdicts, model cards, txt/HTML reports.

## Validation (actual final results)

```
Unit tests:        97/97 PASS
Deep GUI tests:    19/19 PASS (real adversarial widget scenarios)
GUI suite:         PASS
Ruff:              PASS (clean)
Model registry:    25/25 train+predict OK (Extra Trees F1 0.760 baseline)
Leakage:           PASS — 0 patient overlap in every fold; fingerprint
                   probes at chance across 10+ independent runs
Persistence:       PASS — predictions bit-identical across processes
                   (MAX |Δp| = 0.0) and GPU/CPU training
3SSE:              PASS — genuine grouped-OOF chaining verified
GPU implementation: PASS — runtime device proof; strict-mode failures
GPU soak test:     PASS 15/15 — 20× CNN trainings, 100+ predictions,
                   CUDA memory plateau (no leak), 0 CUDA errors,
                   shutdown/restart clean
Scientific invariance: all validated baselines bit-identical after
                   every change (5 audit rounds + formula audit +
                   GPU work)
```

## GPU environment actually tested

NVIDIA GeForce RTX 3050 6GB Laptop GPU, driver 581.86, CUDA 13.0,
torch 2.13.0+cu130, Windows 11. GPU: torch models (auto mode) +
XGBoost (0.56 s vs 1.02 s CPU at n≈300 in explicit gpu mode) +
CatBoost (parent-only). CPU: LightGBM (pip wheel has no GPU) and all
scikit-learn models. 3SSE loky workers intentionally CPU-pinned
(one CUDA context per child would exhaust VRAM).

## Known limitations (documented, not hidden)

- Single-centre dataset; **no external cohort validation** — internal
  CV only; no clinical claim is justified yet.
- Headline numbers are the honest nested estimates (e.g. standard-mode
  honest macro-F1 ≈ 0.56); selection-CV numbers are higher by design.
- Threshold instability exists on some models (per-fold spread up to
  0.01–0.88) — displayed with a warning.
- Patient-level rollup on paired winners uses dominant-class truth
  (patients contribute both site types) — labeled approximation.
- GIGO boundary: finite-signal physically-odd inputs can be
  confidently classified (NaN/no-signal/degenerate inputs are
  rejected with clear errors on every path).
- CNN GPU folds are not bit-identical to CPU folds (CUDA atomics);
  saved-bundle predictions are device-independent.
- Full audit reports live at the repo root (MASTER_AUDIT_REPORT.md,
  FINAL_RELEASE_AUDIT.md, GPU_*.md, METRIC_IMPROVEMENT_*.md,
  FORMULA_*.md, CURRENT_FORMULA_AUDIT.md).
