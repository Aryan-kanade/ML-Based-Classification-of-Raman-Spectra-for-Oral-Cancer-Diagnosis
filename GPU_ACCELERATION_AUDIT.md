# GPU ACCELERATION AUDIT — Raman Spectroscopy ML Application

Date: 2026-09-08 · Tree: `f3be4d0` + this GPU layer (uncommitted).
Baseline = the released CPU/reference configuration; **no scientific
methodology, dataset, preprocessing, folds, seeds, thresholds or
metrics changed — classical baselines verified bit-identical after
the work.**

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | GPU hardware detected | **PASS** — RTX 3050 6GB Laptop, driver CUDA 13.0 | torch probe |
| 2 | CUDA status | **PASS** — torch 2.13.0+cu130, context initializes | `verify_gpu_runtime()` → `cuda:0` |
| 3 | Backend versions | **PASS** — xgb 3.4.1 (USE_CUDA build), catboost 1.2.10, lightgbm 4.7.0 (CPU wheel), torch 2.13.0 | build_info + imports |
| 4 | Model GPU support matrix | **PASS** — torch models GPU (auto); XGB/CatBoost GPU (explicit mode, parent); LightGBM/scikit CPU (honest) | GPU_ACCELERATION.md table |
| 5 | GUI verification | **PASS** — "Compute device" card (Start page): live `Mode: auto · CNN: cuda — probe ran on cuda:0 · XGBoost: cpu …` (cached per session, never hardcoded) | offscreen GUI sanity |
| 6 | CLI verification | **PASS** — sequential.py + reproduce_study print the device summary; strict mode exits 2 with DEVICE ERROR | gpu_3sse.log line 1 |
| 7 | CNN GPU verification | **PASS** — forward-pass probe lands on `cuda:0`; nested CNN trained on cuda (F1 as expected, no error); AMP/TF32/TTA/5-seed unchanged | verify + train run |
| 8 | XGBoost verification | **PASS with caveat** — device='cuda' config verified + benchmarked (0.56 s vs 1.02 s CPU); execution proof = torch-driver probe + USE_CUDA build (XGBoost itself gives no per-fit device signal and silently CPU-falls-back without a GPU — probed and documented) | benchmark output |
| 9 | LightGBM verification | **CPU-ONLY** — pip wheel 4.7.0 has no GPU; source-build requirement documented | GPU_ACCELERATION.md |
| 10 | CatBoost verification | **CPU-ONLY in auto / GPU-capable in gpu mode** (task_type='GPU' configured behind the same strict driver proof; not separately benchmarked this pass — parent-only, same n≈300 caveat) | registry CPU runs |
| 11 | 3SSE verification | **PASS** — real CLI run with GPU CNN layer, exit 0; grouped-OOF genuineness test passes; workers CPU-pinned | gpu_3sse.log |
| 12 | Paired verification | **PASS** — paired registry runs (incl. CNN on cuda) unchanged; reference/pairing untouched | registry 25/25 |
| 13 | Persistence verification | **PASS** — GPU-trained CNN bundle → GPU-process and forced-CPU-process predictions **bit-identical** (MAX\|Δp\|=0.0, labels 100%) | gpu_audit/ |
| 14 | CPU/GPU numerical comparison | **PASS** — identical by design (weights stored/predicted on CPU); CNN *training* folds may differ float-wise (CUDA atomics) — documented, seeds fix splits | GPU_ACCELERATION.md §limitations |
| 15 | Performance benchmark | **PASS (honest)** — CNN 2.1→1.5 s, XGB 1.02→0.56 s at n≈300; modest, as measured | benchmark runs |
| 16 | Memory behavior | **PASS** — CUDA peak 31 MiB flat across 3 train/predict cycles; inference_mode at predict; RSS otherwise NOT RELIABLY MEASURED (stated) | cycle run |
| 17 | Fallback behavior | **PASS** — auto falls back to CPU *with reporting* (not silent); cpu forces; gpu NEVER silently falls back | mode tests |
| 18 | Failure testing | **PASS** — RAMAN_DEVICE=gpu + `CUDA_VISIBLE_DEVICES=-1` → loud RuntimeError; `=99` → same; invalid value → clear error; GUI shows critical dialog | subprocess tests |
| 19 | Full regression | **PASS** — 97/97 unit + 19/19 deep + GUI + ruff, exit 0; RF CM [[88,39],[36,124]] / GNB .7692/.7684 IDENTICAL | gf_*.log |
| 20 | Limitations | documented — booster GPU parent-only; LightGBM wheel; CNN-fold float caveat; XGB-only installs report CPU | GPU_ACCELERATION.md |

## Security / quality

Ruff clean; no subprocess execution added; no hardcoded GPU names
(the RTX 3050 string comes from the driver at runtime); no
credentials; no debug prints; the strict-failure path raises loudly
(no swallowed exceptions); fake-GPU-status impossible by construction
(every reported backend comes from `verify_gpu_runtime()`).

## Scientific invariance (final check)

- RF paired pooled CM, per-fold thresholds, GNB fold-means: identical
- Registry winner Extra Trees 0.760: unchanged (registry CPU path)
- Honest check + supplements: unchanged keys
- Dataset/labels/groups/folds/seeds/threshold algorithm: untouched

## Verdict

**PASS — real GPU acceleration implemented and runtime-proven** for
the torch models (1D-CNN/ensemble, ViT, TabPFN) in the default AUTO
mode, and for XGBoost/CatBoost under explicit strict GPU mode; all
other models honestly CPU. The scientific baseline is bit-invariant.
