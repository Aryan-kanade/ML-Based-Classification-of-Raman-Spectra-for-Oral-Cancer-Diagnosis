# GPU ACCELERATION — Raman Spectroscopy ML Application

Date: 2026-09-08 · Adds a device-mode layer + runtime verification on
top of the existing GPU infrastructure (torch +cu130 was already live
for the 1D-CNN). **Engineering change only — every scientific
baseline is bit-identical (verified).**

## Supported backends (verified at runtime, never claimed from
library capability alone)

| Model family | Backend in AUTO | Backend in GPU mode | Verification |
|---|---|---|---|
| 1D-CNN + 5-seed ensemble (+TTA), ViT, TabPFN (torch) | **cuda** when a CUDA context initializes | cuda | real forward pass; output tensor device asserted (`cuda:0`) |
| XGBoost 3.4.1 | cpu | **cuda** (`device='cuda'`, parent process only) | driver proven via torch probe + `build_info().USE_CUDA` |
| CatBoost 1.2.10 | cpu | **GPU** (`task_type='GPU'`, parent only) | same driver-proof chain |
| LightGBM 4.7.0 | cpu | cpu — **pip wheel has no GPU build** | documented; GPU needs a source build with `-DUSE_GPU=1` (OpenCL) |
| scikit-learn (PCA/SVM/RF/ET/…), PLS-DA, MLP, IsolationForest | cpu | cpu (no genuine GPU backend) | honest CPU status |

**Statement per the final rule:** *GPU acceleration is supported for
the torch models (1D-CNN/ensemble, ViT, TabPFN) and, under explicit
GPU mode, XGBoost and CatBoost; all other models remain CPU-based.*

## Detected hardware (this machine)

NVIDIA GeForce RTX 3050 6GB Laptop GPU · CUDA 13.0 ·
torch 2.13.0+cu130 · probe: `torch.randn(..., device='cuda')` forward
pass lands on `cuda:0` (verified every startup + in tests).

## Modes — `RAMAN_DEVICE` = `auto` (default) | `gpu` | `cpu`

- **auto**: torch models on GPU when CUDA is genuinely present;
  boosters stay CPU — measured reasons: at n≈300 CPU often wins
  (kernel-launch/transfer overhead) and auto-GPU boosters OOM'd loky
  children during full-registry training (2026-09-05). The choice is
  REPORTED in every device summary, never silent.
- **gpu**: STRICT. No real CUDA backend → loud `RuntimeError`
  ("RAMAN_DEVICE=gpu was requested but no usable CUDA backend was
  found…") at GUI startup / CLI entry — verified with
  `CUDA_VISIBLE_DEVICES=-1` and `=99`. Never a silent CPU fallback.
- **cpu**: forces everything CPU (checked on every `gpu_ok()` /
  `torch_device()` call).

## Why XGBoost cannot be the GPU detector (important finding)

With no visible device, `XGBClassifier(device='cuda').fit` runs
**silently on CPU** — no warning, `build_info().USE_CUDA=True`
(probed 2026-09-08). Therefore GPU presence is proven by the torch
CUDA probe (which initializes a real context); the XGBoost canary is
demoted to a build-capability check. GPU-mode XGB additionally
requires the torch probe to pass — conservative by design.

## Verification method

`modeling.verify_gpu_runtime()` — runs an actual `nn.Linear` forward
pass on the selected device and records the OUTPUT TENSOR's device;
reports per-model backends actually configured. Shown in the GUI
"Compute device" card (Start page, live, cached per session) and
printed by both CLIs. A unit test (`test_device_mode_layer_and_
gpu_verification`) pins modes, strictness and probe-device
consistency.

## Benchmarks (honest, measured this machine, real data n≈300)

| Task | CPU | GPU | Winner |
|---|---|---|---|
| 1D-CNN nested 2-fold train (n=299) | 2.1 s | **1.5 s** | GPU |
| XGBoost 200 trees × 287×780 | 1.02 s | **0.56 s** | GPU (explicit gpu mode) |
| Registry (25 models, boosters CPU per AUTO) | 232 s | — | unchanged |

Small-dataset caveat: at n≈300 GPU wins are modest and can reverse
for tiny fits; the numbers above are as measured, not manipulated.

## Persistence & memory

- CNN bundles store weights on CPU (portable design) → predictions
  are device-independent: GPU-process vs forced-CPU-process
  predictions **bit-identical (MAX |Δp| = 0.0, label agreement 1.0)**.
- CUDA peak memory across repeated train/predict cycles: **31 MiB
  flat** (3 cycles, `reset_peak_memory_stats`); no growth.
- Prediction uses `inference_mode`; training state is not retained.

## 3SSE / paired compatibility

3SSE CLI run with a GPU CNN layer (`--models "1D-CNN,PCA + GNB"`):
exit 0, device line printed, chaining + genuine grouped-OOF unchanged
(OOF-leakage test still passes); loky children remain CPU-pinned by
design (one CUDA context per child would blow VRAM). Paired mode
unchanged — reference construction/pairing untouched.

## Known limitations

1. Boosters GPU only in explicit `gpu` mode and only in the PARENT
   process (workers stay CPU-capped — the 2026-09-05 OOM lesson).
2. LightGBM has no GPU in standard wheels.
3. CNN CUDA folds are not bit-identical to CPU folds (CUDA atomics);
   seeds still fix splits (documented since 2026-09-04). Bundle
   predictions ARE identical (weights stored/predicted on CPU).
4. GPU-status detection requires a CUDA torch build (an XGB-only
   install reports CPU — conservative).

See GPU_ACCELERATION_AUDIT.md for the full 20-section evidence.
