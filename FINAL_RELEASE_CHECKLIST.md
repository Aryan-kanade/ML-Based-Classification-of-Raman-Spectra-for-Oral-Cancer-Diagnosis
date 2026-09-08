# FINAL RELEASE CHECKLIST — v1.0.0

Date: 2026-09-08 · Tree: `f3be4d0` + release-preparation changes
(README finalization, release notes, GPU docs) — all uncommitted by
design, awaiting explicit instruction.

## Final release decision

# RELEASE WITH WARNINGS

Zero blockers; the warnings are the documented scientific limitations
(single-centre, no external cohort, honest-number policy,
threshold-instability display, paired patient-rollup approximation,
GIGO boundary) — none affect correctness, reproducibility or the
intended research/demo scope.

## Evidence summary (this finalization pass)

| Category | Result | Evidence |
|---|---|---|
| Production integrity | PASS | diff scans: 0 secrets, 0 debug prints (4 added prints = intentional CLI device summaries), 0 TODO/FIXME blockers, ruff clean |
| Scientific correctness | PASS | invariance re-confirmed this pass: RF cm [[88,39],[36,124]] + thresholds, GNB .7692/.7684, ET 0.760 — unchanged |
| Metrics | PASS | independent reimplementation equals project output (audit_metrics_independent.py ALL PASS); supplements ground-truth-verified |
| Leakage protection | PASS | 0 patient overlap; fingerprint probes at chance (10+ runs) |
| Persistence | PASS | bit-identical across processes and GPU/CPU |
| GUI | PASS (automated; human visual inspection NOT TESTED) | gui_test + 19 deep + 5-cycle GPU soak |
| GPU | PASS | runtime device proof; strict modes; XGB silent-CPU-fallback trap found & fixed pre-release |
| GPU soak | PASS | 15/15 (GPU_STRESS_TEST_AUDIT.md) |
| 3SSE | PASS | genuine grouped-OOF; 3 stress runs exit 0 |
| Paired mode | PASS | deterministic; methodology untouched |
| Security | PASS | no secrets/unsafe code; joblib local-trust boundary documented |
| Dependencies | PASS | 15 pinned requirements; `pip check` clean; Python 3.11–3.14 note accurate |
| Documentation | PASS | README finalized (GPU modes, honest-number policy, validation summary, limitations, counts 56→97); stale artifact pointer fixed; GPU_ACCELERATION*.md accurate to validated behavior |
| Installation | PARTIAL — clean-environment install NOT verified (no clean env available; stated, not claimed); install.bat = pinned requirements (unchanged since §20) |
| Release hygiene | PASS | classification below |

BLOCKERS: 0 · WARNINGS: 1 (Installation PARTIAL — clean-install not
claimable on this machine) + the documented scientific limitations ·
PASS: 14

## Release artifact classification

MUST COMMIT (production): `raman_app/*.py` (6 modified: GPU mode
layer in modeling/gui/sequential/reproduce_study + tests),
`raman_app/README.md`, `Brain.md`.
SHOULD COMMIT (release documentation): `GPU_ACCELERATION.md`,
`GPU_ACCELERATION_AUDIT.md`, `GPU_STRESS_TEST_AUDIT.md`,
`RELEASE_NOTES_v1.0.0.md`, this checklist.
SHOULD NOT COMMIT (remain local): `Data/` (patient data, gitignored),
`.zcode/` evidence logs (gitignored), `audit_metrics_independent.py`
+ `CURRENT_FORMULA_AUDIT.md` + `FORMULA_*.md` +
`METRIC_IMPROVEMENT_IMPLEMENTATION.md` +
`MASTER_AUDIT_REPORT.md` + `FINAL_RELEASE_AUDIT.md` (already
committed in f3be4d0 except audit_metrics_independent.py — reviewer's
choice; recommended: commit it too as a reproducibility script).

## Versioning

**v1.0.0** — first semantic release: scientific validation + software
QA + persistence + GUI + GPU + soak all complete. (Internal
SUITE_VERSION/PARAMS_VERSION machinery unchanged.)

## Recommended commit message

```
v1.0.0: GPU device-mode layer + release documentation

- RAMAN_DEVICE = auto|gpu|cpu (auto default): torch models on CUDA
  when a real context initializes (verified by an actual forward-pass
  probe); strict gpu mode fails loudly without CUDA (never silent
  CPU); XGBoost canary demoted to build-info after discovering
  device='cuda' silently CPU-falls-back with no visible device
- GUI: default mode cpu->auto (boosters stay CPU in auto per the
  measured n~300/OOM lesson), startup mode validation, live
  "Compute device" card; both CLIs print device summaries
- Runtime verification +1 test (97/97); benchmarks CNN 2.1->1.5 s,
  XGB 1.02->0.56 s; 3SSE/paired/persistence verified with GPU CNN;
  CNN bundle predictions device-independent (MAX dp = 0)
- GPU soak test PASS 15/15 (20x train, 100+ predictions, memory
  plateau, 0 CUDA errors) — GPU_STRESS_TEST_AUDIT.md
- README finalized: GPU modes, honest-number policy, validation
  summary, scientific limitations; release notes v1.0.0
```

## GitHub release recommendation

- Version/tag: **v1.0.0**
- Title: "v1.0.0 — Validated Raman ML suite with real GPU acceleration"
- Body: use `RELEASE_NOTES_v1.0.0.md` verbatim (highlights, actual
  validation table, tested GPU environment, known limitations).
