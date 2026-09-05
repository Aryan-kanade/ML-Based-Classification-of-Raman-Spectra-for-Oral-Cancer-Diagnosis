# ROADMAP.md — the complete improvement program (v3, 2026-09-05)

Everything in RESEARCH.md stages 0–4 is DONE (56 unit tests + GUI walk
+ adversarial suite PASS — see §17 of Brain.md for the live counts;
this file's counts supersede any older "N/N" claims). B1/B2/B3 below
are DONE as of the 2026-09-05 audit wave (tests write to a temp dir,
CI workflow exists, LICENSE/CITATION committed) — except the known
leftovers: gui_test/deep_test still assert two in-tree artifacts, and
.github/workflows/ci.yml still needs its first green run.
This file tracks the next program. Effort: S ≤ hours · M ≤ 2 days ·
L > 2 days. Discipline per item: code → targeted test → both suites
green → Brain.md entry.

## A — Scientific credibility
A1 locked one-shot holdout in reproduce_study (S) · A2 3SSE winner
seed-stability 5 seeds + McNemar (M) · A3 ViT vs classical in ONE
harness + ViT predict in GUI (M) · A4 nested selection by default for
top-3 (S) · A5 subsite-stratified metrics (S) · A6 screening-economics
at field prevalence 1/3/5% (S) · A7 active-learning sampler (S) ·
A8 auto methods-section draft (M).

## B — Engineering quality
B1 temp-dir tests (S) · B2 GitHub Actions CI (windows/offscreen/ruff/
logs) (S) · B3 LICENSE+README+CITATION.cff (S) · B5 PyInstaller exe +
portable zip (M) · B6 git-hash+SHA256 run manifests (S) · B7
incremental live mode + measured ms/spectrum (S) · B8 parallel
preprocessing (S) · B9 predict dedup by file hash (S) · B10 cold-start
timing + lazy imports (S) · B4 gui.py page-split LAST (L).

## C — Clinical deployment
C1 Camp mode kiosk skin (M) · C2 drift monitor panel (M) · C3
covariate template CSV (S) · C4 latency benchmark in live mode (S) ·
C5 external-cohort registry (M).

## D — Model/UX polish
D1 interactive threshold/prevalence sweep (M) · D2 QC dashboard canvas
(M) · D3 dark theme (S) · D4 switchable isotonic calibrator + both
ECEs (S) · D5 Grad-CAM/VIP Result-page figure (M).

## E — Advanced ML
E1 TabPFN-2 registry model on PCA scores (M) · E2 self-supervised
spectral pretraining (masked-patch + contrastive) → vit_train
--pretrained (L) · E3 test-time augmentation for CNN/ViT (S) · E4
snapshot ensembling for ViT (S) · E5 label-smoothing/focal options
(S) · E6 Optuna winner tuning pass (M) · E7 band-stability selection
(fold-Jaccard) (S) · E8 GP classifier on PLS scores (S) · E9
multi-task T/N/grade heads behind labels CSV (M).

## F — Data & protocol
F1 public-dataset external validation (raman-data/RamanSPy loaders)
(M) · F2 preprocessing ablation sweep (M) · F3 augmentation ablation
table (S) · F4 leave-one-session-out CV (S) · F5 confounder capture
schema + stratified re-analysis (M) · F6 study-size planner (S) ·
F7 data-collection SOP export (S).

## G — Product & governance
G1 append-only prediction audit trail (JSONL) (S) · G2 model manager
panel (M) · G3 clinical report export HTML→PDF (M) · G4 one-click
table export CSV/Excel (S) · G5 session backup/restore zip (S) · G6
first-run tour (S) · G7 opt-in crash dumps (S).

## I — Publication acceleration
I1 publication figure presets + export-all (S) · I2 LaTeX/Word tables
with CIs (S) · I3 reproducibility artifact builder (M) · I4
standardized benchmark mode (S).

## J — Data-centric AI
J1 label-error detection via winner OOF (confident-learning style) +
Data-page review queue (M) · J2 phantom cohort generator with planted
bands (ground-truth validation of the explainability stack) (M) · J3
data-collection planner from learning curves (S) · J4 CORAL domain
adaptation for external cohorts (unlabeled-only fit) (M) · J5
uncertainty-weighted registry ensemble (S).

## K — Bayesian & deeper statistics
K1 beta-posterior credible bands (sens/spec/AUC/ROC band) (S) · K2
NRI/IDI for fusion (S) · K3 DCA harm slider (S) · K4 permutation CIs
for importance (S) · K5 TOST equivalence for tied models (S).

## L — Power-user GUI
L1 preprocessing timeline scrubber (M) · L2 run comparison view (M) ·
L3 command palette Ctrl+K (S) · L4 interactive Lorentzian peak
fitting (M) · L5 spectral annotation editor (S) · L6 class-mean CI
overlays (S).

## M — Infrastructure & release
M1 Docker image (S) · M2 pre-commit + ruff clean (S) · M3 mypy on
non-GUI modules (M) · M4 CI coverage ≥70% core (S) · M5 performance
regression benchmark (S) · M6 conventional commits → CHANGELOG +
tags (S).

## N — Regulatory-grade governance
N1 ISO-14971 hazard log template (S) · N2 human-factors doc draft
(S) · N3 post-market-surveillance log spec (S) · N4 versioned model
cards + card-hash in audit trail (S).

## Exploratory tier (one-day spikes, go/no-go on grouped-CV gain)
Siamese metric learning for paired mode · band mixture-of-experts ·
S4 state-space sequence models · neural ODE smoothing.

## Anti-plan (final)
GAN/VAE synthesis · SMOTE · EMSC/ComBat/full warping · semi-supervised
labels · full i18n · online training · AutoML suites · ONNX until C4
justifies it.

## Execution order (value-first)
A1 → B1–B3 → J1 → E3+E7 → A2 → J2 → A3 → B7 → F1 → A4–A6 → K1 →
C1+G1 → E1 → F2 → L1 → A7+A8 → E2 → B5+B6 → G2+G3 → M1–M6 → I1–I4 →
C2+C3+F4 → J4 → K2–K5 → L2–L6 → N1–N4 → E6/E8/E9 → B4 last.
