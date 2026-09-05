# RESEARCH.md — deep study: report + literature + this codebase (2026-09-04)

Sources studied: Bidipta Rana's internship report (IIT Madras / BARC, July
2026) + ~120 literature/technical sources (PubMed/PMC, Nature, BMJ, ACS,
RSC, Optica, MDPI, arXiv, IEEE, official library docs). This file is the
implementation map; Brain.md §19 holds the per-stage changelog.

---

## 1. The internship report (Bidipta Rana) — summary & critique

**Pipeline:** crop 700–1800 cm⁻¹ → ASLS baseline → vector normalization →
SURE wavelet denoising (coif5, level 5, soft threshold; garrote discussed) →
Welch t-test + Cohen's d feature selection (p<0.05, |d|>0.1 → 31 features) →
PCA (5 PCs, ~60% var) / PLS (5 comps, r=0.95) → 7 models, 5-fold CV.

**Headline:** PLS+XGBoost best — accuracy 72.4%, precision 73.3%, recall
73.3%, specificity 71.4%, F1 73.3%, AUC 0.721. PLS-based models beat their
PCA twins (PLS+XGB 72.4% vs PCA+XGB 58.6%). PLS+QDA worst (F1 0.364).

**Top discriminative bands (Table 4.1):** 743.6, 739.3, 737.2 (nucleic
acids/phospholipids), 1222.4/1242.3 (amide III), 1317.1/1391.0 (collagen/
protein), 1519.0 (carotenoid C=C), 1569.8, 1608.9 (Phe/C=C), 1752.0/1786.3
(lipid C=O) cm⁻¹ — consistent with the international fingerprint (§3).

**Critique (drives Stage 1):**
- Holdout is tiny (14 healthy + 15 cancer spectra) → ±15pp uncertainty.
- No patient-grouped splitting described — the #1 optimism source in
  Raman ML (Blake review: 12.5pp drop when splitting by sample).
- Neither PLS+XGBoost nor PCA+XGBoost exists in our registry → Stage 2
  adjudicates the claim under grouped nested CV ×3 + Friedman.
- Abstract claims "generative data augmentation" but methodology does not
  detail it → Stage 2 adds safe physics-based synthetic augmentation.

## 2. Gap table — report vs literature vs this codebase (audited)

| Capability | Report | Literature SOTA | This project (before this work) |
|---|---|---|---|
| Wavelet threshold | SURE adaptive | Bayes/SURE > fixed universal | fixed universal, soft only (`preprocessing.py`) |
| Garrote / cycle-spin | mentioned | Gao 1998; Coifman-Donoho | absent |
| Baselines | ASLS | iarPLS ≥ arPLS family; SNIP fast | ALS + arPLS only |
| Welch-t/Cohen-d filter | core | must sit inside inner CV (Cawley-Talbot) | absent |
| PLS+XGB / PCA+XGB | its winner | no head-to-head anywhere | absent |
| Sparse PLS-DA | — | mixOmics practice | absent |
| Patient aggregation | not used | Jeng 2019: +6pp | majority vote only |
| Conformal sets | — | MAPIE-style LAC | absent |
| Grad-CAM / VIP | — | clinical-paper standard | absent (deep winners → RF surrogate) |
| Permutation AUC | — | standard null check | absent (Friedman/McNemar ✓) |
| PR curve / ECE | — | standard | absent (ROC/Brier ✓) |
| DCA net benefit | — | dcurves | **present** ✓ |
| Platt calibration | — | right choice at n<500 | **present** ✓ |
| Covariate fusion | — | confirmed open gap (Hanna 2024) | absent |
| External validation | future work | TRIPOD+AI | absent |
| Real-time mode | future work | 0.1–1 s/spectrum budget | absent |

## 3. Literature anchors (selection; full source list below)

- **Meta-analytic SOTA, oral cancer Raman:** pooled sens 0.89 / spec 0.84 /
  AUC 0.93 (Han 2022, 13 studies; 2026 update ~90/89). Tissue >95/95,
  saliva 93.6/94.0, serum 85/90.
- **Small-n rule:** classical (PLS-DA/SVM) beats DL below ~few thousand
  labeled spectra (2026 review, 12k pathogen spectra crossover study).
- **Augmentation:** GAN spectra ≤ +5pp, leakage-prone, recent evidence
  modest; noise/shift/mixup/masking deliver most benefit safely
  (Bjerrum 2017; SpecAugment 2019; Wu 2021).
- **Calibration alignment > intensity normalization** (Mostafapour 2023):
  wavenumber calibration (Si 520.7 / Phe 1003) was the decisive step.
- **Patient aggregation** (Jeng 2019): 5-spectra averaging 81.75→87.5%.
- **Conformal prediction** enters spectroscopy (ACS EST 2024); coverage
  guarantees enable principled abstention — natural fit for the existing
  INDETERMINATE triage tier.
- **Explainability template** (Bellantuono 2023): SHAP recovers known
  biochemistry (carotenoid 1155 ↓ cancer); cross-model agreement (SHAP vs
  VIP vs Grad-CAM) is what separates accepted clinical ML papers.
- **Biochemical shift narrative:** normal = lipid/carotenoid-dominated;
  cancer = protein + nucleic-acid signatures (Baraga/Feld fingerprint:
  730/745, 1003, 1450, 1655 cm⁻¹).
- **Triage bar (WHO ASSURED):** sens ≥85–90, spec ≥70–80 for
  refer-to-biopsy; position outputs as triage, not diagnosis (already the
  app's framing; Faur 2023's honest saliva AUC 0.65–0.75 supports this).
- **Regulatory:** TRIPOD+AI (BMJ 2024) reporting; DermaSensor = cleared
  spectroscopy+ML template (FDA 2024); EU AI Act high-risk from Aug 2027.

## 4. Implemented roadmap (stages, in Brain.md changelog)

- **Stage 1** — preprocessing power + `reproduce_report.py` head-to-head.
- **Stage 2** — model arsenal: report's winners, leakage-safe t-test
  filter, augmentation parity, physics synth, MC-dropout/ensembles, PR.
- **Stage 3** — conformal, permutation AUC, ECE/isotonic, Grad-CAM/VIP/
  attention, band-agreement panel, patient prob-mean.
- **Stage 4** — covariate fusion, saliva preset, carotenoid explainer,
  external validation, power panel, model card, QC dashboard, real-time
  file-watch, threshold/prevalence sweep.

## 5. Source list (key URLs)

Meta/SOTA: frontiersin.org/articles/10.3389/fonc.2022.925032 (Han 2022) ·
link.springer.com/article/10.1007/s42452-026-08860-2 (2026 meta) ·
pmc.ncbi.nlm.nih.gov/articles/PMC11488342 (Hanna 2024) ·
frontiersin.org/articles/10.3389/fonc.2023.1272305 (Li 2023, multi-task
ResNet, Grad-CAM) · github.com/ISCLab-Bistu/deep-learning-for-OSCC (Chang
2023, 16.2k spectra) · pmc.ncbi.nlm.nih.gov/articles/PMC11846373 (Lin 2025,
3,551-participant serum SERS) · pmc.ncbi.nlm.nih.gov/articles/PMC10219614
(Faur 2023, honest saliva SERS) · pmc.ncbi.nlm.nih.gov/articles/PMC6780219
(Jeng 2019, patient-wise aggregation).

Preprocessing: pybaselines.readthedocs.io · pmc.ncbi.nlm.nih.gov/articles/
PMC11125329 (Han 2024 baseline benchmark) · S0924203123000292 ·
sciencedirect.com/S1386142523007850 (Mostafapour 2023) ·
pubmed.ncbi.nlm.nih.gov/16808434 (PQN, Dieterle 2006) ·
nature.com/articles/srep04751 (Lin 2014, 1003 cm⁻¹) ·
S0169743918301758 (Whitaker-Hayes despiking) · S0003267024001132
(Coca-López 2024) · github.com/mfitzp/icoshift · ASTM E1840 / USP <1120>
(calibration) · PMC9222091 (Blake review) · arxiv.org/abs/1710.01927
(Bjerrum aug) · SpecAugment (Park 2019) · PMC8668947 (Wu 2021 GAN, +5pp).

Methods/stats: link.springer.com/10.1186/s12859-019-3310-7 ("So you think
you can PLS-DA?") · PMC3337399 (Szymanska 2CV) · jmlr.org/papers/v11/
cawley10a.html (Cawley-Talbot) · mapie.readthedocs.io (conformal API
reference) · scikit-learn.org calibration docs ·
pmc.ncbi.nlm.nih.gov/articles/PMC2749250 (Schisterman, Youden) ·
pmc.ncbi.nlm.nih.gov/articles/PMC6123195 (DCA technical note) ·
pmc.ncbi.nlm.nih.gov/articles/PMC10388213 (Bradshaw grouped-CV guide) ·
sklearn permutation_test_score · pROC::power.roc.test (Hanley-McNeil /
Obuchowski) · bmj.com/content/385/bmj-2023-078378 (TRIPOD+AI) ·
github.com/jacobgil/pytorch-grad-cam · nature.com/articles/s41598-023-43856-7
(Bellantuono SHAP-biochemistry template).

Deployment: nature.com/articles/s41598-024-62543-9 (Sentry needle) ·
Jermyn 2015 STM 7:274ra19 (0.1–1 s/spectrum) · barc.gov.in/technologies/
md33drhr (BARC portable oral-screening system) · actrec.gov.in Raman
program · who.int REASSURED · dermasensor.com FDA clearance ·
MDCG 2025-6 (MDR/AI-Act) · pypi.org/project/raman-data ·
ramanspy.readthedocs.io.
