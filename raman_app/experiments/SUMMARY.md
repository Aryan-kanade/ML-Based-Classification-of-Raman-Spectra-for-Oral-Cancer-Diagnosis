# Push-to-0.8 program — results dashboard

Baseline = L4 multi-seed mean of the d2 winner (PLS + XGBoost -> RF -> ET): **0.753**

| lever | arm | F1 | delta | verdict | notes |
|---|---|---|---|---|---|
| L11 label-errors | review-list | 0.757 | +0.004 | tie | 2 flagged (0 likely) — human review next; cleaning confirmed mislabels is a legitimate lift |
| external-data | SMAE pretrain->finetune | 0.571 | -0.182 | drop |  |
| external-data | RamanNet segment-MLP | 0.544 | -0.209 | drop | patient F1 0.388 |
| external-data | ComBat pooled (standard arm) | 0.433 | -0.320 | drop | patient F1 0.585 |
| patient-level | RankNet within-patient | 0.570 | -0.183 | drop | patient sens 0.742 spec 0.424 |
| external-data | center-out ours<->RSClass | 0.429 | -0.324 | drop | AUC 0.542/0.551 = chance; corpus too domain-distant |
| patient-level | MIL attention bag | 0.332 | -0.421 | drop | spectrum F1 0.559 |
| L34 windows | fp_d0 | 0.690 | -0.063 | drop |  |
| L34 windows | fphigh_d0 | 0.699 | -0.054 | drop |  |
| L34 windows | drop_d0 | 0.711 | -0.042 | drop |  |
| L34 windows | fp_d1 | 0.644 | -0.109 | drop |  |
| L34 windows | fphigh_d1 | 0.663 | -0.090 | drop |  |
| L34 windows | drop_d1 | 0.663 | -0.090 | drop |  |
| L34 windows | fp_d2 | 0.630 | -0.123 | drop |  |
| L34 windows | fphigh_d2 | 0.650 | -0.103 | drop |  |
| L34 windows | drop_d2 | 0.693 | -0.060 | drop |  |
| L35 peak-features | top-30 pos/int/width appended | 0.757 | +0.004 | tie |  |
| L38 signed-log | sign(x)*log1p(|x|) before chain | 0.712 | -0.041 | drop |  |
| L41 inner-bagging | layer OOF averaged over 2 seeds | 0.754 | +0.001 | tie |  |
| L42 PCA-denoise | PCA 99.9% reconstruct in-fold | 0.706 | -0.047 | drop |  |
| L39 view-ensemble | 4 views, probs averaged | 0.731 | -0.022 | drop |  |
| L40 logistic stacking | 6 chains+views, LogReg meta in-fold | 0.758 | +0.005 | tie |  |
| L13 augmentation | lorentzian blends 60% in RF/ET layers | 0.730 | -0.023 | drop |  |
| L16 multiview | deriv2|deriv1|vector concat (std-normed) | 0.693 | -0.060 | drop |  |
| L19 OOF-bagging | winner arch, 3 partitions averaged | 0.732 | -0.021 | drop |  |
| L17 TTA | winner arch, 4 jittered copies averaged | 0.749 | -0.004 | tie |  |
| L18 greedy ensemble | 3-chain pool, in-fold greedy selection | 0.765 | +0.012 | KEEP |  |
| closers | L31 residualize (keratin+norm) | 0.710 | -0.043 | drop |  |
| closers | L33 t-gate k=200 (ET-final) | 0.757 | +0.004 | tie |  |
| closers | L28 SAM | 0.556 | -0.197 | drop |  |
| closers | L28 cosine-kNN k=7 | 0.565 | -0.188 | drop |  |
| patient-level | LOO-tuned thr (median) | 0.638 | -0.115 | drop | patient F1 0.638 sens 0.742 spec 0.545 |
| L35 peak-features | top-30 pos/int/width appended | 0.757 | +0.004 | tie |  |
| L36 EMSC | per-fold multiplicative+additive correction | 0.579 | -0.174 | drop |  |
| L51 pretrain->finetune | 24k synthetic phantom spectra -> grouped 5-fold | 0.839 | +0.086 | KEEP |  |
| L51 pretrain | phantom-corpus pretrain (LEAKAGE-FIXED) | 0.647 | -0.106 | drop | original 0.839 was sequential-fold leakage (shared encoder); honest 3-seed 0.647±0.008, McNemar p=1e-4 WORSE |
| L12 wide-tuning | RF/ET wide grids in-fold | 0.744 | -0.009 | drop |  |
| patient-level | LOO-tuned thr (median) | 0.638 | -0.115 | drop | patient F1 0.638 sens 0.742 spec 0.545 |
| L54 bayes-search | final-layer random-15 | 0.752 | -0.001 | tie |  |
| L55 TabPFN-v2 | single model, grouped 5-fold | 0.620 | -0.133 | drop |  |
| L56 RamanNet-style | multi-scale 1D CNN + aug, GPU | 0.591 | -0.162 | drop |  |
| L54 bayes-search | final-layer random-15 | 0.752 | -0.001 | tie |  |
| L55 TabPFN-v2 | single model, grouped 5-fold | 0.620 | -0.133 | drop |  |
| L56 RamanNet-style | multi-scale 1D CNN + aug, GPU | 0.559 | -0.194 | drop |  |
| closers | L31 residualize (keratin+norm) | 0.710 | -0.043 | drop |  |
| closers | L33 t-gate k=200 (gated RF final) | 0.719 | -0.034 | drop |  |
| closers | L28 SAM | 0.556 | -0.197 | drop |  |
| closers | L28 cosine-kNN k=7 | 0.565 | -0.188 | drop |  |
| L54 bayes-search | final-layer random-15 (seed-43 honest) | 0.726 | -0.027 | drop |  |
| L55 TabPFN-v2 | single model, grouped 5-fold | 0.620 | -0.133 | drop |  |
| L56 RamanNet-style | multi-scale 1D CNN + aug, GPU | 0.670 | -0.083 | drop |  |
| patient-level | RankNet within-patient | 0.397 | -0.356 | drop | patient sens 0.548 spec 0.273 |
| audit-fix | L33 t-gate CORRECTED (est_swap was ignored) | 0.719 | -0.034 | drop | old 0.757 measured the plain chain |
| audit-fix | L54 surrogate CORRECTED (honest seed-43) | 0.726 | -0.027 | drop | selection folds 0.750 were biased |
| audit-fix | L56 RamanNet arm (per-class aug labels) | 0.670 | -0.083 | drop | was 0.591 with cross-class label noise |
| audit-fix | RankNet rerun instability | 0.397 | -0.356 | drop | identical seed: 0.570 then 0.397 — GPU nondeterminism |
| push-0.8 L1 | drop worst 5-40% by spike score | 0.686-0.700 | -0.05..-0.07 | drop | label-blind; cleaning HURTS |
| push-0.8 L1 | keratin MAD gates (2.0/2.5/3.0) | 0.736 | -0.021 | drop | |
| push-0.8 L1 | min-2-replicates filter | 0.763 | +0.006 | tie | best cleaner; McNemar p=0.344 NOT significant |
| push-0.8 L1 | suspect-label drop | 0.720 | -0.037 | drop | the 2 flagged labels were probably right |
| push-0.8 L1 | robust refs (median/trimmed, FIXED impl) | 0.676-0.733 | -0.02..-0.08 | drop | mean reference already optimal |
| push-0.8 L1 | R-arms v1 F1=1.000 | - | - | INVALID | degenerate zero-deviation normals — bug, purged |
| push-0.8 L1 | LF drop-influential | 0.750-0.767 | - | diagnostic | outcome-driven removal; p=1.0; not a method |
