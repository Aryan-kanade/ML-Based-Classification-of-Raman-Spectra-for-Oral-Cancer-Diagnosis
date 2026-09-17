# METHOD BANK — 25 unique working methods (D:\BARC\Data only)

Baseline 0.753 · qualified **36/25** (works = F1>=0.740 std<=0.030 or patient F1>=0.500; strong >=0.753; breakthrough >=0.758 + McNemar<0.10)

| id | mechanism | family | F1 mean±std | patient F1 | tier |
|---|---|---|---|---|---|
| LF-5 | drop top-5 influential patients ['S17', 'S46', 'S52', 'S56', 'S58'] | labels | 0.767±0.007 | 0.497 | ⭐ strong |
| MQ1 | greedy Caruana ensemble over 3-chain pool (in-fold) | ensemble | 0.765±0.000 |  | ⭐ strong |
| Q3-2 | require >= 2 replicates per patient x class | winner chain | qc | 0.763±0.009 | 0.561 | ⭐ strong |
| MQ2 | logistic stacking meta over 6 chains+views (in-fold) | ensemble | 0.758±0.000 |  | ⭐ strong |
| MQ3 | find_peaks top-30 pos/int/width appended to deviations | feature | 0.757±0.000 |  | ⭐ strong |
| MQ5 | inner-CV bagging of layer OOFs (2 seeds) | ensemble | 0.754±0.000 |  | ⭐ strong |
| MQ4 | t-test gate k=200 before ET final layer | selection | 0.719±0.000 |  | ⭐ strong |
| LF-1 | drop top-1 influential patients ['S58'] | labels | 0.753±0.009 | 0.446 | ✅ works |
| MG3 | PCA + XGBoost -> Random Forest -> Extra Trees | chain | 0.752±0.002 | 0.419 | ✅ works |
| MH4 | top-3-by-F1 mean fusion | fusion | 0.752±0.000 | 0.653 | ✅ works |
| LF-2 | drop top-2 influential patients ['S52', 'S58'] | labels | 0.750±0.018 | 0.464 | ✅ works |
| MQ6 | TTA: 4 jittered copies averaged at inference | augmentation | 0.749±0.000 |  | ✅ works |
| MF1 | band-integral features appended to winner matrix | feature | 0.749±0.000 |  | ✅ works |
| MG20 | GUI 3SSE XGBoost->ET->RF (adjudicated) | chain | 0.744±0.004 |  | ✅ works |
| MQ9 | wide GridSearchCV grids on RF/ET layers (in-fold) | tuning | 0.744±0.000 |  | ✅ works |
| MH5 | trimmed-mean fusion | fusion | 0.742±0.000 | 0.624 | ✅ works |
| Q6-ker+sus | keratin outliers + suspect labels | winner chain | qc | 0.742±0.018 | 0.474 | ✅ works |
| MH3 | rank fusion (mean per-class rank) | fusion | 0.741±0.000 | 0.619 | ✅ works |
| MH2 | median-probability fusion of bank OOFs | fusion | 0.737±0.000 | 0.622 | ✅ works |
| MH1 | mean-probability fusion of bank OOFs | fusion | 0.736±0.000 | 0.625 | ✅ works |
| RX4-med-reps2 | median references + min 2 replicates | winner chain | reference | 0.733±0.002 | 0.622 | ✅ works |
| MD3 | PLS + XGBoost -> Random Forest -> CatBoost (L3 swap) | chain | 0.731±0.013 | 0.552 | ✅ works |
| ML2 | random-15 surrogate tuning of final layer (Optuna fallback) | tuning | 0.726±0.000 |  | ✅ works |
| MF3 | PLS + XGBoost -> Random Forest -> Extra Trees -> XGBoost (4-layer) | chain | 0.725±0.017 | 0.540 | ✅ works |
| MG1 | PLS + XGBoost -> Extra Trees -> XGBoost | chain | 0.719±0.016 | 0.570 | ✅ works |
| MD1 | PLS + XGBoost -> Random Forest -> XGBoost (L3 swap) | chain | 0.715±0.010 | 0.534 | ✅ works |
| Q8-20+ker+sus+reps2 | all strong QC filters combined | winner chain | qc | 0.710±0.031 | 0.644 | ✅ works |
| MD2 | PLS + XGBoost -> Random Forest -> LightGBM (L3 swap) | chain | 0.696±0.022 | 0.545 | ✅ works |
| MC7 | HGB boosting on band features | hybrid | 0.645±0.006 | 0.559 | ✅ works |
| MB9 | AdaBoost (shallow-tree boosting) | learner | 0.645±0.016 | 0.519 | ✅ works |
| MD5 | PLS + XGBoost -> Random Forest -> PCA+LogReg (L3 swap) | chain | 0.605±0.023 | 0.537 | ✅ works |
| MC3 | band features -> monotone XGBoost (BANDS priors) | hybrid | 0.589±0.027 | 0.524 | ✅ works |
| MD6 | PLS + XGBoost -> Random Forest -> PLS + XGBoost (self-stack L3) | chain | 0.570±0.022 | 0.538 | ✅ works |
| MC1 | RankNet within-patient ranking | objective | 0.397±0.000 | 0.570 | ✅ works |
| MP1 | patient MEDIAN aggregation + LOO-tuned threshold | aggregation |  | 0.638 | ✅ works |
| ME1 | median-of-logit aggregation (LOO-tuned) | aggregation |  | 0.638 | ✅ works |
| MF8 | Peak bands + RF -> Random Forest -> Extra Trees | chain | 0.740±0.015 | 0.432 | ❌ fail |
| Q2-2.0 | keratin MAD outliers z>2.0 dropped | winner chain | qc | 0.736±0.007 | 0.446 | ❌ fail |
| Q2-2.5 | keratin MAD outliers z>2.5 dropped | winner chain | qc | 0.736±0.007 | 0.446 | ❌ fail |
| Q2-3.0 | keratin MAD outliers z>3.0 dropped | winner chain | qc | 0.736±0.007 | 0.446 | ❌ fail |
| MG2 | XGBoost -> Random Forest -> Extra Trees (L1 swap) | chain | 0.735±0.003 | 0.448 | ❌ fail |
| MD7 | PLS + XGBoost -> Extra Trees -> Extra Trees (L2 swap) | chain | 0.734±0.012 | 0.419 | ❌ fail |
| MD11 | Random Forest -> Extra Trees (2-layer, no PLS base) | chain | 0.732±0.012 | 0.433 | ❌ fail |
| LF-ctl-2 | CONTROL: drop 2 LEAST influential patients | labels | 0.732±0.017 | 0.413 | ❌ fail |
| MF2 | PLS + XGBoost -> Random Forest -> Extra Trees -> Extra Trees (4-layer) | chain | 0.731±0.016 | 0.436 | ❌ fail |
| MG4 | t-test filter + XGBoost -> Random Forest -> Extra Trees | chain | 0.730±0.014 | 0.415 | ❌ fail |
| MD8 | PLS + XGBoost -> Extra Trees (2-layer) | chain | 0.730±0.018 | 0.428 | ❌ fail |
| MG8 | Spectral + band features -> Random Forest -> Extra Trees | chain | 0.730±0.011 | 0.427 | ❌ fail |
| MF9 | PLS + XGBoost -> PCA + KNN -> Extra Trees | chain | 0.728±0.005 | 0.430 | ❌ fail |
| MF5 | PLS + XGBoost -> t-test filter + XGBoost -> Extra Trees | chain | 0.727±0.011 | 0.427 | ❌ fail |
| MD12 | PLS + XGBoost -> CatBoost -> Extra Trees (L2 swap) | chain | 0.725±0.012 | 0.444 | ❌ fail |
| MF4 | PLS + XGBoost -> PCA + SVM (RBF) -> Extra Trees | chain | 0.725±0.014 | 0.432 | ❌ fail |
| MG7 | Peak bands + RF -> Extra Trees (2-layer) | chain | 0.725±0.007 | 0.421 | ❌ fail |
| MD9 | PLS + XGBoost -> Random Forest (2-layer) | chain | 0.724±0.007 | 0.447 | ❌ fail |
| MG6 | PLS + XGBoost -> PCA + MLP -> Extra Trees | chain | 0.722±0.016 | 0.413 | ❌ fail |
| MD10 | PLS + XGBoost -> Random Forest -> mixup-ET (augmented L3) | chain | 0.722±0.004 | 0.439 | ❌ fail |
| MF6 | PLS + XGBoost -> LightGBM -> Extra Trees | chain | 0.720±0.015 | 0.413 | ❌ fail |
| MC2 | within-class cross-patient mixup + ET | augmentation | 0.720±0.003 | 0.378 | ❌ fail |
| Q4-suspect | drop 2 suspect labels (TDOC083 TH02 / TDOC085 TH0) | winner chain | qc | 0.720±0.019 | 0.414 | ❌ fail |
| RX1-med | median normal references (both ref sites) | winner chain | reference | 0.720±0.011 | 0.449 | ❌ fail |
| RX1-trm | trimmed-mean normal references | winner chain | reference | 0.720±0.011 | 0.449 | ❌ fail |
| MF7 | PLS + XGBoost -> Hist Gradient Boosting -> Extra Trees | chain | 0.717±0.009 | 0.431 | ❌ fail |
| MG5 | PLS + XGBoost -> PCA + Gaussian Naive Bayes -> Extra Trees | chain | 0.715±0.007 | 0.413 | ❌ fail |
| RX3-med-sus | median references + suspect-drop | winner chain | reference | 0.714±0.001 | 0.396 | ❌ fail |
| Q7-20+ker | drop 20% spike + keratin outliers | winner chain | qc | 0.713±0.017 | 0.442 | ❌ fail |
| MQ8 | drop-silent-region window, deriv-0 | feature | 0.711±0.000 |  | ❌ fail |
| MG10 | PLS + XGBoost -> Isolation Forest (one-vs-rest) -> Extra Trees | chain | 0.709±0.010 | 0.419 | ❌ fail |
| Q9-30+ker+sus | 30% spike + keratin + suspect | winner chain | qc | 0.703±0.022 | 0.490 | ❌ fail |
| Q1-20 | drop worst 20% spectra by spike score | winner chain | qc | 0.700±0.014 | 0.372 | ❌ fail |
| MQ7 | fingerprint+high window, deriv-0 | feature | 0.699±0.000 |  | ❌ fail |
| MS3 | ET+XGB dual with LR arbiter on disagreement | signal | 0.693±0.000 |  | ❌ fail |
| MB3 | random-subspace bagging ET | learner | 0.690±0.006 | 0.359 | ❌ fail |
| Q1-15 | drop worst 15% spectra by spike score | winner chain | qc | 0.687±0.013 | 0.398 | ❌ fail |
| Q1-5 | drop worst 5% spectra by spike score | winner chain | qc | 0.686±0.021 | 0.420 | ❌ fail |
| Q1-25 | drop worst 25% spectra by spike score | winner chain | qc | 0.676±0.014 | 0.337 | ❌ fail |
| Q1-30 | drop worst 30% spectra by spike score | winner chain | qc | 0.676±0.022 | 0.431 | ❌ fail |
| RX2-med-20 | median references + 20% spike drop | winner chain | reference | 0.676±0.026 | 0.377 | ❌ fail |
| RX2-trm-20 | trimmed references + 20% spike drop | winner chain | reference | 0.676±0.026 | 0.377 | ❌ fail |
| Q1-35 | drop worst 35% spectra by spike score | winner chain | qc | 0.675±0.014 | 0.453 | ❌ fail |
| Q1-40 | drop worst 40% spectra by spike score | winner chain | qc | 0.674±0.038 | 0.452 | ❌ fail |
| Q1-10 | drop worst 10% spectra by spike score | winner chain | qc | 0.664±0.014 | 0.366 | ❌ fail |
| Q5-20+sus | drop 20% spike + suspect labels | winner chain | qc | 0.662±0.021 | 0.333 | ❌ fail |
| ML1 | phantom-physics pretrain + per-fold finetune | pretrain | 0.647±0.008 |  | ❌ fail |
| Q3-3 | require >= 3 replicates per patient x class | winner chain | qc | 0.646±0.006 | 0.365 | ❌ fail |
| MA8 | band+ratio+quantile concat -> ET | feature | 0.628±0.022 | 0.382 | ❌ fail |
| MA1 | band+window area vector -> ET | feature | 0.617±0.016 | 0.374 | ❌ fail |
| MC6 | window-corr features -> subspace bag | hybrid | 0.600±0.013 | 0.400 | ❌ fail |
| MC8 | RF on quantile features | hybrid | 0.598±0.013 | 0.379 | ❌ fail |
| MA4 | FFT magnitude band -> ET | feature | 0.597±0.007 | 0.479 | ❌ fail |
| MA3 | wavelet-packet node energies -> ET | feature | 0.595±0.009 | 0.458 | ❌ fail |
| MA5 | entropy/FD/PSD window features -> ET | feature | 0.592±0.025 | 0.437 | ❌ fail |
| MA6 | window correlation (2D-COS-lite) -> ET | feature | 0.590±0.009 | 0.434 | ❌ fail |
| MD4 | PLS + XGBoost -> Random Forest -> HGB (L3 swap) | chain | 0.581±0.008 | 0.487 | ❌ fail |
| MG9 | PLS + XGBoost -> Random Forest -> Hist Gradient Boosting | chain | 0.581±0.008 | 0.487 | ❌ fail |
| MA7 | window quantile summaries -> ET | feature | 0.578±0.024 | 0.393 | ❌ fail |
| MB6 | QDA on PCA-50 (regularized) | learner | 0.575±0.005 | 0.401 | ❌ fail |
| MC4 | quantile+entropy concat -> rotation forest | hybrid | 0.563±0.035 | 0.336 | ❌ fail |
| MB2 | Rotation Forest (PCA subsets) | learner | 0.537±0.012 | 0.326 | ❌ fail |
| MA2 | band-ratio/keratin markers -> ET | feature | 0.495±0.009 | 0.393 | ❌ fail |
| MS2 | class-medoid prototype rule (min-distance) | signal | 0.491±0.000 |  | ❌ fail |
| MB5 | elastic-net logistic (saga) | learner | 0.481±0.030 | 0.474 | ❌ fail |
| MS1 | band-ratio Gaussian likelihood-ratio rule | signal | 0.441±0.000 |  | ❌ fail |
| MB8 | NCA + kNN (PCA-50) | learner | 0.438±0.026 | 0.423 | ❌ fail |
| MB7 | GNB on PCA-50 | learner | 0.425±0.019 | 0.348 | ❌ fail |
| MC5 | FFT features -> GP(PCA-50) | hybrid | 0.423±0.094 | 0.377 | ❌ fail |
| MB4 | SVM polynomial kernel (calibrated) | learner | 0.395±0.017 | 0.343 | ❌ fail |
| MB1 | Gaussian Process (PCA-50) | learner | 0.311±0.003 | 0.338 | ❌ fail |
| MP2 | cascade rule-in/rule-out (0.95/0.90) | aggregation |  | 0.000 | ❌ fail |
| ME2 | winsorized-mean aggregation (clip .1-.9) (LOO-tuned) | aggregation |  | 0.421 | ❌ fail |
| ME3 | mean of 3 most-confident replicates (LOO-tuned) | aggregation |  | 0.483 | ❌ fail |
| ME4 | mean of 3 least-confident replicates (LOO-tuned) | aggregation |  | 0.482 | ❌ fail |
| ME5 | geometric-mean aggregation (LOO-tuned) | aggregation |  | 0.498 | ❌ fail |
| ME6 | midhinge aggregation (q25+q75)/2 (LOO-tuned) | aggregation |  | 0.422 | ❌ fail |
| Q3-4 | require >= 4 replicates per patient x class | winner chain | qc | 0.000±0.000 |  | ❌ fail |

qualified 36/25
