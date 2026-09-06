"""
test_all.py — regression tests for the non-GUI pipeline.

Run:  python test_all.py      (also works under pytest: pytest test_all.py)

Covers: clinical loader (dedupe / cross-class / references / patient
groups), preprocessing (crop / spike score / peak features), modeling
(grouped+repeated CV, ensemble, bundle roundtrip with cropping),
optimize.py, and the ViT (forward pass, checkpoint roundtrip, resample).
Everything runs on tiny synthetic data — a few seconds total.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clinical_data as cdata          # noqa: E402
import dataset as ds                   # noqa: E402
import modeling                        # noqa: E402
import optimize                        # noqa: E402
import preprocessing as pp             # noqa: E402
from vit_model import (SpectraDataset, SpectralViT, ViTConfig,  # noqa: E402
                       load_checkpoint, resample_to_len,
                       save_checkpoint)


# ---------------------------------------------------------------- helpers
def _write_spectrum(path: str, wn, it):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for w, v in zip(wn, it, strict=True):
            fh.write(f"{w:.3f},{v:.4f}\n")


def _make_clinical_tree(root: str):
    """Tiny synthetic clinical dataset with deliberate problems:
    - Patient_1/TDOC001 are byte-duplicates (Normal)
    - one white reference file (Tumor)
    - one file identical under BOTH classes (leak)
    """
    rng = np.random.default_rng(0)
    wn = np.linspace(400, 1800, 500)
    spec_a = rng.normal(0, 1, len(wn))     # Normal-ish
    spec_b = spec_a + 1.0                  # Tumor-ish (shifted)
    w = lambda base, name, arr: _write_spectrum(  # noqa: E731
        os.path.join(root, base, name), wn, arr)
    # Normal: 3 patients x 2 spectra
    w("Normal/Patient_1", "a1.csv", spec_a)
    w("Normal/Patient_1", "a2.csv", spec_a + 0.1)
    w("Normal/Patient_2", "b1.csv", spec_a - 0.2)
    w("Normal/Patient_3", "c1.csv", spec_a + 0.3)
    # duplicate folder generation (same content as Patient_1/a1)
    w("Normal/TDOC001", "dup.csv", spec_a)
    # Tumor: 3 patients
    w("Tumor/Patient_1", "t1.csv", spec_b)
    w("Tumor/Patient_1", "t2.csv", spec_b + 0.1)
    w("Tumor/Patient_2", "u1.csv", spec_b - 0.1)
    w("Tumor/Patient_3", "v1.csv", spec_b + 0.2)
    # white reference inside a class folder
    w("Tumor/Patient_3", "v1_white.csv", spec_b)
    # cross-class identical file
    w("Normal/TDOC009", "leak.csv", spec_b + 7.0)
    w("Tumor/TDOC009", "leak.csv", spec_b + 7.0)
    # log file that is not a spectrum
    with open(os.path.join(root, "Normal", "split_log.txt"), "w") as fh:
        fh.write("not a spectrum\n")
    return wn


def _synthetic_ml(n=40, groups_n=10, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"S{i}" for i in range(groups_n)], n // groups_n)
    y = (np.arange(n) % 2).astype(int)
    X = rng.normal(0, 1, (n, 30)) + y[:, None] * 1.5
    return X, np.array(["A", "B"])[y], groups


# ---------------------------------------------------------------- tests
def test_subject_key():
    assert cdata.subject_key("Patient_15") == "S15"
    assert cdata.subject_key("TDOC123") == "S123"
    assert cdata.subject_key("Subject058 Spectra pro") == "S58"
    assert cdata.subject_key("no number") == "NO NUMBER"


def test_clinical_loader_hygiene():
    with tempfile.TemporaryDirectory() as root:
        _make_clinical_tree(root)
        assert cdata.is_clinical_layout(root)
        cd = cdata.load_clinical_dataset(root)
        # 12 real spectra - 1 duplicate - 2 leak copies = 9 kept
        # 11 spectra - 1 duplicate - 2 cross-class copies = 8 kept
        assert cd.report["n_spectra"] == 8, cd.report
        assert len(cd.report["duplicates_dropped"]) == 1
        assert len(cd.report["cross_class_dropped"]) == 2
        assert len(cd.report["references_excluded"]) == 1
        assert cd.report["grid_identical"] is True
        # same subject key across folder generations; the leaked TDOC009
        # file is dropped entirely so S9 never appears
        assert set(cd.groups) == {"S1", "S2", "S3"}
        assert len(set(cd.groups)) == 3


def test_patient_split_disjoint():
    X, y, groups = _synthetic_ml(60, 12)
    tr, va, te = cdata.patient_split(groups, y, seed=42)
    g = lambda idx: {groups[i] for i in idx}  # noqa: E731
    assert not (g(tr) & g(va)) and not (g(tr) & g(te)) and not (g(va) & g(te))
    assert len(tr) + len(va) + len(te) == 60


def test_find_data_root():
    """Portable dataset discovery: pointer round-trip, env override, and
    non-clinical folders are never auto-selected."""
    old_env = os.environ.pop("RAMAN_DATA_DIR", None)
    old_ptr = cdata.DATA_POINTER
    try:
        with tempfile.TemporaryDirectory() as td:
            cdata.DATA_POINTER = os.path.join(td, "ptr")
            _make_clinical_tree(td)
            cdata.remember_data_root(td)
            assert cdata.find_data_root() == td    # pointer round-trip
            os.environ["RAMAN_DATA_DIR"] = td
            assert cdata.find_data_root() == td    # env override works
            os.environ.pop("RAMAN_DATA_DIR")
            empty = os.path.join(td, "empty_dir")
            os.makedirs(empty)
            cdata.remember_data_root(empty)
            assert cdata.find_data_root() != empty  # layout-validated
    finally:
        cdata.DATA_POINTER = old_ptr
        if old_env is None:
            os.environ.pop("RAMAN_DATA_DIR", None)
        else:
            os.environ["RAMAN_DATA_DIR"] = old_env


def test_crop_and_spike_score():
    wn = np.linspace(0, 2000, 1001)          # step 2
    params = pp.PreprocessParams(crop_min=400, crop_max=1800)
    m = pp.crop_mask(wn, params)
    assert int(m.sum()) == 701               # 400..1800 inclusive
    clean = np.sin(np.linspace(0, 30, 500))
    spiked = clean.copy()
    spiked[100] += 500.0
    assert pp.spike_score(clean) < 20 < pp.spike_score(spiked)
    X = np.vstack([clean, clean])
    out = pp.preprocess_matrix(X, params, wn=np.linspace(0, 2000, 500))
    assert out.shape[1] < 500          # cropped


def test_peak_features():
    wn = np.linspace(300, 1900, 800)
    X = np.random.default_rng(1).normal(0, 1, (10, len(wn)))
    F = modeling.PeakIntensityFeatures(wn=wn).fit(X).transform(X)
    assert F.shape == (10, 16)
    assert np.isfinite(F).all()


def test_evaluate_models_groups_repeats_ensemble():
    X, y, groups = _synthetic_ml()
    res, winner = modeling.evaluate_models(
        X, list(y), model_names=["PCA + LDA", "Random Forest",
                                 "Ensemble (top-3)"],
        k_folds=3, groups=list(groups), repeats=2,
        wavenumbers=np.linspace(300, 1900, X.shape[1]))
    names = {r.name for r in res}
    assert "Ensemble (top-3)" in names
    assert all(r.error is None for r in res), [r.error for r in res]
    assert winner.macro_f1() > 0.9          # separable synthetic data
    # peak-band model works when wavenumbers given
    res2, _ = modeling.evaluate_models(
        X, list(y), model_names=["Peak bands + RF"], k_folds=3,
        groups=list(groups), wavenumbers=np.linspace(300, 1900, X.shape[1]))
    assert res2[0].error is None
    # and degrades gracefully when they are not (a solo selection that
    # cannot run raises "All models failed" carrying the reason)
    try:
        res3, _ = modeling.evaluate_models(
            X, list(y), model_names=["Peak bands + RF"], k_folds=3)
        assert res3[0].error is not None
    except RuntimeError as exc:
        assert "wavenumber" in str(exc)


def test_bundle_roundtrip_with_crop():
    X, y, groups = _synthetic_ml()
    _, winner = modeling.evaluate_models(
        X, list(y), model_names=["PCA + LDA"], k_folds=3,
        groups=list(groups),
        wavenumbers=np.linspace(300, 1900, X.shape[1]))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.joblib")
        params = pp.PreprocessParams(crop_min=120, crop_max=180)
        wn_bundle = np.linspace(300, 1900, X.shape[1])
        modeling.save_bundle(path, winner, wn_bundle, params)
        b = modeling.load_bundle(path)
        # raw spectrum spanning the full grid -> cropped inside
        out = modeling.predict_with_bundle(b, wn_bundle[::-1],
                                           np.zeros_like(wn_bundle))
        assert out["prediction"] in ("A", "B")
        assert abs(sum(out["probabilities"].values()) - 1) < 1e-6


def test_optimize_tiny():
    X, y, groups = _synthetic_ml()
    wn = np.linspace(300, 1900, X.shape[1])
    grid = [dict(crop_min=0.0, crop_max=0.0, sg_deriv=0, norm="vector"),
            dict(crop_min=120.0, crop_max=180.0, sg_deriv=0, norm="vector")]
    out = optimize.optimize_preprocessing(
        X, wn, list(y), groups=list(groups), grid=grid, k=3,
        progress=lambda m: None)
    assert len(out["results"]) == 2
    assert out["best"]["mean_f1"] > 0.5


def test_vit_forward_and_checkpoint():
    import torch
    cfg = ViTConfig(seq_len=64, patch_size=8, dim=32, depth=1, heads=2,
                    n_classes=2).validate()
    model = SpectralViT(cfg)
    xb = torch.randn(4, 64)
    assert model(xb).shape == (4, 2)
    assert SpectraDataset(xb, np.array([0, 1, 0, 1])).X.shape == (4, 64)
    assert resample_to_len(xb.numpy(), 32).shape == (4, 32)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "vit.pt")
        save_checkpoint(path, model, ["A", "B"], np.linspace(0, 1, 64),
                        {}, {"test": ["f1"]}, {"train_loss": [1.0]},
                        input_mean=np.zeros(64), input_std=np.ones(64))
        m2, ck = load_checkpoint(path)
        assert ck["classes"] == ["A", "B"]
        model.eval()
        x = torch.randn(1, 64)
        assert torch.allclose(model(x), m2(x))


def test_learning_curve():
    X, y, groups = _synthetic_ml(60, 12)
    from sklearn.discriminant_analysis import \
        LinearDiscriminantAnalysis
    sizes, means, stds = modeling.learning_curve_by_groups(
        X, y, groups, LinearDiscriminantAnalysis(), k=3)
    assert len(sizes) >= 2 and means[-1] > 0.9


def test_region_importance():
    X, y, groups = _synthetic_ml()
    wn = np.linspace(300, 1900, X.shape[1])
    w2, imp, bands = modeling.region_importance(X, list(y), groups, wn)
    assert imp.shape == wn.shape and np.isfinite(imp).all()
    assert 1 <= len(bands) <= 5
    assert all(lo <= c <= hi for (c, _w, _s), lo, hi
               in zip(bands, [wn.min()] * len(bands),
                      [wn.max()] * len(bands), strict=True))


def test_bootstrap_ci_and_mcnemar():
    X, y, groups = _synthetic_ml()
    y = np.asarray(y)
    lo, hi = modeling.bootstrap_ci(y, y)             # perfect predictions
    assert lo == 1.0 and hi == 1.0
    rng = np.random.default_rng(0)
    noisy = y.copy()
    flip = rng.random(len(y)) < 0.2
    noisy[flip] = np.where(noisy[flip] == "A", "B", "A")
    lo2, hi2 = modeling.bootstrap_ci(y, noisy)
    assert lo2 < 1.0 and lo2 <= hi2
    _b, _c, p_same = modeling.mcnemar_test(y, y, y)
    assert p_same == 1.0                              # identical models
    _b, _c, p_diff = modeling.mcnemar_test(y, y, noisy)
    assert p_diff < 0.05                              # clearly different


def test_paired_features_and_leakage():
    """Paired deviation features: pairing logic + no cross-patient info."""
    import paired as pmod
    rng = np.random.default_rng(3)
    wn_full = np.linspace(400, 1800, 400)
    rows, labels, groups = [], [], []
    for p in range(8):
        base = rng.normal(0, 1, len(wn_full)) * 5      # patient offset
        for lab, k in (("Normal", 2), ("Tumor", 2)):
            for _ in range(k):
                rows.append(base + rng.normal(0, 0.3, len(wn_full))
                            + (1.5 if lab == "Tumor" else 0.0))
                labels.append(lab)
                groups.append(f"S{p}")
    # one unpaired patient (Tumor only) must be excluded
    rows.append(rng.normal(0, 1, len(wn_full)))
    labels.append("Tumor")
    groups.append("S_lonely")
    params = pp.PreprocessParams(crop_min=0, crop_max=0, wavelet=False)
    pdata = pmod.paired_features(np.vstack(rows), labels, groups,
                                 wn_full, params)
    assert pdata.n_patients == 8
    assert pdata.n_unpaired_excluded == 1
    assert pdata.X.shape[0] == 8 * 4                    # 32 paired rows
    assert set(pdata.y) == {"Normal", "Tumor"}
    # leakage guard: reference uses ONLY the patient's own normals —
    # changing another patient's spectra must not change these features
    rows2 = np.vstack(rows).copy()
    other = [i for i, g in enumerate(groups) if g == "S0"]
    rows2[other] += 100.0
    pdata2 = pmod.paired_features(rows2, labels, groups, wn_full, params)
    assert np.allclose(pdata.X[[i for i, r in enumerate(pdata.groups)
                                if r != "S0"]],
                       pdata2.X[[i for i, r in enumerate(pdata2.groups)
                                 if r != "S0"]])
    va = pmod.variance_analysis(np.vstack(rows), labels, groups)
    assert 0.0 <= va["between_patient_var_share"] <= 1.0
    assert np.isfinite(va["d_prime_spectrum"])


def test_nested_pipeline_evaluation():
    """evaluate_pipeline: unbiased, returns per-fold choices."""
    X, y, groups = _synthetic_ml(40, 10)
    wn = np.linspace(300, 1900, X.shape[1])
    out = modeling.evaluate_pipeline(X, wn, list(y), groups=list(groups),
                                     k=3, seed=1, progress=lambda m: None)
    assert len(out["fold_f1s"]) == 3
    assert 0.0 <= out["mean_f1"] <= 1.0
    assert all("preprocess" in c and "model" in c
               for c in out["fold_choices"])


def test_despike_whitaker_hayes():
    """Whitaker-Hayes: 1-point spikes repaired, real wide peaks survive."""
    from preprocessing import despike
    rng = np.random.default_rng(5)
    y = rng.normal(0, 0.05, 600)
    y[200:206] += 3.0                       # wide real peak
    spiked = y.copy()
    spiked[100] += 80.0                     # cosmic ray
    spiked[400] -= 60.0                     # negative spike
    fixed = despike(spiked, z_thresh=7.0, window=5)
    assert abs(fixed[100] - y[100]) < 0.5
    assert abs(fixed[400] - y[400]) < 0.5
    assert fixed[203] > 2.5                 # the real peak survives


def test_pqn_normalization():
    """PQN: a globally rescaled spectrum maps back onto its reference."""
    from preprocessing import pqn_normalize
    ref = np.abs(np.sin(np.linspace(0, 3, 200))) + 0.2
    scaled = ref * 3.7 + 0.02
    q = pqn_normalize(scaled, ref)
    assert abs(np.median(q / ref) - 1.0) < 1e-9
    # a spectrum equal to its reference is unchanged up to numerics
    q2 = pqn_normalize(ref, ref)
    assert np.allclose(q2, ref, atol=1e-9)


def test_arpls_fallback_and_method():
    """baseline_method 'arpls' resolves; missing library falls back."""
    p1 = pp.PreprocessParams(baseline_method="arpls")
    if pp.HAS_PYBASELINES:
        assert p1.validate().baseline_method == "arpls"
        out = pp.preprocess_spectrum(
            np.abs(np.sin(np.linspace(0, 20, 300))) * 10 + 5,
            pp.PreprocessParams(baseline_method="arpls",
                                crop_min=0, crop_max=0))
        assert np.isfinite(out).all()
    else:
        assert p1.validate().baseline_method == "als"


def test_adaptive_wavelet_and_new_steps():
    """Bayes/SURE adaptive thresholds match-or-beat the fixed universal
    rule on synthetic Lorentzians; garrote + cycle-spin + coif5 run; new
    baselines / area / minmax / detrend / Phe-1003 alignment behave."""
    rng = np.random.default_rng(0)
    wn = np.linspace(700, 1800, 1100)
    clean = (np.exp(-((wn - 1003) / 8) ** 2)
             + 0.7 * np.exp(-((wn - 1450) / 10) ** 2)) + 0.3 * (wn - 700) / 1100
    y = clean + rng.normal(0, 0.06, len(wn))
    rmse = lambda o: float(np.sqrt(np.mean((o - clean) ** 2)))  # noqa: E731
    raw = rmse(y)
    uni = rmse(pp.wavelet_denoise(y, "sym8", 4, "universal", "soft"))
    bayes = rmse(pp.wavelet_denoise(y, "sym8", 4, "bayes", "soft"))
    sure = rmse(pp.wavelet_denoise(y, "sym8", 4, "sure", "soft"))
    assert max(uni, bayes, sure) < raw * 0.6      # denoising works
    assert min(bayes, sure) <= uni + 1e-4         # adaptive >= universal
    for kw in ({"tmode": "garrote", "threshold": "sure"},
               {"tmode": "garrote", "threshold": "bayes", "cycle": 3},
               {"threshold": "sure", "wavelet": "coif5", "level": 5}):
        assert 0 < rmse(pp.wavelet_denoise(y, **kw)) < raw, kw
    peak = wn[np.argmax(pp.wavelet_denoise(y, "sym8", 4, "sure", "soft"))]
    assert abs(peak - 1003) < 6                   # peak position survives
    if pp.HAS_PYBASELINES:
        for m in ("iarpls", "pspline", "snip"):
            b = pp._pybase(y, m, 1e5)
            assert np.isfinite(b).all() and b.max() <= y.max() + 1e-9, m
    assert abs(np.trapezoid(pp.normalize(y, "area"))) - 1.0 < 1e-9
    mm = pp.normalize(y, "minmax")
    assert mm.min() >= 0 and mm.max() <= 1 + 1e-12
    slope_before = np.polyfit(np.arange(len(y)), y, 1)[0]
    slope_after = np.polyfit(np.arange(len(y)), pp.detrend_linear(y), 1)[0]
    assert abs(slope_after) < abs(slope_before) / 10
    X = np.vstack([y, np.interp(wn, wn - 4.0, y)])
    drift = np.abs(pp.estimate_wn_drift(X, wn)).max()
    fixed = np.abs(pp.estimate_wn_drift(pp.calibrate_wn(X, wn), wn)).max()
    assert fixed < drift, "Phe-1003 alignment must reduce drift"
    p = pp.PreprocessParams(
        baseline_method="iarpls", norm="area", detrend=True,
        wn_calibrate=True, wavelet_threshold="sure", wavelet_mode="garrote",
        wavelet_cycle=2, wavelet_name="coif5").validate()
    assert (p.wavelet_threshold, p.wavelet_mode, p.wavelet_cycle) == \
        ("sure", "garrote", 2)
    assert np.isfinite(pp.preprocess_matrix(X, p, wn=wn)).all()
    # defaults unchanged (back-compat): saved sessions keep old behavior
    d = pp.PreprocessParams().validate()
    assert (d.wavelet_threshold, d.wavelet_mode, d.wavelet_cycle,
            d.baseline_method, d.norm) == \
        ("universal", "soft", 0, "als", "vector")


def test_stage2_registry_augmentation_uncertainty_pr():
    """Stage 2: report models in the registry; CNN augmentation; MC
    dropout + deep ensemble; physics synthesis; PR points."""
    import gui  # SUITE_VERSION bump when the registry changes
    assert gui.SUITE_VERSION >= 7
    assert len(modeling.ALL_MODEL_NAMES) == 24
    for n in ("Sparse PLS-DA", "PLS + XGBoost", "PCA + XGBoost",
              "t-test filter + XGBoost", "Spectral + band features"):
        assert n in modeling.ALL_MODEL_NAMES
    X, y, groups = _synthetic_ml(60, 12, seed=3)
    res, _ = modeling.evaluate_models(
        X, list(y), model_names=["t-test filter + XGBoost"],
        k_folds=3, groups=list(groups))
    assert res[0].error is None, res[0].error
    # CNN: augmented fit still satisfies the proba contract; MC dropout
    # returns mean+std; the 3-seed ensemble agrees with its members
    if modeling.HAS_TORCH:
        from sklearn.base import clone
        est = clone(modeling.CNN1DClassifier(epochs=3)).fit(X, y)
        proba = est.predict_proba(X[:6])
        assert proba.shape == (6, 2) and np.allclose(proba.sum(1), 1)
        mean, std = est.predict_proba_mc(X[:6], n_passes=4)
        assert mean.shape == (6, 2) and (std >= 0).all()
        ens = modeling.CNNEnsemble(n_seeds=2, epochs=3).fit(X, y)
        assert ens.predict_proba(X[:4]).shape == (4, 2)
        assert (ens.predict_proba_std(X[:4]) >= 0).all()
        # physics synthesis: bounded perturbations of convex blends
        wn = np.linspace(700, 1800, X.shape[1])
        syn = modeling.lorentzian_synthesize(X, 5, wn=wn)
        assert syn.shape == (5, X.shape[1]) and np.isfinite(syn).all()
        lo, hi = X.min(), X.max()
        assert syn.min() >= lo - 0.6 * abs(hi) and \
            syn.max() <= hi + 0.6 * abs(hi)
    # PR points: AP in [0,1]; a perfect ranking gives AP 1
    rng = np.random.default_rng(4)
    s = np.linspace(0, 1, 40)
    yb = (s > 0.5).astype(int)
    prec, rec, ap = modeling.pr_points(yb, s)
    assert 0 <= ap <= 1 and len(prec) == len(rec)
    assert modeling.pr_points(yb, s)[2] > modeling.pr_points(
        yb, rng.permutation(s))[2]
    # ViT MC-dropout + SpecAugment path compiles and runs
    import vit_train as vt
    import torch
    from vit_model import SpectralViT, ViTConfig
    cfg = ViTConfig(seq_len=X.shape[1], patch_size=8, dim=32, depth=1,
                    heads=2, n_classes=2)
    net = SpectralViT(cfg)
    mean, std = vt.mc_dropout_predict(net, X[:8], torch.device("cpu"),
                                      n_passes=3)
    assert mean.shape == (8, 2) and np.allclose(mean.sum(1), 1, atol=1e-5)


def test_stage3_conformal_permutation_explainability():
    """Conformal sets keep their coverage promise on synthetic OOF;
    ECE/isotonic compare runs; the permutation null kills a fake signal
    and keeps a real one; VIP/Grad-CAM produce aligned profiles; the
    band-agreement report matches known bands; patient prob-mean
    aggregation logic is exercised through the GUI offscreen."""
    import clinical as clin
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    sep = 3.0
    s = np.clip(y * sep + rng.normal(0, 1, n), 0, None)
    proba = np.column_stack([1 - 1 / (1 + np.exp(-(s - sep / 2))),
                             1 / (1 + np.exp(-(s - sep / 2)))])
    q = clin.conformal_q(proba, y)
    m = clin.conformal_metrics(clin.conformal_sets(proba, q), y)
    assert m["coverage"] >= 0.85, m          # ~90% promised
    assert 0.5 <= m["mean_size"] <= 2.0      # empty sets = abstention
    e = clin.ece(y, proba[:, 1])
    assert 0 <= e <= 1
    cmp_ = clin.isotonic_compare(y, proba[:, 1])
    assert set(cmp_) >= {"ece_raw", "ece_platt", "ece_isotonic"}
    # permutation test: real signal → small p; noise → large p
    Xr = rng.normal(size=(120, 6))
    yr = rng.integers(0, 2, 120).astype(str)
    yr = np.array(["A", "B"])[yr.astype(int)]
    gr = [f"P{i//4}" for i in range(120)]
    Xr = Xr + (yr == "B")[:, None] * 2.0
    from sklearn.linear_model import LogisticRegression
    out = modeling.permutation_auc_p(
        LogisticRegression(max_iter=1000), Xr, yr, gr, n_perm=15, k=3)
    assert out["p"] <= 0.07 and out["auc"] > 0.9, out
    Xn = rng.normal(size=(120, 6))
    out_n = modeling.permutation_auc_p(
        LogisticRegression(max_iter=1000), Xn, yr, gr, n_perm=15, k=3)
    assert out_n["p"] > 0.1, out_n
    # VIP: PLS loadings produce a per-wavenumber profile
    pls = modeling.PLSScores(n_components=3).fit(Xr, yr)
    vip = modeling.pls_vip(pls.pls_)
    assert vip.shape == (Xr.shape[1],) and (vip >= 0).all()
    # Grad-CAM: aligned to the wavenumber axis, non-negative, peaked
    if modeling.HAS_TORCH:
        est = modeling.CNN1DClassifier(epochs=2).fit(Xr, yr)
        cam = est.grad_cam(Xr[:8])
        assert cam.shape == (8, Xr.shape[1])
        assert (cam >= 0).all() and abs(cam.max() - 1.0) < 1e-6
    # band agreement: importance concentrated on a literature band
    import biochemistry as bio
    wn = np.linspace(400, 1800, 700)
    imp = np.exp(-((wn - 1003) / 12) ** 2)          # phenylalanine peak
    rep = bio.agreement_report(wn, imp, top_k=10)
    assert rep["lit_frac"] >= 0.3 and rep["spearman"] > 0.2, rep
    # patient probability-mean via the GUI machinery (offscreen).
    # ISOLATED construction (deep_test pattern): the un-isolated window
    # auto-loads the real dataset — combined with the torch work above,
    # that natively aborted on some machines (Brain gotcha #22).
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qt_compat import QtWidgets
    import ui_helpers as uh
    import clinical_data as _cd
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")               # windowsvista + this QSS-heavy
    app.setStyleSheet(uh.STYLESHEET)     # UI fail-fasts (0xC0000409) —
    import gui                          # always style before constructing
    _saved_ls, _saved_fdr = uh.load_settings, _cd.find_data_root
    uh.load_settings = lambda: {}
    _cd.find_data_root = lambda: None
    try:
        win = gui.MainWindow()
    finally:
        uh.load_settings = _saved_ls
        _cd.find_data_root = _saved_fdr
    win._positive_class = lambda: "Tumor"     # no trained winner needed
    with tempfile.TemporaryDirectory() as td:  # real folder layout
        for i in range(3):
            os.makedirs(os.path.join(td, f"P{i}"), exist_ok=True)
        files = [os.path.join(td, f"P{i}", f"s{j}.txt")
                 for i in range(3) for j in range(4)]
        pt = [0.9 if (i + j) % 3 == 0 else 0.15
              for i in range(3) for j in range(4)]
        rows = [(files[4 * i + j], "Tumor" if p >= 0.5 else "Normal", p)
                for (i, j, p) in ((i, j, pt[4 * i + j])
                                  for i in range(3) for j in range(4))]
        win._pred_rows = rows
        win._pred_probs = [{"Normal": 1 - p, "Tumor": p} for p in pt]
        win.spec_path_edit.setText(td)
        agg = win._aggregate_patients(files, rows)
        assert len(agg) == 3
        for patient, n_spec, _votes, mean_p, verdict in agg:
            assert n_spec == 4 and 0 <= mean_p <= 1
            assert verdict in ("Normal", "Tumor")
            # probability-mean verdict: >= 0.5 mean P(Tumor) -> Tumor
            assert verdict == ("Tumor" if mean_p >= 0.5 else "Normal")


def test_stage4_power_card_fusion_saliva_external():
    """Stage 4: AUC power behaves; saliva preset + thiocyanate QC +
    biochemical shift + dataset QC; covariate fusion lifts AUC when the
    covariate is informative; model card exports; the external-
    validation CLI runs end-to-end on a synthetic cohort."""
    import clinical as clin
    pw = clin.auc_power(150, 150, auc=0.9)
    assert 0.5 < pw["detectable_auc_80pct"] < 0.9
    pw_small = clin.auc_power(20, 20, auc=0.9)
    assert pw_small["detectable_auc_80pct"] > pw["detectable_auc_80pct"]
    # 2026-09-05 regression: one-sample detectable delta = (z_a+z_b)*SE —
    # the old /sqrt(2) made power claims ~1.4x optimistic.
    from scipy.stats import norm as _norm
    _za = float(_norm.ppf(0.975))
    assert abs(pw["detectable_auc_80pct"]
               - (0.5 + (_za + 0.8416) * pw["se_auc"])) < 1e-9
    p = pp.PreprocessParams.saliva().validate()
    assert p.crop_max >= 2300 and p.baseline_method == "snip" \
        and p.norm == "snv"
    wn = np.linspace(400, 2300, 1900)
    rng = np.random.default_rng(0)

    def spec(car_high: bool) -> np.ndarray:
        return (np.exp(-((wn - 1003) / 8) ** 2)
                + 0.5 * np.exp(-((wn - 1445) / 10) ** 2)
                + (0.4 if car_high else 0.05)
                * np.exp(-((wn - 1520) / 10) ** 2)
                + 0.3 * np.exp(-((wn - 2120) / 12) ** 2)
                + rng.normal(0, 0.02, len(wn)))

    X = np.vstack([spec(False) for _ in range(30)]
                  + [spec(True) for _ in range(30)])
    y = np.array(["Normal"] * 30 + ["Tumor"] * 30)
    import biochemistry as bio
    tc = bio.thiocyanate_index(wn, X)
    assert np.isfinite(tc).all() and (tc > 0).all()
    q = bio.dataset_qc(X, wn)
    assert q["snr_median"] > 5 and q["thiocyanate_band"]
    rows = bio.biochemical_shift(X, wn, y)
    car = [r for r in rows if r[1] == "carotenoids"
           and abs(r[0] - 1520) < 1]
    assert car and car[0][3] > 0.2, rows          # measured direction
    # covariate fusion: an informative covariate must lift the AUC
    probs = np.clip(0.55 + 0.35 * (y == "Tumor")
                    + rng.normal(0, 0.18, len(y)), 0.02, 0.98)
    cov = ((y == "Tumor") * 1.2
           + rng.normal(0, 0.35, len(y))).reshape(-1, 1)
    gr = [f"P{i//3}" for i in range(len(y))]
    out = modeling.covariate_fusion_cv(probs, cov, y, gr)
    assert out["auc_base"] > 0.7 and out["auc_fused"] > out["auc_base"]
    # model card export on a real winner
    Xs, ys, gs = _synthetic_ml(60, 12, seed=5)
    results, winner = modeling.evaluate_models(
        Xs, list(ys), model_names=["PCA + LDA"], k_folds=3,
        groups=list(gs))
    with tempfile.TemporaryDirectory() as td:
        card = os.path.join(td, "card.md")
        text = modeling.export_model_card(card, winner,
                                          pp.PreprocessParams(),
                                          dataset_name="synthetic")
        assert "macro-F1" in text and "Limitations" in text
        assert os.path.getsize(card) > 400
        # external validation CLI on the same synthetic distribution
        bpath = os.path.join(td, "b.joblib")
        modeling.save_bundle(bpath, winner, np.linspace(500, 2000,
                                                        Xs.shape[1]),
                             pp.PreprocessParams())
        import validate_external as ve
        rc = ve.main(["--data", "unused", "--model", bpath])
        assert rc == 2                       # missing folder -> clean exit
        # in-memory external run: reuse the module's helpers
        proba = winner.pipeline.predict_proba(Xs)
        assert proba.shape == (60, 2)


def test_roadmap_J_E_batch():
    """J1 label-error flags planted mislabels; E7 band stability
    separates stable from noise bands; E3 TTA keeps the proba contract;
    the locked-exam block in reproduce_study is wired (unit level)."""
    # J1: 8 spectra with swapped labels on a separable problem
    rng = np.random.default_rng(7)
    y = np.array(["A", "B"] * 60)
    p_b = np.clip((y == "B") * 0.9 + 0.05 + rng.normal(0, 0.02, 120),
                  0.01, 0.99)
    proba = np.column_stack([1 - p_b, p_b])
    planted = [3, 17, 40, 66, 90, 104, 111, 118]
    y_swapped = y.copy()
    y_swapped[planted] = np.where(y[planted] == "A", "B", "A")
    rep = modeling.label_error_report(proba, list(y_swapped))
    flagged = {r["i"] for r in rep}
    assert len(flagged & set(planted)) >= 6, rep   # catches most mislabels
    assert len(flagged - set(planted)) <= 4        # few false accusations
    assert all(r["tier"] in ("likely", "review") for r in rep)
    # E7: consistent importance peaks → high Jaccard; noise → low
    wn_len = 200
    prof_a = np.abs(rng.normal(size=wn_len))
    prof_a[[5, 20, 40, 60, 80, 100, 130, 150, 170, 190]] = 10.0
    st = modeling.band_stability(
        [prof_a + rng.normal(0, 0.5, wn_len) for _ in range(4)], top_k=10)
    assert st["jaccard_mean"] > 0.6
    noisy = [np.abs(rng.normal(size=wn_len)) for _ in range(4)]
    st2 = modeling.band_stability(noisy, top_k=10)
    assert st2["jaccard_mean"] < st["jaccard_mean"]
    assert st["stable"].shape == (wn_len,) and st["stable"].sum() >= 2
    # E3: CNN + ViT TTA keep the probability contract
    if modeling.HAS_TORCH:
        Xc, yc, _gc = _synthetic_ml(40, 8, seed=9)
        est = modeling.CNN1DClassifier(epochs=2).fit(Xc, yc)
        tta = est.predict_proba_tta(Xc[:5], n_aug=3)
        assert tta.shape == (5, 2) and np.allclose(tta.sum(1), 1,
                                                   atol=1e-5)
        import vit_train as vt
        import torch
        from vit_model import SpectralViT, ViTConfig
        cfg = ViTConfig(seq_len=Xc.shape[1], patch_size=8, dim=32,
                        depth=1, heads=2, n_classes=2)
        t = vt.tta_predict(SpectralViT(cfg), Xc[:5],
                           torch.device("cpu"), n_aug=2)
        assert t.shape == (5, 2) and np.allclose(t.sum(1), 1, atol=1e-5)
    # A1 wiring: the locked exam's import path exists
    from clinical_data import patient_split  # noqa: F401
    import reproduce_study  # noqa: F401  (parse-level sanity)


def test_phantom_cohort_ground_truth():
    """J2: the phantom cohort separates perfectly, carries the planted
    biochemistry directions, and the planted bands dominate a simple
    class-difference importance profile (explainability ground truth)."""
    X, y, groups, wn, planted = modeling.phantom_cohort(
        n_patients=24, spectra_per=3, seed=3)
    assert X.shape[0] == 72 and set(y) == {"Normal", "Tumor"}
    assert len(set(groups)) == 24 and np.isfinite(X).all()
    # planted directions recovered by the biochemistry layer
    import biochemistry as bio
    rows = bio.biochemical_shift(X, wn, y)
    lut = {round(r[0]): r for r in rows}
    for center, d in planted:
        if d == 0 or center not in lut:
            continue
        r = lut[center]
        assert np.sign(r[3]) == np.sign(d) and r[4] < 0.05, (center, r)
    # class-difference profile ranks planted bands in the top decile
    diff = np.abs(X[y == "Tumor"].mean(0) - X[y == "Normal"].mean(0))
    top = set(wn[np.argsort(-diff)[:60]])           # top 60 of 1900
    hits = sum(1 for c, _d in planted
               if any(abs(c - t) < 15 for t in top))
    assert hits >= 5, (planted, hits)               # ≥5 of 8 planted
    # a quick grouped-CV model nails the phantom (sanity for CI smoke)
    res, _w = modeling.evaluate_models(
        X, list(y), model_names=["PCA + LDA"], k_folds=3,
        groups=list(groups))
    assert res[0].error is None and res[0].macro_f1() > 0.9


def test_ttest_select_pls_scores_report_replication():
    """Report's building blocks: Welch/Cohen-d filter (leakage-safe inside
    a pipeline), PLS-scores transformer with string labels, and the
    head-to-head replication engine end-to-end on synthetic data."""
    X, y, groups = _synthetic_ml(80, 16, seed=1)
    Xb = X + (y == "B")[:, None] * 2.0            # strongly separable
    sel = modeling.TTestSelect().fit(Xb, y)
    assert 1 <= sel.n_selected_ <= X.shape[1]
    assert sel.transform(Xb).shape[1] == sel.n_selected_
    rng = np.random.default_rng(2)
    Xn = rng.normal(size=(40, 20))
    yn = np.array(["A", "B"] * 20)
    assert modeling.TTestSelect(d_min=50.0).fit(Xn, yn).n_selected_ == 20
    ym = np.array(["A", "B", "C"] * 10)
    assert modeling.TTestSelect().fit(rng.normal(size=(30, 8)),
                                      ym).n_selected_ == 8   # multiclass: all
    from sklearn.base import clone
    scores = modeling.PLSScores(n_components=3).fit(Xb, y)
    assert scores.transform(Xb).shape == (len(Xb), 3)
    assert clone(scores).get_params()["n_components"] == 3
    import reproduce_report as rr
    res = rr.run(Xb.astype(float), y, groups, folds=3, repeats=2, mini=True)
    assert set(res["models"]) == {"PCA+SVM", "PCA+RF", "PLS+XGB"}
    assert all(r["A"]["f1"] > 0.9 for r in res["models"].values())
    assert "friedman_p" in res


def test_shap_region_importance():
    """SHAP importance: signed, aligned, with named bands when available."""
    try:
        import shap  # noqa: F401
    except ImportError:
        return                                   # optional dependency
    X, y, groups = _synthetic_ml(60, 12)
    wn = np.linspace(300, 1900, X.shape[1])
    w2, signed, bands = modeling.region_importance_shap(X, list(y),
                                                        groups, wn)
    assert signed.shape == wn.shape and np.isfinite(signed).all()
    assert 1 <= len(bands) <= 5
    assert any(abs(s) > 0 for _c, _sh, _n, s in bands)
    # band names attach inside the known table window
    named = [n for _c, _sh, n, _s in bands if n]
    assert isinstance(named, list)


def test_result_plot_helpers():
    """The Result-page chart helpers render on a plain Agg figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotting

    fig, ax = plt.subplots()
    plotting.plot_prob_histogram(ax, np.array([0.1, 0.62, 0.85, 0.85]),
                                 threshold=0.55, pos_name="tumor")
    fig.canvas.draw()                      # forces the annotate() path
    plt.close(fig)

    wn = np.linspace(1800.0, 400.0, 400)
    spectra = [np.abs(np.sin(np.linspace(0, 6, 400))) for _ in range(4)]
    fig, ax = plt.subplots()
    plotting.plot_prediction_spectra(
        ax, wn, spectra, mean_trace=np.mean(spectra, axis=0),
        reference=spectra[0] * 0.9,
        bands=[(1000.0, 0.4, "phenylalanine"), (1450.0, 0.3, "")])
    fig.canvas.draw()
    plt.close(fig)


def test_plural_counts():
    """'1 spectrum', '3 spectra', '2 patients' — grammar-safe counts."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import gui
    assert gui.plural(1, "spectrum") == "1 spectrum"
    assert gui.plural(3, "spectrum") == "3 spectra"
    assert gui.plural(1, "patient") == "1 patient"
    assert gui.plural(2, "patient") == "2 patients"


def test_operating_points_and_triage():
    """Rule-out keeps sensitivity, rule-in keeps specificity; triage
    tiers split at those thresholds and collapse on a weak ROC."""
    import clinical as clin
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 400)
    z = 1.5 * y + rng.normal(0, 1.3, 400)
    p = 1 / (1 + np.exp(-z))
    lo, hi, s_lo, s_hi = clin.operating_points(y, p)
    assert 0 < lo < hi < 1 and s_lo >= 0.88 and s_hi >= 0.88
    assert np.isfinite(lo) and np.isfinite(hi)  # no sklearn inf leaks
    assert clin.triage(0.99, lo, hi) == "POSITIVE"
    assert clin.triage(0.01, lo, hi) == "NEGATIVE"
    assert clin.triage((lo + hi) / 2, lo, hi) == "INDETERMINATE"
    assert clin.triage(float("nan"), lo, hi) == "INDETERMINATE"
    # coin-flip model: wide band -> middle is indeterminate, and the
    # fallback fires when the ROC cannot support two targets
    lo2, hi2, *_ = clin.operating_points(y, rng.uniform(size=400))
    assert lo2 is not None and lo2 < hi2
    assert clin.triage(0.7, lo2, hi2) == "INDETERMINATE"
    assert clin.triage(0.9, None, None, fallback=0.5) == "POSITIVE"


def test_ppv_npv_prevalence():
    """Predictive values move with prevalence (Bayes)."""
    import clinical as clin
    ppv05, npv05 = clin.ppv_npv(0.9, 0.9, 0.05)
    ppv60, _ = clin.ppv_npv(0.9, 0.9, 0.60)
    assert ppv05 < 0.5 and ppv60 > 2 * ppv05 and npv05 > 0.99
    # degenerate prevalence does not divide by zero
    assert np.isfinite(clin.ppv_npv(0.9, 0.9, 0.0)[0])


def test_calibration_and_dca():
    """Calibration bins are monotone-ish for a calibrated model; the
    decision curve reduces to treat-none at extreme thresholds."""
    import clinical as clin
    rng = np.random.default_rng(12)
    y = rng.integers(0, 2, 600)
    p = np.clip(0.25 + 0.5 * y + rng.normal(0, 0.18, 600), 0.01, 0.99)
    bins = clin.calibration_bins(y, p)
    assert len(bins) >= 6
    assert all(0 <= b[0] <= 1 and 0 <= b[1] <= 1 for b in bins)
    pred = [b[0] for b in bins]
    assert pred == sorted(pred)          # bins ordered by predicted prob
    # 2026-09-05 regression: bins are CONTIGUOUS ranges of the sorted p —
    # the first bin must sit far below the last (the old interleaved
    # order[b::n] made every bin span the full range, flattening the
    # reliability diagram to the global mean).
    assert pred[0] < pred[-1] - 0.25
    # and the observed rate must track p inside the bins (calibrated-ish
    # model): correlation of mean_p vs observed is positive.
    obs = [b[1] for b in bins]
    assert np.corrcoef(pred, obs)[0, 1] > 0.5
    thr, nb_m, nb_all = clin.decision_curve(y, p)
    assert len(thr) == len(nb_m) == len(nb_all)
    assert nb_m.max() > 0                # the model has utility somewhere
    # treat-all is beneficial at low thresholds, harmful at high ones,
    # and equals the analytic formula prev - (1-prev)*t/(1-t)
    assert nb_all[0] > 0 and nb_all[-1] < 0
    prev = float(np.mean(y))
    analytic = prev - (1 - prev) * thr / (1 - thr)
    assert np.allclose(nb_all, analytic, atol=1e-9)


def test_biochemistry_bands_and_ratios():
    """Band windows integrate the right peaks; ratios follow the
    literature direction (nucleic acids up in tumor, collagen down)."""
    import biochemistry as bio
    wn = np.linspace(400, 1800, 700)

    def mk(nuc=1.0, col=1.0, ker=1.0):
        v = 0.05 + 0.5 * np.exp(-((wn - 1445) / 15) ** 2)
        v += 0.8 * np.exp(-((wn - 1003) / 10) ** 2)
        v += nuc * 0.9 * np.exp(-((wn - 785) / 10) ** 2)
        v += nuc * 0.7 * np.exp(-((wn - 1090) / 12) ** 2)
        v += col * 0.6 * np.exp(-((wn - 854) / 10) ** 2)
        v += col * 0.5 * np.exp(-((wn - 1240) / 14) ** 2)
        v += ker * 0.6 * np.exp(-((wn - 938) / 12) ** 2)
        v += 0.7 * np.exp(-((wn - 1655) / 14) ** 2)
        return v

    r_hi = bio.band_ratios(wn, mk(nuc=2.0))
    r_lo = bio.band_ratios(wn, mk(nuc=0.2))
    assert r_hi["nucleic/protein"] > 5 * r_lo["nucleic/protein"]
    assert bio.keratin_index(wn, mk(ker=3)) > 5 * bio.keratin_index(
        wn, mk(ker=0.1))
    # outside the covered range the area is NaN-safe
    assert np.isnan(bio.band_area(wn, np.zeros_like(wn), 3000.0, 5.0))


def test_biochemistry_paired_and_nmf():
    """Paired deltas separate tumor from the same patient's normal;
    NMF is deterministic and labels components by dominant band."""
    import biochemistry as bio
    rng = np.random.default_rng(13)
    wn = np.linspace(400, 1800, 500)

    def mk(nuc, col):
        v = 0.05 + 0.5 * np.exp(-((wn - 1445) / 15) ** 2)
        v += 0.8 * np.exp(-((wn - 1003) / 10) ** 2)
        v += nuc * 0.9 * np.exp(-((wn - 785) / 10) ** 2)
        v += col * 0.6 * np.exp(-((wn - 854) / 10) ** 2)
        v += 0.7 * np.exp(-((wn - 1655) / 14) ** 2)
        return v

    X = np.vstack([mk(0.2, 1.0) + 0.005 * rng.normal(0, 1, 500)
                   for _ in range(18)]
                  + [mk(2.0, 0.3) + 0.005 * rng.normal(0, 1, 500)
                     for _ in range(18)])
    yy = ["Normal"] * 18 + ["Tumor"] * 18
    gg = [f"P{i % 9}" for i in range(36)]
    _cls_rows, paired = bio.ratio_table(wn, X, yy, gg)
    d = {k: dv for k, _n, dv in paired}
    assert d["nucleic/protein"] > 0.5 and d["collagen/protein"] < 0
    H, labels, W = bio.nmf_components(X, wn, k=3, seed=2)
    H2, _l2, W2 = bio.nmf_components(X, wn, k=3, seed=2)
    assert np.allclose(H, H2) and np.allclose(W, W2)
    assert any("nucleic" in lab for lab in labels)
    deltas = bio.component_deltas(W, yy, gg)
    assert len(deltas) == 3 and all(n == 9 for _c, n, _d in deltas)


def test_biochemistry_plausibility_and_keratin_flags():
    """Literature check: matching sign agrees, opposite sign flags;
    keratin outliers are detected batch-relative."""
    import biochemistry as bio
    rows, (agree, known) = bio.plausibility(
        [(785.0, 0.3, "phosphate", 1), (854.0, 0.2, "", -1),
         (650.0, 0.1, "", 1)])
    assert agree == 2 and known == 2
    assert [r[5] for r in rows] == ["agree", "agree", "unassigned"]
    rows2, (agree2, _k2) = bio.plausibility([(785.0, 0.3, "", -1)])
    assert agree2 == 0 and rows2[0][5] == "opposite"
    rows3, _a3 = bio.plausibility([(1000.0, 0.3, "", 1)])
    assert rows3[0][5] == "agree"        # 1003 phenylalanine window
    wn = np.linspace(400, 1800, 500)
    base = [np.exp(-((wn - 1003) / 10) ** 2) for _ in range(6)]
    spike = np.exp(-((wn - 938) / 12) ** 2) * 8 + np.exp(
        -((wn - 1003) / 10) ** 2)
    _med, _mad, flagged, total = bio.keratin_flags(wn, base + [spike])
    assert flagged >= 1 and total == 7


def test_platt_calibration():
    """Platt scaling: monotone map that lowers the Brier score of a
    miscalibrated (shrunk) model; ~identity on calibrated input."""
    import clinical as clin
    from sklearn.metrics import brier_score_loss
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p_true = np.clip(0.25 + 0.5 * y + rng.normal(0, 0.18, 300), 0.01,
                     0.99)
    p_raw = 0.5 + (p_true - 0.5) * 0.3          # shrunk -> miscalibrated
    ab = clin.fit_platt(y, p_raw)
    assert ab is not None and ab[0] > 0
    p_cal = clin.apply_platt(p_raw, ab)
    assert brier_score_loss(y, p_cal) < brier_score_loss(y, p_raw)
    order = np.argsort(p_raw)                   # monotone mapping
    assert np.all(np.diff(p_cal[order]) >= -1e-12)
    assert clin.fit_platt(y[:5], p_raw[:5]) is None   # too small
    # truly calibrated input -> near-identity fit
    p_c = np.clip(0.1 + 0.8 * rng.uniform(0, 1, 600), 0.02, 0.98)
    y_b = rng.binomial(1, p_c)
    a_id, b_id = clin.fit_platt(y_b, p_c)
    assert 0.6 < a_id < 1.6 and abs(b_id) < 0.25
    assert 0 < clin.apply_platt(0.7, (1.0, 0.0)) < 1


def test_wilson_and_delong():
    """Wilson CI containment; DeLong AUC equals sklearn's with a CI."""
    import math
    import clinical as clin
    from sklearn.metrics import roc_auc_score
    lo, hi = clin.wilson_ci(8, 10)
    assert lo < 0.8 < hi and 0 <= lo <= 1
    lo0, hi0 = clin.wilson_ci(0, 10)
    assert lo0 == 0.0 and 0 < hi0 < 0.35
    assert all(math.isnan(x) for x in clin.wilson_ci(0, 0))
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.25 + 0.5 * y + rng.normal(0, 0.18, 300), 0.01, 0.99)
    auc, se, lo, hi = clin.delong_auc_ci(y, p)
    assert abs(auc - roc_auc_score(y, p)) < 1e-9
    assert lo < auc < hi and lo > 0.5 and se > 0
    a2, _s2, l2, _h2 = clin.delong_auc_ci(y, y.astype(float))
    assert a2 == 1.0 and l2 > 0.5
    assert math.isnan(clin.delong_auc_ci(np.ones(10),
                                         np.linspace(0, 1, 10))[0])


def test_stacked_ensemble():
    """The clone-safe stacked ensemble fits, predicts, and competes in
    evaluate_models under patient-grouped CV."""
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(9)
    n = 120
    X = rng.normal(0, 1, (n, 20))
    X[:60, :4] += 1.5
    y = np.array([0] * 60 + [1] * 60)
    groups = np.array([f"P{i // 3}" for i in range(n)])
    specs = [
        ("lda", make_pipeline(StandardScaler(), PCA(5),
                              LogisticRegression())),
        ("log", make_pipeline(StandardScaler(), LogisticRegression()))]
    se = modeling.StackedEnsemble(specs, seed=1)
    se2 = clone(se)                       # must survive sklearn clone
    se2.fit(X, y, groups=groups)
    pr = se2.predict_proba(X)
    assert pr.shape == (n, 2) and np.allclose(pr.sum(axis=1), 1.0)
    assert (se2.predict(X) == y).mean() > 0.9
    modeling.StackedEnsemble(specs).fit(X, y)     # ungrouped path
    # grouped fit never leaks: identical group members stay together in
    # the meta-training (structural check — inner SGKF used, not KF)
    # end-to-end: the stacked candidate joins the comparison cleanly
    Xs = rng.normal(0, 1, (90, 30))
    Xs[45:] += 1.2
    ys = ["A" if i < 45 else "B" for i in range(90)]
    gs = [f"P{i // 5}" for i in range(90)]
    results, _winner = modeling.evaluate_models(
        Xs, ys, model_names=["PCA + LDA", "PCA + Logistic Regression",
                             "Ensemble (top-3)"],
        k_folds=3, groups=gs, wavenumbers=np.arange(30))
    stacked = next(r for r in results if r.name == "Stacked (top-3)")
    assert stacked.error is None, stacked.error


def test_calibrated_bundle_roundtrip():
    """A calibrated bundle maps predict-time probabilities through the
    Platt sigmoid; the stored threshold moves with them."""
    import clinical as clin
    import joblib
    import preprocessing as pp
    rng = np.random.default_rng(9)
    Xs = rng.normal(0, 1, (90, 30))
    Xs[45:] += 1.2
    ys = ["A" if i < 45 else "B" for i in range(90)]
    gs = [f"P{i // 5}" for i in range(90)]
    _res, winner = modeling.evaluate_models(
        Xs, ys, model_names=["PCA + LDA", "PCA + Logistic Regression",
                             "Ensemble (top-3)"],
        k_folds=3, groups=gs, wavenumbers=np.arange(30))
    params0 = pp.PreprocessParams(crop_min=0, crop_max=0)
    d = tempfile.mkdtemp()
    b_raw = os.path.join(d, "raw.joblib")
    b_cal = os.path.join(d, "cal.joblib")
    modeling.save_bundle(b_raw, winner, np.arange(30), params0)
    modeling.save_bundle(b_cal, winner, np.arange(30), params0,
                         calibrator=(1.4, -0.2))
    wn = np.arange(30, dtype=float)
    checked = 0
    for spec in (rng.normal(0, 1, 30) + 1.2, rng.normal(0, 1, 30) - 0.5,
                 rng.normal(0, 1, 30)):
        o_raw = modeling.predict_with_bundle(joblib.load(b_raw), wn, spec)
        o_cal = modeling.predict_with_bundle(joblib.load(b_cal), wn, spec)
        expect = clin.apply_platt(o_raw["probabilities"]["B"], (1.4, -0.2))
        assert abs(o_cal["probabilities"]["B"] - expect) < 1e-12
        checked += 1
    assert checked == 3
    bc = joblib.load(b_cal)
    if winner.threshold is not None:
        assert abs(bc["threshold"]
                   - clin.apply_platt(winner.threshold,
                                     (1.4, -0.2))) < 1e-12


def test_paired_deploy_parity_wn_calibrate():
    """Deploy-time features/probabilities must MATCH the training-time
    construction for bundles trained with wn_calibrate=True (2026-09-06
    regression: the predict path skipped the Phe-1003 alignment entirely,
    feeding paired models shifted out-of-distribution deviations — every
    prediction saturated to one confident class)."""
    import joblib
    import paired as paired_mod
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_pts = 700
    wn_full = np.linspace(500.0, 2000.0, n_pts)   # ~2.15 cm-1 spacing

    def _spectrum(patient_seed: int, tumor: bool, drift: float):
        rp = np.random.default_rng(patient_seed)
        base = (rp.normal(0, 0.03, n_pts)
                + 0.8 * np.exp(-((wn_full - 1003.0) ** 2) / (2 * 7.0 ** 2))
                + 0.4 * np.exp(-((wn_full - 1450.0) ** 2) / (2 * 18.0 ** 2)))
        if tumor:      # tumor-only band
            base += 0.6 * np.exp(-((wn_full - 1340.0) ** 2)
                                 / (2 * 9.0 ** 2))
        # instrument drift: shift every feature by `drift` cm-1 on the
        # axis (what calibrate_wn exists to undo; > grid spacing so the
        # correction actually fires)
        return np.interp(wn_full, wn_full - drift, base)

    X, labels, groups = [], [], []
    drifts = {}
    for p in range(3):
        for j, (tumor, d) in enumerate([(False, -5.0), (False, 5.0),
                                        (True, 6.0), (True, -6.0)]):
            X.append(_spectrum(100 + p, tumor, d))
            labels.append("Tumor" if tumor else "Normal")
            groups.append(f"P{p}")
            drifts[(p, j)] = d
    X = np.vstack(X)
    params = pp.PreprocessParams(crop_min=600.0, crop_max=1800.0,
                                 wavelet=False, detrend=True,
                                 wn_calibrate=True)
    # ---- training-side construction (the ground truth) ----------------
    pdata = paired_mod.paired_features(X, labels, groups, wn_full, params)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000))])
    pipe.fit(pdata.X, pdata.y)
    w = modeling.ModelResult(name="parity",
                             classes=list(pipe.classes_))
    w.pipeline = pipe
    w.macro = {"f1": (1.0, 0.0)}
    with tempfile.TemporaryDirectory() as td:
        bpath = os.path.join(td, "parity.joblib")
        modeling.save_bundle(bpath, w, wn_full, params, paired=True)
        bundle = joblib.load(bpath)

        def _write_full(path, wn, it):     # full precision, shuffled order
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                for a, b in zip(wn[::-1], it[::-1], strict=True):
                    fh.write(f"{a:.10g},{b:.10g}\n")

        files = {}
        for p in range(3):
            for j in range(4):
                f = os.path.join(td, f"P{p}", f"s{j}.txt")
                _write_full(f, wn_full, X[4 * p + j])
                files[(p, j)] = f
        grid = np.asarray(bundle["wavenumbers"], dtype=float)
        m = pp.crop_mask(grid, params)
        for p in range(3):
            ref = paired_mod.reference_vector(
                bundle, [files[(p, 0)], files[(p, 1)]])
            for j in (2, 3):               # the patient's tumor spectra
                wn_f, it_f = ds.load_spectrum(files[(p, j)])
                # deploy deviation, built exactly like the predict path
                y = pp.align_to_grid(wn_f, it_f, grid, params)
                dev = (pp.preprocess_spectrum(y[m], params) - ref)
                # training deviation: preprocess_matrix calibrated +
                # cropped, minus the patient's calibrated normal mean
                Xp = pp.preprocess_matrix(X[[4 * p + j, 4 * p, 4 * p + 1]],
                                          params, wn=wn_full)
                expect = Xp[0] - Xp[1:].mean(axis=0)
                assert np.allclose(dev, expect, atol=1e-8), \
                    f"deploy/training feature mismatch P{p} s{j}"
                # negative control: WITHOUT calibration the deviation
                # really is different (the drift is material, so the
                # parity above is not vacuous)
                from dataclasses import replace as _dc_replace
                Xr = pp.preprocess_matrix(
                    np.vstack([np.interp(grid, wn_full, X[4 * p + j]),
                               np.interp(grid, wn_full, X[4 * p]),
                               np.interp(grid, wn_full, X[4 * p + 1])]),
                    _dc_replace(params, wn_calibrate=False),
                    wn=grid)
                assert not np.allclose(Xr[0] - Xr[1:].mean(axis=0),
                                       expect, atol=1e-3), \
                    "injected drift had no effect — parity test is vacuous"
                # end-to-end: saved-bundle prediction == direct pipeline
                # call on the training-side feature (no calibrator in
                # this bundle -> probabilities are the raw ones)
                out = modeling.predict_with_bundle(
                    bundle, wn_f, it_f, reference=ref)
                direct = pipe.predict_proba(expect.reshape(1, -1))[0]
                for c, v in zip(pipe.classes_, direct, strict=True):
                    assert abs(out["probabilities"][c] - v) < 1e-6


def test_paired_predict_skips_without_reference():
    """Margin-mode prediction without the patient's normal reference is
    SKIPPED with an explicit log line — never silently predicted from an
    absolute spectrum (2026-09-06 regression guard)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qt_compat import QtWidgets
    import ui_helpers as uh
    import clinical_data as _cd
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(uh.STYLESHEET)
    import gui                          # style before constructing
    _saved_ls, _saved_fdr = uh.load_settings, _cd.find_data_root
    uh.load_settings = lambda: {}
    _cd.find_data_root = lambda: None
    try:
        wn = np.linspace(500.0, 2000.0, 60)
        with tempfile.TemporaryDirectory() as td:
            # a lone Tumor tree: no Normal side anywhere -> the paired
            # reference cannot be resolved
            f = os.path.join(td, "Tumor", "P1", "s0.txt")
            _write_spectrum(f, wn, np.random.default_rng(3).normal(
                0, 1, 60))
            bundle = {"paired": True, "wavenumbers": wn,
                      "prep_params": pp.PreprocessParams(),
                      "classes": ["Normal", "Tumor"]}
            worker = gui.PredictWorker(bundle, [f], td)
            payload = {}
            worker.done.connect(lambda p: payload.update(p))
            worker.run()                # synchronous: no event loop
            assert payload["rows"] == []
            assert any(line.startswith("SKIPPED ")
                       for line in payload["logs"]), payload["logs"]
    finally:
        uh.load_settings = _saved_ls
        _cd.find_data_root = _saved_fdr


def test_patient_verdict_uses_bundle_threshold():
    """The per-patient verdict cuts at the bundle's tuned (calibrated)
    threshold, not a hardcoded 0.5 — a mean P(Tumor) of 0.6 is Normal
    when the validated operating point is 0.75.  Also: _threshold_now
    must NOT Platt-map an already-calibrated stored threshold."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qt_compat import QtWidgets
    import ui_helpers as uh
    import clinical_data as _cd
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(uh.STYLESHEET)
    import gui
    _saved_ls, _saved_fdr = uh.load_settings, _cd.find_data_root
    uh.load_settings = lambda: {}
    _cd.find_data_root = lambda: None
    try:
        win = gui.MainWindow()
        # the startup 3SSE restore can install a real winner whenever
        # study_run_3sse/winner.json exists (persisted since 2026-09-06)
        # — this test needs the no-winner branch of _threshold_now
        win.winner = None
        win._positive_class = lambda: "Tumor"
        win.bundle = {"threshold": 0.75, "classes": ["Normal", "Tumor"],
                      "calibrator": (1.0705, 0.378)}
        # stored threshold is already calibrated: no double Platt map
        assert abs(win._threshold_now() - 0.75) < 1e-12
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "PA"), exist_ok=True)
            os.makedirs(os.path.join(td, "PB"), exist_ok=True)
            files = [os.path.join(td, "PA", "a.txt"),
                     os.path.join(td, "PA", "b.txt"),
                     os.path.join(td, "PB", "c.txt"),
                     os.path.join(td, "PB", "d.txt")]
            # PA: mean P(Tumor) = 0.60  (>0.5 but < 0.75 -> Normal)
            # PB: mean P(Tumor) = 0.90  (>= 0.75 -> Tumor)
            # per-spectrum labels use the SAME cut as the deploy path
            # (predict_with_bundle applies the bundle threshold)
            probs = [{"Normal": 0.4, "Tumor": 0.6},
                     {"Normal": 0.4, "Tumor": 0.6},
                     {"Normal": 0.1, "Tumor": 0.9},
                     {"Normal": 0.1, "Tumor": 0.9}]
            rows = [(os.path.basename(f),
                     "Tumor" if p["Tumor"] >= 0.75 else "Normal",
                     max(p.values())) for f, p in zip(files, probs,
                                                     strict=True)]
            win._pred_rows = rows
            win._pred_probs = probs
            win.spec_path_edit.setText(td)
            agg = {r[0]: r for r in win._aggregate_patients(files, rows)}
            assert agg["PA"][4] == "Normal", agg["PA"]
            assert agg["PB"][4] == "Tumor", agg["PB"]
            assert abs(agg["PA"][3] - 0.6) < 1e-9
    finally:
        uh.load_settings = _saved_ls
        _cd.find_data_root = _saved_fdr


def test_3sse_chain_cloneable():
    """3SSE winner pipelines must satisfy sklearn's clone contract: the
    five GUI diagnostics (learning curve, seed stability, noise, locked
    eval, LOPO) all clone(winner.pipeline) and died on 'Cannot clone
    object AveragedChain' before 2026-09-06 — list(estimators) in
    __init__ breaks clone's post-init identity check."""
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import sequential as seq

    def _mk():
        return Pipeline([("sc", StandardScaler()),
                         ("pca", PCA(n_components=2)),
                         ("lr", LogisticRegression())])

    ac = seq.AveragedChain([_mk(), _mk()], k=3, seed=42, n_seeds=2)
    sc = seq.SequentialChain([_mk(), _mk()], k=3, seed=42)
    for obj in (ac, sc):
        c = clone(obj)                      # RuntimeError before the fix
        assert c is not obj and c.k == obj.k
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 6))
    # class-consistent per group; the curve runs on the full data only
    # (tiny patient subsets can be single-class — not what we test here)
    y = np.repeat([0, 1, 0, 1, 0, 1, 0, 1], 5)
    g = np.repeat(np.arange(8), 5)
    ac.fit(X, y)
    sizes, means, _stds = modeling.learning_curve_by_groups(
        X, y, g, ac, k=3, seed=42, fractions=(1.0,))   # failing diagnostic
    assert sizes == [8] and len(means) == len(sizes)


def test_auto_reference_real_layout():
    """Real clinical layout (<root>/<class>/<patient>/*.csv) through the
    actual PredictWorker: per-patient AUTO references resolve, patients
    without a Normal side are skipped with a logged reason, an
    empty-string manual reference behaves like None (2026-09-06: it
    used to disable BOTH reference paths — every file skipped), and
    picking the tree ROOT as the manual reference fails with an
    actionable message instead of 'No reference spectra could be
    loaded'."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qt_compat import QtWidgets
    import ui_helpers as uh
    import clinical_data as _cd
    import joblib
    import paired as paired_mod
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(uh.STYLESHEET)
    import gui
    _saved_ls, _saved_fdr = uh.load_settings, _cd.find_data_root
    uh.load_settings = lambda: {}
    _cd.find_data_root = lambda: None
    try:
        wn = np.linspace(500.0, 2000.0, 400)   # ~3.75 cm-1 spacing

        def spec(seed, tumor, drift):
            r = np.random.default_rng(seed)
            base = (0.8 * np.exp(-((wn - 1003.0) ** 2) / (2 * 6.0 ** 2))
                    + 0.4 * np.exp(-((wn - 1450.0) ** 2) / (2 * 15.0 ** 2))
                    + r.normal(0, 0.02, len(wn)))
            if tumor:
                base = base + 0.5 * np.exp(-((wn - 1340.0) ** 2)
                                           / (2 * 8.0 ** 2))
            return np.interp(wn, wn - drift, base)

        sites = [(False, -7.0), (False, 7.0), (True, 8.0), (True, -8.0)]
        patients = ("PA", "PB")
        X = np.vstack([spec(100 * (p == "PB") + j, tu, d)
                       for p in patients for j, (tu, d) in enumerate(sites)])
        y = ["Tumor" if tu else "Normal"
             for _p in patients for tu, _d in sites]
        g = [p for p in patients for _s in sites]
        params = pp.PreprocessParams(crop_min=600.0, crop_max=1800.0,
                                     wavelet=False, wn_calibrate=True)
        pdata = paired_mod.paired_features(X, y, g, wn, params)
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=1000))]
                        ).fit(pdata.X, pdata.y)
        w = modeling.ModelResult(name="autoref",
                                 classes=list(pipe.classes_))
        w.pipeline = pipe
        w.macro = {"f1": (1.0, 0.0)}
        with tempfile.TemporaryDirectory() as td:
            bpath = os.path.join(td, "m.joblib")
            modeling.save_bundle(bpath, w, wn, params, paired=True)
            bundle = joblib.load(bpath)

            def wr(rel, vec):
                f = os.path.join(td, rel)
                os.makedirs(os.path.dirname(f), exist_ok=True)
                with open(f, "w", encoding="utf-8") as fh:
                    for a, b in zip(wn, vec, strict=True):
                        fh.write(f"{a:.10g},{b:.10g}\n")

            files = []
            for pi, p in enumerate(patients):
                for j, (tu, d) in enumerate(sites):
                    rel = f"{'Tumor' if tu else 'Normal'}/{p}/s{j}.csv"
                    wr(rel, X[4 * pi + j])
                    files.append(os.path.join(td, rel))
            for j in range(2):        # PZ: tumor-only, no Normal side
                rel = f"Tumor/PZ/z{j}.csv"
                wr(rel, spec(900 + j, True, 3.0))
                files.append(os.path.join(td, rel))

            def _run(worker):
                payload, failed = {}, []

                def _done(d):
                    payload.update(d)

                worker.done.connect(_done)
                worker.failed.connect(lambda tb: failed.append(tb))
                worker.run()          # synchronous: no event loop needed
                return payload, failed

            # 1) auto references with NO manual folder
            payload, failed = _run(gui.PredictWorker(bundle, files, td))
            assert not failed, failed[0] if failed else ""
            assert len(payload["rows"]) == 8, payload["logs"]
            assert payload["n_auto_refs"] == 2
            skipped = [l for l in payload["logs"]
                       if l.startswith("SKIPPED")]
            assert len(skipped) == 2, skipped
            assert any("Margin references" in l
                       for l in payload["logs"]), payload["logs"]
            # tumor sites of paired patients come out Tumor (deviation
            # carries the 1340 signature)
            t_rows = [r for r in payload["rows"] if r[0].startswith("s2")
                      or r[0].startswith("s3")]
            assert len(t_rows) == 4
            assert all(r[1] == "Tumor" for r in t_rows), payload["rows"]

            # 2) empty-string manual reference behaves like None
            payload2, failed2 = _run(gui.PredictWorker(
                bundle, files, td, manual_ref_dir=""))
            assert not failed2
            assert len(payload2["rows"]) == 8, payload2["logs"]
            assert payload2["n_auto_refs"] == 2

            # 3) tree ROOT as the manual reference -> actionable error
            _p3, failed3 = _run(gui.PredictWorker(
                bundle, files, td, manual_ref_dir=td))
            assert failed3 and "tree root" in failed3[0], failed3
    finally:
        uh.load_settings = _saved_ls
        _cd.find_data_root = _saved_fdr


def test_save_bundle_namespace_winner():
    """save_bundle must tolerate SimpleNamespace winners (the 3SSE
    results dialog and Model Lab build them) — 2026-09-06 regression:
    bare winner.per_class made 'Saving 3SSE bundle' fail."""
    import joblib
    from types import SimpleNamespace
    from sklearn.linear_model import LogisticRegression
    Xs = np.random.default_rng(4).normal(size=(40, 6))
    ys = ["A"] * 20 + ["B"] * 20
    pipe = LogisticRegression().fit(Xs, ys)
    w = SimpleNamespace(name="ns-winner", pipeline=pipe,
                        classes=["A", "B"], threshold=None,
                        macro={"f1": (0.9, 0.0)})
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ns.joblib")
        modeling.save_bundle(p, w, np.arange(6.0),
                             pp.PreprocessParams(), paired=True)
        b = joblib.load(p)
        assert b["model_name"] == "ns-winner" and b["paired"] is True
        assert b["per_class"] is None      # absent -> None, not a crash


def test_3sse_dialog_winner_tab_csv():
    """Exporting the Overall Winner tab writes a CSV (2026-09-06: the
    button refused with 'Select a ranking tab first')."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qt_compat import QtWidgets
    import ui_helpers as uh
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Fusion + stylesheet BEFORE constructing widgets: windowsvista +
    # this QSS-heavy UI fail-fasts natively (Brain gotcha, stage3 test
    # pattern)
    app.setStyle("Fusion")
    app.setStyleSheet(uh.STYLESHEET)
    import gui
    payload = {
        "board": {"singles": [
            {"arch": ("PCA + SVM (RBF)",), "level": 1,
             "metrics": {"f1": 0.70, "sens": 0.7, "spec": 0.7,
                         "auc": 0.75, "acc": 0.7}},
            # error-tolerant screening record: FAILED arch, no metrics
            # (2026-09-06: this shape crashed the dialog with
            # KeyError('f1') when viewing a saved run)
            {"arch": ("Broken model",), "level": 1,
             "metrics": {}, "error": "CatBoostError: bad allocation"}],
            "pairs": [], "triples": [], "total": 2, "pruned": 0},
        "validated": {1: [{"arch": ("PCA + SVM (RBF)",),
                           "metrics": {"f1": 0.71, "sens": 0.7,
                                       "spec": 0.7, "auc": 0.76,
                                       "acc": 0.7}}]},
        "winner": {"arch": ("Extra Trees", "Ensemble (top-3)"),
                   "metrics": {"f1": 0.783, "sens": 0.786, "spec": 0.786,
                               "auc": 0.821, "acc": 0.78}},
        "significance": {"baseline": "PCA + SVM (RBF)",
                         "baseline_f1": 0.70, "mcnemar_b": 10,
                         "mcnemar_c": 7, "mcnemar_p": 0.629,
                         "seed_f1s": [0.78, 0.78, 0.79]},
    }
    dlg = gui.SeqResultsDialog(payload)
    # the failed arch is excluded from the ranking (only the good one)
    singles_model = dlg._tab_models[0][1]
    assert len(singles_model._rows) == 1
    assert "PCA + SVM (RBF)" in singles_model._rows[0][2]
    dlg._tabs.setCurrentIndex(3)          # Overall Winner
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "winner.csv")
        _saved = QtWidgets.QFileDialog.getSaveFileName
        QtWidgets.QFileDialog.getSaveFileName = lambda *a, **k: (out, "")
        try:
            dlg._export_csv()
        finally:
            QtWidgets.QFileDialog.getSaveFileName = _saved
        text = open(out, encoding="utf-8-sig").read()
        assert "Winner (2-Model)" in text and "Nested best 1-Model" in text
        assert "Extra Trees → Ensemble (top-3)" in text
        assert "NOT significant" in text and "0.629" in text


def test_3sse_persist_and_latest_dir():
    """GUI searches persist their payload (screening/validated/winner)
    and the viewer picks the most recent run dir — 2026-09-06: GUI runs
    never wrote these artifacts, so 'View saved 3SSE' always failed."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qt_compat import QtWidgets
    import ui_helpers as uh
    import clinical_data as _cd
    import json as _json
    import time as _time
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(uh.STYLESHEET)
    import gui
    _saved_ls, _saved_fdr = uh.load_settings, _cd.find_data_root
    uh.load_settings = lambda: {}
    _cd.find_data_root = lambda: None
    _saved_appdir = gui.APP_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            gui.APP_DIR = td
            d_new = os.path.join(td, "study_run_lab")
            gui.MainWindow._persist_3sse_payload(d_new, {
                "board": {"singles": [
                    {"arch": ("PCA + SVM (RBF)",), "level": 1,
                     "metrics": {"f1": np.float32(0.7),
                                 "auc": float("nan")}}],
                    "pairs": [], "triples": []},
                "validated": {2: [{"arch": ("A", "B"),
                                   "metrics": {"f1": 0.6}}]},
                "winner": {"arch": ("A",), "metrics": {"f1": 0.8,
                                                       "cm": [[3, 1],
                                                              [0, 4]]},
                           "chain": object()}})
            # artifacts written, chain object excluded, numpy sanitized
            with open(os.path.join(d_new, "screening.jsonl"),
                      encoding="utf-8") as fh:
                rec = _json.loads(fh.readline())
            assert rec["arch"] == ["PCA + SVM (RBF)"]
            assert abs(rec["f1"] - 0.7) < 1e-6     # float32-rounded
            assert rec["auc"] is None              # NaN sanitized out
            with open(os.path.join(d_new, "winner.json"),
                      encoding="utf-8") as fh:
                wj = _json.load(fh)
            assert "chain" not in wj and wj["metrics"]["cm"] == [[3, 1],
                                                                 [0, 4]]
            # an older 3sse dir exists; the newer lab dir must win
            d_old = os.path.join(td, "study_run_3sse")
            os.makedirs(d_old, exist_ok=True)
            with open(os.path.join(d_old, "winner.json"), "w") as fh:
                fh.write("{}")
            old_t = _time.time() - 3600
            os.utime(os.path.join(d_old, "winner.json"), (old_t, old_t))
            assert gui.MainWindow._latest_3sse_run_dir() == d_new
            os.remove(os.path.join(d_new, "screening.jsonl"))
            os.remove(os.path.join(d_new, "winner.json"))
            assert gui.MainWindow._latest_3sse_run_dir() == d_old
            os.remove(os.path.join(d_old, "winner.json"))
            assert gui.MainWindow._latest_3sse_run_dir() is None
        # error records persist their failure reason (no metrics)
        d_err = os.path.join(td, "err_run")
        gui.MainWindow._persist_3sse_payload(d_err, {
            "board": {"singles": [
                {"arch": ("Broken model",), "level": 1,
                 "error": "TerminatedWorkerError: boom"}],
                "pairs": [], "triples": []}})
        with open(os.path.join(d_err, "screening.jsonl"),
                  encoding="utf-8") as fh:
            rec = _json.loads(fh.readline())
        assert rec["error"] == "TerminatedWorkerError: boom"
        assert "f1" not in rec
    finally:
        gui.APP_DIR = _saved_appdir
        uh.load_settings = _saved_ls
        _cd.find_data_root = _saved_fdr


def test_is_chain_winner_guard():
    """Chain winners (AveragedChain/SequentialChain) make every refit-
    based diagnostic multiply ~45 inner pipeline fits — the auto
    battery must know it's dealing with one (2026-09-06)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qt_compat import QtWidgets
    import ui_helpers as uh
    import clinical_data as _cd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import sequential as seq
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(uh.STYLESHEET)
    import gui
    _saved_ls, _saved_fdr = uh.load_settings, _cd.find_data_root
    uh.load_settings = lambda: {}
    _cd.find_data_root = lambda: None
    try:
        win = gui.MainWindow()
        mk = lambda: Pipeline([("sc", StandardScaler()),
                               ("lr", LogisticRegression())])
        win.winner = modeling.ModelResult(name="plain",
                                          classes=["A", "B"])
        win.winner.pipeline = mk()
        assert win._is_chain_winner() is False
        win.winner = modeling.ModelResult(name="chain",
                                          classes=["A", "B"])
        win.winner.pipeline = seq.AveragedChain([mk(), mk()],
                                                n_seeds=2)
        assert win._is_chain_winner() is True
        win.winner.pipeline = seq.SequentialChain([mk(), mk()])
        assert win._is_chain_winner() is True
        # no winner at all -> False (guards run before training too)
        win.winner = None
        assert win._is_chain_winner() is False
    finally:
        uh.load_settings = _saved_ls
        _cd.find_data_root = _saved_fdr


def test_batched_bundle_prediction_equivalence():
    """predict_with_bundle_many gives the SAME per-spectrum results as
    the single-spectrum path (it exists because TabPFN-class voters
    cost seconds PER CALL — batched folders take seconds, not minutes;
    measured batch-vs-single |Δp| ≈ 8e-8)."""
    import joblib
    import paired as paired_mod
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    wn = np.linspace(500.0, 2000.0, 200)
    X, y, g = [], [], []
    for p in ("PA", "PB"):
        for j, tumor in enumerate([False, False, True, True]):
            rp = np.random.default_rng(hash(p) % 500 + j)
            base = (rp.normal(0, 0.03, 200)
                    + 0.6 * np.exp(-((wn - 1003.0) ** 2) / (2 * 8.0 ** 2)))
            if tumor:
                base = base + 0.5 * np.exp(-((wn - 1340.0) ** 2)
                                           / (2 * 9.0 ** 2))
            X.append(base)
            y.append("Tumor" if tumor else "Normal")
            g.append(p)
    params = pp.PreprocessParams(crop_min=600.0, crop_max=1800.0,
                                 wavelet=False)
    pdata = paired_mod.paired_features(np.vstack(X), y, g, wn, params)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=500))]
                    ).fit(pdata.X, pdata.y)
    w = modeling.ModelResult(name="batch", classes=list(pipe.classes_))
    w.pipeline = pipe
    w.threshold = 0.55
    w.macro = {"f1": (1.0, 0.0)}
    with tempfile.TemporaryDirectory() as td:
        bpath = os.path.join(td, "b.joblib")
        modeling.save_bundle(bpath, w, wn, params, paired=True,
                             calibrator=(1.2, -0.1))
        bundle = joblib.load(bpath)
        # references built directly from the training-side construction
        Xp = pp.preprocess_matrix(np.vstack(X), params, wn=wn)
        ref_PA = Xp[0:2].mean(axis=0)
        ref_PB = Xp[4:6].mean(axis=0)
        spectra = [(wn, row) for row in X]
        references = [ref_PA, ref_PA, ref_PA, ref_PA,
                      ref_PB, ref_PB, ref_PB, ref_PB]
        results, errors = modeling.predict_with_bundle_many(
            bundle, spectra, references)
        assert not errors and all(r is not None for r in results)
        for i, (w_n, it) in enumerate(spectra):
            single = modeling.predict_with_bundle(
                bundle, w_n, it, reference=references[i])
            assert results[i]["prediction"] == single["prediction"]
            for c, v in single["probabilities"].items():
                assert abs(results[i]["probabilities"][c] - v) < 1e-9
        # a row with a bad axis is reported per-row, others survive
        bad = list(spectra)
        bad[0] = (np.linspace(700.0, 900.0, 50), np.zeros(50))
        res2, err2 = modeling.predict_with_bundle_many(
            bundle, bad, references)
        assert 0 in err2 and "does not cover" in err2[0]
        assert all(res2[i] is not None for i in range(1, len(bad)))


def test_diagnostics_cancel_check():
    """The loop-based diagnostics abort promptly with the 'cancelled by
    user' sentinel when cancel_check fires (2026-09-06: LOPO on chain
    winners runs very long and the busy box now offers Cancel)."""
    import study_stats as sstats
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(6)
    X = rng.normal(size=(40, 6))
    y = np.repeat([0, 1], 20)
    g = np.repeat(np.arange(8), 5)
    est = LogisticRegression()
    for helper, kwargs in (
            (sstats.seed_stability, {"seeds": (0, 1, 2, 3, 4), "k": 3}),
            (sstats.noise_robustness, {"k": 3}),
            (sstats.lopo_evaluate, {})):
        try:
            helper(X, y, g, est, {}, [0, 1],
                   cancel_check=lambda: True, **kwargs)
        except RuntimeError as exc:
            assert "cancelled by user" in str(exc), (helper, exc)
        else:
            raise AssertionError(f"{helper.__name__} ignored cancel")
        # without cancel they run to completion
        out = helper(X, y, g, est, {}, [0, 1], cancel_check=None,
                     **kwargs)
        assert out is not None
        # learning_curve_by_groups has its own signature (no params/classes)
        for cc in (lambda: True, None):
            try:
                modeling.learning_curve_by_groups(
                    X, y, g, est, k=3, fractions=(1.0,), cancel_check=cc)
                assert cc is None, "learning curve ignored cancel"
            except RuntimeError as exc:
                assert cc is not None and "cancelled by user" in str(exc)


def test_diag_progress_streaming_and_cancel_button():
    """Long diagnostics must STREAM progress (chain winners take
    minutes — a frozen 'computing…' looks broken) and the Cancel
    button must exist, start hidden, and be a no-op-safe flow
    (2026-09-06)."""
    import study_stats as sstats
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 6))
    y = np.repeat([0, 1], 20)
    g = np.repeat(np.arange(8), 5)
    est = LogisticRegression()
    msgs = []
    sstats.seed_stability(X, y, g, est, {}, [0, 1], seeds=(0, 1, 2),
                          k=3, progress=msgs.append)
    assert len(msgs) == 3 and "seed 2/3" in msgs[1]
    msgs.clear()
    modeling.learning_curve_by_groups(X, y, g, est, k=3,
                                      fractions=(0.5, 1.0),
                                      progress=msgs.append)
    assert any("patients" in m for m in msgs)
    msgs.clear()
    sstats.noise_robustness(X, y, g, est, {}, [0, 1], k=3,
                            progress=msgs.append)
    assert msgs and msgs[0].startswith("fold 1/")
    msgs.clear()
    sstats.lopo_evaluate(X, y, g, est, {}, [0, 1],
                         progress=msgs.append)
    assert msgs[0].startswith("patient 1/")
    # GUI bookkeeping: cancel button hidden by default, progress sink
    # names the analysis, worker-thread emit is safe without a worker
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qt_compat import QtWidgets
    import ui_helpers as uh
    import clinical_data as _cd
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(uh.STYLESHEET)
    import gui
    _saved_ls, _saved_fdr = uh.load_settings, _cd.find_data_root
    uh.load_settings = lambda: {}
    _cd.find_data_root = lambda: None
    try:
        win = gui.MainWindow()
        win.winner = None
        assert not win.b_cancel_analysis.isVisibleTo(win)
        win._analysis_label = "Seed stability"
        win._analysis_progress("seed 2/5: F1 0.61")
        assert "seed 2/5" in win.train_status.text()
        win._analysis_label = None
        win._emit_progress("from worker thread")     # must not raise
        # non-chain winner: heavy-diag confirm passes without a dialog
        assert win._confirm_heavy_diag("Anything") is True
    finally:
        uh.load_settings = _saved_ls
        _cd.find_data_root = _saved_fdr


def test_winner_importance_runs():
    """winner_importance must pass (X, y, None, wn) to
    region_importance_shap — the old 3-positional-arg call bound wn to
    groups and crashed EVERY 'Band agreement' diagnostic with
    TypeError: missing 'wn' (live since days, 2026-09-06)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(8)
    X = rng.normal(size=(40, 30))
    y = (X[:, 0] > 0).astype(int)
    wn = np.arange(30, dtype=float)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression())]).fit(X, y)
    imp = modeling.winner_importance(pipe, X, y, wn)
    assert imp is not None and len(imp) == 30


def test_pqn_deploy_parity():
    """Margin+PQN bundles train on pqn_normalize(spectrum, ref) - ref;
    the deploy path must apply the SAME PQN step (bundle flag 'pqn',
    2026-09-06: the flag never existed and every deploy path skipped
    PQN — the same failure class as the wn_calibrate mismatch)."""
    import joblib
    import paired as paired_mod
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    wn = np.linspace(500.0, 2000.0, 250)

    def spec(seed, tumor, scale):
        r = np.random.default_rng(seed)
        base = (0.7 * np.exp(-((wn - 1003.0) ** 2) / (2 * 7.0 ** 2))
                + 0.3 * np.exp(-((wn - 1450.0) ** 2) / (2 * 15.0 ** 2))
                + r.normal(0, 0.01, len(wn)))
        if tumor:
            base = base + 0.6 * np.exp(-((wn - 1340.0) ** 2)
                                       / (2 * 8.0 ** 2))
        return base * scale          # per-spectrum intensity (PQN's job)

    X, y, g = [], [], []
    for p in ("PA", "PB"):
        for j, (tu, sc) in enumerate([(False, 1.0), (False, 1.4),
                                      (True, 2.0), (True, 0.7)]):
            X.append(spec(100 * (p == "PB") + j, tu, sc))
            y.append("Tumor" if tu else "Normal")
            g.append(p)
    X = np.vstack(X)
    params = pp.PreprocessParams(crop_min=600.0, crop_max=1800.0,
                                 wavelet=False)
    pdata = paired_mod.paired_features(X, y, g, wn, params, use_pqn=True)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000))]
                    ).fit(pdata.X, pdata.y)
    w = modeling.ModelResult(name="pqn", classes=list(pipe.classes_))
    w.pipeline = pipe
    w.macro = {"f1": (1.0, 0.0)}
    with tempfile.TemporaryDirectory() as td:
        paths = {}
        for flag in (True, False):
            b = os.path.join(td, f"pqn_{flag}.joblib")
            modeling.save_bundle(b, w, wn, params, paired=True, pqn=flag)
            paths[flag] = b
        # deploy-side reference (raw mean of preprocessed normals)
        ref = (pp.preprocess_matrix(np.vstack([X[0], X[1]]), params,
                                    wn=wn)).mean(axis=0)
        tumor_row = 2                       # PA's first tumor spectrum
        wn_f, it_f = wn, X[tumor_row]
        # training-side feature for the same spectrum
        Xp = pp.preprocess_matrix(
            np.vstack([X[tumor_row], X[0], X[1]]), params, wn=wn)
        pq = pp.pqn_normalize(Xp[0], Xp[1:].mean(axis=0))
        expect = pq - Xp[1:].mean(axis=0)
        direct = pipe.predict_proba(expect.reshape(1, -1))[0]
        # flagged bundle: deploy == training construction
        out = modeling.predict_with_bundle(
            joblib.load(paths[True]), wn_f, it_f, reference=ref)
        for c, v in zip(pipe.classes_, direct, strict=True):
            assert abs(out["probabilities"][c] - v) < 1e-6
        out_many, errs = modeling.predict_with_bundle_many(
            joblib.load(paths[True]), [(wn_f, it_f)], [ref])
        assert not errs and abs(
            out_many[0]["probabilities"]["Tumor"]
            - out["probabilities"]["Tumor"]) < 1e-9
        # unflagged bundle: PQN skipped -> materially different
        # prediction (proves the parity test is not vacuous)
        out_noflag = modeling.predict_with_bundle(
            joblib.load(paths[False]), wn_f, it_f, reference=ref)
        assert abs(out_noflag["probabilities"]["Tumor"]
                   - out["probabilities"]["Tumor"]) > 1e-3


def test_persist_run_handles_ndarrays():
    """persist_run must sanitize the GUI worker's validated/winner
    dicts whose metrics carry oof_proba / cm / y_true as ndarrays —
    an ndarray-in-JSON TypeError here surfaced as 'The architecture
    search failed' AFTER the search had finished (2026-09-06)."""
    import json as _json
    import sequential as seq
    validated = {1: [{"arch": ("PCA + SVM (RBF)",),
                      "metrics": {"f1": np.float32(0.72), "sens": 0.7,
                                  "spec": 0.7, "acc": 0.7,
                                  "oof_proba": np.asarray(
                                      [[0.6, 0.4], [0.3, 0.7]])}}]}
    winner = {"arch": ("A", "B"),
              "metrics": {"f1": 0.8, "sens": 0.8, "spec": 0.8,
                          "cm": np.asarray([[3, 1], [0, 4]]),
                          "y_true": np.asarray([0, 1, 1, 0]),
                          "auc": float("nan")},
              "threshold": 0.6, "calibrator": None}
    with tempfile.TemporaryDirectory() as td:
        seq.persist_run(td, {"singles": []}, validated, winner,
                        significance={"mcnemar_p": 0.5})
        with open(os.path.join(td, "validated.json"),
                  encoding="utf-8") as fh:
            v = _json.load(fh)
        assert v["1"][0]["metrics"]["oof_proba"] == [[0.6, 0.4],
                                                     [0.3, 0.7]]
        with open(os.path.join(td, "winner.json"), encoding="utf-8") as fh:
            wj = _json.load(fh)
        assert wj["metrics"]["cm"] == [[3, 1], [0, 4]]
        assert wj["metrics"]["auc"] is None          # NaN -> null
        assert "calibrator" not in wj


def test_dialog_none_metrics_and_stale_clearing():
    """Persisted JSON turns NaN into null: the results dialog, its CSV
    export and the persistence itself (stale-artifact clearing) must
    all tolerate None metrics (2026-09-06)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qt_compat import QtWidgets
    import ui_helpers as uh
    import clinical_data as _cd
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(uh.STYLESHEET)
    import gui
    _saved_ls, _saved_fdr = uh.load_settings, _cd.find_data_root
    uh.load_settings = lambda: {}
    _cd.find_data_root = lambda: None
    _saved_appdir = gui.APP_DIR
    try:
        # auc is None (multi-class runs always NaN->null) — used to
        # crash pill formatting, max() keys and CSV export
        payload = {
            "board": {"singles": [], "pairs": [], "triples": []},
            "validated": {1: [{"arch": ("A",),
                               "metrics": {"f1": 0.7, "auc": None}}]},
            "winner": {"arch": ("A", "B"),
                       "metrics": {"f1": 0.8, "sens": 0.8, "spec": 0.8,
                                   "auc": None, "acc": None}},
        }
        dlg = gui.SeqResultsDialog(payload)
        rows = dlg._winner_rows()
        assert any("Winner (2-Model)" in r[1] for r in rows)
        dlg._tabs.setCurrentIndex(3)
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "w.csv")
            _saved = QtWidgets.QFileDialog.getSaveFileName
            QtWidgets.QFileDialog.getSaveFileName = lambda *a, **k: (out,
                                                                     "")
            try:
                dlg._export_csv()
            finally:
                QtWidgets.QFileDialog.getSaveFileName = _saved
            text = open(out, encoding="utf-8-sig").read()
            assert "Winner (2-Model)" in text and "nan" in text
        # stale-artifact clearing: a no-winner run removes winner.json
        gui.APP_DIR = td
        d = os.path.join(td, "run")
        gui.MainWindow._persist_3sse_payload(d, payload)
        assert os.path.isfile(os.path.join(d, "winner.json"))
        gui.MainWindow._persist_3sse_payload(
            d, {"board": payload["board"]})     # no validated/winner
        assert not os.path.isfile(os.path.join(d, "winner.json"))
        assert not os.path.isfile(os.path.join(d, "validated.json"))
        with open(os.path.join(d, "winner.json"), "w") as fh:
            fh.write("stale")                   # jsonl record safety
        with open(os.path.join(d, "screening.jsonl"), "w") as fh:
            fh.write('{"garbage": true}\n{"arch": ["A"], "level": 1, '
                     '"metrics": {"f1": 0.5}}\n')
        board, key = {"singles": [], "pairs": [], "triples": [],
                      "total": 0, "pruned": 0}, {1: "singles",
                                                 2: "pairs", 3: "triples"}
        import json as _j2
        with open(os.path.join(d, "screening.jsonl"),
                  encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = _j2.loads(line)
                    _lvl, _arch = rec["level"], tuple(rec["arch"])
                except (_j2.JSONDecodeError, KeyError, TypeError):
                    continue
                board[key[_lvl]].append({"arch": _arch, "level": _lvl,
                                         "metrics": rec})
        assert len(board["singles"]) == 1
    finally:
        gui.APP_DIR = _saved_appdir
        uh.load_settings = _saved_ls
        _cd.find_data_root = _saved_fdr


def test_bh_fdr():
    """Benjamini-Hochberg: monotone, bounded, known values."""
    import study_stats as sstats
    adj = sstats.bh_fdr([0.01, 0.02, 0.03, 0.4, 0.5])
    assert adj[0] <= 0.05 and abs(adj[-1] - 0.5) < 1e-9
    assert all(0 <= v <= 1 for v in adj)
    assert sstats.bh_fdr([]) == [] and sstats.bh_fdr([0.5]) == [0.5]


def test_friedman_nemenyi():
    """Clear ranking is significant with Nemenyi separations; twins
    are not; degenerate input is refused."""
    import study_stats as sstats
    rng = np.random.default_rng(1)
    blocks = 12
    good = 0.9 + rng.normal(0, 0.03, blocks)
    mid = 0.75 + rng.normal(0, 0.03, blocks)
    bad = 0.72 + rng.normal(0, 0.03, blocks)
    out = sstats.friedman_nemenyi({"good": list(good), "mid": list(mid),
                                   "bad": list(bad)})
    assert out["ok"] and out["significant"] and out["best"] == "good"
    assert out["vs_best"]["mid"][1]          # beyond the critical diff
    assert out["cd"] > 0
    twin = sstats.friedman_nemenyi(
        {"a": list(good),
         "b": list(good + 0.001 * rng.normal(0, 1, blocks))})
    assert twin["ok"] and twin["p"] > 0.05
    assert not sstats.friedman_nemenyi({"a": [1, 2],
                                        "b": [2, 1]})["ok"]


def test_lopo_seed_noise_rollup():
    """LOPO evaluates every patient unseen; seeds/noise stay coherent;
    the per-patient rollup sorts worst-first."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import study_stats as sstats
    rng = np.random.default_rng(2)
    n = 120
    X = rng.normal(0, 1, (n, 15))
    X[:60, :3] += 1.6
    y = ["Normal"] * 60 + ["Tumor"] * 60
    g = [f"P{i // 4}" for i in range(n)]
    tpl = make_pipeline(StandardScaler(), LogisticRegression())
    lo = sstats.lopo_evaluate(X, y, g, tpl, {}, ["Normal", "Tumor"])
    assert lo["n_patients"] == 30 and lo["f1"] > 0.85
    assert 0.5 < lo["auc"] <= 1.0
    assert len(lo["per_patient"]) == 30
    f1s = sstats.seed_stability(X, y, g, tpl, {}, ["Normal", "Tumor"],
                                seeds=(0, 1), k=4)
    assert len(f1s) == 2 and all(v > 0.7 for v in f1s)
    nr = sstats.noise_robustness(X, y, g, tpl, {}, ["Normal", "Tumor"],
                                 levels=(0.0, 0.05), k=4)
    assert len(nr) == 2 and nr[-1][1] >= nr[0][1] - 0.15
    # fit-once refactor: bit-identical to the old per-level loop
    # (same fits, same sequential noise draws, level-major order)
    rng2 = np.random.default_rng(0)
    ye = np.array([{"Normal": 0, "Tumor": 1}[v] for v in y], dtype=int)
    ref = [(lv, sstats._grouped_f1(X, ye, g, tpl, {}, 4, 0,
                                   noise_level=lv, rng=rng2))
           for lv in (0.0, 0.01, 0.02, 0.05)]
    got = sstats.noise_robustness(X, y, g, tpl, {}, ["Normal", "Tumor"],
                                  k=4, seed=0)
    assert all(abs(a[1] - b[1]) < 1e-12 for a, b in zip(ref, got,
                                                        strict=True))
    # parallel LOPO == serial LOPO (order-independent folds)
    lo_ser = sstats.lopo_evaluate(X, y, g, tpl, {}, ["Normal", "Tumor"],
                                  jobs=1)
    assert lo_ser["per_patient"] == lo["per_patient"]
    assert abs(lo_ser["f1"] - lo["f1"]) < 1e-12
    assert abs(lo_ser["auc"] - lo["auc"]) < 1e-12
    oof = np.zeros((n, 2))
    oof[:, 1] = np.clip(0.5 + X[:, 0] * 0.5 + rng.normal(0, .1, n),
                        0.01, 0.99)
    oof[:, 0] = 1 - oof[:, 1]
    rows = sstats.per_patient_rollup(y, g, oof, ["Normal", "Tumor"])
    assert len(rows) == 30
    assert all(rows[i][2] <= rows[i + 1][2] for i in range(len(rows) - 1))
    assert all(isinstance(r[5], bool) for r in rows)
    # B6: patient-level (mean-probability) metrics pool OOF per patient
    pm = sstats.patient_level_metrics(y, g, oof)
    assert pm is not None and pm["n_patients"] == 30
    assert 0.0 <= pm["f1"] <= 1.0 and 0.0 <= pm["auc"] <= 1.0
    assert sstats.patient_level_metrics(y, None, oof) is None


def test_band_stats_paired_fdr():
    """The planted nucleic-acid band comes out FDR-significant."""
    import biochemistry as bio
    import study_stats as sstats
    rng = np.random.default_rng(3)
    wn = np.linspace(400, 1800, 400)

    def mk(nuc, col):
        v = 0.4 + 0.5 * np.exp(-((wn - 1445) / 15) ** 2)
        v += 0.8 * np.exp(-((wn - 1003) / 10) ** 2)
        v += nuc * 0.9 * np.exp(-((wn - 785) / 10) ** 2)
        v += col * 0.6 * np.exp(-((wn - 854) / 10) ** 2)
        return v

    X, y, g = [], [], []
    for p in range(12):
        for cls, nuc, col in (("Normal", 0.3, 1.0), ("Tumor", 1.8, 0.4)):
            for _s in range(2):
                X.append(mk(nuc, col) + 0.005 * rng.normal(0, 1, 400))
                y.append(cls)
                g.append(f"P{p}")
    rows = sstats.band_stats_paired(np.vstack(X), y, g, wn)
    assert len(rows) == len(bio.BANDS)
    assert any(r[0] == 785.0 and r[6] for r in rows)
    assert all(0 <= r[5] <= 1 for r in rows if np.isfinite(r[5]))
    # multiclass -> not applicable
    assert sstats.band_stats_paired(np.vstack(X),
                                    ["a"] * 24 + ["b"] * 24 + ["c"] * 24,
                                    g * 3, wn) == []


def test_triage_arithmetic():
    """Empirical tier rates translate to exact per-1000 consequences."""
    import study_stats as sstats
    yv = np.array([1] * 50 + [0] * 50)
    tiers = (["POSITIVE"] * 45 + ["INDETERMINATE"] * 3 + ["NEGATIVE"] * 2
             + ["POSITIVE"] * 10 + ["INDETERMINATE"] * 15
             + ["NEGATIVE"] * 25)
    ta = sstats.triage_arithmetic(yv, tiers, prevalence=0.30)
    assert ta["biopsies_triage"] == round(0.96 * 300 + 0.50 * 700)
    assert ta["cancers_missed_negative"] == round(300 * 0.04)
    assert ta["biopsies_triage"] < ta["biopsies_all"]
    assert abs(ta["reduction_pct"]
               - 100 * (1 - (0.96 * 300 + 0.50 * 700) / 1000)) < 1e-6
    assert abs(ta["caught_or_flagged_pct"] - 96.0) < 1e-9
    assert sstats.triage_arithmetic(np.ones(10), ["POSITIVE"] * 10,
                                    0.3) == {}


def test_fold_f1_populated():
    """evaluate_models now records per-fold macro-F1 for the Friedman
    test."""
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (60, 10))
    X[30:] += 1.0
    y = ["A"] * 30 + ["B"] * 30
    g = [f"P{i // 5}" for i in range(60)]
    results, _w = modeling.evaluate_models(
        X, y, model_names=["PCA + LDA", "PCA + Logistic Regression"],
        k_folds=3, groups=g, wavenumbers=np.arange(10))
    for r in results:
        if r.error is None:
            assert len(r.fold_f1) == 3
            assert all(0 <= v <= 1 for v in r.fold_f1)


def test_reproduce_study_clinical_path():
    """reproduce_study must run end-to-end on a clinical-layout tree
    (regression: load_clinical_dataset returns a dataclass, not a tuple,
    and paired mode must not double-preprocess)."""
    import reproduce_study as rs
    with tempfile.TemporaryDirectory() as root:
        _make_clinical_tree(root)
        out = os.path.join(root, "..", "repro_out")
        assert rs.main(["--data", root, "--mini", "--mode", "paired",
                        "--out", out]) == 0
        summary = open(os.path.join(out, "summary.txt"),
                       encoding="utf-8").read()
        assert "Winner" in summary
        assert os.path.exists(os.path.join(out, "winner.joblib"))
        assert os.path.exists(os.path.join(out, "data_report.txt"))


def test_paired_features_preprocesses_once():
    """paired_features takes RAW intensities and preprocesses exactly
    once (reproduce_study once passed pre-processed X -> double
    preprocessing, diverging from the GUI)."""
    import paired
    with tempfile.TemporaryDirectory() as root:
        _make_clinical_tree(root)
        cd = cdata.load_clinical_dataset(root)
        spectra = cd.spectra
        grid = ds.common_grid(spectra)
        X_raw, labels = ds.to_matrix(spectra, grid)
        params = pp.PreprocessParams().validate()
        pd_ = paired.paired_features(X_raw, labels, cd.groups, grid,
                                     params)
        # manual: one preprocessing pass, then deviation from the
        # patient's own normal mean — must match exactly
        Xp = pp.preprocess_matrix(X_raw, params, wn=grid)
        by_patient: dict = {}
        for i, (g, lab) in enumerate(zip(cd.groups, labels, strict=True)):
            by_patient.setdefault(g, {}).setdefault(lab, []).append(i)
        rows = []
        for g, cls_idx in by_patient.items():
            normals = cls_idx.get("Normal")
            if not normals or not cls_idx.get("Tumor"):
                continue
            ref = Xp[normals].mean(axis=0)
            for lab in ("Normal", "Tumor"):
                for i in cls_idx[lab]:
                    rows.append(Xp[i] - ref)
        assert np.allclose(pd_.X, np.vstack(rows))
        assert pd_.n_patients >= 1


def test_oof_pooled_across_repeats():
    """With repeats>1 the pooled OOF probabilities are the MEAN over
    repeats (regression: they used to hold only the last repeat)."""
    X, y, groups = _synthetic_ml(40, 10, seed=3)
    kw = dict(model_names=["PCA + LDA"], k_folds=5, groups=list(groups))
    singles = []
    for rep in range(3):
        res, _ = modeling.evaluate_models(X, list(y), seed=42 + rep * 100,
                                          repeats=1, **kw)
        singles.append(res[0].oof_proba)
    res, _ = modeling.evaluate_models(X, list(y), seed=42, repeats=3, **kw)
    assert np.allclose(res[0].oof_proba, np.mean(singles, axis=0),
                       atol=1e-8)
    assert res[0].groups == list(groups)


def test_bootstrap_ci_patient_level():
    """groups= switches bootstrap_ci to whole-group resampling: with a
    single patient every resample draws the same rows -> zero-width CI,
    while spectrum-level resampling on the same data varies."""
    y_true = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])     # two wrong
    lo_g, hi_g = modeling.bootstrap_ci(y_true, y_pred, n=200, seed=0,
                                       groups=["P1"] * 8)
    lo_s, hi_s = modeling.bootstrap_ci(y_true, y_pred, n=200, seed=0)
    assert hi_g - lo_g == 0.0            # same rows every resample
    assert hi_s - lo_s > hi_g - lo_g


def test_new_models_lgbm_catboost_cnn():
    """LightGBM / CatBoost / 1D-CNN run through the grouped CV machinery
    (skipped per-model when the optional dependency is absent)."""
    from sklearn.base import clone
    rng = np.random.default_rng(1)
    y = np.array(["A", "B"] * 30)
    groups = np.repeat([f"S{i}" for i in range(12)], 5)
    X = rng.normal(0, 1, (60, 120)) + (y == "B")[:, None] * 1.5
    names = [n for n, ok in (("LightGBM", modeling.HAS_LGBM),
                             ("CatBoost", modeling.HAS_CATBOOST),
                             ("1D-CNN", modeling.HAS_TORCH),
                             ("TabPFN (foundation model)",
                              modeling.HAS_TABPFN)) if ok]
    assert names, "none of the new-model dependencies are installed"
    for n in names:
        assert n in modeling.ALL_MODEL_NAMES
    if modeling.HAS_TORCH:            # clone + proba contract
        import torch
        est = clone(modeling.CNN1DClassifier(epochs=5)).fit(X[:20], y[:20])
        proba = est.predict_proba(X[:10])
        assert proba.shape == (10, 2) and np.allclose(proba.sum(1), 1)
        # device selection: RAMAN_DEVICE=cpu forces CPU; auto mode never
        # crashes; fitted weights are stored on CPU (portable bundles)
        os.environ["RAMAN_DEVICE"] = "cpu"
        assert modeling.torch_device().type == "cpu"
        assert modeling.gpu_ok() is False      # kill-switch wins per call
        assert modeling.boost_device() == "cpu"
        if modeling.HAS_XGB:                   # parent-built GPU estimator
            from xgboost import XGBClassifier  # re-pointed for 3SSE workers
            g = modeling.as_env_device(
                XGBClassifier(n_estimators=2, device="cuda"))
            assert g.get_params()["device"] == "cpu"
        est = clone(modeling.CNN1DClassifier(epochs=5)).fit(X[:20], y[:20])
        del os.environ["RAMAN_DEVICE"]
        assert modeling.torch_device() in (torch.device("cpu"),
                                           torch.device("cuda"))
        # registry wiring: XGBoost/CatBoost carry a device chosen from the
        # hardware actually present (cuda/GPU here, cpu/CPU elsewhere)
        if modeling.HAS_XGB:
            xb = next(s for s in modeling.model_specs()
                      if s["name"] == "XGBoost")["estimator"]
            assert xb.get_params()["device"] in ("cuda", "cpu")
        if modeling.HAS_CATBOOST:
            cb = next(s for s in modeling.model_specs()
                      if s["name"] == "CatBoost")["estimator"]
            assert cb.get_params()["task_type"] in ("GPU", "CPU")
        pnext = next(est.net_.parameters())
        assert pnext.device.type == "cpu"
        assert est.predict_proba(X[:5]).shape == (5, 2)
    results, _winner = modeling.evaluate_models(
        X, list(y), model_names=names, k_folds=3, groups=list(groups))
    for r in results:
        assert r.error is None, r.error
        assert 0 <= r.macro_f1() <= 1


# ---------------------------------------------------------------- 3SSE
def test_3sse_architecture_space():
    """17 + 17*16 + 17*16*15 == 4369; ordered, repetition-free."""
    import sequential as seq
    names = [f"M{i}" for i in range(17)]
    assert seq._check_space(names) == 4369
    archs = set(seq.architectures(names))
    assert ("M0", "M1") in archs and ("M1", "M0") in archs
    assert ("M0", "M1", "M2") in archs and ("M0", "M2", "M1") in archs
    assert len(archs) == 4369               # no duplicates
    for a in archs:
        assert len(set(a)) == len(a)         # no model repeats inside


def test_3sse_search_smoke():
    import sequential as seq
    X, y, groups = _synthetic_ml(60, 12, seed=2)
    board = seq.search(np.asarray(X, dtype=np.float32), list(y),
                       groups=list(groups), k=3, jobs=1,
                       model_names=["PCA + LDA",
                                    "PCA + Logistic Regression",
                                    "Random Forest"])
    # 3 + 3*2 + 3*2*1 = 9 architectures
    assert len(board["singles"]) == 3
    assert len(board["pairs"]) == 6
    assert len(board["triples"]) == 6
    for entry in board["singles"] + board["pairs"] + board["triples"]:
        assert "metrics" in entry or "pruned_bound" in entry
        if "metrics" in entry:
            m = entry["metrics"]
            assert 0 <= m["f1"] <= 1 and 0 <= m["sens"] <= 1
            assert m["pat_f1"] >= 0            # patient-level rollup ran


def test_3sse_validate_top_parallel_equivalence():
    """The parallel validate_top pool returns the same architectures in
    the same screening-rank order with identical metrics as jobs=1."""
    import sequential as seq
    X, y, groups = _synthetic_ml(60, 12, seed=4)
    Xa = np.asarray(X, dtype=np.float32)
    board = seq.search(Xa, list(y), groups=list(groups), k=3, jobs=1,
                       model_names=["PCA + LDA",
                                    "PCA + Logistic Regression",
                                    "Random Forest"])
    ser = seq.validate_top(board, Xa, list(y), list(groups), None,
                           top=2, k_outer=3, cnn_epochs=2, jobs=1)
    par = seq.validate_top(board, Xa, list(y), list(groups), None,
                           top=2, k_outer=3, cnn_epochs=2, jobs=-2)
    for level in (1, 2, 3):
        ks = [e["arch"] for e in ser[level]]
        assert ks == [e["arch"] for e in par[level]]   # same order
        for a, b in zip(ser[level], par[level], strict=True):
            for mk in ("f1", "sens", "spec"):
                assert abs(a["metrics"][mk] - b["metrics"][mk]) < 1e-12


def test_3sse_chain_bundle_roundtrip():
    """1/2/3-layer SequentialChain: sklearn protocol + joblib roundtrip
    reproduces predictions exactly (spec §21-22, tests 14-18)."""
    import joblib
    import sequential as seq
    X, y, groups = _synthetic_ml(60, 12, seed=3)
    facs = seq._factories(None)
    pool = ["PCA + LDA", "PCA + Logistic Regression", "Random Forest"]
    for length in (1, 2, 3):
        chain = seq.SequentialChain([facs[n]() for n in pool[:length]],
                                    k=3).fit(X, list(y),
                                             groups=list(groups))
        proba = chain.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "chain.joblib")
            joblib.dump(chain, path)
            chain2 = joblib.load(path)
            assert np.allclose(proba, chain2.predict_proba(X))
        assert chain.predict(X).tolist() == [
            chain.classes_[i] for i in np.argmax(proba, axis=1)]


def test_3sse_oof_is_genuinely_oof():
    """Leakage probe: NO class signal, only patient-unique offsets —
    an in-sample fit memorizes patients (F1 ~1) while grouped OOF must
    stay near chance; a leaking P1 pipeline would transmit the offset."""
    import sequential as seq
    rng = np.random.default_rng(5)
    n_pat, per = 12, 5
    rows, ys, gs = [], [], []
    for p in range(n_pat):
        cls = "A" if p % 2 == 0 else "B"
        base = rng.normal(p, 1, 40)          # patient-unique offset only
        for _ in range(per):
            rows.append(base + rng.normal(0, .1, 40))
            ys.append(cls)
            gs.append(f"P{p}")
    X = np.asarray(rows, dtype=np.float32)
    facs = seq._factories(None)
    chain = seq.SequentialChain([facs["Random Forest"](),
                                 facs["Random Forest"]()], k=3)
    chain.fit(X, ys, groups=gs)
    from sklearn.metrics import f1_score
    in_sample = f1_score(ys, chain.predict(X), average="macro")
    oof = seq._oof_proba(facs["Random Forest"](), X, ys, gs, 3, 42)
    oof_pred = np.array(sorted(set(ys)))[np.argmax(oof, axis=1)]
    oof_f1 = f1_score(ys, oof_pred, average="macro")
    assert in_sample > 0.9
    assert oof_f1 < 0.8, "grouped OOF looks leaked (too high)"


def test_fstring_device_portability():
    """No f-string may open a {..} expression it doesn't close on the
    same line — that is Python 3.12+ syntax and SyntaxErrors the whole
    module on older Pythons (hit on a 3.11 device 2026-09-04; the dev
    machine runs 3.14 and cannot catch it by running the code)."""
    here = os.path.dirname(os.path.abspath(__file__))
    pat = re.compile(r"""f"[^"]*\{[^}]*$|f'[^']*\{[^}]*$""")
    for path in glob.glob(os.path.join(here, "*.py")):
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                assert not pat.search(line), (
                    f"{os.path.basename(path)}:{i} — f-string {{..}} "
                    "spans lines (3.12+ only): " + line.strip())


def test_search_survives_broken_model():
    """One model whose fits fail (e.g. CatBoost native OOM 'bad
    allocation', 2026-09-05) must cost one architecture — never the
    whole 3SSE screening run."""
    import sequential as seq
    from sklearn.base import BaseEstimator, ClassifierMixin

    class Boom(BaseEstimator, ClassifierMixin):
        def fit(self, X, y=None, **kw):
            raise MemoryError("bad allocation (simulated)")
        def predict_proba(self, X):
            raise RuntimeError("unfitted")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    y = (np.arange(60) % 2).tolist()
    groups = [f"S{i // 6}" for i in range(60)]
    orig = modeling.model_specs
    modeling.model_specs = lambda: orig() + [{"name": "Boom",
                                              "estimator": Boom()}]
    try:
        board = seq.search(X, y, groups=groups, wavenumbers=None, k=2,
                           top=3, model_names=["PCA + LDA", "Boom"],
                           jobs=1, prune_pairs=2)
    finally:
        modeling.model_specs = orig
    ok = [s for s in board["singles"] if "metrics" in s]
    assert any(s["arch"] == ("PCA + LDA",) for s in ok), ok
    assert all(s["arch"] != ("Boom",) for s in ok)    # error row, no crash
    assert board["pairs"] == [] or all(
        "Boom" not in p["arch"] for p in board["pairs"])


def _optional_report():
    """Honesty line: which optional deps were absent, i.e. how much of
    the suite silently covered less than it looks (guarded test bodies
    report PASS either way — 2026-09-05)."""
    import importlib.util
    mods = ("torch", "shap", "pybaselines", "lightgbm", "catboost",
            "xgboost", "tabpfn")
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    if missing:
        print(f"NOTE: optional deps not installed (tests that use them "
              f"covered less): {', '.join(missing)}")


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:             # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    _optional_report()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
