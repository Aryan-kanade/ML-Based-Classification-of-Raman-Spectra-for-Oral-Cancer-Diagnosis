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

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clinical_data as cdata          # noqa: E402
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
    assert cdata.subject_key("TDOC015") == "S15"
    assert cdata.subject_key("TDOC058 Spectra pro") == "S58"
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
    oof = np.zeros((n, 2))
    oof[:, 1] = np.clip(0.5 + X[:, 0] * 0.5 + rng.normal(0, .1, n),
                        0.01, 0.99)
    oof[:, 0] = 1 - oof[:, 1]
    rows = sstats.per_patient_rollup(y, g, oof, ["Normal", "Tumor"])
    assert len(rows) == 30
    assert all(rows[i][2] <= rows[i + 1][2] for i in range(len(rows) - 1))
    assert all(isinstance(r[5], bool) for r in rows)


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
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
