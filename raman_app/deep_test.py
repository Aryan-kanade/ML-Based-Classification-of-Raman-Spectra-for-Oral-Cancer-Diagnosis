"""
deep_test.py — adversarial stress test: every error path, worker, legacy
mode and round-trip exercised headless. Run:

    python deep_test.py

Uses an offscreen Qt platform; QMessageBox dialogs are monkeypatched to
RECORD instead of block, so warning paths finish instead of hanging.
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
os.environ.setdefault("MPLBACKEND", "QtAgg")

import numpy as np

import dataset
import modeling

RESULTS: list[tuple[bool, str]] = []


def check(name: str, fn):
    """Run one scenario; record pass/fail without stopping the sweep."""
    try:
        fn()
        RESULTS.append((True, name))
        print(f"  PASS  {name}")
    except Exception as exc:                     # noqa: BLE001
        import traceback
        tb = traceback.extract_tb(exc.__traceback__)
        where = (f" [{os.path.basename(tb[-1].filename)}:{tb[-1].lineno}]"
                 if tb else "")
        RESULTS.append((False, f"{name}: {exc}"))
        print(f"  FAIL  {name}: {exc}{where}")


DIALOG_LOG: list[tuple[str, str]] = []


def _install_dialog_recorder():
    """Record QMessageBox calls instead of showing native (offscreen-
    crashing) modal dialogs. Installed ONCE, never restored."""
    from qt_compat import QtWidgets
    YES = QtWidgets.QMessageBox.Yes
    for n in ("warning", "information", "critical", "about", "question"):
        def make(kind):
            def _box(parent, title, text, *a, **k):
                DIALOG_LOG.append((kind, title, text))
                return YES if kind == "question" else 1024
            return staticmethod(_box)
        setattr(QtWidgets.QMessageBox, n, make(n))


def make_gui():
    from qt_compat import QtWidgets
    import ui_helpers as uh
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    app.setStyle("Fusion")
    app.setStyleSheet(uh.STYLESHEET)
    import gui
    if not DIALOG_LOG and not getattr(make_gui, "_patched", False):
        _install_dialog_recorder()
        make_gui._patched = True
    win = gui.MainWindow()
    win._dialog_log = DIALOG_LOG
    return win


def wait_analysis(win, timeout_ms: int = 120000):
    """The analysis buttons run on a background worker; wait for it and
    flush the queued done-callbacks before asserting on results."""
    from qt_compat import QtWidgets
    if win._analysis_worker is not None:
        win._analysis_worker.wait(timeout_ms)
    for _ in range(10):
        QtWidgets.QApplication.processEvents()


def main() -> int:
    # ---------------------------------------------------------------- GUI
    def s_blank_gui():
        win = make_gui()
        for idx in range(6):                      # navigate all pages blind
            win.stack.setCurrentIndex(idx)
        assert win.stack.count() == 6
        win.run_learning_curve()                  # must warn, not crash
        assert any("Train first" in ti for _k, ti, _t in win._dialog_log)
        win.run_region_importance()
        win.save_result_figures()                 # nothing to save path
        win.freeze_study()                        # no winner: still writes
        win.close()

    check("blank GUI: 6 pages, guards warn not crash", s_blank_gui)

    def s_actions_without_data():
        win = make_gui()
        win.start_training()                      # no spectra
        win.run_prediction()                      # no bundle
        win.run_optimize()                        # no spectra
        win.save_model()                          # no winner
        win.run_honest_check()                    # no data
        assert len(win._dialog_log) >= 4
        win.close()

    check("all actions with NO data: warn, never crash", s_actions_without_data)

    def s_flat_unlabeled():
        win = make_gui()
        td = tempfile.mkdtemp()
        with open(os.path.join(td, "no_class_token.txt"), "w") as fh:
            for w in range(500, 1800, 4):
                fh.write(f"{w}\t{np.sin(w / 200):.4f}\n")
        win.load_folder(td, quiet=True)
        assert len(win.spectra) == 1 and win.labels[0] == ""
        win.close()

    check("flat folder with unlabeled file loads", s_flat_unlabeled)

    def s_one_class_train():
        win = make_gui()
        src = dataset.find_default_source_spectrum()
        demo = dataset.generate_demo_data(
            src, os.path.join(tempfile.mkdtemp(), "d"))
        win.load_folder(demo, quiet=True)
        for name, cb in win.model_checks.items():
            cb.setChecked(name == "PCA + LDA")
        # relabel everything to one class -> cannot-train path
        for i in range(len(win.labels)):
            win.labels[i] = "C1"
        win.start_training()
        assert any("Cannot train" in t or "2 distinct" in t
                   for _k, _ti, t in win._dialog_log)
        win.close()

    check("training with a single class: clean error path", s_one_class_train)

    def s_predict_edge_folders():
        win = make_gui()
        src = dataset.find_default_source_spectrum()
        demo = dataset.generate_demo_data(
            src, os.path.join(tempfile.mkdtemp(), "d2"))
        win.load_folder(demo, quiet=True)
        X = win.get_processed_X()
        res, w = modeling.evaluate_models(
            X, win.labels, model_names=["PCA + LDA"], k_folds=3,
            wavenumbers=win.grid)
        path = os.path.join(tempfile.mkdtemp(), "b.joblib")
        modeling.save_bundle(path, w, win.grid,
                             win.read_params().validate())
        win._set_bundle(modeling.load_bundle(path), path)
        # (a) empty folder
        empty = tempfile.mkdtemp()
        win.spec_path_edit.setText(empty)
        got = win._prediction_files()
        assert got is None
        # (b) folder with ONLY a reference file -> skipped, none left
        refonly = tempfile.mkdtemp()
        wn, it = dataset.load_spectrum(win.spectra[0].path)
        with open(os.path.join(refonly, "x_white.txt"), "w") as fh:
            for a, b in zip(wn, it, strict=True):
                fh.write(f"{a}\t{b}\n")
        win.spec_path_edit.setText(refonly)
        assert win._prediction_files() is None
        # (c) paired bundle without a reference folder set
        modeling.save_bundle(path, w, win.grid,
                             win.read_params().validate(), paired=True)
        win._set_bundle(modeling.load_bundle(path), path)
        win.spec_path_edit.setText(demo)
        win.run_prediction()
        assert any("Reference required" in ti for _k, ti, _t in win._dialog_log)
        win.close()

    check("predict: empty folder / refs-only / paired-no-ref paths",
          s_predict_edge_folders)

    def s_positive_row_renders():
        # regression for the QtGui crash: a POSITIVE prediction row must
        # render through the red-highlight branch on the Result page
        win = make_gui()
        src = dataset.find_default_source_spectrum()
        demo = dataset.generate_demo_data(
            src, os.path.join(tempfile.mkdtemp(), "d3"))
        win.load_folder(demo, quiet=True)
        X = win.get_processed_X()
        res, w = modeling.evaluate_models(
            X, win.labels, model_names=["PCA + LDA"], k_folds=3,
            wavenumbers=win.grid)
        path = os.path.join(tempfile.mkdtemp(), "b2.joblib")
        modeling.save_bundle(path, w, win.grid,
                             win.read_params().validate())
        win._set_bundle(modeling.load_bundle(path), path)
        win.spec_path_edit.setText(demo)
        win.run_prediction()                      # demo -> C1/C5/C8 preds
        if win._pred_worker is not None:          # prediction is async
            from qt_compat import QtWidgets
            win._pred_worker.wait(120000)
            for _ in range(10):
                QtWidgets.QApplication.processEvents()
        win.stack.setCurrentIndex(win.stack.count() - 1)   # Result page
        for _ in range(6):
            win._result_app_process() if hasattr(win, "_result_app_process") \
                else None
        # force a positive-class hit: relabel one row to the demo 'pos'
        pos = win._positive_class() or sorted(
            {c for _f, c, _p in win._pred_rows})[0]
        win._pred_rows = [(win._pred_rows[0][0], pos, 0.9)] \
            + win._pred_rows[1:]
        win.render_result_page()                  # must not raise
        assert win.r_pred_table.rowCount() > 0
        assert win.r_pred_table.cellWidget(0, 3) is not None  # prob bar
        win.close()

    check("Result page renders a POSITIVE prediction row (QtGui regression)",
          s_positive_row_renders)

    def s_result_binary_paths():
        # binary verdict path end-to-end: likelihood meter with clinical
        # cut-offs, P(positive) histogram, patient table, spectra overlay
        # with paired reference + band shading, probability bars, triage
        # tiers — plus the empty/degenerate cases
        from qt_compat import QtWidgets
        win = make_gui()
        wn = np.linspace(1800.0, 400.0, 60)
        spec = np.abs(np.sin(np.linspace(0, 5, 60)))
        # real out-of-fold probabilities -> operating points + triage
        rng = np.random.default_rng(3)
        yt = rng.integers(0, 2, 80)
        pof = np.clip(1 / (1 + np.exp(-(2.0 * yt
                                       + rng.normal(0, 1.5, 80)))), 0.01,
                      0.99)
        oof = np.full((80, 2), np.nan)
        oof[:, 0] = 1 - pof
        oof[:, 1] = pof
        cm2 = np.zeros((2, 2), int)
        for t, q in zip(yt, (pof >= 0.5).astype(int), strict=True):
            cm2[t, q] += 1

        class _W:                                  # minimal winner stand-in
            name = "shim"
            classes = ["Normal", "Tumor"]
            threshold = 0.5
            macro = {"sens": (0.7, 0.05), "spec": (0.6, 0.05)}
            oof_proba = oof
            y_true_encoded = yt
            cm = cm2
            per_class: dict = {}

            def macro_f1(self):
                return 0.65

        win.winner = _W()
        vals = (0.95, 0.30, 0.65, 0.10)
        win._pred_rows = [(f"P{i}/a.csv", "Tumor" if v >= 0.5 else "Normal",
                           v) for i, v in enumerate(vals)]
        win._pred_probs = [{"Tumor": v, "Normal": 1 - v} for v in vals]
        win._pred_spectra = [spec, spec * 0.9]
        win._pred_wn = wn
        win._pred_reference = spec * 0.95
        win._region_bands = [(1000.0, 0.5, "phenylalanine", 1),
                             (1450.0, 0.2, "", 0)]
        win._patient_rows = [("P0", 2, "Tumor:1 / Normal:1", 0.625,
                              "Tumor"),
                             ("P1", 2, "Normal:2", 0.2, "Normal")]
        win.bundle = {"classes": ["Normal", "Tumor"], "threshold": 0.5,
                      "paired": True, "macro": {}}
        win.stack.setCurrentIndex(win.stack.count() - 1)   # Result page
        win.render_result_page()
        QtWidgets.QApplication.processEvents()
        # meter got the mean P(tumor) = (0.95+0.30+0.65+0.10)/4
        assert win.result_meter._value is not None
        assert abs(win.result_meter._value - 0.5) < 1e-9
        assert win.result_meter._rule_out is not None
        assert win.result_meter._rule_in is not None
        assert win.result_meter._rule_out < win.result_meter._rule_in
        assert win.result_meter.isVisibleTo(win.stack)
        # triage tiers must come from the ROC cut-offs, NOT the CI bounds
        # (regression: a shadowed lo/hi once fed the bootstrap CI here)
        lo, hi = win._op_points_now()
        import clinical as clin
        expect = [clin.triage(v, lo, hi) for v in vals]
        got = [win.r_pred_table.item(r, 2).text() for r in range(4)]
        assert got == expect, (got, expect, lo, hi)
        assert win.r_pat_table.rowCount() == 2
        assert win.r_pat_table.isVisibleTo(win.stack)
        assert win.r_spec_card.isVisibleTo(win.stack)
        bar = win.r_pred_table.cellWidget(0, 3)
        assert isinstance(bar, QtWidgets.QProgressBar)
        assert bar.value() == 950
        assert win.r_cal_canvas is not None        # calibration chart exists
        # empty patients -> table hidden, hint kept (flat-folder case)
        win._patient_rows = []
        win.render_result_page()
        assert not win.r_pat_table.isVisibleTo(win.stack)
        assert win.r_pat_hint.isVisibleTo(win.stack)
        # no full probs -> meter hidden again, spectra overlay stays
        win._pred_probs = []
        win.render_result_page()
        assert not win.result_meter.isVisibleTo(win.stack)
        assert win.r_spec_card.isVisibleTo(win.stack)
        win.close()

    check("Result page binary paths: meter, triage, histogram, overlay",
          s_result_binary_paths)

    def s_workers_programmatic():
        # sender() is None when workers are started programmatically —
        # the run must still work and the callbacks must not crash
        win = make_gui()
        src = dataset.find_default_source_spectrum()
        demo = dataset.generate_demo_data(
            src, os.path.join(tempfile.mkdtemp(), "d4"))
        win.load_folder(demo, quiet=True)
        win.run_optimize()                        # sender() is None here
        win.run_honest_check()                    # sender() is None here
        win._opt_worker.wait(120000)
        win._honest_worker.wait(180000)
        assert win._opt_worker.isFinished()
        assert win._honest_worker.isFinished()
        win.close()

    check("Optimize + Honest workers run with sender()=None", s_workers_programmatic)

    def s_patient_aggregation_misaligned():
        # a file that FAILS to predict must not shift patient grouping
        win = make_gui()
        td = tempfile.mkdtemp()
        ok1 = os.path.join(td, "patA"); ok2 = os.path.join(td, "patB")
        os.makedirs(ok1); os.makedirs(ok2)
        src = dataset.find_default_source_spectrum()
        wn, it = dataset.load_spectrum(win.spectra[0].path) if win.spectra \
            else dataset.load_spectrum(src)
        for d, tag in ((ok1, "A"), (ok2, "B")):
            with open(os.path.join(d, f"good_{tag}.txt"), "w") as fh:
                for a, b in zip(wn, it, strict=True):
                    fh.write(f"{a}\t{b}\n")
        # one unreadable file in patA that will fail to load
        with open(os.path.join(ok1, "broken.txt"), "w") as fh:
            fh.write("not,a,spectrum\n")
        win.spec_path_edit.setText(td)
        files = win._prediction_files()
        assert files is not None and len(files) == 3
        # simulate: one failure, two successes -> aggregate must be aligned
        rows = [("good_A.txt", "C1", 0.9), ("good_B.txt", "C1", 0.8)]
        pats = win._aggregate_patients(files[:1] + files[2:], rows)
        assert len(pats) == 2
        assert pats[0][0] == "patA" and pats[1][0] == "patB"
        win.close()

    check("patient grouping survives a failed prediction (zip alignment)",
          s_patient_aggregation_misaligned)

    # ---------------------------------------------------------------- CLI
    def s_cli_flat_mode():
        import subprocess
        env = dict(os.environ)
        src = dataset.find_default_source_spectrum()
        demo = dataset.generate_demo_data(
            src, os.path.join(tempfile.mkdtemp(), "d5"))
        out = os.path.join(tempfile.mkdtemp(), "o")
        r = subprocess.run(
            [sys.executable, "vit_train.py", "--data", demo,
             "--epochs", "2", "--out", out, "--preset", "default"],
            capture_output=True, text=True, timeout=600, env=env)
        assert r.returncode == 0, r.stderr[-400:]
        r2 = subprocess.run(
            [sys.executable, "vit_test.py", "--data", demo,
             "--model", os.path.join(out, "vit_model.pt")],
            capture_output=True, text=True, timeout=600, env=env)
        assert r2.returncode == 0, r2.stderr[-400:]

    check("CLI legacy FLAT mode: vit_train + vit_test round-trip",
          s_cli_flat_mode)

    def s_cli_bad_inputs():
        import subprocess
        r = subprocess.run(
            [sys.executable, "vit_train.py", "--data",
             os.path.join(tempfile.mkdtemp(), "nope")],
            capture_output=True, text=True, timeout=120)
        assert r.returncode != 0 and "not found" in (r.stdout + r.stderr)
        r2 = subprocess.run(
            [sys.executable, "vit_test.py", "--model",
             os.path.join(tempfile.mkdtemp(), "missing.pt")],
            capture_output=True, text=True, timeout=120)
        assert r2.returncode != 0

    check("CLI bad inputs fail cleanly", s_cli_bad_inputs)

    def s_old_bundle_compat():
        # a bundle saved BEFORE crop existed (params without crop keys)
        # must still predict via the uncropped fallback
        src = dataset.find_default_source_spectrum()
        demo = dataset.generate_demo_data(
            src, os.path.join(tempfile.mkdtemp(), "d6"))
        spectra = dataset.load_folder(demo)
        grid = dataset.common_grid(spectra)
        X, y = dataset.to_matrix(spectra, grid)
        from sklearn.discriminant_analysis import \
            LinearDiscriminantAnalysis
        est = LinearDiscriminantAnalysis().fit(X, y)
        class _W:                                   # minimal winner stub
            name = "LDA-legacy"
            pipeline = est
            classes = sorted(set(y))
            threshold = None
            macro = {"sens": (0.9,), "spec": (0.9,)}
        import preprocessing as pp
        from dataclasses import asdict, replace
        legacy_params = asdict(replace(pp.PreprocessParams(),
                                       crop_min=0.0, crop_max=0.0))
        path = os.path.join(tempfile.mkdtemp(), "legacy.joblib")
        modeling.save_bundle(path, _W, grid, legacy_params)
        b = modeling.load_bundle(path)
        wn, it = dataset.load_spectrum(spectra[0].path)
        out = modeling.predict_with_bundle(b, wn, it)
        assert out["prediction"] in b["classes"]

    check("legacy (uncropped) bundle still predicts", s_old_bundle_compat)

    def _clinical_tree(tmp: str, n_pat: int = 8):
        """Synthetic <root>/<class>/<patient> tree with a real signal."""
        import numpy as np
        rng = np.random.default_rng(4)
        wn_t = np.linspace(400.0, 1800.0, 300)
        root = os.path.join(tmp, "clin")
        for p in range(n_pat):
            for cls, lift in (("Normal", 0.0), ("Tumor", 1.0)):
                d = os.path.join(root, cls, f"P{p:02d}")
                os.makedirs(d, exist_ok=True)
                for s in range(2):
                    v = (0.5 + 0.4 * np.exp(-((wn_t - 1450) / 20) ** 2)
                         + lift * 0.5 * np.exp(-((wn_t - 785) / 15) ** 2)
                         + 0.02 * rng.normal(0, 1, len(wn_t)))
                    with open(os.path.join(d, f"s{s}.txt"), "w") as fh:
                        for a, b in zip(wn_t, v, strict=True):
                            fh.write(f"{a}\t{b}\n")
        return root

    def s_clinical_tree_and_auto_refs():
        # <root>/<class>/<patient> prediction + auto per-patient margin
        # references + patient-grouped verdicts keyed by PATIENT
        from qt_compat import QtWidgets
        import preprocessing as pp
        win = make_gui()
        tmp = tempfile.mkdtemp()
        root = _clinical_tree(tmp)
        win.load_folder(root, quiet=True)
        # standard bundle first: two-level walk + patient grouping
        X = win.get_processed_X()
        _res, w = modeling.evaluate_models(
            X, win.labels, model_names=["PCA + LDA"], k_folds=3,
            wavenumbers=win.grid)
        path = os.path.join(tmp, "std.joblib")
        modeling.save_bundle(path, w, win.grid,
                             win.read_params().validate())
        win._set_bundle(modeling.load_bundle(path), path)
        win.spec_path_edit.setText(root)
        win.run_prediction()
        if win._pred_worker is not None:
            win._pred_worker.wait(120000)
            for _ in range(10):
                QtWidgets.QApplication.processEvents()
        assert len(win._pred_rows) == 32          # 8 patients x 2 classes x 2
        assert len(win._patient_rows) == 8        # grouped by PATIENT
        assert all(r[0] == f"P{i:02d}" for i, r in
                   enumerate(win._patient_rows))
        # margin mode with AUTO references (no manual folder)
        import paired as pmod
        params = win.read_params().validate()
        m_crop = pp.crop_mask(win.grid, params)
        wn_c = np.asarray(win.grid)[m_crop]
        keep = [i for i, lab in enumerate(win.labels) if lab.strip()]
        pd_ = pmod.paired_features(
            win.get_processed_X()[keep],
            [win.labels[i] for i in keep],
            [win.groups[i] for i in keep], wn_c, params)
        _res2, w2 = modeling.evaluate_models(
            pd_.X, pd_.y, model_names=["PCA + LDA"], k_folds=3,
            groups=pd_.groups, wavenumbers=pd_.wn)
        path2 = os.path.join(tmp, "paired.joblib")
        modeling.save_bundle(path2, w2, win.grid, params, paired=True)
        win._set_bundle(modeling.load_bundle(path2), path2)
        win.ref_path_edit.setText("")             # forces auto mode
        win.spec_path_edit.setText(root)
        win.run_prediction()
        if win._pred_worker is not None:
            win._pred_worker.wait(120000)
            for _ in range(10):
                QtWidgets.QApplication.processEvents()
        assert len(win._pred_rows) == 32          # all files predicted
        win.close()

    check("clinical tree prediction + auto per-patient margin refs",
          s_clinical_tree_and_auto_refs)

    def s_locked_eval():
        # the one-shot held-out final exam runs and surfaces FINAL
        win = make_gui()
        tmp = tempfile.mkdtemp()
        root = _clinical_tree(tmp)
        win.load_folder(root, quiet=True)
        X = win.get_processed_X()
        _res, w = modeling.evaluate_models(
            X, win.labels, model_names=["PCA + LDA"], k_folds=3,
            groups=win.groups, wavenumbers=win.grid)
        win.winner = w
        params = win.read_params().validate()
        import preprocessing as pp
        win._lc_data = (X, win.labels, win.groups)
        win._lc_wn = np.asarray(win.grid)[pp.crop_mask(win.grid, params)]
        win.run_locked_eval()                     # question -> Yes
        wait_analysis(win)
        out = win._locked_result
        assert out is not None and out["n_test_patients"] >= 1
        assert 0.0 <= out["f1"] <= 1.0 and "auc" in out
        win.render_locked_result()
        assert "FINAL" in win.r_locked_label.text()
        win.close()

    check("locked test-set evaluation (one-shot FINAL)", s_locked_eval)

    def s_html_report():
        # self-contained HTML report with embedded figures
        win = make_gui()
        tmp = tempfile.mkdtemp()
        root = _clinical_tree(tmp, n_pat=7)
        win.load_folder(root, quiet=True)
        X = win.get_processed_X()
        _res, w = modeling.evaluate_models(
            X, win.labels, model_names=["PCA + LDA"], k_folds=3,
            groups=win.groups, wavenumbers=win.grid)
        win.winner = w
        path = os.path.join(tmp, "m.joblib")
        modeling.save_bundle(path, w, win.grid,
                             win.read_params().validate())
        win._set_bundle(modeling.load_bundle(path), path)
        win.spec_path_edit.setText(root)
        win.run_prediction()
        if win._pred_worker is not None:
            win._pred_worker.wait(120000)
            from qt_compat import QtWidgets
            for _ in range(10):
                QtWidgets.QApplication.processEvents()
        win.save_result_report_html()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "result_report.html")
        assert os.path.isfile(out)
        html = open(out, encoding="utf-8").read()
        assert "<html" in html and "data:image/png;base64" in html
        assert "Locked" in html or "Predictive" in html
        win.close()

    check("HTML report writes with embedded figures", s_html_report)

    def s_deep_diagnostics():
        # LOPO + seed stability + noise robustness + local explanation +
        # file logging, end-to-end on a clinical-style tree
        from qt_compat import QtWidgets
        win = make_gui()
        tmp = tempfile.mkdtemp()
        root = _clinical_tree(tmp, n_pat=8)
        win.load_folder(root, quiet=True)
        X = win.get_processed_X()
        _res, w = modeling.evaluate_models(
            X, win.labels, model_names=["PCA + LDA", "PCA + Logistic "
                                        "Regression"],
            k_folds=3, groups=win.groups, wavenumbers=win.grid)
        win.on_train_done(_res, w)                # Friedman hook fires
        import preprocessing as pp
        params = win.read_params().validate()
        win._lc_data = (X, win.labels, win.groups)
        win._lc_wn = np.asarray(win.grid)[pp.crop_mask(win.grid, params)]
        # deep diagnostics (each runs on the shared analysis worker)
        win.run_lopo()
        wait_analysis(win)
        assert win._lopo_result is not None
        assert win._lopo_result["n_patients"] == 8
        win.run_seed_stability()
        wait_analysis(win)
        assert win._seed_result and len(win._seed_result) == 5
        win.run_noise_check()
        wait_analysis(win)
        assert win._noise_result and len(win._noise_result) == 4
        # predict something, then explain it locally
        path = os.path.join(tmp, "m.joblib")
        modeling.save_bundle(path, w, win.grid, params)
        win._set_bundle(modeling.load_bundle(path), path)
        win.spec_path_edit.setText(root)
        win.run_prediction()
        if win._pred_worker is not None:
            win._pred_worker.wait(120000)
            for _ in range(10):
                QtWidgets.QApplication.processEvents()
        assert len(win._pred_rows) == 32
        try:
            import shap  # noqa: F401
            win.run_local_explain()
            wait_analysis(win)
            assert win._local_bands, "local explanation produced rows"
        except ImportError:
            pass                        # optional dependency
        # Result page renders everything new
        win.render_result_page()
        assert win.r_deep_label.text() != ""
        assert win.r_pp_table.rowCount() >= 0
        # file log received lines
        log_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "session.log")
        assert os.path.isfile(log_path)
        win.close()

    check("deep diagnostics: LOPO, seeds, noise, local SHAP, file log",
          s_deep_diagnostics)

    def s_reproduce_cli():
        # headless one-shot reproduction on demo data
        import subprocess
        out = os.path.join(tempfile.mkdtemp(), "repro")
        r = subprocess.run(
            [sys.executable, "reproduce_study.py", "--demo", "--mini",
             "--folds", "3", "--repeats", "1", "--out", out],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
        assert os.path.isfile(os.path.join(out, "summary.txt"))
        assert os.path.isfile(os.path.join(out, "winner.joblib"))

    check("reproduce_study.py CLI (demo, mini)", s_reproduce_cli)

    # ---------------------------------------------------------------- done
    failed = [msg for ok, msg in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} deep checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
