"""
gui_test.py — automated GUI verification (offscreen, no window shown).

Exercises the full six-page flow: window construction, clinical-style
data load, preprocess preview, training handlers, model save/load,
folder prediction (with reference-file skipping) and the Result page.

Run:  set QT_QPA_PLATFORM=offscreen && python gui_test.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before Qt import
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
os.environ.setdefault("MPLBACKEND", "QtAgg")

import dataset
import modeling


def _flat_folder(root: str, n_per_class: int = 10) -> str:
    """Synthetic flat labeled folder (classes parsed from the C-number
    filename token, like the real loader expects)."""
    import numpy as np
    rng = np.random.default_rng(0)
    wn = np.linspace(400, 1800, 400)
    os.makedirs(root, exist_ok=True)
    for cls, gain in (("C1", 0.4), ("C5", 1.0), ("C8", 1.6)):
        for i in range(n_per_class):
            peaks = (np.exp(-((wn - 1003) / 15) ** 2)
                     + gain * np.exp(-((wn - 1450) / 15) ** 2))
            it = 0.1 + peaks + rng.normal(0, 0.02, len(wn))
            with open(os.path.join(root, f"SYN_785_{cls}_{i + 1}.txt"),
                      "w", encoding="utf-8") as fh:
                for w, v in zip(wn, it, strict=True):
                    fh.write(f"{w:.2f},{v:.4f}\n")
    return root


def main() -> int:
    import qt_compat
    from qt_compat import QtWidgets
    import gui

    print(f"Qt binding: {qt_compat.BINDING}")
    # keep test artifacts (reports, logs) OUT of the repo tree (B1)
    import tempfile as _tf
    import gui as _gui
    _gui.APP_DIR = _tf.mkdtemp(prefix="raman_guitest_")
    # keep a reference: an unreferenced QApplication gets garbage-collected
    # and every later Qt call crashes natively
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    win = gui.MainWindow()
    win.show()

    # --- [0] Start page: no void — columns stretch, F1 walkthrough on page -
    start = win.stack.widget(gui.TAB_START).widget()
    assert start.layout().stretch(0) == 1, "columns must absorb extra height"
    texts = [l.text() for l in start.findChildren(QtWidgets.QLabel)]
    assert any("How to use" in t for t in texts), "walkthrough card missing"
    print("[0] Start page fills the window (walkthrough card present)")

    # --- [0b] toolbar palette must be light or mpl icons paint white -----
    tb = win.prep_canvas.parent().findChild(QtWidgets.QToolBar)
    tb.ensurePolished()
    assert tb.palette().color(tb.backgroundRole()).value() >= 128, \
        "toolbar QSS background resolves dark -> white mpl icons"
    print("[0b] toolbar palette light — mpl icons black project-wide")

    # --- [1] load data through the GUI path (synthetic flat folder) -------
    import tempfile
    folder = _flat_folder(os.path.join(tempfile.mkdtemp(), "syn"))
    win.load_folder(folder, quiet=True)
    assert len(win.spectra) == 30, len(win.spectra)
    assert win.grid is not None and win.X_raw.shape[0] == 30
    print(f"[1] loaded {len(win.spectra)} synthetic spectra via GUI")

    # --- [2] preprocess preview with crop ---------------------------------
    win.stack.setCurrentIndex(gui.TAB_PREP)
    win.preview_preprocess()
    axes = win.prep_canvas.figure.axes
    assert axes, "no preview axes"
    titles = " ".join(a.get_title() for a in axes)
    for cls in ("C1", "C5", "C8"):  # one row per class in the preview
        assert cls in titles, f"class {cls} missing from preview titles"
    print("[2] preprocess preview OK (crop active, one row per class)")

    # --- [3] training handlers (synchronous, 2 models) --------------------
    X = win.get_processed_X()
    keep = [i for i, l in enumerate(win.labels) if l.strip()]
    results, winner = modeling.evaluate_models(
        X[keep], [win.labels[i] for i in keep],
        model_names=["PCA + LDA", "Random Forest"], k_folds=3,
        wavenumbers=win.grid)
    win.results = results
    win.winner = winner
    win._lc_data = (X[keep], [win.labels[i] for i in keep], None)
    win.on_train_done(results, winner)
    assert win.b_save.isEnabled()
    print(f"[3] training handlers OK — winner {winner.name} "
          f"(F1 {winner.macro_f1():.3f})")

    # --- [4] save + load model bundle, predict a folder --------------------
    path = os.path.join(os.environ.get("TEMP", "."), "gui_test_model.joblib")
    modeling.save_bundle(path, winner, win.grid,
                         win.read_params().validate(), "gui-test")
    win._set_bundle(modeling.load_bundle(path), path)
    # a folder with one spectrum + one reference file that must be skipped
    pdir = os.path.join(os.path.dirname(path), "pred_folder")
    os.makedirs(pdir, exist_ok=True)
    wn, it = dataset.load_spectrum(win.spectra[0].path)
    with open(os.path.join(pdir, "spec1.txt"), "w") as fh:
        for w, v in zip(wn, it, strict=True):
            fh.write(f"{w}\t{v}\n")
    with open(os.path.join(pdir, "spec2_white.txt"), "w") as fh:
        for w, v in zip(wn, it, strict=True):
            fh.write(f"{w}\t{v}\n")
    win.spec_path_edit.setText(pdir)
    win.run_prediction()
    if win._pred_worker is not None:            # prediction is async
        win._pred_worker.wait(120000)
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
    assert len(win._pred_rows) == 1, win._pred_rows   # reference skipped
    assert win.pred_table.columnCount() == 5          # triage+bar+conformal set
    assert win.pred_table.cellWidget(0, 3) is not None
    assert not win.pred_pills[0].isHidden()          # summary pill shown
    # (page auto-jumped to Result — isVisible() would be False)
    print(f"[4] prediction OK (reference file skipped): "
          f"{win._pred_rows[0][0]} -> {win._pred_rows[0][1]}")

    # --- [5] Result page renders from state -------------------------------
    win.stack.setCurrentIndex(gui.TAB_RESULT)
    QtWidgets.QApplication.processEvents()
    assert win.result_verdict.text() != ""
    assert winner.name in win.r_model_title.text()
    assert win.r_pred_table.rowCount() == 1
    assert win.dist_canvas.figure.axes
    print(f"[5] Result page OK — verdict: {win.result_verdict.text()!r}")

    # --- [6] report writer -------------------------------------------------
    win.save_result_report()
    # APP_DIR was redirected to a temp dir at startup — the report lands
    # THERE, never in the source tree (the old assertion read the stale
    # in-tree copy and passed even when the fresh write failed)
    rep = os.path.join(_gui.APP_DIR, "result_report.txt")
    assert os.path.isfile(rep) and "Predictions" in open(
        rep, encoding="utf-8").read()
    print("[6] result report written (temp APP_DIR, out of tree)")

    # --- [7] sidebar completion ticks + nav bounds -------------------------
    win.refresh_nav()
    win.go_next()                      # on Result: must stay on Result
    assert win.stack.currentIndex() == gui.TAB_RESULT
    win.go_back()
    assert win.stack.currentIndex() == gui.TAB_PRED
    print("[7] navigation bounds OK")

    # --- [8] global excepthook: an exception in a Qt callback must reach
    # the hook (log + one dialog) instead of qFatal-aborting the process —
    # patch the modal box so the offscreen test cannot block on it ------
    import sys as _sys
    from qt_compat import QtCore
    assert _sys.excepthook is not win._sys_hook_prev
    warned = []
    _orig_warn = QtWidgets.QMessageBox.warning

    def _fake_warning(*a, **k):
        warned.append(a)
        return QtWidgets.QMessageBox.Ok
    QtWidgets.QMessageBox.warning = staticmethod(_fake_warning)
    try:
        def _boom():
            raise ValueError("excepthook probe")
        QtCore.QTimer.singleShot(0, _boom)
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
    finally:
        QtWidgets.QMessageBox.warning = _orig_warn
    assert win._excepthook_shown and warned, "excepthook did not fire"
    assert not win.isHidden()          # app survived the exception
    # [8b] 2026-09-12: the error must also be ON SCREEN — persistent
    # banner strip + ⚠ counter in the status bar (nothing fails into
    # session.log invisibly anymore)
    assert win._err_banner.isVisible(), "error banner did not show"
    assert "excepthook probe" in win._err_banner.msg_lbl.text()
    assert win._err_chip.isVisible() and win._err_chip.text() == "⚠ 1"
    win._err_banner.b_dismiss.click()
    assert not win._err_banner.isVisible()
    assert win._err_chip.isVisible()   # counter survives dismissal
    print("[8] excepthook OK — slot exception logged, banner + ⚠ shown, "
          "app alive")

    win.close()
    print("\nGUI TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
