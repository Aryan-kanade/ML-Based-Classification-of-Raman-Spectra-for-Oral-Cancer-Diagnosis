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


def main() -> int:
    import qt_compat
    from qt_compat import QtWidgets
    import gui

    print(f"Qt binding: {qt_compat.BINDING}")
    # keep a reference: an unreferenced QApplication gets garbage-collected
    # and every later Qt call crashes natively
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    win = gui.MainWindow()
    win.show()

    # --- [1] load data through the GUI path (fresh demo dataset) ----------
    import tempfile
    source = dataset.find_default_source_spectrum()
    assert source, "no real source spectrum found"
    demo = dataset.generate_demo_data(
        source, os.path.join(tempfile.mkdtemp(), "demo"))
    win.load_folder(demo, quiet=True)
    assert len(win.spectra) == 45, len(win.spectra)
    assert win.grid is not None and win.X_raw.shape[0] == 45
    print(f"[1] loaded {len(win.spectra)} demo spectra via GUI")

    # --- [2] preprocess preview with crop ---------------------------------
    win.stack.setCurrentIndex(gui.TAB_PREP)
    win.preview_preprocess()
    assert win.prep_canvas.figure.axes, "no preview axes"
    print("[2] preprocess preview OK (crop active)")

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
    rep = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "result_report.txt")
    assert os.path.isfile(rep) and "Predictions" in open(rep).read()
    print("[6] result report written")

    # --- [7] sidebar completion ticks + nav bounds -------------------------
    win.refresh_nav()
    win.go_next()                      # on Result: must stay on Result
    assert win.stack.currentIndex() == gui.TAB_RESULT
    win.go_back()
    assert win.stack.currentIndex() == gui.TAB_PRED
    print("[7] navigation bounds OK")

    win.close()
    print("\nGUI TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
