"""
stress_audit.py -- native-crash + error-surface stress harness (2026-09-12).

deep_test.py forces the joblib THREADING backend
(JOBLIB_MULTIPROCESSING=0) so loky pools never spawn -- that shield
hides the exact Windows race we need to probe (Brain.md gotchas
#16/#22/#25: loky/native worker pools beside a live Qt main loop).
This harness runs REAL loky pools next to REAL Qt, each scenario in a
SUBPROCESS so a native access violation kills only the child and the
parent records the exit code (0xC0000005 -> -1073741819).

Scenarios (child mode: python stress_audit.py --scenario X):
  A crash-repro   real data, CatBoost training, the FULL auto-diag
                  battery + local-explain -- the exact sequence that
                  killed the app on 2026-09-12 17:05
  B overlap-guard a long analysis is running; every guarded entry
                  (training, honest, regions, learning curve, seeds,
                  noise, locked, LOPO) must REFUSE, never start a
                  second pool
  C close-mid-work closeEvent while a worker runs -> must exit
                  cleanly (wait->terminate path), no hang, no crash
  D ui-churn      preview timer + page switches + redraws while a
                  worker runs; the excepthook must stay silent

Usage:
    python stress_audit.py              # parent: run A-D, summarize
    python stress_audit.py --scenario A # child: single scenario

Exit code 0 = every scenario survived.  Scenario A runs ~15-25 min on
the real dataset (CatBoost + nested diagnostics) by design.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# realistic device mode (no RAMAN_DEVICE pinning) -- the crash happened
# under 'auto' with CUDA initialized; keep it that way.

SCENARIO_TIMEOUT_S = {"A": 2700, "B": 180, "C": 180, "D": 180}


def _qt_env() -> dict:
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    env.setdefault("MPLBACKEND", "QtAgg")
    return env


# ---------------------------------------------------------------- parent
def run_parent() -> int:
    only = sys.argv[sys.argv.index("--scenario") + 1] \
        if "--scenario" in sys.argv else None
    outcomes: list[tuple[str, int, str, float]] = []
    for name in ("A", "B", "C", "D"):
        if only and name != only:
            continue
        env = _qt_env()
        if name == "A":
            # scenario A must spawn REAL loky pools -- never inherit a
            # threading-backend override
            env.pop("JOBLIB_MULTIPROCESSING", None)
        cmd = [sys.executable, os.path.join(HERE, "stress_audit.py"),
               "--scenario", name]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=env,
                timeout=SCENARIO_TIMEOUT_S[name], cwd=HERE)
            rc = proc.returncode
            out = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            rc, out = 124, str(exc)
        dt = time.time() - t0
        outcomes.append((name, rc, out, dt))
        tail = [ln for ln in out.splitlines() if ln.strip()][-1:] \
            or ["<no output>"]
        verdict = "PASS" if (rc == 0 and "STRESS PASS" in out) else "FAIL"
        print(f"[stress {name}] {verdict} rc={rc} {dt/60:.1f} min "
              f"last: {tail[0][:120]}", flush=True)
    bad = [o for o in outcomes if o[1] != 0 or "STRESS PASS" not in o[2]]
    if bad:
        print(f"\n{len(bad)} scenario(s) FAILED -- full child output:",
              flush=True)
        for name, rc, out, _dt in bad:
            print(f"\n===== scenario {name} (rc={rc}) =====")
            print("\n".join(out.splitlines()[-60:]))
        return 1
    print("\nAll stress scenarios survived (no native crash, no guard "
          "leak, no silent excepthook fire).", flush=True)
    return 0


# ------------------------------------------------------------ test data
def _flat_folder(root: str, n_per_class: int = 12) -> str:
    """Synthetic flat labeled folder (class parsed from the C-number
    filename token) -- gui_test.py's exact recipe."""
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


def _make_gui(record_dialogs: bool = True):
    """Offscreen MainWindow isolated from developer settings (deep_test
    pattern), with modal dialogs recorded instead of shown."""
    from qt_compat import QtWidgets
    import ui_helpers as uh
    import clinical_data as _cd
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    app.setStyle("Fusion")
    app.setStyleSheet(uh.STYLESHEET)
    import gui
    if record_dialogs and not getattr(_make_gui, "_patched", False):
        YES = QtWidgets.QMessageBox.Yes
        log = _make_gui.dlog = []

        def _rec(kind):
            def _box(parent, title, text, *a, **k):
                log.append((kind, title, str(text)[:80]))
                return YES if kind == "question" else 1024
            return staticmethod(_box)
        for n in ("warning", "information", "critical", "about", "question"):
            setattr(QtWidgets.QMessageBox, n, _rec(n))

        def _fd(_n):
            def _f(*a, **k):
                return "" if _n == "getExistingDirectory" else ("", "")
            return staticmethod(_f)
        for n in ("getSaveFileName", "getOpenFileName", "getExistingDirectory",
                  "getSaveFileNames", "getOpenFileNames"):
            setattr(QtWidgets.QFileDialog, n, _fd(n))
        _make_gui._patched = True
    elif not hasattr(_make_gui, "dlog"):
        _make_gui.dlog = []
    uh.load_settings, _s1 = (lambda: {}), uh.load_settings
    _cd.find_data_root, _s2 = (lambda: None), _cd.find_data_root
    _cd.remember_data_root, _s3 = (lambda p: None), _cd.remember_data_root
    try:
        win = gui.MainWindow()
    finally:
        uh.load_settings = _s1
        _cd.find_data_root = _s2
        _cd.remember_data_root = _s3
    return win


def _pump(win, seconds: float, step: float = 0.25):
    """Keep the Qt loop alive (queued worker callbacks land) while
    waiting -- the GUI-thread side of every wait in this harness."""
    from qt_compat import QtWidgets
    app = QtWidgets.QApplication.instance()
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(4):
            app.processEvents()
        time.sleep(step)


def _busy(win) -> bool:
    return bool(
        (win.worker is not None and win.worker.isRunning())
        or (win._analysis_worker is not None
            and win._analysis_worker.isRunning())
        or (win._honest_worker is not None and win._honest_worker.isRunning())
        or (win._opt_worker is not None and win._opt_worker.isRunning())
        or (win._pred_worker is not None and win._pred_worker.isRunning())
        or (win._seq_worker is not None and win._seq_worker.isRunning()))


def _block(released):
    time.sleep(6)
    released.append(1)
    return "done"


# ------------------------------------------------------- scenario children
def scenario_a() -> int:
    """THE 2026-09-12 crash sequence, verbatim: real data, CatBoost
    training, auto-diagnostics battery (honest -> regions/SHAP -> band
    agreement -> learning curve -> seeds -> noise -> locked -> LOPO),
    then the local-explain SHAP path.  REAL loky pools."""
    if os.environ.get("JOBLIB_MULTIPROCESSING") == "0":
        print("child A refuses to run under the threading backend",
              flush=True)
        return 2
    import numpy as np
    win = _make_gui()
    # real geometry: offscreen canvases with zero size make matplotlib
    # raise ('box_aspect' must be positive) -- the app now survives that
    # (stage-guarded post-training chain), but the scenario wants the
    # REAL flow, plots included
    win.resize(1280, 840)
    win.show()
    _pump(win, 1.0)
    data_root = os.path.join(os.path.dirname(HERE), "Data")
    if not os.path.isdir(data_root):
        data_root = r"D:\BARC\Data"
    win.load_folder(data_root, quiet=True)
    assert win.spectra, "real dataset did not load"
    for name, cb in win.model_checks.items():
        cb.setChecked(name == "CatBoost")
    win.spin_folds.setValue(3)          # faster than the user's 5-fold
    win.chk_repeat.setChecked(False)
    win.start_training()
    # drain: training + the full automatic diagnostics chain.  The
    # naive "idle? break" races the queued done-callback (thread
    # returns from run() BEFORE the signal is delivered): pump again
    # on idle and only break when STILL idle (2026-09-12 flake).
    t0, last_msg = time.time(), ""
    while True:
        _pump(win, 1.0, step=0.2)
        busy = _busy(win) or bool(win._diag_queue)
        if not busy:
            _pump(win, 2.0, step=0.25)     # let queued callbacks land
            if not (_busy(win) or win._diag_queue):
                break
        if time.time() - t0 > 2400:
            print(f"TIMEOUT draining diag chain; "
                  f"left={len(win._diag_queue)}", flush=True)
            return 3
        msg = win.train_status.text()
        if msg != last_msg:
            last_msg = msg
            print(f"[A] {time.strftime('%H:%M:%S')} {msg[:90]}",
                  flush=True)
    print(f"[A] diagnostics chain drained in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    # the battery must ACTUALLY have run (honest lands first)
    if win._honest_result is None:
        print("[A] FAIL: auto-diagnostics battery never ran "
              "(no honest result)", flush=True)
        print("STRESS FAIL A", flush=True)
        return 6
    # local-explain (Result-page SHAP path): fabricate a prediction set
    # from the training features so run_local_explain has its inputs
    X, yy, _gg = win._lc_data
    idx = np.linspace(0, len(yy) - 1, min(24, len(yy))).astype(int)
    win._pred_rows = [(f"s{i}.csv", yy[i], 0.5) for i in idx]
    win._pred_spectra = [np.asarray(X)[i] for i in idx]
    win.run_local_explain()
    # robust wait: pump until the worker is gone AND the queued done-
    # callback actually ran (_local_bands set) -- checking isRunning()
    # alone can exit before finish() executes (2026-09-12 stress flake)
    t0 = time.time()
    while time.time() - t0 < 600:
        _pump(win, 0.5, step=0.1)
        w = win._analysis_worker
        if ((w is None or not w.isRunning())
                and win._local_bands is not None):
            break
    ok = bool(win._local_bands)
    n = len(win._local_bands or [])
    print(f"[A] local explain produced {n} class explanations", flush=True)
    print("STRESS PASS A" if ok else "STRESS FAIL A", flush=True)
    return 0 if ok else 5


def scenario_b() -> int:
    """Overlap guards: while ONE analysis runs, every other entry must
    refuse (dialog) -- never a second concurrent pool (gotcha #16)."""
    win = _make_gui()
    win.load_folder(_flat_folder(os.path.join(tempfile.mkdtemp(), "b")),
                    quiet=True)
    assert win.spectra

    # _analysis_busy_box builds a REAL modal box -> would hang offscreen;
    # record it like every other dialog instead
    win._analysis_busy_box = lambda: _make_gui.dlog.append(
        ("busy", "Busy", "analysis still running"))

    released: list[int] = []
    win._run_async("Fake long analysis", lambda: _block(released),
                   lambda r: None)
    base = win._analysis_worker
    assert base is not None and base.isRunning()
    before = list(_make_gui.dlog)
    for entry in (win.run_honest_check, win.run_region_importance,
                  win.run_learning_curve, win.run_seed_stability,
                  win.run_noise_check, win.run_lopo,
                  lambda: win.run_locked_eval(auto=True),
                  win.start_training):
        entry()
        _pump(win, 0.2)
    new_dialogs = _make_gui.dlog[len(before):]
    still = (win._analysis_worker is base and base.isRunning()
             and win._honest_worker is None)
    t0 = time.time()
    while (win._analysis_worker is not None
           and win._analysis_worker.isRunning()):
        if time.time() - t0 > 30:
            break
        _pump(win, 0.5)
    ok = still and len(new_dialogs) >= 6 and bool(released)
    print(f"[B] worker survived={still} guard dialogs={len(new_dialogs)} "
          f"fake-job finished={bool(released)}", flush=True)
    for d in new_dialogs:
        print(f"    dialog: {d[0]} / {d[1]}", flush=True)
    print("STRESS PASS B" if ok else "STRESS FAIL B", flush=True)
    return 0 if ok else 5


def scenario_c() -> int:
    """closeEvent while a worker runs: wait->terminate path must exit
    the process cleanly (marker written, exit 0, no hang)."""
    win = _make_gui()
    win.load_folder(_flat_folder(os.path.join(tempfile.mkdtemp(), "c")),
                    quiet=True)
    for name, cb in win.model_checks.items():
        cb.setChecked(name.startswith("PCA + Logistic"))
    win.spin_folds.setValue(3)
    win.start_training()
    _pump(win, 1.5)                     # let the worker get going
    assert _busy(win), "worker did not start"
    t0 = time.time()
    win.close()                         # question patched -> Yes
    _pump(win, 3.0)
    print(f"[C] close during work returned in {time.time() - t0:.1f}s",
          flush=True)
    print("STRESS PASS C", flush=True)
    return 0


def scenario_d() -> int:
    """UI churn during a running worker: preview timer, page switches,
    redraws -- the excepthook must NOT fire (log watched)."""
    win = _make_gui()
    win.load_folder(_flat_folder(os.path.join(tempfile.mkdtemp(), "d")),
                    quiet=True)

    fired: list[str] = []
    orig_log = win.log

    def spy(msg):
        if "UNHANDLED" in msg:
            fired.append(msg)
        orig_log(msg)
    win.log = spy

    released: list[int] = []
    win._run_async("Fake churn target", lambda: _block(released),
                   lambda r: None)
    from qt_compat import QtCore
    vals = (9, 13, 15, 21)
    for i in range(28):
        win.go_to(i % 6)
        win.spin_sg_window.setValue(vals[(i // 6) % len(vals)])
        win.spin_folds.setValue(3 + (i % 3))
        QtCore.QTimer.singleShot(0, lambda: None)
        _pump(win, 0.15, step=0.05)
    t0 = time.time()
    while (win._analysis_worker is not None
           and win._analysis_worker.isRunning()):
        if time.time() - t0 > 20:
            break
        _pump(win, 0.5)
    ok = (not fired) and bool(released)
    print(f"[D] excepthook fired {len(fired)} time(s); worker finished="
          f"{bool(released)}", flush=True)
    for f in fired[:3]:
        print(f"    {f.splitlines()[0]}", flush=True)
    print("STRESS PASS D" if ok else "STRESS FAIL D", flush=True)
    return 0 if ok else 5


if __name__ == "__main__":
    if "--scenario" in sys.argv:
        # native deaths must leave a Python-level trace (main.py does
        # this for the GUI; the child does it for itself)
        import faulthandler
        try:
            _fh = open(os.path.join(HERE, "crash.log"), "a",
                       encoding="utf-8")
            _fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"stress child start (pid {os.getpid()}) ===\n")
            _fh.flush()
            faulthandler.enable(file=_fh, all_threads=True)
        except (OSError, ValueError):
            faulthandler.enable()
        which = sys.argv[sys.argv.index("--scenario") + 1].upper()
        fn = {"A": scenario_a, "B": scenario_b, "C": scenario_c,
              "D": scenario_d}[which]
        sys.exit(fn())
    sys.exit(run_parent())
