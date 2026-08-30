"""
gui.py — user-friendly Qt main window with sidebar wizard navigation
(PyQt5 / PyQt6 / PySide6 via qt_compat).

Pages (left sidebar steps):
  Start        — welcome, progress checklist, one-click actions
  Data         — load a folder of spectra, fix class labels
  Preprocess   — tune the Raman pipeline, live preview
  Train        — cross-validate the model suite, best sens/spec/F1
  Predict      — load a saved model and classify a folder of spectra

Extras: card layout, big result banner with color-coded stats,
color-coded metric tables, live status in the window title, interactive
matplotlib plots (zoom / pan / save), remembered settings, menu
shortcuts, wait cursors.
"""

from __future__ import annotations

import os
import threading
import traceback
from dataclasses import asdict
from math import log10

import numpy as np

import qt_compat as qc
from qt_compat import (QtWidgets, QtCore, Signal, QAction, ALIGN_CENTER,
                       ALIGN_LEFT, ALIGN_RIGHT, ALIGN_VCENTER, NO_PEN,
                       PEN_DASH, PAINTER_AA, HEADER_STRETCH,
                       HEADER_RESIZE, IS_SELECTABLE,
                       IS_ENABLED, IS_EDITABLE, QT_VERTICAL, QT_HORIZONTAL,
                       BINDING, EDIT_NO, EDIT_DC, EDIT_SC, SELECT_ROWS,
                       WAIT_CURSOR, TEXT_RICH)

import dataset
import modeling
import preprocessing
import biochemistry as bio
import clinical as clin
import clinical_data as cdata
import study_stats as sstats
from preprocessing import PreprocessParams, HAS_PYWT
import plotting
import ui_helpers as uh
from ui_helpers import TIPS, STYLESHEET

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(APP_DIR, "demo_data")

# bump when the model suite changes so saved checkbox states don't hide
# newly added models
SUITE_VERSION = 5

# every GUI log line is mirrored to a rotating session log on disk
import logging                                    # noqa: E402
import logging.handlers                           # noqa: E402

FILE_LOG = logging.getLogger("raman_app.session")
if not FILE_LOG.handlers:
    _fh = logging.handlers.RotatingFileHandler(
        os.path.join(APP_DIR, "session.log"), maxBytes=1_000_000,
        backupCount=2, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                       "%Y-%m-%d %H:%M:%S"))
    FILE_LOG.addHandler(_fh)
    FILE_LOG.setLevel(logging.INFO)
    FILE_LOG.propagate = False

# page indexes (sidebar order)
TAB_START, TAB_DATA, TAB_PREP, TAB_TRAIN, TAB_PRED, TAB_RESULT = range(6)
NAV_LABELS = ["Start", "Data", "Preprocess", "Train", "Predict", "Result"]
NEXT_LABELS = ["Next: load your data  →", "Next: preprocess  →",
               "Next: train && evaluate  →", "Next: predict  →",
               "Next: final result  →", ""]


def sanitize(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in name)


def busy(on: bool):
    """Show/restore the wait cursor during long operations."""
    if on:
        QtWidgets.QApplication.setOverrideCursor(WAIT_CURSOR)
    else:
        try:
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            pass


# ==========================================================================
# Training worker thread
# ==========================================================================
class TrainWorker(QtCore.QThread):
    progress = Signal(int, str)
    done = Signal(object, object)      # results, winner
    failed = Signal(str)

    def __init__(self, X, y, model_names, k_folds, seed, parent=None,
                 groups=None, repeats=1, wavenumbers=None):
        super().__init__(parent)
        self.X, self.y = X, y
        self.model_names = model_names
        self.k_folds, self.seed = k_folds, seed
        self.groups = groups
        self.repeats = repeats
        self.wavenumbers = wavenumbers

    def run(self):
        try:
            results, winner = modeling.evaluate_models(
                self.X, self.y, self.model_names, self.k_folds, self.seed,
                progress_cb=lambda pct, msg: self.progress.emit(pct, msg),
                groups=self.groups, repeats=self.repeats,
                wavenumbers=self.wavenumbers)
            self.done.emit(results, winner)
        except Exception:
            self.failed.emit(traceback.format_exc())


class OptimizeWorker(QtCore.QThread):
    """Background preprocessing auto-tune (see optimize.py)."""
    progress = Signal(str)
    done = Signal(object)              # optimize result dict
    failed = Signal(str)

    def __init__(self, X_raw, wn, y, groups, exclude, parent=None):
        super().__init__(parent)
        self.X_raw, self.wn, self.y = X_raw, wn, y
        self.groups, self.exclude = groups, exclude

    def run(self):
        try:
            import optimize
            out = optimize.optimize_preprocessing(
                self.X_raw, self.wn, self.y, groups=self.groups,
                exclude=self.exclude, progress=lambda m: self.progress.emit(m))
            self.done.emit(out)
        except Exception:
            self.failed.emit(traceback.format_exc())


class PipelineWorker(QtCore.QThread):
    """Nested (honest) pipeline evaluation in the background."""
    progress = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, X_raw, wn, y, groups, exclude, parent=None):
        super().__init__(parent)
        self.X_raw, self.wn, self.y = X_raw, wn, y
        self.groups, self.exclude = groups, exclude

    def run(self):
        try:
            if self.exclude is not None:
                keep = [i for i, bad in enumerate(self.exclude)
                        if not bad]
                X_raw = self.X_raw[keep]
                y = [self.y[i] for i in keep]
                g = ([self.groups[i] for i in keep]
                     if self.groups is not None else None)
            else:
                X_raw, y, g = self.X_raw, self.y, self.groups
            out = modeling.evaluate_pipeline(
                X_raw, self.wn, y, groups=g,
                progress=lambda m: self.progress.emit(m))
            self.done.emit(out)
        except Exception:
            self.failed.emit(traceback.format_exc())


class FuncWorker(QtCore.QThread):
    """Run any callable off the UI thread (diagnostics buttons)."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception:
            self.failed.emit(traceback.format_exc())


class SeqSearchWorker(QtCore.QThread):
    """
    3SSE architecture search in the background: quick screening of all
    single/pair/triple chains, nested validation of the top-N per level,
    the deployable winner chain, and automatic significance testing
    (McNemar vs the best single model + 3-seed stability).  Supports
    cooperative cancellation and JSONL-checkpoint resume.
    """
    progress = Signal(int, str)
    search_progress = Signal(int, int, str, float, float)  # done,total,best,f1,eta
    leaderboard = Signal(list)          # [(arch_str, f1), ...] top-5
    done = Signal(object)              # {"board", "validated", "winner", ...}
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, X, y, groups, wavenumbers, model_names, k=3,
                 seed=42, top=20, fast=False, resume_path=None,
                 parent=None):
        super().__init__(parent)
        self.X, self.y, self.groups = X, y, groups
        self.wavenumbers = wavenumbers
        self.model_names = model_names
        self.k, self.seed, self.top = k, seed, top
        self.fast = fast
        self.resume_path = resume_path
        self._cancel = threading.Event()

    def cancel(self):
        """Cooperative stop; finished triples stay in the checkpoint."""
        self._cancel.set()

    def run(self):
        try:
            import sequential

            def prog(done, total, best, f1, eta):
                self.progress.emit(
                    int(100 * done / max(total, 1)),
                    f"{done}/{total} architectures · best {best} "
                    f"(F1 {f1:.3f}) · ETA {eta / 60:.0f} min")
                self.search_progress.emit(done, total, best, f1, eta)

            names = self.model_names
            k_eff, cnn_eff = self.k, 15
            if self.fast:               # quick screening preset
                names = [n for n in names
                         if n not in sequential.SLOW_MODELS]
                k_eff, cnn_eff = 2, 8
            board = sequential.search(
                self.X, self.y, groups=self.groups,
                wavenumbers=self.wavenumbers, k=k_eff, seed=self.seed,
                top=self.top, model_names=names,
                cnn_epochs=cnn_eff, resume_path=self.resume_path,
                progress_cb=prog, cancel_check=self._cancel.is_set,
                leaderboard_cb=lambda t5: self.leaderboard.emit(t5))
        except sequential.SearchCancelled:
            self.cancelled.emit()
            return
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        try:
            validated = sequential.validate_top(
                board, self.X, self.y, self.groups, self.wavenumbers,
                top=self.top, seed=self.seed,
                progress=lambda m: self.progress.emit(-1, m))
            winner = sequential.finalize_winner(
                validated, self.X, self.y, self.groups,
                self.wavenumbers, board=board, k=k_eff, seed=self.seed)
            sig = sequential.significance_of(
                validated, winner, board, self.X, self.y, self.groups,
                self.wavenumbers, seed=self.seed,
                progress=lambda m: self.progress.emit(-1, m))
            if sig is not None and self.resume_path:
                try:
                    import json as _json
                    with open(os.path.join(
                            os.path.dirname(self.resume_path),
                            "significance.json"), "w",
                            encoding="utf-8") as fh:
                        _json.dump(sig, fh, indent=2)
                except OSError:
                    pass
            # a completed run clears its checkpoint: the next Run starts
            # fresh instead of resuming stale triples
            if self.resume_path and os.path.isfile(self.resume_path):
                try:
                    os.remove(self.resume_path)
                except OSError:
                    pass
            self.done.emit({"board": board, "validated": validated,
                            "winner": winner, "k": k_eff,
                            "seed": self.seed, "groups": self.groups,
                            "significance": sig})
        except Exception:
            self.failed.emit(traceback.format_exc())


class PredictWorker(QtCore.QThread):
    """
    Prediction in the background: the file loop, per-patient paired
    references and preprocessing all run off the UI thread.  GUI work
    (plots, tables) is deferred to signals handled on the main thread.
    """
    progress = Signal(int, str)
    first_plot = Signal(object)          # dict for the Predict-page preview
    done = Signal(object)                # full payload
    failed = Signal(str)

    NORMAL_SYNONYMS = ("normal", "norm", "benign", "negative")

    def __init__(self, bundle, files, root, manual_ref_dir=None,
                 parent=None):
        super().__init__(parent)
        self.bundle, self.files, self.root = bundle, files, root
        self.manual_ref_dir = manual_ref_dir

    def _auto_reference(self, path: str, cache: dict):
        """
        Margin mode on a clinical tree: build the reference from the
        SAME patient's normal-class folder (<root>/Normal/<patient>).
        Returns None when the tree has no matching normal side.
        """
        import paired as paired_mod
        try:
            rel = os.path.relpath(os.path.abspath(path),
                                  os.path.abspath(self.root))
            parts = rel.replace("\\", "/").split("/")
            if len(parts) < 3:
                return None                    # not <class>/<patient>/file
            patient = parts[-2]
            if patient in cache:
                return cache[patient]
            for cls_dir in sorted(os.listdir(self.root)):
                if not cls_dir.lower().startswith(self.NORMAL_SYNONYMS):
                    continue
                ref_dir = os.path.join(self.root, cls_dir, patient)
                if not os.path.isdir(ref_dir):
                    continue
                ref_files = [os.path.join(ref_dir, fn)
                             for fn in sorted(os.listdir(ref_dir))
                             if fn.lower().endswith((".txt", ".dat",
                                                    ".csv"))]
                if ref_files:
                    vec = paired_mod.reference_vector(self.bundle,
                                                      ref_files)
                    cache[patient] = vec
                    return vec
            cache[patient] = None
        except Exception:
            cache[patient] = None
        return None

    def run(self):
        try:
            import paired as paired_mod
            reference = None
            ref_map: dict[str, object] = {}
            if self.bundle.get("paired"):
                if self.manual_ref_dir and os.path.isdir(
                        self.manual_ref_dir):
                    ref_files = [
                        os.path.join(self.manual_ref_dir, fn)
                        for fn in sorted(os.listdir(self.manual_ref_dir))
                        if fn.lower().endswith((".txt", ".dat", ".csv"))]
                    reference = paired_mod.reference_vector(self.bundle,
                                                            ref_files)
                # else: per-patient auto references resolved per file
            rows, prob_list, spec_list, logs = [], [], [], []
            spec_wn = None
            ok_paths = []
            first = True
            n = len(self.files)
            for i, path in enumerate(self.files, start=1):
                self.progress.emit(
                    int(100 * i / n),
                    f"Predicting {i}/{n}: {os.path.basename(path)}")
                ref = reference
                if (ref is None and self.bundle.get("paired")
                        and self.manual_ref_dir is None):
                    ref = self._auto_reference(path, ref_map)
                try:
                    wn, it = dataset.load_spectrum(path)
                    out = modeling.predict_with_bundle(self.bundle, wn, it,
                                                       reference=ref)
                except Exception as exc:
                    logs.append(f"Prediction failed for {path}: {exc}")
                    continue
                pmax = max(out["probabilities"].values())
                rows.append((os.path.basename(path), out["prediction"],
                             pmax))
                prob_list.append({k: float(v)
                                  for k, v in out["probabilities"].items()})
                ok_paths.append(path)
                logs.append(f"{os.path.basename(path)} → "
                            f"{out['prediction']} (p={pmax:.3f})")
                try:
                    grid = self.bundle["wavenumbers"]
                    y = np.interp(grid, wn, it)
                    m = preprocessing.crop_mask(
                        grid, self.bundle["prep_params"])
                    yproc = preprocessing.preprocess_spectrum(
                        y[m], self.bundle["prep_params"])
                    spec_list.append(np.asarray(yproc, dtype=float))
                    spec_wn = np.asarray(grid)[m]
                except Exception as exc:
                    logs.append(f"Spectrum preview failed for {path}: "
                                f"{exc}")
                if first:
                    first = False
                    if spec_list and spec_wn is not None:
                        self.first_plot.emit({
                            "wn": spec_wn, "raw": y[m], "proc": yproc,
                            "pred": out["prediction"],
                            "probs": out["probabilities"],
                            "title": os.path.basename(path)})
            self.done.emit({
                "rows": rows, "probs": prob_list, "spectra": spec_list,
                "wn": spec_wn, "reference": reference,
                "n_auto_refs": len([v for v in ref_map.values()
                                    if v is not None]),
                "ok_paths": ok_paths, "logs": logs,
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


# ==========================================================================
# Sidebar navigation
# ==========================================================================
class SidebarNav(QtWidgets.QWidget):
    def __init__(self, on_nav, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(215)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 18, 12, 14)
        v.setSpacing(4)

        title = QtWidgets.QLabel("Raman Classifier")
        title.setObjectName("BrandTitle")
        v.addWidget(title)
        sub = QtWidgets.QLabel("Oral cancer · SERS spectra ·\n"
                               "max sens / spec / F1")
        sub.setObjectName("BrandSub")
        v.addWidget(sub)
        v.addSpacing(14)

        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: list[QtWidgets.QPushButton] = []
        for idx, label in enumerate(NAV_LABELS):
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setProperty("nav", True)
            self.group.addButton(b, idx)
            b.clicked.connect(lambda _=False, i=idx: on_nav(i))
            v.addWidget(b)
            self.buttons.append(b)
        self.buttons[0].setChecked(True)

        v.addStretch(1)
        v.addWidget(QtWidgets.QLabel("PROGRESS"))
        self.overall = QtWidgets.QProgressBar()
        self.overall.setTextVisible(False)
        self.overall.setFixedHeight(8)
        v.addWidget(self.overall)
        self.overall_label = QtWidgets.QLabel("0 of 3 steps done")
        self.overall_label.setObjectName("BrandSub")
        v.addWidget(self.overall_label)

    def set_status(self, done: list[bool]):
        """done: per-page completion flags; updates ticks + progress."""
        for b, d in zip(self.buttons, done, strict=True):
            base = NAV_LABELS[self.buttons.index(b)]
            b.setText(base + ("   ✓" if d else ""))
        steps = sum(done[1:4])          # data / train / predict
        self._animate_progress(int(100 * steps / 3))
        self.overall_label.setText(f"{steps} of 3 steps done")

    def _animate_progress(self, target: int):
        """Smoothly ramp the progress bar to `target` percent."""
        if target == self.overall.value():
            return
        anim = QtCore.QVariantAnimation(self, duration=400)
        anim.setStartValue(self.overall.value())
        anim.setEndValue(target)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.valueChanged.connect(self.overall.setValue)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        self._prog_anim = anim            # keep a reference alive


# ==========================================================================
# Animated page stack (fade-in on page change)
# ==========================================================================
class PageStack(QtWidgets.QStackedWidget):
    """QStackedWidget that fades the incoming page in (170 ms, OutCubic)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._anim = None
        self._fade = None
        self.currentChanged.connect(self._fade_in)

    def _fade_in(self, idx: int):
        page = self.widget(idx)
        if page is None:
            return
        if self._anim is not None:           # rapid page switch: snap
            self._anim.stop()
        self._fade = QtWidgets.QGraphicsOpacityEffect(page)
        self._fade.setOpacity(0.0)
        page.setGraphicsEffect(self._fade)
        self._anim = QtCore.QPropertyAnimation(self._fade, b"opacity", self)
        self._anim.setDuration(170)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._anim.finished.connect(self._drop_effect)
        self._anim.start()

    def _drop_effect(self):
        """Remove the effect so the page repaints at full speed later."""
        page = self.currentWidget()
        if page is not None and page.graphicsEffect() is self._fade:
            page.setGraphicsEffect(None)
        self._fade = None
        self._anim = None


# ==========================================================================
# Counting helpers
# ==========================================================================
_IRREGULAR_PLURALS = {"spectrum": "spectra"}


def plural(n: int, word: str) -> str:
    """"1 spectrum", "3 spectra", "2 patients" — grammar-safe counts."""
    if n == 1:
        return f"1 {word}"
    return f"{n} {_IRREGULAR_PLURALS.get(word, word + 's')}"


# ==========================================================================
# Tumor-likelihood meter (custom-painted 0-100% gauge)
# ==========================================================================
class LikelihoodMeter(QtWidgets.QWidget):
    """
    Slim painted gauge for binary verdicts: colored likelihood zones
    (green / amber / red), a dark marker at the mean P(positive), the
    two clinical cut-off notches (rule-out / rule-in), and a confidence
    tier label.
    """

    ZONES = ((0.0, 0.4, "#bbf7d0"), (0.4, 0.6, "#fde68a"),
             (0.6, 1.0, "#fecaca"))
    TIERS = ((0.66, "HIGH", "#b91c1c"), (0.4, "MODERATE", "#b45309"),
             (0.0, "LOW", "#15803d"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value: float | None = None
        self._rule_out: float | None = None
        self._rule_in: float | None = None
        self._pos_name = "positive"
        self.setMinimumHeight(78)

    def set_data(self, value: float | None, rule_out: float | None,
                 rule_in: float | None, pos_name: str):
        self._value = value
        self._rule_out = rule_out
        self._rule_in = rule_in
        self._pos_name = pos_name or "positive"
        self.update()

    def _tier(self) -> tuple[str, str]:
        if self._value is None:
            return ("", "#475569")
        for lo, label, color in self.TIERS:
            if self._value >= lo:
                return (f"{label} · mean P = {self._value:.0%}", color)
        return ("", "#475569")

    def paintEvent(self, _event):
        p = qc.QtGui.QPainter(self)
        p.setRenderHint(PAINTER_AA)
        w = self.width()
        bar_y, bar_h = 34, 12

        # caption + confidence tier
        name = self._pos_name.capitalize()
        f = qc.QtGui.QFont(self.font())
        f.setBold(True)
        f.setPointSizeF(10.0)
        p.setFont(f)
        p.setPen(qc.QtGui.QColor("#1e293b"))
        p.drawText(qc.QtCore.QRect(0, 0, w // 2, 24),
                   ALIGN_LEFT | ALIGN_VCENTER,
                   f"{name} likelihood")
        tier, tier_col = self._tier()
        if tier:
            p.setPen(qc.QtGui.QColor(tier_col))
            p.drawText(qc.QtCore.QRect(w // 2, 0, w // 2, 24),
                       ALIGN_RIGHT | ALIGN_VCENTER,
                       tier)

        # track + likelihood zones
        track = qc.QtCore.QRectF(0.5, bar_y, w - 1.0, bar_h)
        p.setPen(qc.QtGui.QPen(qc.QtGui.QColor("#cbd5e1"), 1))
        p.setBrush(qc.QtGui.QColor("#f8fafc"))
        p.drawRoundedRect(track, 6, 6)
        for a, b, color in self.ZONES:
            p.setPen(NO_PEN)
            p.setBrush(qc.QtGui.QColor(color))
            p.drawRect(qc.QtCore.QRectF(a * w, bar_y, (b - a) * w, bar_h))

        # clinical cut-off notches (rule-out below, rule-in above)
        for frac, color in ((self._rule_out, "#15803d"),
                            (self._rule_in, "#b91c1c")):
            if frac is None:
                continue
            x = max(2.0, min(w - 2.0, frac * w))
            pen = qc.QtGui.QPen(qc.QtGui.QColor(color), 2)
            pen.setStyle(PEN_DASH)
            p.setPen(pen)
            p.drawLine(qc.QtCore.QPointF(x, bar_y - 5),
                       qc.QtCore.QPointF(x, bar_y + bar_h + 5))

        # marker at the mean P(positive)
        if self._value is not None:
            x = max(2.0, min(w - 2.0, self._value * w))
            p.setPen(NO_PEN)
            p.setBrush(qc.QtGui.QColor("#0f172a"))
            tri = qc.QtGui.QPolygonF([
                qc.QtCore.QPointF(x - 5, bar_y - 9),
                qc.QtCore.QPointF(x + 5, bar_y - 9),
                qc.QtCore.QPointF(x, bar_y)])
            p.drawPolygon(tri)
            p.setPen(qc.QtGui.QPen(qc.QtGui.QColor("#0f172a"), 2))
            p.drawLine(qc.QtCore.QPointF(x, bar_y),
                       qc.QtCore.QPointF(x, bar_y + bar_h))

        # scale ticks
        f2 = qc.QtGui.QFont(self.font())
        f2.setPointSizeF(7.5)
        p.setFont(f2)
        p.setPen(qc.QtGui.QColor("#94a3b8"))
        for frac, text in ((0.0, "0"), (0.5, "50%"), (1.0, "100%")):
            p.drawText(qc.QtCore.QRectF((frac * w) - 30, bar_y + bar_h + 2,
                                        60, 16),
                       ALIGN_CENTER, text)
        p.end()


# ==========================================================================
# Main window
# ==========================================================================
class ChainFlowWidget(QtWidgets.QWidget):
    """Painted 3SSE architecture flow, e.g.
    [Spectrum] → [Model A] → P₁ → [Model B] → [Verdict].
    Scales to 1-3 layers; set_arch() to update."""

    BOX_H = 30

    def __init__(self, arch, parent=None):
        super().__init__(parent)
        self._arch = list(arch)
        self.setMinimumHeight(52)

    def set_arch(self, arch):
        self._arch = list(arch)
        self.setVisible(bool(self._arch))
        self.update()

    def paintEvent(self, _ev):
        p = qc.QtGui.QPainter(self)
        p.setRenderHint(PAINTER_AA)
        f = p.font()
        f.setPointSizeF(8.5)
        p.setFont(f)
        fm = qc.QtGui.QFontMetricsF(p.font())
        h = self.height()
        bh = min(self.BOX_H, h - 14)
        by = (h - bh) // 2
        # items: (label, color, is_badge)
        items = [("Spectrum", "#475569", False)]
        for i, model in enumerate(self._arch):
            items.append((model, "#4f46e5", False))
            if i < len(self._arch) - 1:
                items.append((f"P{i + 1}+", "#b45309", True))
        items.append(("Verdict", "#15803d", False))
        # widths: boxes sized to text (elided), badges tiny, arrows 14px
        avail = self.width() - 8
        n_arrows = len(items) - 1
        arrow_w = 14 if n_arrows else 0
        box_budget = (avail - n_arrows * arrow_w) / len(items)
        x = 4.0
        for idx, (label, color, is_badge) in enumerate(items):
            if is_badge:
                w = fm.horizontalAdvance(label) + 10
                rect = QtCore.QRectF(x, by + bh / 2 - 9, w, 18)
                p.setPen(qc.QtGui.QColor("#b45309"))
                p.setBrush(qc.QtGui.QColor("#fef3c7"))
                p.drawRoundedRect(rect, 9, 9)
                p.drawText(rect, ALIGN_CENTER, label)
            else:
                w = min(box_budget, fm.horizontalAdvance(label) + 18)
                text = label
                while (text and fm.horizontalAdvance(text)
                       > w - 14):
                    text = text[:-2]
                if text != label:
                    text = text.rstrip() + "…"
                rect = QtCore.QRectF(x, by, w, bh)
                p.setPen(qc.QtGui.QColor(color))
                p.setBrush(qc.QtGui.QColor(color))
                p.setOpacity(0.14)
                p.drawRoundedRect(rect, 6, 6)
                p.setOpacity(1.0)
                p.drawText(rect, ALIGN_CENTER | ALIGN_VCENTER, text)
            x += w
            if idx < n_arrows:
                ax0, ax1 = x + 2, x + arrow_w - 2
                yc = by + bh / 2
                p.setPen(qc.QtGui.QPen(qc.QtGui.QColor("#94a3b8"), 1.4))
                p.drawLine(QtCore.QPointF(ax0, yc),
                           QtCore.QPointF(ax1 - 4, yc))
                p.drawPolygon(QtCore.QPointF(ax1, yc),
                              QtCore.QPointF(ax1 - 5, yc - 3),
                              QtCore.QPointF(ax1 - 5, yc + 3))
                x += arrow_w
        p.end()


class ArchTableModel(QtCore.QAbstractTableModel):
    """Sortable read-only table of architecture results (handles the
    4,080-row tab without per-row widgets)."""

    HEADERS = ["Rank", "Type", "Architecture", "Macro-F1", "Sens",
               "Spec", "AUC", "Acc"]

    def __init__(self, rows, parent=None):
        super().__init__(parent)
        self._rows = rows              # [rank, type, arch, f1, sens, ...]

    def rowCount(self, _parent=None):
        return len(self._rows)

    def columnCount(self, _parent=None):
        return len(self.HEADERS)

    def headerData(self, sec, orient, role=QtCore.Qt.DisplayRole):
        if (role == QtCore.Qt.DisplayRole
                and orient == QtCore.Qt.Horizontal):
            return self.HEADERS[sec]
        return None

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or not index.isValid():
            return None
        return str(self._rows[index.row()][index.column()])


class SeqResultsDialog(QtWidgets.QDialog):
    """3SSE results: ranking tabs per level + the overall winner."""

    def __init__(self, payload, parent=None):
        super().__init__(parent)
        self.payload = payload
        board = payload["board"]
        self.setWindowTitle("3SSE — Sequential Architecture Search")
        self.resize(880, 600)
        lay = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs)
        level_meta = ((1, "singles", "Single Models"),
                      (2, "pairs", "2-Model"),
                      (3, "triples", "3-Model"))
        self._tabs = tabs
        self._tab_models: list[tuple[str, ArchTableModel]] = []
        for level, key, title in level_meta:
            rows = self._rows(board[key], level)
            model = ArchTableModel(rows)
            tabs.addTab(self._table(model), f"{title} ({len(rows)})")
            self._tab_models.append((title, model))
        tabs.addTab(self._winner_tab(payload), "Overall Winner")
        btn_row = QtWidgets.QHBoxLayout()
        b_csv = QtWidgets.QPushButton("Export this tab to CSV…")
        b_csv.setToolTip("Writes the currently selected ranking tab "
                         "(as displayed, sorted) to a CSV file.")
        b_csv.clicked.connect(self._export_csv)
        btn_row.addWidget(b_csv)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

    @staticmethod
    def _table(model: ArchTableModel):
        view = QtWidgets.QTableView()
        proxy = QtCore.QSortFilterProxyModel()
        proxy.setSourceModel(model)
        view.setModel(proxy)
        view.setSortingEnabled(True)
        view.horizontalHeader().setStretchLastSection(True)
        view.verticalHeader().setVisible(False)
        view.setAlternatingRowColors(True)
        return view

    def _export_csv(self):
        import csv as _csv
        idx = self._tabs.currentIndex()
        if idx < 0 or idx >= len(self._tab_models):
            QtWidgets.QMessageBox.information(
                self, "Nothing to export",
                "Select a ranking tab (Single / 2-Model / 3-Model) "
                "first.")
            return
        title, model = self._tab_models[idx]
        default = f"3sse_{title.lower().replace(' ', '_')}.csv"
        path, _f = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export ranking to CSV",
            os.path.join(os.path.expanduser("~"), "Desktop", default),
            "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                w = _csv.writer(fh)
                w.writerow(ArchTableModel.HEADERS)
                w.writerows(model._rows)
            self.parent().log(f"3SSE ranking exported: {path} "
                              f"({len(model._rows)} rows)") \
                if self.parent() else None
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Export failed", f"Could not write CSV:\n{exc}")

    @staticmethod
    def _rows(entries, level):
        scored = sorted((e for e in entries if "metrics" in e),
                        key=lambda e: (-e["metrics"]["f1"],
                                       -e["metrics"]["sens"],
                                       -e["metrics"]["spec"]))
        label = {1: "Single", 2: "2-Model", 3: "3-Model"}[level]
        return [[i + 1, label, " → ".join(e["arch"]),
                 f"{e['metrics']['f1']:.3f}",
                 f"{e['metrics']['sens']:.3f}",
                 f"{e['metrics']['spec']:.3f}",
                 f"{e['metrics'].get('auc', float('nan')):.3f}",
                 f"{e['metrics']['acc']:.3f}"]
                for i, e in enumerate(scored)]

    def _winner_tab(self, payload):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        winner = payload.get("winner")
        if winner and winner.get("arch"):
            lay.addWidget(ChainFlowWidget(winner["arch"]))
            if winner.get("metrics"):
                m = winner["metrics"]
                pills = QtWidgets.QHBoxLayout()
                for label, val, tone in (
                        ("Macro-F1", m["f1"], "green" if m["f1"] >= .7
                         else "amber"),
                        ("Sensitivity", m["sens"], "indigo"),
                        ("Specificity", m["spec"], "indigo"),
                        ("ROC-AUC", m.get("auc", float("nan")), "slate")):
                    pills.addWidget(uh.pill(
                        f"{label}: {val:.3f}" if val == val
                        else f"{label}: n/a", tone))
                lay.addLayout(pills)
        validated = payload.get("validated") or {}
        if validated or winner:
            lines = ["NESTED VALIDATION — best per level", ""]
            for level in (1, 2, 3):
                cands = validated.get(level, [])
                if cands:
                    b = max(cands, key=lambda e: e["metrics"]["f1"])
                    m = b["metrics"]
                    lines.append(f"{level}-Model:  "
                                 f"{' → '.join(b['arch'])}")
                    lines.append(f"          F1 {m['f1']:.3f} · "
                                 f"sens {m['sens']:.3f} · "
                                 f"spec {m['spec']:.3f} · "
                                 f"AUC {m.get('auc', float('nan')):.3f} ·"
                                 f" acc {m['acc']:.3f}")
                    lines.append("")
            if winner:
                m = winner["metrics"]
                lines += ["=" * 56,
                          f"OVERALL WINNER ({len(winner['arch'])}-Model):"
                          f" {' → '.join(winner['arch'])}",
                          f"  Macro-F1 {m['f1']:.3f} · sensitivity "
                          f"{m['sens']:.3f} · specificity "
                          f"{m['spec']:.3f}",
                          f"  ROC-AUC {m.get('auc', float('nan')):.3f} · "
                          f"accuracy {m['acc']:.3f}",
                          "  Baseline (paired Extra Trees): F1 0.702 · "
                          "AUC 0.788"]
            sig = payload.get("significance")
            if sig:
                f1s = sig["seed_f1s"]
                mean = sum(f1s) / len(f1s)
                sd = (sum((x - mean) ** 2 for x in f1s)
                      / max(len(f1s) - 1, 1)) ** 0.5
                verdict = ("SIGNIFICANT" if sig["mcnemar_p"] < 0.05
                           else "NOT significant")
                lines += ["",
                          "SIGNIFICANCE (auto)",
                          f"  vs best single ({sig['baseline']}, F1 "
                          f"{sig['baseline_f1']:.3f}): McNemar "
                          f"b={sig['mcnemar_b']} c={sig['mcnemar_c']} "
                          f"→ p={sig['mcnemar_p']:.3f} — the chain's "
                          f"improvement is {verdict}.",
                          f"  Seed stability ({len(f1s)} seeds): F1 "
                          f"{mean:.3f} ± {sd:.3f} "
                          f"({', '.join(f'{x:.3f}' for x in f1s)})"]
        elif payload.get("report_text"):
            # loaded from a saved run without persisted validation data
            text = payload["report_text"]
            start = text.find("FULL NESTED VALIDATION")
            lines = ([text[start:].rstrip()] if start >= 0
                     else [text])
        else:
            lines = ["No validated winner available."]
        lbl = QtWidgets.QLabel("\n".join(lines))
        lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(lbl)
        lay.addStretch(1)
        btn = QtWidgets.QPushButton("Save winner as model bundle…")
        # only live search results carry the fitted chain
        btn.setEnabled(bool(winner and winner.get("chain")))
        btn.clicked.connect(self._on_save)
        lay.addWidget(btn)
        return w

    def _on_save(self):
        pw = self.parent()
        if pw is not None and hasattr(pw, "on_seq_save"):
            pw.on_seq_save(self.payload)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raman Spectra Classifier")
        self.resize(1280, 840)

        # ---- state ----
        self.spectra: list[dataset.Spectrum] = []
        self.labels: list[str] = []
        self.groups: list[str] | None = None   # patient key per spectrum
        self.spike_flags: list[bool] | None = None  # low-quality per spectrum
        self._opt_worker = None
        self._pred_worker = None
        self._lc_data: tuple | None = None     # (X, y, groups) for curves
        self._lc_wn: np.ndarray | None = None  # cropped wavenumber axis
        self._pred_rows: list[tuple[str, str, float]] = []  # last predictions
        self._pred_probs: list[dict[str, float]] = []  # full per-class probs
        self._pred_spectra: list[np.ndarray] = []  # preprocessed spectra
        self._pred_wn: np.ndarray | None = None     # cropped wavenumber axis
        self._pred_reference: np.ndarray | None = None  # paired normal mean
        self._patient_rows: list[tuple] = []   # per-patient verdicts
        self._region_bands: list | None = None  # (center, share, name, sign)
        self._op_points: tuple | None = None    # (rule_out, rule_in) probs
        self._locked_result: dict | None = None  # one-shot held-out result
        self._lopo_result: dict | None = None    # leave-one-patient-out
        self._seed_result: list | None = None    # F1 across seeds
        self._noise_result: list | None = None   # (noise, f1) curve
        self._friedman_result: dict | None = None  # model comparison test
        self._local_bands: list | None = None    # per-prediction SHAP rows
        self._band_stats: list | None = None     # FDR band statistics
        self._paired_mode = False              # last training mode
        self._honest_worker = None
        self._honest_result: dict | None = None
        self._honest_btn = None
        self.grid: np.ndarray | None = None
        self.X_raw: np.ndarray | None = None
        self._data_key: str | None = None
        self._proc_cache: tuple[str, np.ndarray] | None = None
        self.results: list | None = None
        self.winner = None
        self.worker: TrainWorker | None = None
        self._seq_worker: SeqSearchWorker | None = None
        self._seq_payload: dict | None = None
        self._analysis_worker: FuncWorker | None = None
        self.bundle: dict | None = None
        self._loading_table = False
        self._previewed_for: str | None = None
        self._source_folder: str | None = None
        self.settings: dict = uh.load_settings()

        self._build_ui()
        self._build_menu()
        self._apply_settings()
        self.update_footer()

        # in-app activity log (same lines as session.log), toggled from
        # a permanent status-bar button — errors are never invisible
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setFont(qc.QtGui.QFont("Consolas", 9))
        self.log_dock = QtWidgets.QDockWidget("Activity log", self)
        self.log_dock.setWidget(self.log_view)
        dock_area = (qc.Qt.BottomDockWidgetArea
                     if hasattr(qc.Qt, "BottomDockWidgetArea")
                     else qc.Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(dock_area, self.log_dock)
        self.log_dock.hide()
        b_log = QtWidgets.QPushButton("Activity log")
        b_log.setFlat(True)
        b_log.setToolTip("Show or hide the in-app log — the same "
                         "lines that are written to session.log, "
                         "including error details.")
        b_log.clicked.connect(
            lambda: self.log_dock.setVisible(
                not self.log_dock.isVisible()))
        self.statusBar().addWidget(b_log)

        self.log("Ready. Open your spectra folder, or use One-click "
                 "demo on the Start page.")

        # restore last session's data folder, if still there
        self.settings.pop("files", None)          # legacy setting
        folder = self.settings.get("folder", "")
        if folder and os.path.isdir(folder):
            self.log(f"Restoring last session: {folder}")
            self.load_folder(folder, quiet=True)
        self.update_welcome()
        self.refresh_nav()

    # ================================================================= UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QtWidgets.QWidget()
        body.setObjectName("PageHost")
        hbody = QtWidgets.QHBoxLayout(body)
        hbody.setContentsMargins(12, 12, 12, 4)
        hbody.setSpacing(12)

        self.sidebar = SidebarNav(self.go_to)
        hbody.addWidget(self.sidebar)

        main_col = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(main_col)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(10)

        self.stack = PageStack()
        mv.addWidget(self.stack, 1)

        # footer: back / next navigation
        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(10)
        self.b_back = QtWidgets.QPushButton("Back")
        self.b_back.setMinimumHeight(36)
        self.b_back.clicked.connect(self.go_back)
        footer.addWidget(self.b_back)
        footer.addStretch(1)
        self.next_btn = QtWidgets.QPushButton("")
        self.next_btn.setProperty("primary", True)
        self.next_btn.setMinimumHeight(38)
        self.next_btn.setMinimumWidth(230)
        self.next_btn.clicked.connect(self.go_next)
        footer.addWidget(self.next_btn)
        mv.addLayout(footer)

        hbody.addWidget(main_col, 1)

        root.addWidget(body, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Start: load your data folder")

        # every page is wrapped in a scroll area: content taller than the
        # window (Result page in particular) stays reachable
        for page in (self._build_welcome_page, self._build_data_page,
                     self._build_preprocess_page, self._build_train_page,
                     self._build_predict_page, self._build_result_page):
            self.stack.addWidget(self._wrap_scroll(page()))
        self.stack.setCurrentIndex(TAB_START)
        self.stack.currentChanged.connect(self.on_page_changed)

    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        for text, slot, scut in (
                ("&Open data folder…", self.browse_folder, "Ctrl+O"),
                ("&Reload data", self.reload_folder, "F5"),
                (None, None, None),
                ("&Load model…", self.browse_model, "Ctrl+L"),
                ("&Save best model…", self.save_model, "Ctrl+S"),
                (None, None, None),
                ("E&xit", self.close, "Ctrl+Q")):
            if text is None:
                m_file.addSeparator()
                continue
            act = QAction(text, self)
            if scut:
                act.setShortcut(scut)
            act.triggered.connect(slot)
            m_file.addAction(act)

        m_help = mb.addMenu("&Help")
        for text, slot, scut in (
                ("How to use this app", self.show_howto, "F1"),
                ("What the metrics mean", self.show_metric_help, None),
                ("About", self.show_about, None)):
            act = QAction(text, self)
            if scut:
                act.setShortcut(scut)
            act.triggered.connect(slot)
            m_help.addAction(act)

    # ------------------------------------------------------ shared helpers
    @staticmethod
    def _wrap_scroll(page: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        """Make a page scrollable when its content exceeds the window."""
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        return scroll

    @staticmethod
    def card(title: str, hint: str = ""):
        """White rounded card with header; returns (frame, vbox)."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("Card")
        uh.attach_shadow(frame)
        v = QtWidgets.QVBoxLayout(frame)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(8)
        head = QtWidgets.QLabel(title)
        head.setObjectName("CardHeader")
        v.addWidget(head)
        if hint:
            h = QtWidgets.QLabel(hint)
            h.setObjectName("CardHint")
            h.setWordWrap(True)
            v.addWidget(h)
        return frame, v

    @staticmethod
    def section_label(text: str) -> QtWidgets.QLabel:
        lab = QtWidgets.QLabel(text)
        lab.setObjectName("SectionLabel")
        return lab

    # ----------------------------------------------------------- Start page
    def _build_welcome_page(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setSpacing(10)

        # gradient hero panel with live status pills
        hero = QtWidgets.QFrame()
        hero.setObjectName("Hero")
        hv = QtWidgets.QHBoxLayout(hero)
        hv.setContentsMargins(22, 18, 22, 18)
        hv.setSpacing(16)
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(6)
        title = QtWidgets.QLabel("Raman Spectra Classifier")
        title.setObjectName("HeroTitle")
        left.addWidget(title)
        sub = QtWidgets.QLabel(
            "Oral-cancer detection from SERS Raman spectra — automatically "
            "picks the model with the best <b>sensitivity, specificity and "
            "F1 score</b>.")
        sub.setObjectName("HeroSub")
        sub.setWordWrap(True)
        left.addWidget(sub)
        self.hero_pills = [uh.pill("no data", "white"),
                           uh.pill("0 classes", "white"),
                           uh.pill("0 patients", "white")]
        pills_row = QtWidgets.QHBoxLayout()
        pills_row.setSpacing(8)
        for p in self.hero_pills:
            pills_row.addWidget(p)
        pills_row.addStretch(1)
        left.addSpacing(4)
        left.addLayout(pills_row)
        hv.addLayout(left, 1)
        v.addWidget(hero)

        prog, pv = self.card("Your progress")
        self.w_lbl_data = QtWidgets.QLabel("")
        self.w_lbl_model = QtWidgets.QLabel("")
        self.w_lbl_saved = QtWidgets.QLabel("")
        for w in (self.w_lbl_data, self.w_lbl_model, self.w_lbl_saved):
            w.setTextFormat(TEXT_RICH)
            pv.addWidget(w)
        v.addWidget(prog)

        actions, av = self.card("Quick actions")
        grid = QtWidgets.QGridLayout()
        big = [("Load data folder", self.act_load_folder,
                "Browse for the folder with your labeled spectra.", False),
               ("Generate demo data", self.act_demo,
                "Create a synthetic 3-class dataset to try the app.",
                False),
               ("One-click demo", self.quick_start,
                "Demo data, preprocessing, model comparison, best model "
                "— all in one go.", True),
               ("Predict with saved model", self.act_predict,
                "Skip training — load a saved model and classify new "
                "spectra.", False)]
        for i, (text, slot, tip, primary) in enumerate(big):
            b = QtWidgets.QPushButton(text)
            b.setProperty("primary", primary)
            b.setMinimumHeight(46)
            b.setToolTip(tip)
            b.setCursor(qc.POINTING_HAND)
            b.clicked.connect(slot)
            grid.addWidget(b, i // 2, i % 2)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(10)
        av.addLayout(grid)
        v.addWidget(actions)

        tips, tv = self.card("Typical workflow")
        tv.addWidget(QtWidgets.QLabel(
            "1 Load data  →  2 check preprocessing  →  3 train & save "
            "the best model  →  4 predict new spectra.\n"
            "Use the sidebar (or Back / Next below) to move between "
            "steps. Data and settings are remembered."))
        tv.addWidget(QtWidgets.QLabel(
            "Note: the demo dataset is synthetic — load your real "
            "labeled data for actual results."))
        v.addWidget(tips)
        v.addStretch(1)
        return page

    # ------------------------------------------------------------- Data page
    def _build_data_page(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setSpacing(10)

        src, sv = self.card("Data source")
        top = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText(
            "Folder containing spectra .txt files …")
        self.folder_edit.setToolTip(TIPS["folder"])
        top.addWidget(self.folder_edit, 1)
        b_browse = QtWidgets.QPushButton("Browse…")
        b_browse.setProperty("primary", True)
        b_browse.setToolTip(TIPS["folder"])
        b_browse.clicked.connect(self.browse_folder)
        top.addWidget(b_browse)
        b_reload = QtWidgets.QPushButton("Reload")
        b_reload.setToolTip("Re-read the loaded folder (e.g. after adding "
                            "more spectra).")
        b_reload.clicked.connect(self.reload_folder)
        top.addWidget(b_reload)
        sv.addLayout(top)
        self.count_label = QtWidgets.QLabel("No spectra loaded — browse "
                                            "above.")
        self.count_label.setObjectName("CardHint")
        sv.addWidget(self.count_label)
        v.addWidget(src)

        mid = QtWidgets.QSplitter(QT_HORIZONTAL)

        tbl_card, tv = self.card("Spectra && classes")
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["File", "Patient", "Class (double-click to edit)", "Points",
             "WN min", "WN max", "Quality"])
        self.table.horizontalHeader().setSectionResizeMode(0, HEADER_STRETCH)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(EDIT_DC | EDIT_SC)
        self.table.setSelectionBehavior(SELECT_ROWS)
        self.table.setAlternatingRowColors(True)
        self.table.cellChanged.connect(self.on_cell_changed)
        self.table.setToolTip(
            "One row per spectrum. The Class column is auto-filled from "
            "the filename token (e.g. _C8_) and can be corrected by "
            "double-clicking.")
        tv.addWidget(self.table)
        mid.addWidget(tbl_card)

        view_card, vv = self.card("Viewer")
        row = QtWidgets.QHBoxLayout()
        b_plot = QtWidgets.QPushButton("Plot selected")
        b_plot.setToolTip("Ctrl-click several table rows to overlay them "
                          "in the plot.")
        b_plot.clicked.connect(self.plot_selected)
        row.addWidget(b_plot)
        b_means = QtWidgets.QPushButton("Plot class means")
        b_means.clicked.connect(self.plot_class_means)
        row.addWidget(b_means)
        row.addStretch(1)
        vv.addLayout(row)
        holder, self.data_canvas = plotting.canvas_with_toolbar(
            width=7, height=4)
        vv.addWidget(holder, 1)
        mid.addWidget(view_card)
        mid.setStretchFactor(0, 3)
        mid.setStretchFactor(1, 4)
        v.addWidget(mid, 1)
        return page

    # ------------------------------------------------------ Preprocess page
    def _build_preprocess_page(self):
        page = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(page)
        h.setSpacing(10)

        param_card, pv = self.card("Pipeline parameters",
                                   "Defaults suit most Raman data — tweak "
                                   "and press Preview.")
        form = QtWidgets.QFormLayout()
        form.setSpacing(6)

        form.addRow(self.section_label("REGION (cm-1)"))
        self.spin_crop_min = QtWidgets.QDoubleSpinBox()
        self.spin_crop_min.setRange(0, 5000)
        self.spin_crop_min.setDecimals(0)
        self.spin_crop_min.setSingleStep(50)
        self.spin_crop_min.setValue(400)
        self.spin_crop_min.setToolTip(
            "Keep only this wavenumber range (0 = no limit). The "
            "fingerprint region 400-1800 usually works best — outside it "
            "the signal is mostly noise.")
        self.spin_crop_max = QtWidgets.QDoubleSpinBox()
        self.spin_crop_max.setRange(0, 5000)
        self.spin_crop_max.setDecimals(0)
        self.spin_crop_max.setSingleStep(50)
        self.spin_crop_max.setValue(1800)
        self.spin_crop_max.setToolTip(
            "Upper end of the kept range (0 = no limit).")
        crop_row = QtWidgets.QHBoxLayout()
        crop_row.setContentsMargins(0, 0, 0, 0)
        crop_row.addWidget(self.spin_crop_min)
        crop_row.addWidget(QtWidgets.QLabel("to"))
        crop_row.addWidget(self.spin_crop_max)
        wrap = QtWidgets.QWidget()
        wrap.setLayout(crop_row)
        form.addRow("Keep range", wrap)

        form.addRow(self.section_label("DENOISING"))
        self.chk_despike = QtWidgets.QCheckBox(
            "Despike cosmic rays (Whitaker-Hayes)")
        self.chk_despike.setChecked(False)
        self.chk_despike.setToolTip(
            "Repairs single-point cosmic-ray spikes (Whitaker & Hayes "
            "2018). Measured on this dataset, heavily spiked spectra are "
            "not recoverable this way — EXCLUDING them scored better, so "
            "despiking is off by default and spiked spectra are excluded "
            "(Quality column) instead.")
        form.addRow(self.chk_despike)
        self.spin_despike_z = QtWidgets.QDoubleSpinBox()
        self.spin_despike_z.setRange(3.0, 20.0)
        self.spin_despike_z.setSingleStep(0.5)
        self.spin_despike_z.setValue(7.0)
        self.spin_despike_z.setToolTip("Spike sensitivity: lower = more "
                                       "aggressive (7 is the paper default).")
        form.addRow("Spike Z threshold", self.spin_despike_z)
        self.chk_wavelet = QtWidgets.QCheckBox("Wavelet denoising")
        self.chk_wavelet.setChecked(True)
        self.chk_wavelet.setEnabled(HAS_PYWT)
        self.chk_wavelet.setToolTip(TIPS["wavelet"])
        if not HAS_PYWT:
            self.chk_wavelet.setToolTip("PyWavelets not installed "
                                        "(pip install PyWavelets)")
        form.addRow(self.chk_wavelet)
        self.combo_wavelet = QtWidgets.QComboBox()
        self.combo_wavelet.addItems(["sym4", "sym6", "sym8", "db4", "db6",
                                     "coif3", "coif5"])
        self.combo_wavelet.setCurrentIndex(
            self.combo_wavelet.findText("sym8"))  # match PreprocessParams
        self.combo_wavelet.setToolTip(TIPS["wavelet_name"])
        form.addRow("Wavelet", self.combo_wavelet)
        self.spin_level = QtWidgets.QSpinBox()
        self.spin_level.setRange(1, 8)
        self.spin_level.setValue(4)
        self.spin_level.setToolTip(TIPS["level"])
        form.addRow("Decomposition level", self.spin_level)

        form.addRow(self.section_label("SMOOTHING"))
        self.spin_sg_window = QtWidgets.QSpinBox()
        self.spin_sg_window.setRange(5, 51)
        self.spin_sg_window.setSingleStep(2)
        self.spin_sg_window.setValue(11)
        self.spin_sg_window.setToolTip(TIPS["sg_window"])
        form.addRow("Savitzky–Golay window", self.spin_sg_window)
        self.spin_sg_poly = QtWidgets.QSpinBox()
        self.spin_sg_poly.setRange(1, 5)
        self.spin_sg_poly.setValue(3)
        self.spin_sg_poly.setToolTip(TIPS["sg_poly"])
        form.addRow("SG polynomial order", self.spin_sg_poly)
        self.combo_deriv = QtWidgets.QComboBox()
        self.combo_deriv.addItems(["0 (smooth only)", "1 (1st derivative)",
                                   "2 (2nd derivative)"])
        self.combo_deriv.setToolTip(TIPS["deriv"])
        form.addRow("SG derivative", self.combo_deriv)

        form.addRow(self.section_label("BASELINE"))
        self.combo_baseline = QtWidgets.QComboBox()
        self.combo_baseline.addItems(["als", "arpls"])
        self.combo_baseline.setToolTip(
            "ALS: Eilers & Boelens asymmetric least squares (fast).\n"
            "arpls: asymmetrically reweighted PLS (Baek 2015) — usually "
            "handles broad fluorescence humps better.")
        if not preprocessing.HAS_PYBASELINES:
            self.combo_baseline.setItemText(1, "arpls (needs pybaselines)")
        form.addRow("Method", self.combo_baseline)
        self.spin_lambda_exp = QtWidgets.QSpinBox()
        self.spin_lambda_exp.setRange(2, 9)
        self.spin_lambda_exp.setValue(5)
        self.spin_lambda_exp.setToolTip(TIPS["als_lambda"])
        form.addRow("ALS lambda (10^)", self.spin_lambda_exp)
        self.spin_als_p = QtWidgets.QDoubleSpinBox()
        self.spin_als_p.setRange(0.001, 0.5)
        self.spin_als_p.setDecimals(3)
        self.spin_als_p.setSingleStep(0.005)
        self.spin_als_p.setValue(0.01)
        self.spin_als_p.setToolTip(TIPS["als_p"])
        form.addRow("ALS p (asymmetry)", self.spin_als_p)
        self.spin_als_iter = QtWidgets.QSpinBox()
        self.spin_als_iter.setRange(5, 30)
        self.spin_als_iter.setValue(10)
        self.spin_als_iter.setToolTip(TIPS["als_niter"])
        form.addRow("ALS iterations", self.spin_als_iter)

        form.addRow(self.section_label("NORMALIZATION"))
        self.combo_norm = QtWidgets.QComboBox()
        self.combo_norm.addItems(["vector", "snv", "none"])
        self.combo_norm.setToolTip(TIPS["norm"])
        form.addRow("Method", self.combo_norm)

        b_preview = QtWidgets.QPushButton("Preview spectrum")
        b_preview.setProperty("primary", True)
        b_preview.setToolTip("Show raw vs preprocessed for the spectrum "
                             "selected in the Data page.")
        b_preview.clicked.connect(self.preview_preprocess)
        form.addRow(b_preview)
        b_optimize = QtWidgets.QPushButton("Optimize preprocessing")
        b_optimize.setToolTip(
            "Tries crop-range / derivative / normalization combinations "
            "under patient-grouped cross-validation and applies the best "
            "one. Typically 1-3 minutes.")
        b_optimize.clicked.connect(self.run_optimize)
        form.addRow(b_optimize)
        self.optimize_status = QtWidgets.QLabel(
            "Optimize tries combinations on your data and applies the "
            "winner (grouped CV, no leakage).")
        self.optimize_status.setObjectName("CardHint")
        self.optimize_status.setWordWrap(True)
        pv.addWidget(self.optimize_status)
        pv.addLayout(form)
        h.addWidget(param_card, 1)

        prev_card, cv = self.card("Preview",
                                  "Left: raw on the common grid · "
                                  "right: after the pipeline.")
        holder, self.prep_canvas = plotting.canvas_with_toolbar(
            width=9, height=5)
        cv.addWidget(holder, 1)
        h.addWidget(prev_card, 3)
        return page

    # ---------------------------------------------------------- Train page
    def _build_train_page(self):
        page = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(page)
        h.setSpacing(10)

        # ---- left: controls ----
        ctrl, cv = self.card("Controls")
        cv.addWidget(QtWidgets.QLabel("Classification mode:"))
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems([
            "Standard — each spectrum independently",
            "Margin mode — vs patient's own normal",
            "Margin + PQN — scale-corrected (Dieterle 2006)",
            "3SSE search — sequential stacking (standard data)",
            "3SSE search — sequential stacking (paired data)"])
        self.combo_mode.setToolTip(
            "Standard: every spectrum is classified on its own (works for "
            "any new spectrum).\nMargin mode (paired reference): features "
            "are the DEVIATION from the same patient's mean normal "
            "spectrum — measurably stronger (~+0.1 F1 on the clinical "
            "data) and it models intraoperative tumor-margin assessment "
            "(fiber-optic Raman, in-vivo literature); needs a normal "
            "reference from the same patient at prediction time.\n"
            "Margin + PQN: the deviation is additionally scale-corrected "
            "with probabilistic quotient normalization against the "
            "patient's own normal (Dieterle 2006).")
        cv.addWidget(self.combo_mode)
        cv.addSpacing(4)
        # ---- 3SSE card: run button + live search status ----------------
        seq_card, scv = self.card(
            "3SSE — sequential architecture search",
            "Screens every chain (single → 2-model → 3-model), then "
            "nested-validates the top 20 per level; the overall winner "
            "becomes THE trained model (Save → Predict like any "
            "training).")
        self.seq_card = seq_card
        self.seq_counts = QtWidgets.QLabel("")
        self.seq_counts.setObjectName("CardHint")
        self.seq_counts.setWordWrap(True)
        scv.addWidget(self.seq_counts)
        self.b_3sse_run = QtWidgets.QPushButton(
            "Run 3SSE architecture search now")
        self.b_3sse_run.setObjectName("Primary")
        self.b_3sse_run.setToolTip(
            "Runs the full search on the loaded data with the checked "
            "models (paired data when patient groups exist, standard "
            "otherwise). Takes a while — progress is shown below and "
            "the app stays responsive.")
        self.b_3sse_run.clicked.connect(self.run_3sse_now)
        scv.addWidget(self.b_3sse_run)
        run_row = QtWidgets.QHBoxLayout()
        self.chk_3sse_fast = QtWidgets.QCheckBox(
            "Fast screening (2-fold, skip 1D-CNN/CatBoost/XGBoost)")
        self.chk_3sse_fast.setToolTip(
            "Quick preset: 2-fold screening and the three slowest "
            "models skipped — several times faster, good for "
            "exploration; uncheck for the full search.")
        self.chk_3sse_fast.toggled.connect(self._update_seq_card)
        run_row.addWidget(self.chk_3sse_fast)
        self.b_3sse_cancel = QtWidgets.QPushButton("Cancel search")
        self.b_3sse_cancel.setToolTip(
            "Stop the search. Finished architectures are kept in the "
            "checkpoint — pressing Run again resumes where it stopped.")
        self.b_3sse_cancel.clicked.connect(self.cancel_3sse)
        self.b_3sse_cancel.hide()
        run_row.addWidget(self.b_3sse_cancel)
        run_row.addStretch(1)
        scv.addLayout(run_row)
        self.seq_estimate = QtWidgets.QLabel("")
        self.seq_estimate.setObjectName("CardHint")
        scv.addWidget(self.seq_estimate)
        self.seq_counter = QtWidgets.QLabel("")
        self.seq_counter.setStyleSheet(
            "font-size:15pt; font-weight:700; color:#312e81;")
        self.seq_phase = QtWidgets.QLabel(
            "Ready — press Run (or pick a 3SSE mode and Start training).")
        self.seq_phase.setObjectName("CardHint")
        self.seq_best = QtWidgets.QLabel("")
        self.seq_best.setWordWrap(True)
        self.seq_best.setStyleSheet("color:#475569;")
        scv.addWidget(self.seq_counter)
        scv.addWidget(self.seq_phase)
        scv.addWidget(self.seq_best)
        self.seq_top5 = QtWidgets.QLabel("")
        self.seq_top5.setTextFormat(QtCore.Qt.RichText)
        self.seq_top5.setStyleSheet(
            "color:#334155; font-family:Consolas, monospace;")
        scv.addWidget(self.seq_top5)
        self.b_3sse_view = QtWidgets.QPushButton(
            "View saved 3SSE results (last architecture search)…")
        self.b_3sse_view.setFlat(True)
        self.b_3sse_view.setToolTip(
            "Open the ranking tables (single / 2-model / 3-model layers "
            "and the overall winner) of the last completed 3SSE search "
            "in study_run_3sse/ — no re-run needed.")
        self.b_3sse_view.clicked.connect(self.view_saved_3sse)
        scv.addWidget(self.b_3sse_view)
        self.combo_mode.currentIndexChanged.connect(
            self._update_seq_card)
        cv.addSpacing(4)
        cv.addWidget(QtWidgets.QLabel("Models to compare "
                                      "(best F1 wins — try all!):"))
        self.model_checks: dict[str, QtWidgets.QCheckBox] = {}
        grid = QtWidgets.QGridLayout()
        grid.setVerticalSpacing(2)
        for i, name in enumerate(modeling.ALL_MODEL_NAMES):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            self.model_checks[name] = cb
            grid.addWidget(cb, i // 2, i % 2)
            cb.toggled.connect(self._update_seq_card)
        cv.addLayout(grid)
        self._update_seq_card()               # initial counts
        sel_row = QtWidgets.QHBoxLayout()
        b_all = QtWidgets.QPushButton("All")
        b_all.setFlat(True)
        b_all.clicked.connect(
            lambda: [cb.setChecked(True)
                     for cb in self.model_checks.values()])
        b_none = QtWidgets.QPushButton("None")
        b_none.setFlat(True)
        b_none.clicked.connect(
            lambda: [cb.setChecked(False)
                     for cb in self.model_checks.values()])
        sel_row.addWidget(b_all)
        sel_row.addWidget(b_none)
        sel_row.addStretch(1)
        cv.addLayout(sel_row)
        self.spin_folds = QtWidgets.QSpinBox()
        self.spin_folds.setRange(2, 10)
        self.spin_folds.setValue(5)
        self.spin_folds.setToolTip(TIPS["folds"])
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("CV folds"))
        row.addWidget(self.spin_folds)
        cv.addLayout(row)
        self.spin_seed = QtWidgets.QSpinBox()
        self.spin_seed.setRange(0, 9999)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip(TIPS["seed"])
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Random seed"))
        row.addWidget(self.spin_seed)
        cv.addLayout(row)
        self.chk_repeat = QtWidgets.QCheckBox("Repeat CV ×3 (stabler winner)")
        self.chk_repeat.setToolTip(
            "Repeats the whole cross-validation with three different fold "
            "shuffles and pools the folds — the winner is picked on the "
            "pooled mean, so fold luck matters less. Takes 3x longer.")
        cv.addWidget(self.chk_repeat)
        self.chk_exclude_flagged = QtWidgets.QCheckBox(
            "Exclude spiked spectra (quality flag)")
        self.chk_exclude_flagged.setChecked(True)
        self.chk_exclude_flagged.setToolTip(
            "Spectra with cosmic-ray spikes (see Quality column on the "
            "Data page) are left out of training. Usually helps.")
        if self.spike_flags is None:
            self.chk_exclude_flagged.setEnabled(False)
        cv.addWidget(self.chk_exclude_flagged)
        self.chk_avg_replicates = QtWidgets.QCheckBox(
            "Average replicate spectra per patient")
        self.chk_avg_replicates.setToolTip(
            "Patients with several spectra of the same class are averaged "
            "into one spectrum per (patient, class). Slightly fewer rows, "
            "much less noise — makes cross-validation more stable.")
        self.chk_avg_replicates.setEnabled(False)
        cv.addWidget(self.chk_avg_replicates)
        self.b_train = QtWidgets.QPushButton("Start training")
        self.b_train.setProperty("primary", True)
        self.b_train.setToolTip("Compares every selected model with "
                                "stratified k-fold cross-validation "
                                "(typically under a minute for small "
                                "datasets).")
        self.b_train.setEnabled(False)          # until data is loaded
        self.b_train.clicked.connect(self.start_training)
        cv.addWidget(self.b_train)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        cv.addWidget(self.progress)
        self.train_status = QtWidgets.QLabel("Load data first (Start or "
                                             "Data page).")
        self.train_status.setWordWrap(True)
        self.train_status.setObjectName("CardHint")
        cv.addWidget(self.train_status)
        self.b_save = QtWidgets.QPushButton("Save best model…")
        self.b_save.setProperty("success", True)
        self.b_save.setEnabled(False)
        self.b_save.setToolTip(TIPS["save_model"])
        self.b_save.clicked.connect(self.save_model)
        cv.addWidget(self.b_save)
        b_metrics = QtWidgets.QPushButton("What do these metrics mean?")
        b_metrics.setFlat(True)
        b_metrics.clicked.connect(self.show_metric_help)
        cv.addWidget(b_metrics)
        cv.addSpacing(6)
        # 3SSE power feature sits BELOW the everyday train flow
        cv.addWidget(seq_card)
        cv.addStretch(1)
        h.addWidget(ctrl, 1)

        # ---- right: results (banner, comparison, charts, per-class) ----
        right = QtWidgets.QSplitter(QT_VERTICAL)

        banner, bv = self.card("Result")
        self.banner_title = QtWidgets.QLabel(
                                  "No model trained yet — press Start training")
        self.banner_title.setStyleSheet(
            "font-size:13pt; font-weight:800; color:#1e3a8a; "
            "border:none; background:transparent;")
        bv.addWidget(self.banner_title)
        stats_row = QtWidgets.QHBoxLayout()
        self.stat_values: dict[str, QtWidgets.QLabel] = {}
        for key, name, note in (
                ("sens", "SENSITIVITY", "of positive cases caught"),
                ("spec", "SPECIFICITY", "of negatives ruled out"),
                ("f1", "F1 SCORE", "balance of both")):
            block = QtWidgets.QWidget()
            bvx = QtWidgets.QVBoxLayout(block)
            bvx.setContentsMargins(0, 0, 0, 0)
            bvx.setSpacing(0)
            lab = QtWidgets.QLabel(name)
            lab.setObjectName("StatName")
            lab.setAlignment(ALIGN_CENTER)
            val = QtWidgets.QLabel("–")
            val.setObjectName("StatValue")
            val.setAlignment(ALIGN_CENTER)
            nt = QtWidgets.QLabel(note)
            nt.setObjectName("StatNote")
            nt.setAlignment(ALIGN_CENTER)
            bvx.addWidget(lab)
            bvx.addWidget(val)
            bvx.addWidget(nt)
            stats_row.addWidget(block)
            self.stat_values[key] = val
        bv.addLayout(stats_row)
        self.banner_plain = QtWidgets.QLabel("")
        self.banner_plain.setObjectName("CardHint")
        self.banner_plain.setWordWrap(True)
        bv.addWidget(self.banner_plain)
        # 3SSE winners: painted chain diagram X → model → P → model → …
        self.chain_flow = ChainFlowWidget([])
        self.chain_flow.hide()
        bv.addWidget(self.chain_flow)
        right.addWidget(banner)

        cmp_card, cpv = self.card("Model comparison",
                                  "Mean ± std across CV folds; cells are "
                                  "color-coded (green ≥ 0.90, amber ≥ 0.70, "
                                  "red < 0.70). ★ = winner.")
        self.compare_table = QtWidgets.QTableWidget(0, 4)
        self.compare_table.setHorizontalHeaderLabels(
            ["Model", "Sensitivity (macro)", "Specificity (macro)",
             "F1 (macro)"])
        self.compare_table.horizontalHeader().setSectionResizeMode(
            0, HEADER_STRETCH)
        self.compare_table.verticalHeader().setVisible(False)
        self.compare_table.setEditTriggers(EDIT_NO)
        self.compare_table.setAlternatingRowColors(True)
        self.compare_table.setToolTip(
            "Sensitivity: " + TIPS["sens"] + "\nSpecificity: " +
            TIPS["spec"] + "\nF1: " + TIPS["f1"])
        cpv.addWidget(self.compare_table)
        right.addWidget(cmp_card)

        charts, chv = self.card("Confusion matrix && ROC / per-class")
        lc_row = QtWidgets.QHBoxLayout()
        b_lc = QtWidgets.QPushButton("Data value: learning curve")
        b_lc.setToolTip(
            "Computes macro-F1 at 25/50/75/100% of the patients with the "
            "winning model — shows whether collecting more patients "
            "would improve results. Draws into the right chart.")
        b_lc.clicked.connect(self.run_learning_curve)
        b_regions = QtWidgets.QPushButton("Spectral regions that matter")
        b_regions.setToolTip(
            "RandomForest importance mapped back onto the wavenumber "
            "axis, overlaid on the class-mean spectra — shows WHERE the "
            "discriminative signal lives. Draws into the right chart.")
        b_regions.clicked.connect(self.run_region_importance)
        b_honest = QtWidgets.QPushButton("Honest check (nested)")
        b_honest.setToolTip(
            "Re-chooses the preprocessing INSIDE every CV fold (training "
            "patients only) and reports the unbiased macro-F1 — removes "
            "the optimism from having tuned preprocessing on the same "
            "data. Takes a couple of minutes.")
        b_honest.clicked.connect(self.run_honest_check)
        b_locked = QtWidgets.QPushButton("Locked test-set eval")
        b_locked.setToolTip(
            "TRIPOD-style final exam: splits the PATIENTS 70/15/15 "
            "(train/val/test), refits the winner on the training "
            "patients only and evaluates ONE-SHOT on the untouched "
            "test patients. Honest but noisy — run it once per "
            "configuration, not repeatedly. Needs clinical data (7+ "
            "patients).")
        b_locked.clicked.connect(self.run_locked_eval)
        lc_row.addWidget(b_lc)
        lc_row.addWidget(b_regions)
        lc_row.addWidget(b_honest)
        lc_row.addWidget(b_locked)
        lc_row.addStretch(1)
        chv.addLayout(lc_row)
        deep_row = QtWidgets.QHBoxLayout()
        b_lopo = QtWidgets.QPushButton("Leave-one-patient-out")
        b_lopo.setToolTip(
            "The strictest generalization check for small paired "
            "datasets: refit the winner once per left-out PATIENT "
            "(fixed parameters) and evaluate every patient unseen. "
            "Draws per-patient accuracy into the right chart.")
        b_lopo.clicked.connect(self.run_lopo)
        b_seeds = QtWidgets.QPushButton("Seed stability")
        b_seeds.setToolTip(
            "Macro-F1 of the fixed winner across 5 CV seeds — "
            "quantifies how much the numbers move just by re-running. "
            "Boxplot into the right chart.")
        b_seeds.clicked.connect(self.run_seed_stability)
        b_noise = QtWidgets.QPushButton("Noise robustness")
        b_noise.setToolTip(
            "Adds calibrated measurement noise (1/2/5% of signal "
            "scale) at inference and reports the macro-F1 degradation "
            "curve — the practical robustness statement.")
        b_noise.clicked.connect(self.run_noise_check)
        deep_row.addWidget(b_lopo)
        deep_row.addWidget(b_seeds)
        deep_row.addWidget(b_noise)
        deep_row.addStretch(1)
        chv.addLayout(deep_row)
        plots = QtWidgets.QSplitter(QT_HORIZONTAL)
        cm_holder, self.cm_canvas = plotting.canvas_with_toolbar(
            width=5, height=4)
        roc_holder, self.roc_canvas = plotting.canvas_with_toolbar(
            width=5, height=4)
        plots.addWidget(cm_holder)
        plots.addWidget(roc_holder)
        chv.addWidget(plots)
        right.addWidget(charts)

        # biochemistry: what the signal is MADE of (proteins, nucleic
        # acids, collagen, keratin) and whether the model's bands match
        # the medical literature
        bio_card, bcv = self.card(
            "Biochemistry",
            "The signal is a sum of molecular signatures. Tumor tissue "
            "raises nucleic-acid and protein bands and loses collagen "
            "(literature direction); keratin varies by site and is "
            "flagged as a confounder.")
        bio_row = QtWidgets.QHBoxLayout()
        b_bio = QtWidgets.QPushButton("Analyze biochemistry")
        b_bio.setToolTip(
            "Band-ratio markers with paired patient deltas, NMF "
            "unmixing into biochemical components, a keratin "
            "site-effect check, and a plausibility check of the "
            "model's discriminative bands against the literature. "
            "Uses the last training data (train first).")
        b_bio.clicked.connect(self.run_biochemistry)
        bio_row.addWidget(b_bio)
        bio_row.addStretch(1)
        bcv.addLayout(bio_row)
        self.bio_label = QtWidgets.QLabel("")
        self.bio_label.setObjectName("CardHint")
        self.bio_label.setWordWrap(True)
        bcv.addWidget(self.bio_label)
        bio_plots = QtWidgets.QSplitter(QT_HORIZONTAL)
        bh1, self.bio_canvas = plotting.canvas_with_toolbar(width=5,
                                                            height=3.4)
        bh2, self.bio_comp_canvas = plotting.canvas_with_toolbar(width=5,
                                                                 height=3.4)
        bio_plots.addWidget(bh1)
        bio_plots.addWidget(bh2)
        bcv.addWidget(bio_plots, 1)
        self.bio_table = QtWidgets.QTableWidget(0, 6)
        self.bio_table.setHorizontalHeaderLabels(
            ["Band", "Molecule", "Assignment", "Literature", "Model",
             "Verdict"])
        self.bio_table.horizontalHeader().setSectionResizeMode(
            2, HEADER_STRETCH)
        self.bio_table.verticalHeader().setVisible(False)
        self.bio_table.setEditTriggers(EDIT_NO)
        self.bio_table.setAlternatingRowColors(True)
        self.bio_table.setMaximumHeight(180)
        bcv.addWidget(self.bio_table)
        right.addWidget(bio_card)

        pc_card, pcv = self.card("Per-class metrics")
        self.perclass_table = QtWidgets.QTableWidget(0, 6)
        self.perclass_table.setHorizontalHeaderLabels(
            ["Class", "n (spectra)", "Sensitivity", "Specificity",
             "Precision", "F1"])
        self.perclass_table.horizontalHeader().setSectionResizeMode(
            0, HEADER_STRETCH)
        self.perclass_table.verticalHeader().setVisible(False)
        self.perclass_table.setEditTriggers(EDIT_NO)
        self.perclass_table.setAlternatingRowColors(True)
        pcv.addWidget(self.perclass_table)
        right.addWidget(pc_card)

        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 3)
        right.setStretchFactor(2, 3)
        right.setStretchFactor(3, 4)
        right.setStretchFactor(4, 2)
        right.setSizes([240, 220, 220, 260, 160])
        h.addWidget(right, 3)
        return page

    # -------------------------------------------------------- Predict page
    def _build_predict_page(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setSpacing(10)

        hint = QtWidgets.QLabel(
            "After predicting, the app jumps to the Result page with the "
            "final verdict, the full prediction table and validation "
            "charts.")
        hint.setObjectName("CardHint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        g, gv = self.card("1 — Trained model")
        row = QtWidgets.QHBoxLayout()
        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("model .joblib file …")
        row.addWidget(self.model_path_edit, 1)
        b_model = QtWidgets.QPushButton("Load model…")
        b_model.clicked.connect(self.browse_model)
        row.addWidget(b_model)
        gv.addLayout(row)
        self.model_info = QtWidgets.QLabel("No model loaded.")
        self.model_info.setWordWrap(True)
        self.model_info.setTextFormat(TEXT_RICH)
        gv.addWidget(self.model_info)
        v.addWidget(g)

        # paired-reference row (visible only for paired-trained models)
        self.ref_row_widget = QtWidgets.QWidget()
        rr = QtWidgets.QHBoxLayout(self.ref_row_widget)
        rr.setContentsMargins(0, 4, 0, 0)
        rr.addWidget(QtWidgets.QLabel("Normal reference:"))
        self.ref_path_edit = QtWidgets.QLineEdit()
        self.ref_path_edit.setReadOnly(True)
        self.ref_path_edit.setPlaceholderText(
            "folder with the SAME patient's normal spectra …")
        rr.addWidget(self.ref_path_edit, 1)
        b_ref = QtWidgets.QPushButton("Browse…")
        b_ref.clicked.connect(self.browse_reference)
        rr.addWidget(b_ref)
        self.ref_row_widget.setVisible(False)
        v.addWidget(self.ref_row_widget)

        g2, g2v = self.card("2 — Spectra to classify",
                            "Pick a folder of spectra or individual "
                            ".txt / .dat / .csv files — every spectrum "
                            "is classified.")
        row2 = QtWidgets.QHBoxLayout()
        self.spec_path_edit = QtWidgets.QLineEdit()
        self.spec_path_edit.setReadOnly(True)
        self.spec_path_edit.setPlaceholderText(
            "folder or spectrum file(s) to classify …")
        row2.addWidget(self.spec_path_edit, 1)
        b_spec = QtWidgets.QPushButton("Browse folder…")
        b_spec.clicked.connect(self.browse_spectra)
        row2.addWidget(b_spec)
        b_spec_files = QtWidgets.QPushButton("Add files…")
        b_spec_files.setToolTip("Select individual spectrum file(s) "
                                "instead of a whole folder.")
        b_spec_files.clicked.connect(self.browse_spectra_files)
        row2.addWidget(b_spec_files)
        self.b_predict = QtWidgets.QPushButton("Predict")
        self.b_predict.setProperty("primary", True)
        self.b_predict.setEnabled(False)
        self.b_predict.setToolTip(TIPS["predict"])
        self.b_predict.clicked.connect(self.run_prediction)
        row2.addWidget(self.b_predict)
        g2v.addLayout(row2)
        v.addWidget(g2)

        res, rv = self.card("3 — Result")
        splitter = QtWidgets.QSplitter(QT_HORIZONTAL)
        pred_holder, self.pred_canvas = plotting.canvas_with_toolbar(
            width=6, height=4)
        splitter.addWidget(pred_holder)
        self.pred_table = QtWidgets.QTableWidget(0, 3)
        self.pred_table.setHorizontalHeaderLabels(
            ["File", "Prediction", "P(prediction)"])
        self.pred_table.horizontalHeader().setSectionResizeMode(
            0, HEADER_STRETCH)
        self.pred_table.verticalHeader().setVisible(False)
        self.pred_table.setEditTriggers(EDIT_NO)
        self.pred_table.setAlternatingRowColors(True)
        splitter.addWidget(self.pred_table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        rv.addWidget(splitter, 1)
        v.addWidget(res, 1)
        return page

    # --------------------------------------------------------- Result page
    def _build_result_page(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setSpacing(10)

        # 1. verdict banner (gradient panel, like the hero)
        self.result_banner = QtWidgets.QFrame()
        self.result_banner.setObjectName("Hero")
        bv = QtWidgets.QHBoxLayout(self.result_banner)
        bv.setContentsMargins(22, 16, 22, 16)
        self.result_verdict = QtWidgets.QLabel("No predictions yet")
        self.result_verdict.setObjectName("HeroTitle")
        self.result_verdict.setWordWrap(True)
        bv.addWidget(self.result_verdict, 1)
        self.result_pills = [uh.pill("—", "white") for _ in range(3)]
        pv_row = QtWidgets.QVBoxLayout()
        pills_row = QtWidgets.QHBoxLayout()
        pills_row.setSpacing(8)
        for p in self.result_pills:
            pills_row.addWidget(p)
        pills_row.addStretch(1)
        pv_row.addLayout(pills_row)
        self.result_note = QtWidgets.QLabel("")
        self.result_note.setObjectName("HeroSub")
        self.result_note.setWordWrap(True)
        pv_row.addWidget(self.result_note)
        bv.addLayout(pv_row, 2)
        v.addWidget(self.result_banner)

        # 1b. tumor-likelihood meter (binary verdicts)
        self.result_meter = LikelihoodMeter()
        v.addWidget(self.result_meter)

        # 2. model performance card
        perf, perfv = self.card("Model performance (cross-validated)")
        self.r_model_title = QtWidgets.QLabel("No model trained yet")
        self.r_model_title.setStyleSheet(
            "font-size:13pt; font-weight:800; color:#1e3a8a; "
            "border:none; background:transparent;")
        perfv.addWidget(self.r_model_title)
        stats_row = QtWidgets.QHBoxLayout()
        self.r_stats: dict[str, QtWidgets.QLabel] = {}
        for key, name, note in (
                ("sens", "SENSITIVITY", "positives caught"),
                ("spec", "SPECIFICITY", "negatives cleared"),
                ("f1", "MACRO-F1", "balance of both"),
                ("auc", "ROC AUC", "ranking quality")):
            block = QtWidgets.QWidget()
            bx = QtWidgets.QVBoxLayout(block)
            bx.setContentsMargins(0, 0, 0, 0)
            bx.setSpacing(0)
            lab = QtWidgets.QLabel(name)
            lab.setObjectName("StatName")
            lab.setAlignment(ALIGN_CENTER)
            val = QtWidgets.QLabel("–")
            val.setObjectName("StatValue")
            val.setAlignment(ALIGN_CENTER)
            nt = QtWidgets.QLabel(note)
            nt.setObjectName("StatNote")
            nt.setAlignment(ALIGN_CENTER)
            bx.addWidget(lab)
            bx.addWidget(val)
            bx.addWidget(nt)
            stats_row.addWidget(block)
            self.r_stats[key] = val
        perfv.addLayout(stats_row)
        self.r_model_note = QtWidgets.QLabel("")
        self.r_model_note.setObjectName("CardHint")
        self.r_model_note.setWordWrap(True)
        perfv.addWidget(self.r_model_note)
        # predictive values by clinical setting (prevalence-driven)
        self.r_ppv_table = QtWidgets.QTableWidget(
            len(clin.PREVALENCE_SCENARIOS), 4)
        self.r_ppv_table.setHorizontalHeaderLabels(
            ["Clinical setting", "Prevalence", "PPV", "NPV"])
        self.r_ppv_table.horizontalHeader().setSectionResizeMode(
            0, HEADER_STRETCH)
        self.r_ppv_table.verticalHeader().setVisible(False)
        self.r_ppv_table.setEditTriggers(EDIT_NO)
        self.r_ppv_table.setAlternatingRowColors(True)
        self.r_ppv_table.setMaximumHeight(120)
        self.r_ppv_table.setToolTip(
            "The same test has very different predictive values at "
            "different disease prevalence (Bayes). PPV/NPV are what a "
            "clinician experiences; sensitivity/specificity are "
            "properties of the test.")
        perfv.addWidget(self.r_ppv_table)
        # triage consequences per 1000 patients (empirical tier rates)
        self.r_triage_table = QtWidgets.QTableWidget(
            len(clin.PREVALENCE_SCENARIOS), 5)
        self.r_triage_table.setHorizontalHeaderLabels(
            ["Setting", "Biopsies (triage)", "Cancers caught",
             "Missed at negative", "Biopsy reduction"])
        self.r_triage_table.horizontalHeader().setSectionResizeMode(
            0, HEADER_STRETCH)
        self.r_triage_table.verticalHeader().setVisible(False)
        self.r_triage_table.setEditTriggers(EDIT_NO)
        self.r_triage_table.setAlternatingRowColors(True)
        self.r_triage_table.setMaximumHeight(120)
        self.r_triage_table.setToolTip(
            "Per 1,000 patients at each prevalence: biopsies the "
            "three-tier triage would order (positive + indeterminate), "
            "cancers caught at the positive tier, cancers released at "
            "the negative tier, and the biopsy reduction versus "
            "biopsy-everyone.")
        perfv.addWidget(self.r_triage_table)
        # locked one-shot held-out evaluation (visible after running it)
        locked, lockedv = self.card("Locked test-set evaluation (FINAL)")
        self.r_locked_label = QtWidgets.QLabel(
            "Not run yet — Train page: 'Locked test-set eval' splits the "
            "patients 70/15/15 and evaluates once on the untouched side.")
        self.r_locked_label.setObjectName("CardHint")
        self.r_locked_label.setWordWrap(True)
        lockedv.addWidget(self.r_locked_label)
        self.r_locked_card = locked
        v.addWidget(perf)
        # deep evaluation summary (LOPO / seeds / noise / Friedman)
        deep, deepv = self.card("Deep evaluation",
                                "Leave-one-patient-out, seed stability "
                                "and noise robustness — run from the "
                                "Train page ('deep diagnostics' row); "
                                "Friedman is computed after every "
                                "training.")
        self.r_deep_label = QtWidgets.QLabel("Not run yet.")
        self.r_deep_label.setObjectName("CardHint")
        self.r_deep_label.setWordWrap(True)
        deepv.addWidget(self.r_deep_label)
        self.r_deep_card = deep
        v.addWidget(deep)
        # per-patient out-of-fold performance (hard-patient analysis)
        pmap, pmapv = self.card("Per-patient performance (out-of-fold)",
                                "Which patients does the model actually "
                                "get right? Rows sorted worst-first; "
                                "red = below 50% of spectra correct.")
        self.r_pp_table = QtWidgets.QTableWidget(0, 6)
        self.r_pp_table.setHorizontalHeaderLabels(
            ["Patient", "Spectra", "Accuracy", "Mean P(positive)",
             "True class", "Flag"])
        self.r_pp_table.horizontalHeader().setSectionResizeMode(
            0, HEADER_STRETCH)
        self.r_pp_table.verticalHeader().setVisible(False)
        self.r_pp_table.setEditTriggers(EDIT_NO)
        self.r_pp_table.setAlternatingRowColors(True)
        self.r_pp_table.setMaximumHeight(240)
        pmapv.addWidget(self.r_pp_table)
        self.r_pp_card = pmap
        v.addWidget(pmap)
        # local (per-prediction) explanation
        local, localv = self.card(
            "Why this call",
            "Signed band contributions behind each predicted class "
            "(tree-surrogate SHAP on the mean spectrum of each predicted "
            "group). Red pushes toward the positive class, blue away.")
        lo_row = QtWidgets.QHBoxLayout()
        b_local = QtWidgets.QPushButton("Explain this prediction")
        b_local.setToolTip(
            "Computes per-spectrum SHAP band contributions for the mean "
            "spectrum of every predicted class. Uses a tree surrogate "
            "over the training features — an honest approximation, not "
            "the winner's internal weights.")
        b_local.clicked.connect(self.run_local_explain)
        lo_row.addWidget(b_local)
        lo_row.addStretch(1)
        localv.addLayout(lo_row)
        lh, self.r_local_canvas = plotting.canvas_with_toolbar(width=7,
                                                               height=3.2)
        lh.setMinimumHeight(240)
        localv.addWidget(lh, 1)
        self.r_local_card = local
        v.addWidget(local)

        # 3. prediction details + distribution chart
        mid = QtWidgets.QSplitter(QT_HORIZONTAL)
        det, detv = self.card("Predictions",
                              "Every classified spectrum with its "
                              "confidence.")
        self.r_pred_table = QtWidgets.QTableWidget(0, 4)
        self.r_pred_table.setHorizontalHeaderLabels(
            ["File", "Predicted class", "Triage", "Probability"])
        self.r_pred_table.horizontalHeader().setSectionResizeMode(
            0, HEADER_STRETCH)
        for col in (1, 2, 3):
            self.r_pred_table.horizontalHeader().setSectionResizeMode(
                col, HEADER_RESIZE)
        self.r_pred_table.verticalHeader().setVisible(False)
        self.r_pred_table.verticalHeader().setDefaultSectionSize(26)
        self.r_pred_table.setEditTriggers(EDIT_NO)
        self.r_pred_table.setAlternatingRowColors(True)
        detv.addWidget(self.r_pred_table, 1)
        mid.addWidget(det)
        dist, distv = self.card("Prediction distribution",
                                "Left: predicted class counts · right: "
                                "per-spectrum P(positive) with the decision "
                                "threshold (binary models).")
        dh, self.dist_canvas = plotting.canvas_with_toolbar(width=4,
                                                            height=3.4)
        dh.setMinimumHeight(230)
        distv.addWidget(dh, 1)
        mid.addWidget(dist)
        mid.setStretchFactor(0, 3)
        mid.setStretchFactor(1, 2)
        mid.setSizes([620, 380])
        v.addWidget(mid, 1)

        # 3a. predicted spectra overlay (mean + optional paired reference)
        spec, specv = self.card("Predicted spectra",
                                "The classified spectra after "
                                "preprocessing: faint individual traces, "
                                "bold mean; dashed = the patient's normal "
                                "reference (paired mode); amber spans = top "
                                "discriminative bands.")
        sph, self.r_spec_canvas = plotting.canvas_with_toolbar(width=7,
                                                               height=3.2)
        sph.setMinimumHeight(250)
        specv.addWidget(sph, 1)
        self.r_spec_card = spec
        v.addWidget(spec, 1)

        # 3b. per-patient verdicts (clinical-style inputs)
        pat, patv = self.card("Patient verdicts",
                              "Shown when the predicted folder contains "
                              "patient subfolders: votes and mean "
                              "positive probability per patient.")
        self.r_pat_table = QtWidgets.QTableWidget(0, 6)
        self.r_pat_table.setHorizontalHeaderLabels(
            ["Patient", "Spectra", "Votes", "Mean P(+)", "Verdict",
             "Triage"])
        self.r_pat_table.horizontalHeaderItem(3).setToolTip(
            "mean probability of the positive class across that "
            "patient's spectra")
        self.r_pat_table.horizontalHeaderItem(5).setToolTip(
            "clinical tier from the mean P(positive) against the "
            "rule-out / rule-in thresholds: NEGATIVE / INDETERMINATE / "
            "POSITIVE")
        self.r_pat_table.horizontalHeader().setSectionResizeMode(
            0, HEADER_STRETCH)
        self.r_pat_table.horizontalHeader().setSectionResizeMode(
            4, HEADER_STRETCH)
        for col in (1, 2, 3, 5):
            self.r_pat_table.horizontalHeader().setSectionResizeMode(
                col, HEADER_RESIZE)
        self.r_pat_table.verticalHeader().setVisible(False)
        self.r_pat_table.setEditTriggers(EDIT_NO)
        self.r_pat_table.setAlternatingRowColors(True)
        patv.addWidget(self.r_pat_table)
        self.r_pat_hint = QtWidgets.QLabel("")
        self.r_pat_hint.setObjectName("CardHint")
        patv.addWidget(self.r_pat_hint)
        self.r_pat_card = pat
        v.addWidget(pat, 1)

        # 4. validation charts (confusion, ROC, calibration, DCA)
        charts, chv = self.card("Validation of the winning model",
                                "Discrimination, calibration and clinical "
                                "utility — the TRIPOD+AI triad. Confusion "
                                "matrix + ROC (binary) or per-class "
                                "metrics; calibration (reliability) and "
                                "decision curve when out-of-fold "
                                "probabilities are available.")
        grid = QtWidgets.QWidget()
        gl = QtWidgets.QGridLayout(grid)
        gl.setContentsMargins(0, 0, 0, 0)
        for r in range(2):
            for c in range(2):
                holder, canvas = plotting.canvas_with_toolbar(
                    width=5, height=3.4)
                holder.setMinimumHeight(260)
                gl.addWidget(holder, r, c)
                if (r, c) == (0, 0):
                    self.r_cm_canvas = canvas
                elif (r, c) == (0, 1):
                    self.r_roc_canvas = canvas
                elif (r, c) == (1, 0):
                    self.r_cal_canvas = canvas
                else:
                    self.r_dca_canvas = canvas
        chv.addWidget(grid, 1)
        v.addWidget(charts, 1)

        # 4b. biochemistry of this prediction (markers + confounders)
        bioc, biocv = self.card(
            "Biochemistry of this prediction",
            "Classical intensity markers of the classified spectra; with "
            "a paired normal reference, deltas vs the patient's own "
            "normal. High keratin can indicate a site effect, not "
            "disease.")
        self.r_bio_table = QtWidgets.QTableWidget(0, 4)
        self.r_bio_table.setHorizontalHeaderLabels(
            ["Marker", "Predicted", "Reference", "Delta"])
        self.r_bio_table.horizontalHeader().setSectionResizeMode(
            0, HEADER_STRETCH)
        self.r_bio_table.verticalHeader().setVisible(False)
        self.r_bio_table.setEditTriggers(EDIT_NO)
        self.r_bio_table.setAlternatingRowColors(True)
        self.r_bio_table.setMaximumHeight(200)
        biocv.addWidget(self.r_bio_table)
        self.r_bio_hint = QtWidgets.QLabel("")
        self.r_bio_hint.setObjectName("CardHint")
        self.r_bio_hint.setWordWrap(True)
        biocv.addWidget(self.r_bio_hint)
        self.r_bio_card = bioc
        v.addWidget(bioc, 1)

        # 5. plain-language reading + save report
        read, readv = self.card("What this means")
        self.r_reading = QtWidgets.QLabel("Train a model and run a "
                                          "prediction to see the full "
                                          "reading.")
        self.r_reading.setObjectName("CardHint")
        self.r_reading.setWordWrap(True)
        readv.addWidget(self.r_reading)
        row = QtWidgets.QHBoxLayout()
        b_freeze = QtWidgets.QPushButton("Freeze study")
        b_freeze.setToolTip(
            "Writes study_manifest.json — data + code hashes, "
            "preprocessing, model metrics and confidence interval — so "
            "every number on this page can be reproduced and verified "
            "later.")
        b_freeze.clicked.connect(self.freeze_study)
        b_figs = QtWidgets.QPushButton("Save figures…")
        b_figs.setToolTip(
            "Saves the Result-page charts as 200-dpi PNGs (confusion "
            "matrix, validation chart, distribution, predicted spectra) "
            "into a folder you pick — ready for reports.")
        b_figs.clicked.connect(self.save_result_figures)
        b_save_report = QtWidgets.QPushButton("Save report…")
        b_save_report.setProperty("success", True)
        b_save_report.setToolTip(
            "Writes a plain-text report of everything on this page "
            "(data, preprocessing, models, predictions) to "
            "result_report.txt.")
        b_save_report.clicked.connect(self.save_result_report)
        b_html = QtWidgets.QPushButton("Save report (HTML)…")
        b_html.setToolTip(
            "One self-contained, printable HTML file with every figure "
            "and table from this page — share it or print it as-is.")
        b_html.clicked.connect(self.save_result_report_html)
        row.addStretch(1)
        row.addWidget(b_freeze)
        row.addWidget(b_figs)
        row.addWidget(b_save_report)
        row.addWidget(b_html)
        readv.addLayout(row)
        v.addWidget(read)
        return page

    # ------------------------------------------------------ study manifest
    def freeze_study(self):
        """Snapshot data + code hashes, params and results for exact
        reproducibility of everything on the Result page."""
        import hashlib
        import json
        from datetime import datetime

        def sha1_file(path):
            h = hashlib.sha1()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
            return h.hexdigest()

        manifest: dict = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "app": "Raman Spectra Classifier",
            "data_folder": self._source_folder,
            "n_spectra": len(self.spectra),
            "n_patients": len(set(self.groups)) if self.groups else 0,
            "paired_mode": self._paired_mode,
            "preprocessing": asdict(self.read_params().validate()),
            "code_hashes": {fn: sha1_file(os.path.join(APP_DIR, fn))
                            for fn in sorted(os.listdir(APP_DIR))
                            if fn.endswith(".py")},
        }
        if self._source_folder and os.path.isdir(self._source_folder):
            manifest["data_hashes"] = {
                rel: sha1_file(os.path.join(root, fn))
                for root, _dirs, fns in os.walk(self._source_folder)
                for fn in sorted(fns)
                if fn.lower().endswith((".txt", ".csv", ".dat"))
                for rel in [os.path.relpath(os.path.join(root, fn),
                                            self._source_folder)]}
        if self.winner is not None:
            manifest["winner"] = {
                "model": self.winner.name,
                "macro_f1": round(self.winner.macro_f1(), 4),
                "sensitivity": round(
                    self.winner.macro.get("sens", (float("nan"),))[0], 4),
                "specificity": round(
                    self.winner.macro.get("spec", (float("nan"),))[0], 4),
            }
        if self._honest_result is not None:
            manifest["nested_honest_f1"] = round(
                self._honest_result["mean_f1"], 4)
        if self._pred_rows:
            manifest["predictions"] = [
                {"file": f, "class": c, "p": round(p, 4)}
                for f, c, p in self._pred_rows]
        path = os.path.join(APP_DIR, "study_manifest.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
            n_files = len(manifest.get("data_hashes", {}))
            self.log(f"Study frozen: {path} "
                     f"({n_files} data files, "
                     f"{len(manifest['code_hashes'])} code files hashed)")
            self.statusBar().showMessage(f"Study manifest: {path}")
        except Exception as exc:
            self.friendly_error("Freezing the study failed", exc)

    POSITIVE_SYNONYMS = {"tumor", "tumour", "cancer", "malignant",
                         "positive"}

    def _positive_class(self) -> str | None:
        """The 'positive' class name for verdicts (Tumor/Cancer/...)."""
        classes = set(self.winner.classes) if self.winner is not None else (
            set(l for l in self.labels if l))
        pos = {c for c in classes if c.lower() in self.POSITIVE_SYNONYMS}
        return sorted(pos)[0] if pos else None

    def _op_points_now(self) -> tuple[float | None, float | None]:
        """
        Clinical operating points (rule-out, rule-in) from the winner's
        pooled out-of-fold probabilities, mapped through the bundle's
        Platt calibrator when present so the cut-offs live in the SAME
        probability space as predict-time outputs; falls back to the
        points persisted in a loaded bundle.  (None, None) otherwise.
        """
        w = self.winner
        if (w is not None and len(w.classes) == 2
                and getattr(w, "oof_proba", None) is not None
                and getattr(w, "y_true_encoded", None) is not None):
            valid = ~np.isnan(w.oof_proba[:, 1])
            if valid.any():
                yv = np.asarray(w.y_true_encoded[valid])
                pv = w.oof_proba[valid, 1]
                cal = (self.bundle or {}).get("calibrator")
                if cal:
                    pv = clin.apply_platt(pv, cal)
                try:
                    lo, hi, _s, _sp = clin.operating_points(yv, pv)
                    return lo, hi
                except Exception:
                    return None, None
        if self._op_points:
            lo, hi = self._op_points
            return lo, hi
        return None, None

    def render_result_page(self):
        """Fill the Result page from the current state (called on entry)."""
        pos = self._positive_class()
        rows = self._pred_rows

        # ---- verdict banner ------------------------------------------------
        pats = self._patient_rows
        if self.winner is not None:
            n_classes = len(self.winner.classes)
        elif self.bundle is not None:
            n_classes = len(self.bundle.get("classes") or [])
        else:
            n_classes = len(set(l for l in self.labels if l))
        if pats:
            pos_cls = pos or ""
            pos_pat = [r for r in pats if r[4] == pos_cls] if pos_cls else []
            n = len(pats)
            if pos_cls and pos_pat:
                head = (f"{plural(len(pos_pat), 'patient')} of {n} "
                        f"classified as {pos_cls.upper()} "
                        f"({len(pos_pat) / n:.0%})")
                note = ("Patient-level verdicts below — votes are the "
                        "spectrum counts per class")
            elif pos_cls:
                head = "Every patient classified as clear / negative"
                note = "Patient-level verdicts below"
            else:
                head = f"{plural(n, 'patient')} classified"
                note = "Patient-level verdicts below"
            self.result_verdict.setText(head)
            self.result_note.setText(note)
            pills = ([f"{len(pos_pat)} {pos_cls}",
                      f"{n - len(pos_pat)} negative",
                      f"n = {plural(n, 'patient')}"]
                     if pos_cls else [f"n = {plural(n, 'patient')}"])
            for i, p in enumerate(self.result_pills):
                if i < len(pills):
                    p.setText(pills[i])
                    p.setVisible(True)
                else:
                    p.setVisible(False)
        elif rows:
            n = len(rows)
            counts: dict[str, int] = {}
            for _, cls, _p in rows:
                counts[cls] = counts.get(cls, 0) + 1
            if pos and pos in counts:
                k = counts[pos]
                pct = k / n
                if k == 0:
                    head = "Every spectrum classified as clear / negative"
                else:
                    head = (f"{plural(k, 'spectrum')} of {n} classified "
                            f"as {pos.upper()} ({pct:.0%})")
                mean_p = (sum(p for _f, c, p in rows if c == pos) / k
                          if k else 0.0)
                note = (f"Average {pos} probability: {mean_p:.0%} · "
                        "positive calls are highlighted in the table below"
                        if k else "No positive findings")
                pills = [f"{k} {pos}", f"{n - k} negative",
                         f"n = {plural(n, 'spectrum')}"]
            else:
                top = max(counts.items(), key=lambda kv: kv[1])
                head = (f"{plural(n, 'spectrum')} classified — most "
                        f"common: {top[0]} "
                        f"({top[1]} of {n}, {top[1] / n:.0%})")
                note = "Distribution shown in the chart below"
                pills = [f"{c}: {k}" for c, k in
                         sorted(counts.items(), key=lambda kv: -kv[1])][:3]
            self.result_verdict.setText(head)
            self.result_note.setText(note)
            for i, p in enumerate(self.result_pills):
                if i < len(pills):
                    p.setText(pills[i])
                    p.setVisible(True)
                else:
                    p.setVisible(False)
        else:
            self.result_verdict.setText("No predictions yet")
            self.result_note.setText(
                "Run a prediction on the Predict page — the final verdict "
                "appears here.")
            for p in self.result_pills:
                p.setVisible(False)

        # ---- clinical operating points (rule-out / rule-in) ---------------
        lo, hi = self._op_points_now()

        # ---- tumor-likelihood meter (binary models with full probs) --------
        mean_p_all: float | None = None
        if (rows and pos and n_classes == 2
                and len(self._pred_probs) == len(rows)):
            ps = [pr.get(pos) for pr in self._pred_probs]
            ps = [x for x in ps if x is not None]
            if ps:
                mean_p_all = float(np.mean(ps))
        threshold = None
        if (self.winner is not None
                and getattr(self.winner, "threshold", None) is not None):
            threshold = float(self.winner.threshold)
        elif (self.bundle is not None
                and self.bundle.get("threshold") is not None):
            threshold = float(self.bundle["threshold"])
        # show/use the threshold in the calibrated probability space
        _cal = (self.bundle or {}).get("calibrator")
        if (threshold is not None and _cal and n_classes == 2):
            threshold = clin.apply_platt(threshold, _cal)
        if mean_p_all is not None:
            self.result_meter.set_data(mean_p_all, lo, hi, pos)
        self.result_meter.setVisible(mean_p_all is not None)

        # triage counts over all spectra (for the banner note)
        tri_counts = {clin.TRIAGE_POSITIVE: 0,
                      clin.TRIAGE_INDETERMINATE: 0,
                      clin.TRIAGE_NEGATIVE: 0}
        if (lo is not None and hi is not None and pos and rows
                and n_classes == 2
                and len(self._pred_probs) == len(rows)):
            for pr in self._pred_probs:
                tier = clin.triage(pr.get(pos, float("nan")), lo, hi,
                                   fallback=threshold)
                tri_counts[tier] = tri_counts.get(tier, 0) + 1

        n_triaged = sum(tri_counts.values())
        if n_triaged:
            self.result_note.setText(
                self.result_note.text()
                + " · Triage: "
                + f"{tri_counts[clin.TRIAGE_POSITIVE]} positive · "
                + f"{tri_counts[clin.TRIAGE_INDETERMINATE]} indeterminate"
                + f" · {tri_counts[clin.TRIAGE_NEGATIVE]} negative")

        # ---- model performance ---------------------------------------------
        w = self.winner
        if w is not None:
            self.r_model_title.setText(f"Winner: {w.name}")
            auc_val = None
            try:
                if (len(w.classes) == 2
                        and getattr(w, "oof_proba", None) is not None
                        and getattr(w, "y_true_encoded", None) is not None):
                    valid = ~np.isnan(w.oof_proba[:, 1])
                    if valid.any():
                        _fpr, _tpr, auc_val = modeling.roc_points(
                            w.y_true_encoded[valid], w.oof_proba[valid, 1])
            except Exception:
                auc_val = None
            for key, val in (("sens", w.macro.get("sens", (None, 0))[0]),
                             ("spec", w.macro.get("spec", (None, 0))[0]),
                             ("f1", w.macro_f1()),
                             ("auc", auc_val)):
                lbl = self.r_stats[key]
                lbl.setText(f"{val:.3f}" if val is not None else "–")
                if val is not None:
                    lbl.setStyleSheet(f"color:{uh.metric_fg(val)};")
            bits = [f"{self.spin_folds.value()}-fold CV"]
            if self.chk_repeat.isChecked():
                bits.append("repeated x3")
            if self.groups:
                bits.append(f"grouped by {len(set(self.groups))} patients")
            if w.threshold is not None:
                bits.append(f"decision threshold {threshold:.3f}"
                            + (" (calibrated)" if _cal else ""))
            if lo is not None and hi is not None:
                bits.append(f"rule-out p≤{lo:.2f} (sens ≥90%) · "
                            f"rule-in p≥{hi:.2f} (spec ≥90%)")
            # Wilson CIs from the pooled confusion matrix (binary)
            try:
                if w.cm is not None and len(w.classes) == 2:
                    tp = int(w.cm[1, 1])
                    fn_ = int(w.cm[1].sum()) - tp
                    tn = int(w.cm[0, 0])
                    fp_ = int(w.cm[0].sum()) - tn
                    s_lo_, s_hi_ = clin.wilson_ci(tp, tp + fn_)
                    p_lo_, p_hi_ = clin.wilson_ci(tn, tn + fp_)
                    bits.append(f"sens {tp / (tp + fn_):.2f} "
                                f"[{s_lo_:.2f}–{s_hi_:.2f}] · "
                                f"spec {tn / (tn + fp_):.2f} "
                                f"[{p_lo_:.2f}–{p_hi_:.2f}] (Wilson)")
            except Exception:
                pass
            if auc_val is not None:
                try:
                    _a, _se, d_lo, d_hi = clin.delong_auc_ci(
                        (np.asarray(w.y_true_encoded[valid]) == 1),
                        w.oof_proba[valid, 1])
                    if np.isfinite(d_lo):
                        bits.append(f"AUC {auc_val:.3f} "
                                    f"(DeLong 95% CI "
                                    f"{d_lo:.2f}–{d_hi:.2f})")
                except Exception:
                    pass
            # 95% bootstrap CI for the pooled out-of-fold macro-F1
            try:
                if (getattr(w, "oof_proba", None) is not None
                        and getattr(w, "y_true_encoded", None) is not None):
                    valid = ~np.isnan(w.oof_proba).any(axis=1)
                    pred_oof = np.argmax(w.oof_proba[valid], axis=1)
                    w_groups = (np.asarray(w.groups)[valid]
                                if getattr(w, "groups", None) else None)
                    ci_lo, ci_hi = modeling.bootstrap_ci(
                        w.y_true_encoded[valid], pred_oof,
                        groups=w_groups)
                    bits.append(f"macro-F1 95% CI "
                                f"[{ci_lo:.2f}–{ci_hi:.2f}]")
            except Exception:
                pass
            self.r_model_note.setText(" · ".join(bits))
        else:
            self.r_model_title.setText("No model trained yet")
            for lbl in self.r_stats.values():
                lbl.setText("–")
                lbl.setStyleSheet("")
            self.r_model_note.setText("")
        # predictive values by clinical setting (prevalence matters)
        sens_v = w.macro.get("sens", (None,))[0] if w is not None else None
        spec_v = w.macro.get("spec", (None,))[0] if w is not None else None
        for r_i, (setting, prev) in enumerate(clin.PREVALENCE_SCENARIOS):
            ppv = npv = None
            if sens_v is not None and spec_v is not None:
                ppv, npv = clin.ppv_npv(sens_v, spec_v, prev)
            for c_i, txt in enumerate((
                    setting, f"{prev:.0%}",
                    f"{ppv:.0%}" if ppv is not None
                    and np.isfinite(ppv) else "–",
                    f"{npv:.0%}" if npv is not None
                    and np.isfinite(npv) else "–")):
                it = QtWidgets.QTableWidgetItem(txt)
                if c_i:
                    it.setTextAlignment(ALIGN_CENTER)
                self.r_ppv_table.setItem(r_i, c_i, it)

        # triage consequences per 1000 patients (empirical tier rates
        # from the out-of-fold predictions)
        tri_rows_filled = False
        if (w is not None and len(w.classes) == 2
                and getattr(w, "oof_proba", None) is not None
                and getattr(w, "y_true_encoded", None) is not None
                and lo is not None and hi is not None):
            try:
                valid = ~np.isnan(w.oof_proba[:, 1])
                yv_t = (np.asarray(w.y_true_encoded[valid]) == 1)
                pv_t = w.oof_proba[valid, 1]
                cal_c = (self.bundle or {}).get("calibrator")
                if cal_c:
                    pv_t = clin.apply_platt(pv_t, cal_c)
                tiers = [clin.triage(v, lo, hi) for v in pv_t]
                for r_i, (setting, prev) in enumerate(
                        clin.PREVALENCE_SCENARIOS):
                    ar = sstats.triage_arithmetic(yv_t, tiers, prev)
                    if not ar:
                        continue
                    tri_rows_filled = True
                    for c_i, txt in enumerate((
                            setting, f"{ar['biopsies_triage']}",
                            f"{ar['cancers_caught']} "
                            f"({ar['caught_or_flagged_pct']:.0f}%+flag)",
                            f"{ar['cancers_missed_negative']}",
                            f"-{ar['reduction_pct']:.0f}%")):
                        it = QtWidgets.QTableWidgetItem(txt)
                        if c_i:
                            it.setTextAlignment(ALIGN_CENTER)
                        self.r_triage_table.setItem(r_i, c_i, it)
            except Exception:
                pass
        if not tri_rows_filled:
            for r_i in range(self.r_triage_table.rowCount()):
                for c_i in range(5):
                    it = QtWidgets.QTableWidgetItem("–"
                                                     if c_i else
                                                     clin.PREVALENCE_SCENARIOS
                                                     [r_i][0])
                    if c_i:
                        it.setTextAlignment(ALIGN_CENTER)
                    self.r_triage_table.setItem(r_i, c_i, it)

        # ---- deep evaluation summary ---------------------------------------
        bits = []
        if self._lopo_result:
            lo_ = self._lopo_result
            bits.append(
                f"LOPO: macro-F1 {lo_['f1']:.3f} over "
                f"{lo_['n_patients']} unseen patients"
                + (f", AUC {lo_['auc']:.3f}"
                   if np.isfinite(lo_["auc"]) else ""))
        if self._seed_result:
            f1s_ = self._seed_result
            bits.append(f"seed stability: {np.mean(f1s_):.3f} ± "
                        f"{np.std(f1s_):.3f} (5 seeds)")
        if self._noise_result:
            nr_ = self._noise_result
            bits.append(f"noise: F1 {nr_[0][1]:.3f} → {nr_[-1][1]:.3f} "
                        "at 5% noise")
        if self._friedman_result and self._friedman_result.get("ok"):
            fr_ = self._friedman_result
            bits.append("Friedman: differences "
                        + ("significant" if fr_["significant"]
                           else "not significant (models tie)"))
        self.r_deep_label.setText(" · ".join(bits) if bits
                                  else "Not run yet.")
        self.r_deep_card.setVisible(any(
            [self._lopo_result, self._seed_result, self._noise_result,
             self._friedman_result and
             self._friedman_result.get("ok")]))

        # ---- per-patient out-of-fold performance map -----------------------
        if (w is not None and getattr(w, "oof_proba", None) is not None
                and self._lc_data is not None and self._lc_data[2]):
            try:
                _Xl, yyl, ggl = self._lc_data
                rows_pp = sstats.per_patient_rollup(
                    yyl, ggl, w.oof_proba, list(w.classes),
                    pos_idx=1)
                self.r_pp_card.setVisible(True)
                self.r_pp_table.setRowCount(len(rows_pp))
                for r_i, (pat, n_sp, acc, mp, tcls, hard) in \
                        enumerate(rows_pp):
                    for c_i, txt in enumerate((
                            pat, str(n_sp), f"{acc:.0%}",
                            f"{mp:.2f}" if np.isfinite(mp) else "–",
                            tcls, "hard" if hard else "")):
                        it = QtWidgets.QTableWidgetItem(txt)
                        if c_i:
                            it.setTextAlignment(ALIGN_CENTER)
                        if hard and c_i in (0, 5):
                            it.setForeground(qc.QtGui.QColor("#b91c1c"))
                            f_ = it.font()
                            f_.setBold(True)
                            it.setFont(f_)
                        self.r_pp_table.setItem(r_i, c_i, it)
            except Exception:
                self.r_pp_card.setVisible(False)
        else:
            self.r_pp_card.setVisible(False)
        self.r_local_card.setVisible(bool(self._pred_spectra))

        # ---- prediction table + distribution -------------------------------
        tri_color = {clin.TRIAGE_POSITIVE: "#b91c1c",
                     clin.TRIAGE_INDETERMINATE: "#b45309",
                     clin.TRIAGE_NEGATIVE: "#15803d"}
        self.r_pred_table.setRowCount(len(rows))
        for r, (fname, cls, p) in enumerate(rows):
            it0 = QtWidgets.QTableWidgetItem(fname)
            it1 = QtWidgets.QTableWidgetItem(cls)
            it1.setTextAlignment(ALIGN_CENTER)
            if pos and cls == pos:
                it1.setForeground(qc.QtGui.QColor("#b91c1c"))
                f = it1.font()
                f.setBold(True)
                it1.setFont(f)
            # clinical triage tier from the full probability dict
            tier = None
            if (pos and n_classes == 2
                    and len(self._pred_probs) == len(rows)):
                tier = clin.triage(
                    self._pred_probs[r].get(pos, float("nan")),
                    lo, hi, fallback=threshold)
            it2 = QtWidgets.QTableWidgetItem(tier or "–")
            it2.setTextAlignment(ALIGN_CENTER)
            it2.setToolTip(clin.TRIAGE_ACTIONS.get(tier, ""))
            if tier in tri_color:
                it2.setForeground(qc.QtGui.QColor(tri_color[tier]))
                f2 = it2.font()
                f2.setBold(True)
                it2.setFont(f2)
            self.r_pred_table.setItem(r, 0, it0)
            self.r_pred_table.setItem(r, 1, it1)
            self.r_pred_table.setItem(r, 2, it2)
            # probability as a slim confidence bar for the predicted class
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(round(max(0.0, min(1.0, p)) * 1000)))
            bar.setFormat(f"{p:.0%}")
            bar.setFixedHeight(14)
            color = "#dc2626" if (pos and cls == pos) else "#6366f1"
            bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: {color}; "
                "border-radius: 5px; }}")
            if len(self._pred_probs) == len(rows):
                bar.setToolTip(" · ".join(
                    f"{k} {v:.2f}" for k, v in sorted(
                        self._pred_probs[r].items(),
                        key=lambda kv: -kv[1])))
            self.r_pred_table.setCellWidget(r, 3, bar)
        # ---- patient verdict table -----------------------------------------
        pats = self._patient_rows
        self.r_pat_table.setVisible(bool(pats))
        self.r_pat_table.setRowCount(len(pats))
        for r, (patient, n_spec, votes, mean_p, verdict) in enumerate(pats):
            tier = None
            if (pos and n_classes == 2 and mean_p):
                tier = clin.triage(mean_p, lo, hi, fallback=threshold)
            for c, txt in enumerate((patient, str(n_spec), votes,
                                     f"{mean_p:.3f}" if mean_p else "–",
                                     verdict, tier or "–")):
                it = QtWidgets.QTableWidgetItem(txt)
                if c:
                    it.setTextAlignment(ALIGN_CENTER)
                if pos and c == 4 and verdict == pos:
                    it.setForeground(qc.QtGui.QColor("#b91c1c"))
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                if c == 5 and tier in tri_color:
                    it.setForeground(qc.QtGui.QColor(tri_color[tier]))
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                if c == 3 and mean_p:
                    col = ("#b91c1c" if mean_p >= 0.6 else
                           "#b45309" if mean_p >= 0.4 else "#15803d")
                    it.setForeground(qc.QtGui.QColor(col))
                self.r_pat_table.setItem(r, c, it)
        if pats:
            n_pos = sum(1 for p in pats if pos and p[4] == pos)
            self.r_pat_hint.setText(
                f"{plural(len(pats), 'patient')} — {n_pos} called "
                f"{pos if pos else 'positive'} (majority vote per "
                "patient; mean probability of the positive class)")
        else:
            self.r_pat_hint.setText(
                "No patient grouping detected — predict a clinical-style "
                "folder (one subfolder per patient) to see per-patient "
                "verdicts.")

        # ---- distribution: class counts (+ P(positive) histogram) ----------
        fig = self.dist_canvas.figure
        fig.clf()
        binary_probs = (rows and pos and n_classes == 2
                        and len(self._pred_probs) == len(rows))
        if binary_probs:
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
        else:
            ax1 = fig.add_subplot(111)
            ax2 = None
        if rows:
            classes_seen = sorted({c for _f, c, _p in rows})
            vals = [sum(1 for _f, c, _p in rows if c == k)
                    for k in classes_seen]
            colors = [("#dc2626" if pos and k == pos
                       else plotting.class_color(i))
                      for i, k in enumerate(classes_seen)]
            bars = ax1.bar(classes_seen, vals, color=colors,
                           edgecolor="white", linewidth=0.8, width=0.62)
            total = sum(vals) or 1
            ax1.bar_label(bars, labels=[f"{v} ({v / total:.0%})"
                                        for v in vals],
                          fontsize=9, fontweight="bold", color="#334155",
                          padding=3)
            ax1.set_ylabel("spectra")
            ax1.set_title("Class counts", fontsize=10)
            ax1.set_ylim(0, max(vals) + max(1, int(max(vals) * 0.25)))
            ax1.margins(y=0.1)
        else:
            ax1.text(0.5, 0.5, "no predictions yet", ha="center", va="center",
                     transform=ax1.transAxes, color="#94a3b8")
            ax1.set_axis_off()
        if ax2 is not None:
            p_arr = np.array([pr.get(pos, np.nan)
                              for pr in self._pred_probs], dtype=float)
            plotting.plot_prob_histogram(ax2, p_arr, threshold=threshold,
                                         pos_name=pos)
        self.dist_canvas.draw_idle()

        # ---- predicted spectra overlay -------------------------------------
        if self._pred_spectra and self._pred_wn is not None:
            self.r_spec_card.setVisible(True)
            ax = plotting.clear(self.r_spec_canvas)
            ref = (self._pred_reference
                   if (self.bundle is not None
                       and self.bundle.get("paired")) else None)
            plotting.plot_prediction_spectra(
                ax, self._pred_wn, self._pred_spectra,
                mean_trace=np.mean(
                    np.vstack(self._pred_spectra), axis=0),
                reference=ref, bands=self._region_bands)
            self.r_spec_canvas.draw_idle()
        else:
            self.r_spec_card.setVisible(False)

        # ---- validation charts ----------------------------------------------
        if w is not None and getattr(w, "cm", None) is not None:
            axc = plotting.clear(self.r_cm_canvas)
            plotting.plot_confusion_matrix(
                axc, w.cm, w.classes,
                title="Confusion matrix (pooled CV folds)")
            self.r_cm_canvas.draw_idle()
        if w is not None and getattr(w, "oof_proba", None) is not None:
            try:
                axr = plotting.clear(self.r_roc_canvas)
                ye = w.y_true_encoded
                if len(w.classes) == 2:
                    valid = ~np.isnan(w.oof_proba[:, 1])
                    fpr, tpr, auc = modeling.roc_points(
                        ye[valid], w.oof_proba[valid, 1])
                    plotting.plot_roc(axr, fpr, tpr, auc)
                    axr.set_title("ROC — out-of-fold", fontsize=10)
                else:
                    plotting.plot_perclass_metrics(
                        axr, w.classes, w.per_class,
                        title="Per-class metrics — out-of-fold")
                self.r_roc_canvas.draw_idle()
            except Exception:
                self.log("Result-page validation chart failed:\n"
                         + traceback.format_exc())
        # calibration + decision curve (TRIPOD+AI calibration/utility)
        have_oof = (w is not None
                    and getattr(w, "oof_proba", None) is not None
                    and getattr(w, "y_true_encoded", None) is not None
                    and len(w.classes) == 2)
        if have_oof:
            try:
                valid = ~np.isnan(w.oof_proba[:, 1])
                yv = (np.asarray(w.y_true_encoded[valid]) == 1)
                pv = w.oof_proba[valid, 1]
                cal_bins_raw = clin.calibration_bins(yv, pv)
                cal_c = (self.bundle or {}).get("calibrator")
                brier_raw = float(np.mean((pv - yv.astype(float)) ** 2))
                cal_bins_cal = None
                title = f"Calibration — Brier {brier_raw:.3f}"
                if cal_c:
                    pvc = clin.apply_platt(pv, cal_c)
                    cal_bins_cal = clin.calibration_bins(yv, pvc)
                    brier_cal = float(np.mean(
                        (pvc - yv.astype(float)) ** 2))
                    title = (f"Calibration — Brier {brier_raw:.3f} raw / "
                             f"{brier_cal:.3f} calibrated")
                ax_cal = plotting.clear(self.r_cal_canvas)
                plotting.plot_calibration(ax_cal, cal_bins_raw,
                                           cal_bins=cal_bins_cal,
                                           title=title)
                self.r_cal_canvas.draw_idle()
                ax_dca = plotting.clear(self.r_dca_canvas)
                thr_nb, nb_m, nb_all = clin.decision_curve(
                    yv, clin.apply_platt(pv, cal_c) if cal_c else pv)
                plotting.plot_dca(ax_dca, thr_nb, nb_m, nb_all)
                self.r_dca_canvas.draw_idle()
            except Exception:
                self.log("Calibration / DCA chart failed:\n"
                         + traceback.format_exc())
        else:
            for canvas in (self.r_cal_canvas, self.r_dca_canvas):
                axp = plotting.clear(canvas)
                axp.text(0.5, 0.5, "binary model with out-of-fold\n"
                         "probabilities required", ha="center", va="center",
                         transform=axp.transAxes, color="#94a3b8",
                         fontsize=9)
                axp.set_axis_off()
                canvas.draw_idle()

        # ---- biochemistry of this prediction --------------------------------
        if self._pred_spectra and self._pred_wn is not None:
            self.r_bio_card.setVisible(True)
            wn_p = self._pred_wn
            mean_spec = np.mean(np.vstack(self._pred_spectra), axis=0)
            pred_r = bio.band_ratios(wn_p, mean_spec)
            ref_r = (bio.band_ratios(wn_p, self._pred_reference)
                     if self._pred_reference is not None else None)
            meanings = {key: meaning
                        for key, _n, _d, meaning in bio.RATIOS}
            bio_rows = []
            for key, _num, _den, _meaning in bio.RATIOS:
                v = pred_r.get(key, float("nan"))
                rv = (ref_r or {}).get(key, float("nan"))
                dv = (v - rv) if (np.isfinite(v) and np.isfinite(rv)) \
                    else float("nan")
                bio_rows.append((f"{key} — {meanings[key]}", v, rv, dv))
            ker_med, _ker_mad, ker_flag, ker_tot = bio.keratin_flags(
                wn_p, self._pred_spectra)
            self.r_bio_table.setRowCount(len(bio_rows) + 1)
            for r_i, (name, v, rv, dv) in enumerate(bio_rows):
                for c_i, txt in enumerate((
                        name,
                        f"{v:.3f}" if np.isfinite(v) else "–",
                        f"{rv:.3f}" if np.isfinite(rv) else "–",
                        f"{dv:+.3f}" if np.isfinite(dv) else "–")):
                    it = QtWidgets.QTableWidgetItem(txt)
                    if c_i:
                        it.setTextAlignment(ALIGN_CENTER)
                    if c_i == 3 and np.isfinite(dv) and dv != 0:
                        it.setForeground(qc.QtGui.QColor(
                            "#b91c1c" if dv > 0 else "#2563eb"))
                        fnt = it.font()
                        fnt.setBold(True)
                        it.setFont(fnt)
                    self.r_bio_table.setItem(r_i, c_i, it)
            ker_txt = (f"{ker_flag} of {ker_tot} flagged"
                       if ker_tot else "–")
            for c_i, txt in enumerate(("Keratin index (site-effect guard)",
                                       f"{ker_med:.3f}"
                                       if np.isfinite(ker_med) else "–",
                                       "", ker_txt)):
                it = QtWidgets.QTableWidgetItem(txt)
                if c_i:
                    it.setTextAlignment(ALIGN_CENTER)
                self.r_bio_table.setItem(len(bio_rows), c_i, it)
            hint = ("Red = higher than the patient's own normal, blue = "
                    "lower. Literature: nucleic/protein and amide I/III "
                    "rise in tumor, collagen/protein falls.")
            if ref_r is None:
                hint = ("No paired reference — absolute marker values. "
                        + hint)
            if ker_flag:
                hint += (f" {ker_flag} keratin-high spectrum(s): the "
                         "site (keratinised vs not) may explain part of "
                         "the difference — interpret with care.")
            self.r_bio_hint.setText(hint)
        else:
            self.r_bio_card.setVisible(False)

        # ---- plain-language reading -----------------------------------------
        self.r_reading.setText(self._result_reading(pos))
        self.render_locked_result()

    def _result_reading(self, pos: str | None) -> str:
        w = self.winner
        if w is None:
            return ("Train a model first — the reading summarizes how "
                    "trustworthy the predictions are.")
        sens = w.macro.get("sens", (None,))[0]
        spec = w.macro.get("spec", (None,))[0]
        parts = []
        if sens is not None and spec is not None:
            parts.append(f"The winning model (<b>{w.name}</b>) catches "
                         f"<b>{sens:.0%}</b> of positive cases "
                         "(sensitivity) and correctly clears "
                         f"<b>{spec:.0%}</b> of negative cases "
                         "(specificity) under patient-grouped "
                         "cross-validation.")
        if self._paired_mode:
            parts.append("<b>Margin mode (paired normal reference)</b>: "
                         "every spectrum is classified by its deviation "
                         "from the same patient's normal tissue — the "
                         "same principle as intraoperative margin "
                         "assessment with fiber-optic Raman (~91% "
                         "accuracy in 2025 in-vivo work). New "
                         "predictions need a normal reference from the "
                         "same patient.")
        if (sens is not None and spec is not None):
            ppv30, npv30 = clin.ppv_npv(sens, spec, 0.30)
            ppv05, _ = clin.ppv_npv(sens, spec, 0.05)
            parts.append(
                "What a positive call is worth depends on where you "
                "stand: at a lesion-clinic prevalence (~30%) a positive "
                f"is right about <b>{ppv30:.0%}</b> of the time (PPV); "
                f"in low-prevalence screening (~5%) only <b>{ppv05:.0%}"
                "</b>. Negative calls stay reliable in both "
                f"(<b>{npv30:.0%}</b> NPV at 30%).")
        if (len(w.classes) == 2 and getattr(w, "oof_proba", None) is not None
                and getattr(w, "y_true_encoded", None) is not None):
            valid = ~np.isnan(w.oof_proba[:, 1])
            if valid.any():
                p = w.oof_proba[valid, 1]
                yv = (w.y_true_encoded[valid] == 1).astype(float)
                brier = float(np.mean((p - yv) ** 2))
                parts.append(f"Calibration (Brier score): <b>{brier:.3f}</b> "
                             "— 0 is perfect, 0.25 is chance; the closer "
                             "the probabilities are to reality, the more "
                             "you can trust a 0.8 vs a 0.55.")
        if self._pred_rows and pos:
            n = len(self._pred_rows)
            k = sum(1 for _f, c, _p in self._pred_rows if c == pos)
            if k:
                parts.append(f"On the {n} spectra you just classified, "
                             f"<b>{k} were called {pos}</b>. Spectra near "
                             "the decision threshold (probability 0.4-0.6) "
                             "are the least certain — check them "
                             "individually.")
            else:
                parts.append(f"On the {n} spectra you just classified, "
                             "<b>none were called positive</b>.")
        n_eval_pat = (len(set(w.groups)) if getattr(w, "groups", None)
                      else None)
        parts.append("These numbers come from patient-grouped "
                     "cross-validation"
                     + (f" on {n_eval_pat} patients" if n_eval_pat else "")
                     + "; expect larger "
                     "variation on new patients. This is research "
                     "triage support for trained clinicians — not a "
                     "medical diagnosis; histopathology remains the "
                     "reference standard.")
        # is the winner significantly better than the runner-up?
        try:
            ok = [r for r in (self.results or [])
                  if r.error is None and getattr(r, "oof_proba", None)
                  is not None and r.name != w.name]
            if ok and w.oof_proba is not None:
                runner = max(ok, key=lambda r: r.macro_f1())
                valid = (~(np.isnan(w.oof_proba).any(axis=1))
                         & ~(np.isnan(runner.oof_proba).any(axis=1)))
                pa = np.argmax(w.oof_proba[valid], axis=1)
                pb = np.argmax(runner.oof_proba[valid], axis=1)
                _b, _c, p = modeling.mcnemar_test(
                    w.y_true_encoded[valid], pa, pb)
                verdict = ("statistically significant"
                           if p < 0.05 else "NOT statistically "
                           "significant — treat them as equal")
                parts.append(f"Winner vs runner-up ({runner.name}): "
                             f"McNemar p = {p:.2f} ({verdict}).")
        except Exception:
            pass
        return "<br><br>".join(parts)

    def save_result_figures(self):
        """Export the three Result-page charts as 200-dpi PNGs."""
        if not self._pred_rows and self.winner is None:
            QtWidgets.QMessageBox.information(
                self, "Nothing to save yet",
                "Train or predict first — then the charts can be saved.")
            return
        start = self.settings.get("last_figure_dir",
                                  os.path.dirname(APP_DIR))
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder for the result figures", start)
        if not folder:
            return
        self.settings["last_figure_dir"] = folder
        uh.save_settings(self.settings)
        saved = []
        for canvas, name in ((self.r_cm_canvas, "result_confusion_matrix"),
                             (self.r_roc_canvas, "result_validation"),
                             (self.dist_canvas, "result_distribution"),
                             (self.r_spec_canvas,
                              "result_predicted_spectra")):
            try:
                path = os.path.join(folder, f"{name}.png")
                canvas.figure.savefig(path, dpi=200, bbox_inches="tight")
                saved.append(path)
            except Exception as exc:
                self.log(f"Saving {name} failed: {exc}")
        if saved:
            self.log(f"Saved {len(saved)} result figures to {folder}")
            self.statusBar().showMessage(
                f"Saved {len(saved)} figures to {folder}")

    def save_result_report(self):
        """Write everything on the Result page to result_report.txt."""
        from datetime import datetime
        w = self.winner
        lines = ["Raman Classifier — final result report",
                 f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
                 "=" * 60]
        if self.spectra:
            n_sub = len(set(self.groups)) if self.groups else 0
            lines.append(f"Dataset: {len(self.spectra)} spectra, "
                         f"{len(set(l for l in self.labels if l))} classes"
                         + (f", {n_sub} patients" if n_sub else ""))
        try:
            lines.append(f"Preprocessing: {self.read_params().validate()}")
        except Exception:
            pass
        if w is not None:
            lines.append("")
            lines.append(f"Winning model: {w.name}")
            lines.append(f"  sensitivity {w.macro.get('sens', (0,))[0]:.3f}"
                         f" · specificity "
                         f"{w.macro.get('spec', (0,))[0]:.3f}"
                         f" · macro-F1 {w.macro_f1():.3f}")
            if w.threshold is not None:
                lines.append(f"  decision threshold {w.threshold:.3f}")
            if self._region_bands:
                tops = ", ".join(f"{b[0]:.0f} cm-1 {b[2]}".strip()
                                 for b in self._region_bands[:3])
                lines.append(f"  top discriminative bands: {tops}")
            if self.results:
                lines.append("")
                lines.append("Model comparison (macro-F1):")
                for r in sorted((x for x in self.results
                                 if x.error is None),
                                key=lambda x: -x.macro_f1()):
                    lines.append(f"  {r.name:<32} {r.macro_f1():.3f}")
        lines.append("")
        if self._pred_rows:
            pos = self._positive_class()
            counts: dict[str, int] = {}
            for _f, c, _p in self._pred_rows:
                counts[c] = counts.get(c, 0) + 1
            lines.append(f"Predictions ({plural(len(self._pred_rows), 'spectrum')}):")
            for c, k in sorted(counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {c}: {k}")
            lo_r, hi_r = self._op_points_now()
            if (pos and len(self._pred_probs) == len(self._pred_rows)):
                ps = [pr.get(pos) for pr in self._pred_probs]
                ps = [x for x in ps if x is not None]
                if ps:
                    mean_p = float(np.mean(ps))
                    tier = ("HIGH" if mean_p >= 0.66 else
                            "MODERATE" if mean_p >= 0.4 else "LOW")
                    thr = (f" (threshold "
                           f"{self.winner.threshold:.3f})"
                           if (self.winner is not None
                               and self.winner.threshold is not None) else "")
                    lines.append(f"  mean P({pos}) = {mean_p:.3f} — "
                                 f"{tier} signal{thr}")
                if lo_r is not None and hi_r is not None:
                    tiers = [clin.triage(pr.get(pos, float("nan")), lo_r,
                                         hi_r) for pr in self._pred_probs]
                    lines.append("  triage: "
                                 + " / ".join(
                                     f"{t} {tiers.count(t)}"
                                     for t in ("POSITIVE", "INDETERMINATE",
                                               "NEGATIVE")))
            lines.append("")
            lines.append(f"{'file':<40} {'class':<12} {'prob':>6}  triage")
            for i, (f, c, p) in enumerate(self._pred_rows):
                tier = (clin.triage(self._pred_probs[i].get(
                    pos, float("nan")), lo_r, hi_r)
                        if (pos and lo_r is not None and hi_r is not None
                            and len(self._pred_probs)
                            == len(self._pred_rows)) else "-")
                lines.append(f"{f:<40} {c:<12} {p:>6.3f}  {tier}")
        else:
            lines.append("Predictions: none yet")
        lines.append("")
        lines.append(self._result_reading(self._positive_class()).
                     replace("<b>", "").replace("</b>", "").
                     replace("<br><br>", "\n\n"))
        # honest literature context (meta-analyses, spectrum-level splits)
        if w is not None:
            lines.extend([
                "", "Compared with the published literature",
                "-" * 60,
                f"{'Test (validation method)':<38} {'sens':>6} {'spec':>6}",
                f"{'This study (patient-grouped CV)':<38} "
                f"{w.macro.get('sens', (0,))[0]:>6.2f} "
                f"{w.macro.get('spec', (0,))[0]:>6.2f}",
                f"{'Han 2022 meta (13 studies)':<38} {'0.89':>6} "
                f"{'0.84':>6}",
                f"{'2025 meta-analysis (OSCC subgroup)':<38} {'0.89':>6} "
                f"{'0.91':>6}",
                f"{'Purohit 2026 review (pooled)':<38} {'0.90':>6} "
                f"{'0.89':>6}",
                f"{'VELscope autofluorescence (typical)':<38} {'~0.84':>6} "
                f"{'~0.45':>6}",
                f"{'Toluidine blue (typical)':<38} {'~0.63':>6} "
                f"{'~0.83':>6}",
                "",
                "Caveat: most published Raman studies validate with "
                "spectrum-level random splits, which lets spectra from "
                "the SAME patient appear in both train and test and "
                "optimistically inflates the numbers. This project "
                "evaluates with PATIENT-GROUPED cross-validation — a "
                "stricter standard; the two kinds of numbers are not "
                "directly comparable. Adjunct devices (VELscope, "
                "toluidine blue) are shown for clinical context.",
            ])
            # predictive values at plausible prevalence
            sens_r = w.macro.get("sens", (None,))[0]
            spec_r = w.macro.get("spec", (None,))[0]
            if sens_r is not None and spec_r is not None:
                lines.extend([
                    "", "Predictive values by clinical setting",
                    "-" * 60,
                    f"{'setting':<20} {'prevalence':>10} {'PPV':>6} "
                    f"{'NPV':>6}"])
                for setting, prev in clin.PREVALENCE_SCENARIOS:
                    ppv_r, npv_r = clin.ppv_npv(sens_r, spec_r, prev)
                    lines.append(f"{setting:<20} {prev:>10.0%} "
                                 f"{ppv_r:>6.0%} {npv_r:>6.0%}")
                lines.append("PPV/NPV move strongly with prevalence "
                             "(Bayes) — read them against the setting "
                             "you work in.")
            # TRIPOD+AI-style reporting section
            n_pat = len(set(self.groups)) if self.groups else 0
            tri = ["", "Reporting (TRIPOD+AI style)", "-" * 60,
                   f"Participants: {len(self.spectra)} spectra"
                   + (f" from {n_pat} patients (paired design)"
                      if n_pat else " (flat layout)"),
                   "Outcomes: class labels from the top-level folders "
                   "(histopathology-confirmed tissue sites)",
                   "Analysis: patient-grouped CV "
                   "(StratifiedGroupKFold); repeated x3 when enabled; "
                   "preprocessing re-chosen inside every fold for the "
                   "nested honest estimate — no patient leaks between "
                   "train and test (PROBAST+AI analysis domain)",
                   "Intended use: research triage support for trained "
                   "clinicians; not a diagnosis — histopathology is "
                   "the reference standard",
                   "Limitations: single centre; no external validation; "
                   "spectrum-level sample sizes overstate statistical "
                   "power; anatomical site, keratinisation and tobacco "
                   "history are not recorded and may confound"]
            lines.extend(tri)
        # ---- deep evaluation / clinical consequences --------------------
        deep = []
        if self._locked_result:
            lk = self._locked_result
            extra = (f", sens {lk['sens']:.3f}, spec {lk['spec']:.3f}, "
                     f"AUC {lk['auc']:.3f} (CI "
                     f"{lk['auc_ci'][0]:.2f}-{lk['auc_ci'][1]:.2f})"
                     if "sens" in lk else "")
            deep.append(
                f"Locked test set (FINAL, {lk['when']}): {lk['n_test_spectra']} "
                f"spectra / {lk['n_test_patients']} unseen patients — "
                f"macro-F1 {lk['f1']:.3f}{extra}")
        if self._lopo_result:
            lo_ = self._lopo_result
            deep.append(
                f"Leave-one-patient-out ({lo_['n_patients']} patients): "
                f"macro-F1 {lo_['f1']:.3f}"
                + (f", AUC {lo_['auc']:.3f}"
                   if np.isfinite(lo_["auc"]) else "")
                + f", mean per-patient accuracy "
                  f"{np.mean(lo_['accs']):.3f}")
        if self._seed_result:
            deep.append(f"Seed stability (5 seeds): macro-F1 "
                        f"{np.mean(self._seed_result):.3f} ± "
                        f"{np.std(self._seed_result):.3f}")
        if self._noise_result:
            deep.append("Noise robustness: " + " -> ".join(
                f"{int(l * 100)}%:{f:.3f}"
                for l, f in self._noise_result))
        if self._friedman_result and self._friedman_result.get("ok"):
            fr_ = self._friedman_result
            deep.append(
                f"Friedman test: p={fr_['p']:.3f} — differences "
                + ("significant" if fr_["significant"]
                   else "NOT significant (models statistically tie)")
                + f"; best by average rank: {fr_['best']}")
        if deep:
            lines += ["", "Deep evaluation", "-" * 60] + \
                [f"  {d}" for d in deep]
        # per-patient out-of-fold performance (worst first)
        if (self.winner is not None
                and getattr(self.winner, "oof_proba", None) is not None
                and self._lc_data is not None and self._lc_data[2]):
            try:
                rows_pp = sstats.per_patient_rollup(
                    self._lc_data[1], self._lc_data[2],
                    self.winner.oof_proba,
                    list(self.winner.classes), pos_idx=1)
                hard = [r for r in rows_pp if r[5]]
                lines += ["", "Per-patient performance (out-of-fold)",
                          "-" * 60,
                          "  worst 5: " + ", ".join(
                              f"{r[0]} {r[2]:.0%} (n={r[1]})"
                              for r in rows_pp[:5])]
                if hard:
                    lines.append(
                        "  HARD patients (<50% spectra correct): "
                        + ", ".join(f"{r[0]} {r[2]:.0%}" for r in hard))
            except Exception:
                pass
        # FDR-corrected band statistics
        if self._band_stats:
            lines += ["", "Band statistics (patient-paired, BH-FDR)",
                      "-" * 60,
                      f"{'band':>6} {'molecule':<16} {'median Δ':>9} "
                      f"{'p(FDR)':>8}  significant"]
            for center, mol, _assign, _dir, med, p_fdr, sig in \
                    self._band_stats:
                lines.append(f"{center:>6.0f} {mol:<16} {med:>+9.3f} "
                             f"{p_fdr:>8.4f}  {'*' if sig else ''}")
        path = os.path.join(APP_DIR, "result_report.txt")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            self.log(f"Result report saved: {path}")
            self.statusBar().showMessage(f"Result report saved: {path}")
        except Exception as exc:
            self.friendly_error("Saving the report failed", exc)

    def save_result_report_html(self):
        """One self-contained, printable HTML report with the figures."""
        import base64
        import io
        from datetime import datetime

        w = self.winner
        if w is None and not self._pred_rows:
            QtWidgets.QMessageBox.information(
                self, "Nothing to save yet",
                "Train or predict first — then the report can be built.")
            return

        def fig_b64(canvas) -> str:
            try:
                buf = io.BytesIO()
                canvas.figure.savefig(buf, format="png", dpi=150,
                                      bbox_inches="tight")
                return ("data:image/png;base64,"
                        + base64.b64encode(buf.getvalue()).decode())
            except Exception:
                return ""

        def esc(v) -> str:
            return (str(v).replace("&", "&amp;")
                    .replace("<", "&lt;").replace(">", "&gt;"))

        parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>Raman Classifier — final result report</title>",
            "<style>",
            "body{font-family:'Segoe UI',system-ui,sans-serif;max-width:"
            "960px;margin:24px auto;color:#1e293b;line-height:1.45}",
            "h1{font-size:1.5rem;border-bottom:3px solid #2563eb;"
            "padding-bottom:8px}h2{font-size:1.1rem;color:#1e3a8a;"
            "margin-top:28px}table{border-collapse:collapse;width:100%;"
            "font-size:.9rem;margin:10px 0}",
            "th,td{border:1px solid #e2e8f0;padding:6px 9px;"
            "text-align:left}th{background:#f1f5f9}",
            "img{max-width:100%;border:1px solid #e2e8f0;margin:6px 0;"
            "border-radius:6px}.muted{color:#64748b;font-size:.85rem}",
            ".warn{background:#fef9c3;border-left:4px solid #b45309;"
            "padding:8px 12px;border-radius:6px}",
            "@media print{body{margin:8mm}}",
            "</style></head><body>",
            "<h1>Raman Spectra Classifier — final result report</h1>",
            f"<p class='muted'>Generated {datetime.now():%Y-%m-%d %H:%M}"
            " · Raman Classifier (patient-grouped evaluation)</p>",
        ]
        if self.spectra:
            n_pat = len(set(self.groups)) if self.groups else 0
            parts.append(
                "<h2>Dataset</h2><p>"
                + esc(f"{len(self.spectra)} spectra, "
                      f"{len(set(l for l in self.labels if l))} classes"
                      + (f", {n_pat} patients (paired design)"
                         if n_pat else ""))
                + "</p>")
        if w is not None:
            sens = w.macro.get("sens", (0,))[0]
            spec = w.macro.get("spec", (0,))[0]
            parts.append(
                "<h2>Winning model</h2><table><tr><th>Model</th>"
                f"<td><b>{esc(w.name)}</b></td></tr>"
                f"<tr><th>Sensitivity / Specificity / macro-F1</th>"
                f"<td>{sens:.3f} / {spec:.3f} / {w.macro_f1():.3f}"
                "</td></tr>")
            lo_r, hi_r = self._op_points_now()
            if lo_r is not None and hi_r is not None:
                parts.append(
                    "<tr><th>Clinical cut-offs</th><td>"
                    f"rule-out p≤{lo_r:.2f} (sens ≥90%) · rule-in "
                    f"p≥{hi_r:.2f} (spec ≥90%)</td></tr>")
            if (self.bundle or {}).get("calibrator"):
                parts.append("<tr><th>Probabilities</th><td>Platt-"
                             "calibrated (out-of-fold fit)</td></tr>")
            parts.append("</table>")
            # predictive values
            if sens is not None and spec is not None:
                rows_ppv = "".join(
                    f"<tr><td>{esc(s_)}</td><td>{pv_:.0%}</td>"
                    f"<td>{ppv_:.0%}</td><td>{npv_:.0%}</td></tr>"
                    for s_, pv_ in clin.PREVALENCE_SCENARIOS
                    for ppv_, npv_ in [clin.ppv_npv(sens, spec, pv_)])
                parts.append(
                    "<h2>Predictive values by clinical setting</h2>"
                    "<table><tr><th>Setting</th><th>Prevalence</th>"
                    f"<th>PPV</th><th>NPV</th></tr>{rows_ppv}</table>")
            # locked final exam
            if self._locked_result:
                lk = self._locked_result
                extra = (f", sens {lk['sens']:.2f}, spec {lk['spec']:.2f}, "
                         f"AUC {lk['auc']:.3f} (95% CI "
                         f"{lk['auc_ci'][0]:.2f}–{lk['auc_ci'][1]:.2f})"
                         if "sens" in lk else "")
                parts.append(
                    "<h2>Locked test-set evaluation (FINAL)</h2><p>"
                    + esc(f"One-shot on {lk['n_test_spectra']} spectra / "
                          f"{lk['n_test_patients']} unseen patients "
                          f"({lk['when']}): macro-F1 {lk['f1']:.3f}"
                          + extra) + "</p>")
        deep_bits = []
        if self._lopo_result:
            lo_ = self._lopo_result
            deep_bits.append(
                f"Leave-one-patient-out ({lo_['n_patients']} patients): "
                f"macro-F1 {lo_['f1']:.3f}"
                + (f", AUC {lo_['auc']:.3f}"
                   if np.isfinite(lo_["auc"]) else ""))
        if self._seed_result:
            deep_bits.append(f"Seed stability: {np.mean(self._seed_result):.3f}"
                             f" ± {np.std(self._seed_result):.3f} (5 seeds)")
        if self._noise_result:
            deep_bits.append("Noise robustness: " + " → ".join(
                f"{int(l * 100)}%: {f:.3f}"
                for l, f in self._noise_result))
        if self._friedman_result and self._friedman_result.get("ok"):
            fr_ = self._friedman_result
            deep_bits.append(f"Friedman p={fr_['p']:.3f} — differences "
                             + ("significant"
                                if fr_["significant"]
                                else "not significant (models tie)"))
        if deep_bits:
            parts.append("<h2>Deep evaluation</h2><p>"
                         + "<br>".join(esc(b) for b in deep_bits)
                         + "</p>")
        if self._pred_rows:
            pos = self._positive_class()
            lo_r, hi_r = self._op_points_now()
            body = ""
            for i, (f, c, p) in enumerate(self._pred_rows[:300]):
                tier = (clin.triage(self._pred_probs[i].get(
                    pos, float("nan")), lo_r, hi_r)
                        if (pos and lo_r is not None and hi_r is not None
                            and len(self._pred_probs)
                            == len(self._pred_rows)) else "–")
                body += (f"<tr><td>{esc(f)}</td><td>{esc(c)}</td>"
                         f"<td>{p:.3f}</td><td>{esc(tier)}</td></tr>")
            parts.append(
                "<h2>Predictions</h2><table><tr><th>File</th>"
                f"<th>Class</th><th>Probability</th><th>Triage</th>"
                f"</tr>{body}</table>")
        parts.append("<h2>Figures</h2>")
        for canvas, cap in (
                (self.r_cm_canvas, "Confusion matrix (pooled CV folds)"),
                (self.r_roc_canvas, "ROC / per-class metrics"),
                (self.r_cal_canvas, "Calibration"),
                (self.r_dca_canvas, "Decision curve (net benefit)"),
                (self.dist_canvas, "Prediction distribution"),
                (self.r_spec_canvas, "Predicted spectra")):
            b64 = fig_b64(canvas)
            if b64:
                parts.append(f"<p class='muted'>{esc(cap)}</p>"
                             f"<img src='{b64}' alt='{esc(cap)}'>")
        parts.append(
            "<h2>Context</h2><div class='warn'><b>Intended use:</b> "
            "research triage support for trained clinicians — not a "
            "medical diagnosis; histopathology remains the reference "
            "standard.</div>"
            "<p class='muted'>Evaluated with patient-grouped "
            "cross-validation (no patient leaks); published oral-cancer "
            "Raman meta-analyses (Han 2022; 2025 meta; Purohit 2026) "
            "pool ~0.89–0.90 sensitivity / 0.84–0.91 specificity but "
            "mostly under spectrum-level splits, which inflate them. "
            "Single-centre data; no external validation; PPV/NPV move "
            "with prevalence.</p>")
        parts.append("</body></html>")
        path = os.path.join(APP_DIR, "result_report.html")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(parts))
            self.log(f"HTML report saved: {path}")
            self.statusBar().showMessage(f"HTML report saved: {path}")
        except Exception as exc:
            self.friendly_error("Saving the HTML report failed", exc)

    # ===================================================== navigation
    def go_to(self, idx: int):
        self.stack.setCurrentIndex(idx)

    def go_next(self):
        idx = self.stack.currentIndex()
        if idx < TAB_RESULT:
            self.stack.setCurrentIndex(idx + 1)

    def go_back(self):
        idx = self.stack.currentIndex()
        if idx > TAB_START:
            self.stack.setCurrentIndex(idx - 1)

    def update_footer(self):
        idx = self.stack.currentIndex()
        self.b_back.setVisible(idx > TAB_START)
        if idx < TAB_RESULT:
            self.next_btn.setText(NEXT_LABELS[idx])
            self.next_btn.setVisible(True)
        else:
            self.next_btn.setVisible(False)

    def on_page_changed(self, idx: int):
        self.update_footer()
        if (idx == TAB_PREP and self.spectra
                and self._previewed_for != self._data_key):
            self.preview_preprocess()
        if idx == TAB_TRAIN and self.spectra and self.winner is None:
            self.statusBar().showMessage("Ready to train — press "
                                         "“Start training”.")
        if idx == TAB_RESULT:
            self.render_result_page()

    # ======================================================= start actions
    def act_load_folder(self):
        self.go_to(TAB_DATA)
        self.browse_folder()

    def act_demo(self):
        self.go_to(TAB_DATA)
        self.generate_demo(ask=True)

    def act_predict(self):
        self.go_to(TAB_PRED)
        if self.bundle is None:
            self.browse_model()

    def quick_start(self):
        """One-click: ensure demo data → train."""
        if not self.spectra:
            source = dataset.find_default_source_spectrum()
            if not source:
                QtWidgets.QMessageBox.information(
                    self, "One-click demo",
                    "I could not find a real spectrum to derive the demo "
                    "from.\nPlace a .txt spectrum next to the app folder "
                    "(e.g. D:\\BARC) or load your own folder instead.")
                self.go_to(TAB_DATA)
                return
            try:
                busy(True)
                out = dataset.generate_demo_data(source, DEMO_DIR,
                                                 n_per_class=15)
            except Exception as exc:
                busy(False)
                self.friendly_error("Demo data generation failed", exc)
                return
            busy(False)
            self.folder_edit.setText(out)
            self.load_folder(out)
            self.log("One-click demo: dataset generated.")
        self.go_to(TAB_TRAIN)
        self.start_training()

    # ================================================================ Data
    def browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with spectra files",
            os.path.dirname(APP_DIR))
        if folder:
            self.folder_edit.setText(folder)
            self.load_folder(folder)

    def reload_folder(self):
        if self._source_folder and os.path.isdir(self._source_folder):
            self.load_folder(self._source_folder)
        else:
            self.go_to(TAB_DATA)

    def generate_demo(self, ask: bool = True):
        source = dataset.find_default_source_spectrum()
        if not source:
            source, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select a REAL spectrum to derive demo classes from",
                os.path.dirname(APP_DIR), "Text (*.txt *.dat *.csv)")
            if not source:
                return
        n = 15
        if ask:
            n, ok = QtWidgets.QInputDialog.getInt(
                self, "Demo dataset", "Spectra per class:", 15, 5, 100, 1)
            if not ok:
                return
        try:
            busy(True)
            out = dataset.generate_demo_data(source, DEMO_DIR, n_per_class=n)
        except Exception as exc:
            busy(False)
            self.friendly_error("Demo generation failed", exc)
            return
        busy(False)
        self.log(f"Demo dataset generated in {out} (3 classes C1/C5/C8, "
                 f"{n} spectra each, derived from "
                 f"{os.path.basename(source)}).")
        self.folder_edit.setText(out)
        self.load_folder(out)

    def load_folder(self, folder: str, quiet: bool = False):
        import clinical_data
        if clinical_data.is_clinical_layout(folder):
            try:
                cd = clinical_data.load_clinical_dataset(folder)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Could not read clinical data",
                    f"{folder}\n\n{exc}")
                return
            rep = cd.report
            self.log(
                f"Clinical dataset: {rep['n_spectra']} spectra / "
                f"{rep['n_subjects']} subjects "
                f"(kept {rep['class_counts']}); excluded "
                f"{len(rep['references_excluded'])} reference spectra, "
                f"{len(rep['duplicates_dropped'])} duplicate copies, "
                f"{len(rep['cross_class_dropped'])} cross-class identical "
                f"files")
            try:
                rep_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "data_report.txt")
                clinical_data.write_report(rep_path, rep)
                self.log(f"Data hygiene report: {rep_path}")
            except Exception:
                pass
            self._install_dataset(cd.spectra, source_folder=folder,
                                  quiet=quiet, groups=cd.groups,
                                  spike_flags=cd.flagged)
            return
        spectra = dataset.load_folder(folder)
        if not spectra:
            QtWidgets.QMessageBox.warning(
                self, "No data found",
                "No loadable .txt/.dat/.csv spectra were found in:\n"
                f"{folder}\n\nA spectrum file has 2 columns: "
                "wavenumber and intensity.")
            return
        self._install_dataset(spectra, source_folder=folder, quiet=quiet)

    def _install_dataset(self, spectra: list, source_folder: str,
                         quiet: bool = False, groups: list[str] | None = None,
                         spike_flags: list[bool] | None = None):
        try:
            busy(True)
            self.grid = dataset.common_grid(spectra)
            self.X_raw, _ = dataset.to_matrix(spectra, self.grid)
            busy(False)
            self.spectra = spectra
            self.labels = [s.label for s in spectra]
            self.groups = list(groups) if groups is not None else None
            self.spike_flags = (list(spike_flags)
                                if spike_flags is not None else None)
            self.chk_exclude_flagged.setEnabled(
                self.spike_flags is not None
                and any(self.spike_flags))
            self.chk_avg_replicates.setEnabled(self.groups is not None)
            self._source_folder = source_folder
            self.settings["folder"] = source_folder
            shown = source_folder
            self.folder_edit.setText(shown)
            self.folder_edit.setToolTip(shown)
            self._data_key = f"{source_folder}|{len(spectra)}|{self.X_raw.shape}"
            self._proc_cache = None
            self._previewed_for = None
            self._fill_table()
            self._draw_distribution()
            n_unlabeled = sum(1 for l in self.labels if not l)
            n_classes = len(set(l for l in self.labels if l))
            subj = (f" | {len(set(self.groups))} patients — CV grouped "
                    f"by patient" if self.groups else "")
            self.count_label.setText(
                f"{len(spectra)} spectra | {n_classes} classes | "
                f"grid {self.grid.min():.0f}–{self.grid.max():.0f} "
                f"cm⁻¹ ({len(self.grid)} pts)" + subj
                + (f" | {n_unlabeled} unlabeled — fix the Class column"
                   if n_unlabeled else ""))
            self.log(f"Loaded {len(spectra)} spectra from {shown}; "
                     f"classes: {sorted(set(l for l in self.labels if l))}")
            self.b_train.setEnabled(True)
            self.train_status.setText(
                f"{len(spectra)} spectra ready with the current "
                "preprocessing — press Start training.")
            uh.save_settings(self.settings)
            if n_classes < 2:
                self.log("WARNING: fewer than 2 classes detected — "
                         "training needs ≥ 2 classes. Fix labels in the "
                         "table (e.g. files named *_C8_*.txt → class C8).")
                if not quiet:
                    QtWidgets.QMessageBox.warning(
                        self, "Only one class",
                        "All spectra have the same class label. Training "
                        "needs at least 2 classes.\n\nCheck the Class "
                        "column — filenames like P01_cAg_785_C8_3.txt "
                        "give class C8 automatically.")
            else:
                self.statusBar().showMessage(
                    f"Data ready: {len(spectra)} spectra, {n_classes} "
                    "classes — next: Preprocess")
            self.update_welcome()
            self.refresh_nav()
        except Exception as exc:
            busy(False)
            self.log("Failed to load data:\n" + traceback.format_exc())
            if not quiet:
                self.friendly_error("Could not load that data", exc)

    def _fill_table(self):
        self._loading_table = True
        self.table.setRowCount(len(self.spectra))
        for i, s in enumerate(self.spectra):
            it_name = QtWidgets.QTableWidgetItem(s.name)
            it_name.setFlags(IS_SELECTABLE | IS_ENABLED)
            it_name.setToolTip(TIPS["class_col"])
            self.table.setItem(i, 0, it_name)
            it_pat = QtWidgets.QTableWidgetItem(
                self.groups[i] if self.groups else "")
            it_pat.setFlags(IS_SELECTABLE | IS_ENABLED)
            it_pat.setToolTip("Patient / subject this spectrum belongs to "
                              "(from the folder name; CV keeps all spectra "
                              "of one patient together)")
            self.table.setItem(i, 1, it_pat)
            it_cls = QtWidgets.QTableWidgetItem(s.label)
            it_cls.setFlags(IS_SELECTABLE | IS_ENABLED | IS_EDITABLE)
            it_cls.setToolTip(TIPS["class_col"])
            if not s.label:
                it_cls.setForeground(QtGui_red())
            self.table.setItem(i, 2, it_cls)
            for col, val in ((3, str(len(s))), (4, f"{s.wavenumbers.min():.2f}"),
                             (5, f"{s.wavenumbers.max():.2f}")):
                it = QtWidgets.QTableWidgetItem(val)
                it.setFlags(IS_SELECTABLE | IS_ENABLED)
                self.table.setItem(i, col, it)
            bad = bool(self.spike_flags and self.spike_flags[i])
            it_q = QtWidgets.QTableWidgetItem("spiked" if bad else "ok")
            it_q.setFlags(IS_SELECTABLE | IS_ENABLED)
            it_q.setTextAlignment(ALIGN_CENTER)
            if bad:
                it_q.setForeground(QtGui_red())
            it_q.setToolTip(
                "Spike score flags cosmic-ray artifacts (very sharp "
                "jumps). Spiked spectra can be excluded from training on "
                "the Train page.")
            self.table.setItem(i, 6, it_q)
        self._loading_table = False

    def on_cell_changed(self, row: int, col: int):
        if self._loading_table or col != 2:
            return
        item = self.table.item(row, 2)
        txt = item.text().strip()
        self.labels[row] = txt
        item.setForeground(QtGui_black() if txt else QtGui_red())
        self._draw_distribution()
        self.update_welcome()

    def _selected_indices(self) -> list[int]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return rows if rows else [0]

    def plot_selected(self):
        if not self.spectra:
            return
        idx = self._selected_indices()
        to_plot = [(self.grid, self.X_raw[i],
                    f"{self.spectra[i].name} [{self.labels[i] or '?'}]")
                   for i in idx[:10]]
        ax = plotting.clear(self.data_canvas)
        plotting.plot_spectra(ax, to_plot,
                              title=f"Raw spectra (n={len(to_plot)})")
        self.data_canvas.draw_idle()

    def plot_class_means(self):
        if not self.spectra:
            return
        classes = sorted(set(l for l in self.labels if l))
        if not classes:
            self.log("No labeled spectra to average.")
            return
        ax = plotting.clear(self.data_canvas)
        to_plot = []
        for c in classes:
            mask = [i for i, l in enumerate(self.labels) if l == c]
            mean = self.X_raw[mask].mean(axis=0)
            to_plot.append((self.grid, mean, f"{c} (n={len(mask)})"))
        plotting.plot_spectra(ax, to_plot, title="Class mean spectra (raw)")
        self.data_canvas.draw_idle()

    def _draw_distribution(self):
        ax = plotting.clear(self.data_canvas)
        plotting.plot_class_distribution(
            ax, [l if l else "unlabeled" for l in self.labels])
        self.data_canvas.draw_idle()

    # ========================================================= Preprocess
    def read_params(self) -> PreprocessParams:
        baseline = self.combo_baseline.currentText()
        if not preprocessing.HAS_PYBASELINES:
            baseline = "als"
        return PreprocessParams(
            crop_min=self.spin_crop_min.value(),
            crop_max=self.spin_crop_max.value(),
            despike=self.chk_despike.isChecked(),
            despike_z=self.spin_despike_z.value(),
            wavelet=self.chk_wavelet.isChecked(),
            wavelet_name=self.combo_wavelet.currentText(),
            wavelet_level=self.spin_level.value(),
            sg_window=self.spin_sg_window.value(),
            sg_poly=self.spin_sg_poly.value(),
            sg_deriv=self.combo_deriv.currentIndex(),
            baseline_method=baseline,
            als_lambda=float(10 ** self.spin_lambda_exp.value()),
            als_p=self.spin_als_p.value(),
            als_niter=self.spin_als_iter.value(),
            norm=self.combo_norm.currentText(),
        )

    def _apply_params(self, p: dict | None):
        if not p:
            return
        fields = set(PreprocessParams.__dataclass_fields__)
        merged = {**asdict(PreprocessParams()),
                  **{k: v for k, v in p.items() if k in fields}}
        pr = PreprocessParams(**merged)
        self.spin_crop_min.setValue(pr.crop_min)
        self.spin_crop_max.setValue(pr.crop_max)
        self.chk_despike.setChecked(bool(pr.despike))
        self.spin_despike_z.setValue(float(pr.despike_z))
        self.combo_baseline.setCurrentText(
            "arpls" if (pr.baseline_method == "arpls"
                        and preprocessing.HAS_PYBASELINES) else "als")
        self.chk_wavelet.setChecked(pr.wavelet and HAS_PYWT)
        idx = self.combo_wavelet.findText(pr.wavelet_name)
        if idx >= 0:
            self.combo_wavelet.setCurrentIndex(idx)
        self.spin_level.setValue(pr.wavelet_level)
        self.spin_sg_window.setValue(pr.sg_window)
        self.spin_sg_poly.setValue(pr.sg_poly)
        self.combo_deriv.setCurrentIndex(int(pr.sg_deriv))
        try:
            self.spin_lambda_exp.setValue(
                int(max(2, min(9, round(log10(max(pr.als_lambda, 1e-2)))))))
        except Exception:
            pass
        self.spin_als_p.setValue(pr.als_p)
        self.spin_als_iter.setValue(pr.als_niter)
        self.combo_norm.setCurrentText(pr.norm)

    def get_processed_X(self) -> np.ndarray:
        params = self.read_params().validate()
        key = f"{self._data_key}|{params}"
        if self._proc_cache and self._proc_cache[0] == key:
            return self._proc_cache[1]
        busy(True)
        try:
            X = preprocessing.preprocess_matrix(self.X_raw, params,
                                                wn=self.grid)
        finally:
            busy(False)
        self._proc_cache = (key, X)
        return X

    def preview_preprocess(self):
        if not self.spectra:
            self.log("Load data first (Start page).")
            self.statusBar().showMessage("Load data first — nothing to "
                                         "preview yet.")
            return
        i = self._selected_indices()[0]
        params = self.read_params().validate()
        busy(True)
        try:
            m = preprocessing.crop_mask(self.grid, params)
            wn = self.grid[m]
            raw = self.X_raw[i][m]
            proc = preprocessing.preprocess_spectrum(raw, params)
        finally:
            busy(False)
        fig = self.prep_canvas.figure
        fig.clf()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        plotting.plot_preprocess_preview(
            ax1, ax2, wn, raw, proc,
            title=f"{self.spectra[i].name} [{self.labels[i] or '?'}]")
        self.prep_canvas.draw_idle()
        self._previewed_for = self._data_key
        self.statusBar().showMessage(
            f"Preview of {self.spectra[i].name} — adjust settings or "
            "continue to Train.")

    # ================================================== preprocessing auto-tune
    def run_optimize(self):
        if not self.spectra:
            QtWidgets.QMessageBox.information(
                self, "No data yet", "Load data first (Start page).")
            return
        exclude = None
        if (self.spike_flags is not None
                and self.chk_exclude_flagged.isChecked()):
            exclude = self.spike_flags
        self._opt_btn = self.sender()          # None when called w/o a button
        if self._opt_btn is not None:
            self._opt_btn.setEnabled(False)
        self.optimize_status.setText("Optimizing — this typically takes "
                                     "1-3 minutes…")
        self.log("Preprocessing optimization started "
                 "(grouped CV over crop/deriv/norm combinations)")
        self._opt_worker = OptimizeWorker(
            self.X_raw, self.grid, self.labels, self.groups, exclude)
        self._opt_worker.progress.connect(
            lambda m: self.optimize_status.setText(m))
        self._opt_worker.progress.connect(self.log)
        self._opt_worker.done.connect(self.on_optimize_done)
        self._opt_worker.failed.connect(self.on_optimize_failed)
        self._opt_worker.start()

    def on_optimize_done(self, out: dict):
        best = out["best"]
        self._apply_params(asdict(out["params"]))
        self._proc_cache = None
        rank = "\n".join(
            f"  {i}. {r['label']} — {r['model']} "
            f"F1 {r['mean_f1']:.3f} ± {r['std_f1']:.3f}"
            for i, r in enumerate(out["results"][:5], start=1))
        self.optimize_status.setText(
            f"Best: {best['label']} with {best['model']} "
            f"(macro-F1 {best['mean_f1']:.3f} ± {best['std_f1']:.3f}) — "
            "applied to the parameters.")
        self.log(f"Preprocessing optimized. Top 5:\n{rank}")
        if self._opt_btn is not None:
            self._opt_btn.setEnabled(True)
        try:
            import optimize
            path = optimize.save_best_params(out["params"],
                                             best["mean_f1"],
                                             best["model"])
            self.log(f"Saved best preprocessing: {path}")
        except Exception:
            pass

    def on_optimize_failed(self, tb: str):
        self.optimize_status.setText("Optimization failed — see the log.")
        self.log(f"Optimization failed:\n{tb}")
        QtWidgets.QMessageBox.warning(
            self, "Optimization failed",
            "Preprocessing optimization failed.\n\nTechnical details "
            "are in session.log next to the app.")
        if getattr(self, "_opt_btn", None) is not None:
            self._opt_btn.setEnabled(True)

    # ========================================================= learning curve
    def _run_async(self, label: str, fn, on_done):
        """
        Run fn() on a worker thread; on_done(result) back on the UI
        thread. The triggering button (self.sender()) is disabled while
        running so a double-click cannot queue a second job behind a
        frozen UI. Failures land in a dialog + session.log.
        """
        if (self._analysis_worker is not None
                and self._analysis_worker.isRunning()):
            QtWidgets.QMessageBox.information(
                self, "Busy",
                "Another analysis is still running — wait for it to "
                "finish.")
            return
        btn = self.sender()
        has_btn = isinstance(btn, QtWidgets.QAbstractButton)
        if has_btn:
            btn.setEnabled(False)
        self.train_status.setText(f"{label}…")
        worker = FuncWorker(fn)

        def finish(result):
            self._analysis_worker = None
            if has_btn:
                btn.setEnabled(True)
            on_done(result)

        def fail(tb: str):
            self._analysis_worker = None
            if has_btn:
                btn.setEnabled(True)
            self.train_status.setText(f"{label} failed.")
            self.log(f"{label} failed:\n{tb}")
            QtWidgets.QMessageBox.warning(
                self, f"{label} failed",
                f"{label} failed.\n\nTechnical details are in "
                "session.log next to the app.")

        worker.done.connect(finish)
        worker.failed.connect(fail)
        self._analysis_worker = worker
        worker.start()

    def run_learning_curve(self):
        """Macro-F1 vs number of patients — how much is more data worth?"""
        if self.winner is None or self._lc_data is None:
            QtWidgets.QMessageBox.information(
                self, "Train first",
                "Train a model first — the learning curve uses the "
                "winning pipeline.")
            return
        X, yy, gg = self._lc_data
        if gg is None:
            QtWidgets.QMessageBox.information(
                self, "No patient groups",
                "The learning curve needs patient groups (clinical data "
                "layout).")
            return
        classes = sorted(set(yy))
        ye = [classes.index(v) for v in yy]
        winner_pl, k, seed = (self.winner.pipeline,
                              self.spin_folds.value(),
                              self.spin_seed.value())

        def fn():
            return modeling.learning_curve_by_groups(
                X, ye, gg, winner_pl, k=k, seed=seed)

        def done(res):
            sizes, means, stds = res
            ax = plotting.clear(self.roc_canvas)
            ax.errorbar(sizes, means, yerr=stds, marker="o", ms=5, lw=1.6,
                        capsize=4, color=plotting.COL_MAIN,
                        ecolor=plotting.COL_RAW)
            ax.set_xlabel("patients used")
            ax.set_ylabel("macro-F1 (grouped CV)")
            ax.set_title("Learning curve — the slope says what more "
                         "data is worth")
            ax.set_ylim(0, 1)
            self.roc_canvas.draw_idle()
            verdict = ("still rising — more patients should help"
                       if means[-1] > means[0] + 0.02 else
                       "flat — more of the same patients adds little")
            self.log(f"Learning curve: {len(sizes)} points from "
                     f"{sizes[0]} to {sizes[-1]} patients, "
                     f"F1 {means[0]:.3f} -> "
                     f"{means[-1]:.3f} ({verdict})")
            self.train_status.setText(f"Learning curve done — {verdict}.")

        self._run_async("Computing learning curve", fn, done)

    def run_region_importance(self):
        """WHERE does the model look?  RF importance over the spectrum."""
        if self._lc_data is None or self._lc_wn is None:
            QtWidgets.QMessageBox.information(
                self, "Train first",
                "Train a model first — the region plot uses the "
                "preprocessed training data.")
            return
        X, yy, gg = self._lc_data
        if X.shape[1] != len(self._lc_wn):
            QtWidgets.QMessageBox.warning(
                self, "Preprocessing changed",
                "The preprocessing changed since training — retrain "
                "first so the wavenumber axis matches.")
            return
        self.train_status.setText("Computing region importances…")
        use_shap = False
        try:
            import shap  # noqa: F401
            use_shap = True
        except ImportError:
            pass

        def fn():
            if use_shap:
                try:
                    wn_s, signed_s, bands_s = modeling.region_importance_shap(
                        X, yy, gg, self._lc_wn)
                    return True, wn_s, signed_s, bands_s, ""
                except Exception as exc:
                    # SHAP failed (e.g. non-tree winner) -> Gini fallback
                    wn_g, imp_g, bands_g = modeling.region_importance(
                        X, yy, gg, self._lc_wn)
                    return (False, wn_g, imp_g, bands_g,
                            f"SHAP importance failed ({exc}); using Gini")
            wn_g, imp_g, bands_g = modeling.region_importance(
                X, yy, gg, self._lc_wn)
            return False, wn_g, imp_g, bands_g, ""

        def done(res):
            use_shap_w, wn, signed, bands, note = res
            if note:
                self.log(note)
            # keep the top bands for the Result-page spectra overlay +
            # bundle (center, share, name, sign toward the positive class)
            if use_shap_w:
                self._region_bands = [(float(c), float(sh), str(nm), int(s))
                                      for c, sh, nm, s in bands]
            else:
                self._region_bands = [(float(c), float(sh), "", 0)
                                      for c, _w, sh in bands]
            ax = plotting.clear(self.roc_canvas)
            if use_shap_w:
                # signed SHAP: red pushes toward the positive class,
                # blue away
                mag = np.abs(signed) / (np.abs(signed).max() or 1.0)
                sgn = np.sign(signed)
                ax.fill_between(wn, 0, mag * sgn, where=sgn >= 0,
                                color="#dc2626", alpha=0.45,
                                label=f"toward {sorted(set(yy))[1]}")
                ax.fill_between(wn, 0, mag * sgn, where=sgn < 0,
                                color="#2563eb", alpha=0.45,
                                label=f"toward {sorted(set(yy))[0]}")
                ax.axhline(0, color="#64748b", lw=0.8)
                top_share = max(b[1] for b in bands) or 1.0
                for center, share, name, s in bands[:3]:
                    ax.axvline(center, color="#b45309", lw=1.0, ls="--",
                               alpha=0.8)
                    ax.annotate(
                        f"{center:.0f}{name}\n{share / top_share:.0%}",
                        xy=(center, 0.97 if s >= 0 else -0.97),
                        ha="center",
                        va="top" if s >= 0 else "bottom",
                        fontsize=7.5, color="#b45309")
                ax.set_ylim(-1.15, 1.15)
                ax.set_ylabel("SHAP contribution (signed)")
                ax.set_title("Which spectral regions drive the "
                             "classification (SHAP)")
            else:
                # class-mean spectra (scaled to [0,1]) as context
                for i, cls in enumerate(sorted(set(yy))):
                    m = np.asarray(X)[
                        [j for j, c in enumerate(yy) if c == cls]]
                    mean = m.mean(axis=0)
                    lo, hi = mean.min(), mean.max()
                    scaled = ((mean - lo) / (hi - lo)
                              if hi > lo else mean * 0)
                    ax.plot(wn, scaled, lw=1.0, alpha=0.55,
                            color=plotting.class_color(i),
                            label=f"mean {cls}")
                hi_s = signed / (signed.max() or 1)
                ax.fill_between(wn, 0, hi_s * 0.9, color="#4f46e5",
                                alpha=0.30, label="importance")
                ax.plot(wn, hi_s * 0.9, lw=1.2, color="#4f46e5")
                top_share = max(s for _c, _w, s in bands) or 1.0
                for center, _width, share in bands[:3]:
                    ax.axvline(center, color="#b45309", lw=1.0, ls="--",
                               alpha=0.8)
                    ax.annotate(f"{center:.0f}\n{share / top_share:.0%}",
                                xy=(center, 0.97), ha="center", va="top",
                                fontsize=8, color="#b45309")
                ax.set_ylim(0, 1.05)
                ax.set_title("Which spectral regions drive the "
                             "classification")
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.legend(loc="upper right", fontsize=8)
            self.roc_canvas.draw_idle()
            if use_shap_w:
                tops = ", ".join(f"{c:.0f} cm⁻¹{name} "
                                 f"({'+' if s >= 0 else '-'})"
                                 for c, _sh, name, s in bands[:3])
            else:
                tops = ", ".join(f"{c:.0f} cm⁻¹ ({sh / top_share:.0%})"
                                 for c, _w, sh in bands[:3])
            self.log(f"Top discriminative regions "
                     f"({'SHAP' if use_shap_w else 'Gini'}): {tops}")
            self.train_status.setText(
                f"Region importance done — top: {tops}.")

        self._run_async("Computing region importance", fn, done)

    def run_biochemistry(self):
        """
        What is the signal MADE of — and does the model look at the
        biochemistry the literature says should change in tumor tissue?
        """
        if self._lc_data is None or self._lc_wn is None:
            QtWidgets.QMessageBox.information(
                self, "Train first",
                "Train a model first — the biochemistry panel uses the "
                "preprocessed training data.")
            return
        X, yy, gg = self._lc_data
        wn = np.asarray(self._lc_wn)
        if X.shape[1] != len(wn):
            QtWidgets.QMessageBox.warning(
                self, "Preprocessing changed",
                "The preprocessing changed since training — retrain "
                "first so the wavenumber axis matches.")
            return
        seed = self.spin_seed.value()

        def fn():
            # model bands needed for the plausibility check — compute
            # them here if the user hasn't run region importance yet
            region_bands = self._region_bands
            bands_note = ""
            if not region_bands:
                try:
                    _wn_s, _signed, bands_s = modeling.region_importance_shap(
                        X, yy, gg, wn)
                    region_bands = [(float(c), float(sh), str(nm), int(s))
                                    for c, sh, nm, s in bands_s]
                    bands_note = ("Biochemistry: computed signed SHAP "
                                  "bands (region importance had not been "
                                  "run)")
                except Exception:
                    try:
                        _wn_g, _imp, bands_g = modeling.region_importance(
                            X, yy, gg, wn)
                        region_bands = [(float(c), float(sh), "", 0)
                                        for c, _w, sh in bands_g]
                        bands_note = ("Biochemistry: computed Gini bands "
                                      "(SHAP unavailable)")
                    except Exception:
                        region_bands = []
            class_rows, paired_rows = bio.ratio_table(wn, X, yy, gg)
            _H, labels, W = bio.nmf_components(X, wn, k=5, seed=seed)
            deltas = bio.component_deltas(W, yy, gg)
            ker_med, _mad, ker_flag, ker_tot = bio.keratin_flags(wn, X)
            # FDR-corrected patient-paired band statistics (2 classes
            # with patient groups only)
            band_rows = (sstats.band_stats_paired(X, yy, gg, wn)
                         if (gg and len(set(yy)) == 2) else [])
            plaus_rows, (agree, known) = bio.plausibility(
                region_bands or [])
            return {"region_bands": region_bands, "bands_note": bands_note,
                    "class_rows": class_rows, "paired_rows": paired_rows,
                    "labels": labels, "deltas": deltas,
                    "ker": (ker_med, ker_flag, ker_tot),
                    "band_rows": band_rows,
                    "plaus": (plaus_rows, agree, known)}

        def done(res):
            if res["bands_note"]:
                self.log(res["bands_note"])
            if not self._region_bands:
                self._region_bands = res["region_bands"]
            self._band_stats = res["band_rows"] or None
            class_rows = res["class_rows"]
            paired_rows = res["paired_rows"]
            labels = res["labels"]
            deltas = res["deltas"]
            ker_med, ker_flag, ker_tot = res["ker"]
            plaus_rows, agree, known = res["plaus"]

            # left chart: paired marker deltas (tumor minus OWN normal) —
            # vertical bars: short two-line labels never overlap
            ax = plotting.clear(self.bio_canvas)
            short = {"nucleic/protein": "nuc/protein", "collagen/protein":
                     "coll/protein", "lipid/protein": "lipid/protein",
                     "amide I/III": "amide I/III"}
            expected = {"nucleic/protein": "↑", "collagen/protein": "↓",
                        "lipid/protein": "·", "amide I/III": "↑"}
            if paired_rows:
                keys = [f"{short.get(k, k)}\n(exp {expected.get(k, '·')})"
                        for k, _n, _d in paired_rows]
                vals = [d for _k, _n, d in paired_rows]
                bars = ax.bar(keys, vals,
                              color=["#dc2626" if v >= 0 else "#2563eb"
                                     for v in vals],
                              edgecolor="white", linewidth=0.6, width=0.6)
                ax.bar_label(bars, fmt="%+.2f", fontsize=9,
                             fontweight="bold",
                             color="#334155", padding=4)
                n_pat = max(n for _k, n, _d in paired_rows)
                ax.set_title(f"Marker deltas: tumor − own normal\n"
                             f"({n_pat} patients)", fontsize=10)
            else:
                # no patient grouping -> class means side by side
                classes = sorted(set(yy))
                keys = [short.get(k, k) for k, _m, _means in class_rows]
                x = np.arange(len(keys))
                w_ = 0.8 / max(len(classes), 1)
                for i, c in enumerate(classes):
                    vals = [means.get(c, 0) or 0
                            for _k, _m, means in class_rows]
                    ax.bar(x + i * w_ - 0.4 + w_ / 2, vals, w_,
                           color=plotting.class_color(i), label=c,
                           edgecolor="white", linewidth=0.6)
                ax.set_xticks(x, keys)
                ax.legend(fontsize=8)
                ax.set_title("Marker values per class (no pairing)",
                             fontsize=10)
            ax.axhline(0, color="#64748b", lw=0.8)
            ax.set_ylabel("Δ marker (paired)")
            ax.margins(y=0.18)
            ax.tick_params(axis="x", labelsize=9)

            # right chart: NMF biochemical component shifts (paired)
            ax2 = plotting.clear(self.bio_comp_canvas)
            comp_vals = [d for _c, _n, d in deltas]
            x2 = np.arange(len(labels))
            bars2 = ax2.bar(x2, comp_vals,
                            color=["#dc2626" if v >= 0 else "#2563eb"
                                   for v in comp_vals],
                            edgecolor="white", linewidth=0.6, width=0.6)
            ax2.bar_label(bars2, fmt="%+.3f", fontsize=9,
                          fontweight="bold",
                          color="#334155", padding=4)
            ax2.set_xticks(x2, labels, rotation=25, ha="right",
                           fontsize=8.5)
            ax2.axhline(0, color="#64748b", lw=0.8)
            ax2.set_title("Biochemical components (NMF)\n"
                          "paired deltas, tumor − own normal", fontsize=10)
            ax2.set_ylabel("Δ component weight")
            ax2.margins(y=0.18)
            self.bio_canvas.draw_idle()
            self.bio_comp_canvas.draw_idle()

            # plausibility table (literature reference panel when no model
            # bands exist — the table is never left empty)
            lit_txt = {1: "up in tumor", -1: "down in tumor", 0: "mixed"}
            mod_txt = {1: "toward tumor", -1: "away", 0: "–"}
            if plaus_rows:
                table_rows = plaus_rows
            else:
                table_rows = [(c, mol, assign, direction, 0, "reference")
                              for c, _hw, mol, assign, direction
                              in bio.BANDS]
            self.bio_table.setRowCount(len(table_rows))
            for r, (center, mol, assign, direction, sign, verdict) in \
                    enumerate(table_rows):
                for c, txt in enumerate((
                        f"{center:.0f}", mol, assign,
                        lit_txt.get(direction, "–"),
                        mod_txt.get(sign, "–"), verdict)):
                    it = QtWidgets.QTableWidgetItem(txt)
                    if c != 2:
                        it.setTextAlignment(ALIGN_CENTER)
                    if c == 5:                    # color the verdict
                        col = {"agree": "#15803d", "opposite": "#b91c1c",
                               "unknown-sign": "#b45309",
                               "unassigned": "#64748b",
                               "reference": "#475569"}.get(verdict,
                                                           "#334155")
                        it.setForeground(qc.QtGui.QColor(col))
                        f = it.font()
                        f.setBold(True)
                        it.setFont(f)
                    self.bio_table.setItem(r, c, it)

            bits = []
            if known:
                bits.append(f"plausibility: {agree} of {known} model "
                            "bands match the literature direction")
            elif self._region_bands:
                bits.append("plausibility: no model band fell inside a "
                            "literature window")
            else:
                bits.append("model bands unavailable — literature "
                            "reference panel shown")
            if ker_tot and np.isfinite(ker_med):
                bits.append(f"keratin: {ker_flag} of {ker_tot} spectra "
                            "keratin-high (possible site effect)")
            sig_bands = [r for r in (self._band_stats or []) if r[6]]
            if sig_bands:
                txt = ", ".join(
                    f"{r[0]:.0f} {r[1]} "
                    f"({'up' if r[4] > 0 else 'down'})"
                    for r in sig_bands[:4])
                bits.append(f"FDR-significant bands (p<0.05): {txt}")
            self.bio_label.setText(
                " · ".join(bits) + "\n"
                "Red = higher in tumor, blue = lower · exp ↑/↓ = "
                "literature direction")
            self.train_status.setText("Biochemistry done.")
            self.log("Biochemistry: " + " · ".join(bits))

        self._run_async("Analyzing biochemistry", fn, done)

    def run_honest_check(self):
        """Nested evaluation: preprocessing re-chosen inside each fold."""
        if not self.spectra or self.X_raw is None:
            QtWidgets.QMessageBox.information(
                self, "No data", "Load data first.")
            return
        if self._honest_worker is not None and self._honest_worker.isRunning():
            return
        exclude = (self.spike_flags
                   if (self.spike_flags is not None
                       and self.chk_exclude_flagged.isChecked()) else None)
        self.b_honest_btn = self.sender()      # None when called w/o a button
        if self.b_honest_btn is not None:
            self.b_honest_btn.setEnabled(False)
        self.train_status.setText("Honest (nested) evaluation running — "
                                  "a couple of minutes…")
        self.log("Nested honest evaluation started (preprocessing "
                 "re-chosen inside every fold)")
        self._honest_worker = PipelineWorker(
            self.X_raw, self.grid, self.labels, self.groups, exclude)
        self._honest_worker.progress.connect(
            lambda m: (self.statusBar().showMessage(m),
                       self.train_status.setText(m)))
        self._honest_worker.progress.connect(self.log)
        self._honest_worker.done.connect(self.on_honest_done)
        self._honest_worker.failed.connect(self.on_honest_failed)
        self._honest_worker.start()

    def on_honest_done(self, out: dict):
        if self.b_honest_btn is not None:
            self.b_honest_btn.setEnabled(True)
        optimistic = self.winner.macro_f1() if self.winner else float("nan")
        choices = "; ".join(
            f"fold{c['fold']}: {c['preprocess']}+{c['model']} "
            f"F1 {c['f1']:.2f}" for c in out["fold_choices"])
        self.log(f"Honest (nested) evaluation: macro-F1 "
                 f"{out['mean_f1']:.3f} ± {out['std_f1']:.3f} "
                 f"(optimistic single-choice number: {optimistic:.3f})")
        self.log(f"  per-fold choices: {choices}")
        self._honest_result = out
        self.train_status.setText(
            f"Honest macro-F1 {out['mean_f1']:.3f} ± {out['std_f1']:.3f} "
            f"(vs optimistic {optimistic:.3f}) — see log.")

    def on_honest_failed(self, tb: str):
        if self.b_honest_btn is not None:
            self.b_honest_btn.setEnabled(True)
        self.log(f"Honest evaluation failed:\n{tb}")
        self.train_status.setText("Honest evaluation failed — see log.")
        QtWidgets.QMessageBox.warning(
            self, "Honest evaluation failed",
            "Nested honest evaluation failed.\n\nTechnical details are "
            "in session.log next to the app.")

    def run_locked_eval(self):
        """
        TRIPOD-style final exam: split PATIENTS 70/15/15, refit the
        winner on the training patients only, evaluate ONE-SHOT on the
        untouched test patients.  Honest but noisy — guard against
        re-running until the numbers look good.
        """
        if self.winner is None or self.winner.pipeline is None:
            QtWidgets.QMessageBox.information(
                self, "Train first",
                "Train a model first — the locked evaluation refits the "
                "winner on a patient subset.")
            return
        if not self.groups or len(set(self.groups)) < 7:
            QtWidgets.QMessageBox.information(
                self, "Clinical data needed",
                "The locked split needs at least 7 patients (clinical "
                "folder layout with patient subfolders).")
            return
        if self._lc_data is None:
            QtWidgets.QMessageBox.information(
                self, "Train first",
                "Train first so the feature matrix matches the winner.")
            return
        ans = QtWidgets.QMessageBox.question(
            self, "Run the locked final exam?",
            "This splits the patients 70/15/15, refits the winner on the "
            "training patients and evaluates ONE-SHOT on the untouched "
            "test patients.\n\nIt is meant to be run ONCE per "
            "configuration — re-running until the numbers look good "
            "defeats its purpose. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if ans != QtWidgets.QMessageBox.Yes:
            return
        X, yy, gg = self._lc_data
        groups_l, labels_l = list(self.groups), list(self.labels)
        winner_pl, winner_classes, seed = (self.winner.pipeline,
                                           list(self.winner.classes),
                                           self.spin_seed.value())

        def fn():
            from datetime import datetime
            from sklearn.base import clone
            # patient sets from the ORIGINAL labels/groups (works for
            # spike-excluded / replicate-averaged / paired matrices)
            tr_idx, _va, te_idx = cdata.patient_split(
                groups_l, labels_l, seed=seed)
            tr_pats = {groups_l[i] for i in tr_idx}
            te_pats = {groups_l[i] for i in te_idx}
            rows_tr = [i for i in range(len(yy)) if gg[i] in tr_pats]
            rows_te = [i for i in range(len(yy)) if gg[i] in te_pats]
            if not rows_tr or not rows_te:
                raise RuntimeError(
                    "the split left an empty side — too few patients")
            est = clone(winner_pl).fit(
                np.asarray(X)[rows_tr], [yy[i] for i in rows_tr])
            X_te = np.asarray(X)[rows_te]
            y_te = [yy[i] for i in rows_te]
            pred = est.predict(X_te)
            from sklearn.metrics import confusion_matrix, f1_score
            cm = confusion_matrix(y_te, pred, labels=winner_classes)
            f1 = float(f1_score(y_te, pred, average="macro"))
            out = {"when": f"{datetime.now():%Y-%m-%d %H:%M}",
                   "n_test_patients": len(te_pats),
                   "n_test_spectra": len(rows_te),
                   "f1": f1, "cm": cm, "classes": winner_classes}
            if len(winner_classes) == 2:
                tp = int(cm[1, 1]); fn_ = int(cm[1].sum()) - tp
                tn = int(cm[0, 0]); fp_ = int(cm[0].sum()) - tn
                out["sens"] = tp / (tp + fn_) if tp + fn_ else float("nan")
                out["spec"] = tn / (tn + fp_) if tn + fp_ else float("nan")
                proba = est.predict_proba(X_te)[:, 1]
                yv = (np.asarray(y_te) == winner_classes[1]).astype(int)
                _a, _se, lo, hi = clin.delong_auc_ci(yv, proba)
                out["auc"] = float(_a)
                out["auc_ci"] = (float(lo), float(hi))
            return out

        def done(out):
            self._locked_result = out
            self.log(f"Locked FINAL evaluation ({out['when']}): "
                     f"{out['n_test_spectra']} spectra / "
                     f"{out['n_test_patients']} unseen patients — "
                     f"macro-F1 {out['f1']:.3f}"
                     + (f", AUC {out.get('auc', float('nan')):.3f}"
                        if "auc" in out else ""))
            self.train_status.setText(
                f"Locked FINAL: macro-F1 {out['f1']:.3f} on "
                f"{out['n_test_patients']} unseen patients")
            self.render_locked_result()

        self._run_async("Locked evaluation", fn, done)

    def render_locked_result(self):
        """Fill the FINAL card on the Result page from _locked_result."""
        out = getattr(self, "_locked_result", None)
        if not out:
            self.r_locked_card.setVisible(False)
            return
        self.r_locked_card.setVisible(True)
        txt = (f"FINAL — one-shot on {out['n_test_spectra']} spectra / "
               f"{out['n_test_patients']} UNSEEN patients "
               f"({out['when']}): macro-F1 {out['f1']:.3f}")
        if "sens" in out:
            txt += (f" · sens {out['sens']:.2f} · spec {out['spec']:.2f}"
                    f" · AUC {out['auc']:.3f} (95% CI "
                    f"{out['auc_ci'][0]:.2f}–{out['auc_ci'][1]:.2f})")
        txt += (". Model refit on the training patients only; no "
                "threshold or preprocessing tuning touched the test "
                "patients.")
        self.r_locked_label.setText(txt)

    # ==================================================== deep diagnostics
    def _deep_guard(self, need_groups: bool = True) -> bool:
        if self.winner is None or self.winner.pipeline is None:
            QtWidgets.QMessageBox.information(
                self, "Train first",
                "Train a model first — the deep diagnostics use the "
                "winner.")
            return False
        if self._lc_data is None:
            QtWidgets.QMessageBox.information(
                self, "Train first",
                "Train first so the feature matrix matches the winner.")
            return False
        if need_groups and not self._lc_data[2]:
            QtWidgets.QMessageBox.information(
                self, "Clinical data needed",
                "This diagnostic needs patient groups (clinical folder "
                "layout with patient subfolders).")
            return False
        return True

    def run_lopo(self):
        """Leave-one-PATIENT-out: every patient judged by a model that
        never saw them (fixed winner parameters, no re-tuning)."""
        if not self._deep_guard():
            return
        from sklearn.base import clone
        X, yy, gg = self._lc_data
        winner_pl, winner_classes, seed = (self.winner.pipeline,
                                           list(self.winner.classes),
                                           self.spin_seed.value())

        def fn():
            return sstats.lopo_evaluate(
                X, yy, gg, clone(winner_pl), {}, winner_classes,
                seed=seed)

        def done(out):
            self._lopo_result = out
            worst = sorted(out["per_patient"], key=lambda r: r[2])[:3]
            self.log(f"LOPO ({out['n_patients']} patients): macro-F1 "
                     f"{out['f1']:.3f}"
                     + (f", AUC {out['auc']:.3f}"
                        if np.isfinite(out["auc"]) else "")
                     + f", mean per-patient accuracy "
                     f"{np.mean(out['accs']):.3f}")
            self.log("  hardest: " + ", ".join(
                f"{p} ({a:.0%})" for p, _n, a, _pp, _t in worst))
            self.train_status.setText(
                f"LOPO: macro-F1 {out['f1']:.3f} over "
                f"{out['n_patients']} unseen patients")
            ax = plotting.clear(self.roc_canvas)
            pp = sorted(out["per_patient"], key=lambda r: r[2])
            names = [r[0] for r in pp]
            accs = [r[2] for r in pp]
            ax.bar(names, accs,
                   color=["#dc2626" if a < 0.5 else "#2563eb"
                          for a in accs],
                   edgecolor="white", linewidth=0.6)
            ax.axhline(np.mean(accs), color="#64748b", lw=1.0, ls="--",
                       label=f"mean {np.mean(accs):.2f}")
            ax.set_ylabel("per-patient accuracy")
            ax.set_title(f"Leave-one-patient-out ({len(pp)} patients, "
                         f"F1 {out['f1']:.3f})", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=60, labelsize=7)
            ax.legend(loc="lower right", fontsize=8)
            self.roc_canvas.draw_idle()

        self._run_async("Leave-one-patient-out", fn, done)

    def run_seed_stability(self):
        """How much do the winner's numbers move just by re-running?"""
        if not self._deep_guard():
            return
        from sklearn.base import clone
        X, yy, gg = self._lc_data
        winner_pl, winner_classes, k = (self.winner.pipeline,
                                        list(self.winner.classes),
                                        self.spin_folds.value())

        def fn():
            return sstats.seed_stability(
                X, yy, gg, clone(winner_pl), {}, winner_classes,
                seeds=(0, 1, 2, 3, 4), k=k)

        def done(f1s):
            self._seed_result = f1s
            self.log(f"Seed stability: macro-F1 {np.mean(f1s):.3f} ± "
                     f"{np.std(f1s):.3f} over 5 seeds "
                     f"(min {min(f1s):.3f}, max {max(f1s):.3f})")
            self.train_status.setText(
                f"Seed stability: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
            ax = plotting.clear(self.roc_canvas)
            ax.boxplot([f1s], tick_labels=["winner"], showmeans=True,
                       widths=0.35)
            ax.scatter([1] * len(f1s), f1s, color="#2563eb", zorder=3,
                       s=18)
            ax.set_ylabel("macro-F1 (grouped CV)")
            ax.set_title(f"Seed stability — {np.mean(f1s):.3f} ± "
                         f"{np.std(f1s):.3f} (5 seeds)", fontsize=10)
            self.roc_canvas.draw_idle()

        self._run_async("Seed stability check", fn, done)

    def run_noise_check(self):
        """Calibrated measurement noise at inference: F1 degradation."""
        if not self._deep_guard():
            return
        from sklearn.base import clone
        X, yy, gg = self._lc_data
        winner_pl, winner_classes, k, seed = (
            self.winner.pipeline, list(self.winner.classes),
            self.spin_folds.value(), self.spin_seed.value())

        def fn():
            return sstats.noise_robustness(
                X, yy, gg, clone(winner_pl), {}, winner_classes,
                levels=(0.0, 0.01, 0.02, 0.05), k=k, seed=seed)

        def done(curve):
            self._noise_result = curve
            drop = curve[0][1] - curve[-1][1]
            self.log("Noise robustness: " + " -> ".join(
                f"{int(lv * 100)}%: {f:.3f}" for lv, f in curve)
                + f" (drop {drop:.3f} at 5% noise)")
            self.train_status.setText(
                f"Noise check: F1 {curve[0][1]:.3f} -> {curve[-1][1]:.3f} "
                "at 5% noise")
            ax = plotting.clear(self.roc_canvas)
            lv = [f"{int(l * 100)}%" for l, _f in curve]
            fv = [f for _l, f in curve]
            ax.plot(lv, fv, "o-", color="#dc2626", lw=1.8, ms=6)
            for x, y in zip(lv, fv, strict=True):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8.5,
                            color="#334155")
            ax.set_xlabel("added noise (fraction of signal scale)")
            ax.set_ylabel("macro-F1")
            ax.set_ylim(min(fv) - 0.05, max(fv) + 0.05)
            ax.set_title(f"Noise robustness — drop {drop:.3f} at 5% "
                         "noise", fontsize=10)
            self.roc_canvas.draw_idle()

        self._run_async("Noise robustness check", fn, done)

    def run_local_explain(self):
        """
        Signed band contributions behind THIS prediction: tree-surrogate
        SHAP evaluated on the mean spectrum of every predicted class.
        """
        if (self._lc_data is None or self._lc_wn is None
                or not self._pred_spectra):
            QtWidgets.QMessageBox.information(
                self, "Nothing to explain yet",
                "Train and predict first — the explanation needs both "
                "the training features and the predicted spectra.")
            return
        if len(self._pred_rows) != len(self._pred_spectra):
            QtWidgets.QMessageBox.information(
                self, "Nothing to explain yet",
                "Predict first — the explanation works on the predicted "
                "spectra.")
            return
        try:
            import shap  # noqa: F401
        except ImportError:
            QtWidgets.QMessageBox.information(
                self, "shap not installed",
                "Local explanations need the optional `shap` package "
                "(pip install shap).")
            return
        from sklearn.ensemble import RandomForestClassifier
        X, yy, gg = self._lc_data
        wn = np.asarray(self._lc_wn)
        pred_rows = list(self._pred_rows)
        pred_spectra = list(self._pred_spectra)

        def fn():
            import shap as shap_mod
            est = RandomForestClassifier(
                n_estimators=300, class_weight="balanced", n_jobs=-1,
                random_state=0).fit(np.asarray(X),
                                    modeling._encode(list(yy),
                                                     sorted(set(yy))))
            expl = shap_mod.TreeExplainer(est)
            # mean spectrum per predicted class
            groups_cls: dict[str, list[np.ndarray]] = {}
            for (_fname, cls, _p), spec in zip(pred_rows, pred_spectra,
                                               strict=True):
                groups_cls.setdefault(cls, []).append(spec)
            classes_enc = sorted(set(yy))
            pos_idx_enc = len(classes_enc) - 1
            rows = []
            for cls in sorted(groups_cls):
                mean_spec = np.mean(np.vstack(groups_cls[cls]), axis=0)
                sv = expl.shap_values(mean_spec.reshape(1, -1),
                                      check_additivity=False)
                if isinstance(sv, list):           # per-class layout
                    sv_pos = np.asarray(sv[pos_idx_enc])[0]
                else:
                    sv_a = np.asarray(sv)
                    sv_pos = (sv_a[0, :, pos_idx_enc]
                              if sv_a.ndim == 3 else sv_a[0])
                contribs = []
                for center, hw, mol, assign, direction in bio.BANDS:
                    m = bio._window(wn, center, hw)
                    if m.sum() < 2:
                        continue
                    contribs.append((float(np.mean(sv_pos[m])), center,
                                     mol, assign, direction))
                contribs.sort(key=lambda r: -abs(r[0]))
                rows.append((cls, contribs[:5]))
            return rows

        def done(rows):
            self._local_bands = rows
            ax = plotting.clear(self.r_local_canvas)
            colors = {1: "#dc2626", -1: "#2563eb"}
            ybase = 0.0
            yticks, yticklabs = [], []
            for cls, contribs in rows:
                for i, (v, center, mol, _assign, _dir) in enumerate(contribs):
                    y = ybase + i
                    ax.barh(y, v, height=0.7,
                            color=colors[1 if v >= 0 else -1],
                            edgecolor="white", linewidth=0.5)
                    ax.text(v, y, f"  {center:.0f} {mol}",
                            va="center", ha="left" if v >= 0 else "right",
                            fontsize=8, color="#334155")
                yticks.append(ybase + len(contribs) / 2 - 0.5)
                yticklabs.append(f"predicted: {cls}")
                ybase += len(contribs) + 1.2
            ax.set_yticks(yticks, yticklabs, fontsize=9,
                          fontweight="bold")
            ax.axvline(0, color="#64748b", lw=0.8)
            ax.invert_yaxis()
            ax.set_xlabel("signed SHAP contribution (surrogate)")
            ax.set_title("Why this call — band contributions per "
                         "predicted class", fontsize=10)
            self.r_local_canvas.draw_idle()
            tops = "; ".join(
                f"{cls}: " + ", ".join(
                    f"{c:.0f}{'+' if v >= 0 else '−'}"
                    for v, c, _m, _a, _d in contribs[:3])
                for cls, contribs in rows)
            self.log(f"Local explanation ({len(pred_spectra)} "
                     f"spectra): {tops}")
            self.train_status.setText("Local explanation done — see "
                                      "Result page.")

        self._run_async("Explaining prediction", fn, done)

    # ============================================================== Train
    def start_training(self):
        if not self.spectra:
            QtWidgets.QMessageBox.information(
                self, "No data yet",
                "Load a spectra folder (or generate the demo dataset) "
                "first — use the Start page.")
            self.go_to(TAB_START)
            return
        unlabeled = sum(1 for l in self.labels if not l.strip())
        if unlabeled:
            QtWidgets.QMessageBox.warning(
                self, "Unlabeled spectra",
                f"{unlabeled} spectra have no class label (see Data "
                "page). They will be EXCLUDED from training.")
        keep = [i for i, l in enumerate(self.labels) if l.strip()]
        params_now0 = self.read_params().validate()
        if (self.spike_flags is not None
                and self.chk_exclude_flagged.isChecked()
                and not params_now0.despike):
            # despiking RECOVERS spiked spectra; exclusion only applies
            # when the user turned despiking off
            n_flagged = sum(1 for i in keep if self.spike_flags[i])
            if n_flagged:
                self.log(f"Excluding {n_flagged} spiked spectra "
                         "(quality flag, despiking off) from training.")
            keep = [i for i in keep if not self.spike_flags[i]]
        yy = [self.labels[i].strip() for i in keep]
        if len(set(yy)) < 2:
            QtWidgets.QMessageBox.critical(
                self, "Cannot train",
                "Need at least 2 distinct classes among labeled spectra.\n"
                "Check the Class column in the Data page.")
            return
        names = [n for n, cb in self.model_checks.items() if cb.isChecked()]
        if not names:
            QtWidgets.QMessageBox.warning(self, "No models",
                                          "Select at least one model.")
            return
        self.b_train.setEnabled(False)
        self.progress.setValue(0)
        self.train_status.setText("Preprocessing spectra…")
        mode_idx = self.combo_mode.currentIndex()
        seq_mode = mode_idx >= 3                     # 3SSE search
        paired_mode = mode_idx in (1, 2, 4)
        if paired_mode and self.groups is None:
            self.b_train.setEnabled(True)
            QtWidgets.QMessageBox.warning(
                self, "Paired mode needs patient groups",
                "Paired-reference classification needs clinical data "
                "(patient folders). Load D:\\BARC\\Data-style data first "
                "or switch back to Standard mode.")
            return
        try:
            if paired_mode:
                import paired as paired_mod
                pd_ = paired_mod.paired_features(
                    self.X_raw, self.labels, self.groups, self.grid,
                    self.read_params().validate(),
                    exclude=(self.spike_flags
                             if (self.spike_flags is not None
                                 and self.chk_exclude_flagged.isChecked()
                                 and not self.read_params().despike)
                             else None),
                    use_pqn=(mode_idx == 2))
                X, yy, gg = pd_.X, pd_.y, pd_.groups
                self._lc_wn = pd_.wn
                self._paired_mode = True
                self.log(
                    f"Paired-reference mode{' + PQN' if mode_idx == 2 else ''}: "
                    f"{pd_.n_patients} patients "
                    f"with both classes, {pd_.X.shape[0]} deviation "
                    f"spectra; {pd_.n_unpaired_excluded} patients "
                    "skipped (no normal reference).")
            else:
                X = self.get_processed_X()
                X = X[keep]
                self._paired_mode = False
        except Exception as exc:
            self.b_train.setEnabled(True)
            self.friendly_error("Preprocessing failed", exc)
            return
        if not paired_mode:
            self.train_status.setText(f"Preprocessing done (X={X.shape}). "
                                      "Training…")
            gg = ([self.groups[i] for i in keep]
                  if self.groups is not None else None)
        # optional: average replicate spectra per (patient, class) —
        # fewer, cleaner rows; the pilot measured ~4x lower fold variance
        if (not paired_mode and gg is not None
                and self.chk_avg_replicates.isChecked()):
            agg: dict[tuple[str, str], list[int]] = {}
            for i in range(len(keep)):
                agg.setdefault((gg[i], yy[i]), []).append(i)
            n_before = len(yy)
            X = np.vstack([X[rows].mean(axis=0)
                           for rows in agg.values()])
            yy = [lab for (_pat, lab) in agg.keys()]
            gg = [pat for (pat, _lab) in agg.keys()]
            self.log(f"Averaged replicates: {len(agg)} (patient, class) "
                     f"rows from {n_before} spectra")
        repeats = 3 if self.chk_repeat.isChecked() else 1
        self.log(f"Training {len(names)} models, "
                 f"{self.spin_folds.value()}-fold CV"
                 + (f" x{repeats}" if repeats > 1 else "")
                 + f", n={X.shape[0]}, "
                 f"features={X.shape[1]}, classes={sorted(set(yy))}"
                 + (f", grouped by {len(set(gg))} patients" if gg else ""))
        self._lc_data = (X, yy, gg)
        self._region_bands = None      # stale after any retrain
        # the wavenumber axis for the peak-band model must match the
        # (cropped) feature columns of X — also reused by the
        # region-importance plot (paired mode set it already)
        if not paired_mode:
            params_now = self.read_params().validate()
            wn_for_models = np.asarray(self.grid)[
                preprocessing.crop_mask(self.grid, params_now)]
            self._lc_wn = wn_for_models
        else:
            wn_for_models = self._lc_wn
        if seq_mode:
            # ---- 3SSE: screen every single/pair/triple chain ----------
            import sequential
            n_arch = sequential._check_space(names)
            self.train_status.setText(
                f"3SSE search: {n_arch} architectures — this takes a "
                "while (see progress).")
            self.log(f"3SSE search started: {len(names)} models · "
                     f"{n_arch} architectures "
                     f"({len(names)} singles + "
                     f"{len(names) * (len(names) - 1)} pairs + "
                     f"{len(names) * (len(names) - 1) * (len(names) - 2)}"
                     " triples), patient-grouped OOF chaining")
            self._seq_worker = SeqSearchWorker(
                X, yy, gg, wn_for_models, model_names=names,
                k=self.spin_folds.value(), seed=self.spin_seed.value(),
                fast=self.chk_3sse_fast.isChecked(),
                resume_path=os.path.join(APP_DIR, "study_run_3sse",
                                         "archs.jsonl"))
            self._seq_worker.progress.connect(self.on_seq_progress)
            self._seq_worker.search_progress.connect(
                self.on_seq_search_progress)
            self._seq_worker.leaderboard.connect(self.on_seq_leaderboard)
            self._seq_worker.done.connect(self.on_seq_done)
            self._seq_worker.cancelled.connect(self.on_seq_cancelled)
            self._seq_worker.failed.connect(self.on_seq_failed)
            self._seq_worker.start()
            self.b_3sse_run.hide()
            self.b_3sse_cancel.show()
            return
        self.worker = TrainWorker(X, yy, names, self.spin_folds.value(),
                                  self.spin_seed.value(), groups=gg,
                                  repeats=repeats,
                                  wavenumbers=wn_for_models)
        self.worker.progress.connect(self.on_train_progress)
        self.worker.done.connect(self.on_train_done)
        self.worker.failed.connect(self.on_train_failed)
        self.worker.start()

    def on_train_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.train_status.setText(msg)
        self.statusBar().showMessage(msg)

    def on_train_done(self, results, winner):
        self.results, self.winner = results, winner
        self.chain_flow.hide()          # 3SSE path re-shows with arch
        self.b_train.setEnabled(bool(self.spectra))
        self.progress.setValue(100)
        sens = winner.macro["sens"][0]
        spec = winner.macro["spec"][0]
        f1 = winner.macro_f1()
        # big result banner
        self.banner_title.setText(f"Winner: {winner.name}")
        for key, v in (("sens", sens), ("spec", spec), ("f1", f1)):
            lbl = self.stat_values[key]
            lbl.setText(f"{v:.3f}")
            lbl.setStyleSheet(f"color:{uh.metric_fg(v)};")
        extra = (f" Decision threshold tuned to maximize F1: "
                 f"{winner.threshold:.3f}." if winner.threshold is not None
                 else "")
        self.banner_plain.setText(
            f"Detects {sens:.0%} of true positive cases · correctly rules "
            f"out {spec:.0%} of negatives · {self.spin_folds.value()}-fold "
            f"cross-validated. Save the model to start predicting."
            f"{extra}")
        self.train_status.setText(
            f"Done — best: {winner.name} (macro-F1 {f1:.3f}). Save it, "
            "then go to Predict.")
        self.log(f"Training complete. Winner: {winner.name} "
                 f"(macro sens {sens:.3f} / spec {spec:.3f} / F1 {f1:.3f})"
                 + (f", threshold={winner.threshold:.3f}"
                    if winner.threshold is not None else ""))
        # Friedman + Nemenyi: is the model ranking real or noise?
        try:
            scores = {r.name: r.fold_f1 for r in results
                      if r.error is None and len(r.fold_f1) > 2}
            if len(scores) >= 2:
                fr = sstats.friedman_nemenyi(scores)
                if fr.get("ok"):
                    self._friedman_result = fr
                    worse = [n for n, (_d, sig) in fr["vs_best"].items()
                             if sig]
                    self.log(
                        f"Friedman p={fr['p']:.3f} — model differences "
                        + ("ARE statistically significant"
                           if fr["significant"]
                           else "are NOT statistically significant "
                                "(treat as equal)")
                        + (f"; {fr['best']} beats {', '.join(worse)} "
                           f"beyond the Nemenyi CD ({fr['cd']:.2f})"
                           if worse else ""))
        except Exception:
            pass
        self._fill_compare_table()
        self._fill_perclass_table()
        self._draw_winner_plots()
        self.b_save.setEnabled(True)
        self.statusBar().showMessage(
            f"Best model: {winner.name} — save it to start predicting")
        self.update_welcome()
        self.refresh_nav()

    def on_train_failed(self, err: str):
        self.b_train.setEnabled(bool(self.spectra))
        self.log("Training FAILED:\n" + err)
        QtWidgets.QMessageBox.critical(
            self, "Training failed",
            "Training ran into an error:\n\n" + err.splitlines()[-1] +
            "\n\nFull details are printed to the console.")

    # ================================================== 3SSE handlers
    def _seq_effective_models(self) -> list[str]:
        """Checked models, minus the fast-screening skips."""
        import sequential
        checked = [n for n, cb in self.model_checks.items()
                   if cb.isChecked()]
        if self.chk_3sse_fast.isChecked():
            checked = [n for n in checked
                       if n not in sequential.SLOW_MODELS]
        return checked

    def _update_seq_card(self, *_args):
        """Refresh the 3SSE card: architecture counts + rough time
        estimate for the currently checked (and fast-filtered) models."""
        checked = self._seq_effective_models()
        base = [n for n in checked if n != "Ensemble (top-3)"]
        n = len(base) + (1 if "Ensemble (top-3)" in checked else 0)
        singles, pairs, triples = n, n * (n - 1), n * (n - 1) * (n - 2)
        total_arch = singles + pairs + triples
        if n < 3:
            # not enough models for the chain search — say so clearly
            self.seq_counts.setText(
                f"Only {n} model{'s' if n > 1 else ''} selected — the "
                "architecture search needs at least 2 (3+ for two- and "
                "three-model chains). Tick more models above (the "
                "'All' button checks everything).")
            self.seq_counts.setStyleSheet("color:#b91c1c;")
            self.seq_estimate.setText("")
            self.b_3sse_run.setEnabled(n >= 2)
            return
        self.seq_counts.setStyleSheet("")
        self.b_3sse_run.setEnabled(True)
        k = 2 if self.chk_3sse_fast.isChecked() else (
            self.spin_folds.value()
            if hasattr(self, "spin_folds") else 5)
        # calibrated on the measured full run (4,369 @ k=3 ≈ 55 min
        # screening + ~25 min validation) — rough by design
        est_min = total_arch * k / 240 + 60 * 0.4
        data_note = ("paired data" if self.groups is not None
                     else "standard data (no patient groups loaded)")
        fast_note = (" · FAST: 2-fold, slow models skipped"
                     if self.chk_3sse_fast.isChecked() else "")
        self.seq_counts.setText(
            f"Search space: {singles:,} single models · "
            f"{pairs:,} two-model chains · {triples:,} three-model "
            f"chains = {total_arch:,} architectures "
            f"(ordered, no repeats; probabilities chained between "
            f"layers, patient-grouped OOF). Will run on {data_note}"
            f"{fast_note}.")
        self.seq_estimate.setText(
            f"Rough time estimate: ≈ {est_min:.0f} min "
            "(screening + nested validation + significance tests)")
        ckpt = os.path.join(APP_DIR, "study_run_3sse", "archs.jsonl")
        if os.path.isfile(ckpt):
            self.seq_phase.setText(
                "Checkpoint from an interrupted search found — Run "
                "resumes it instead of starting over.")

    def run_3sse_now(self):
        """Run button: pick the right 3SSE mode for the loaded data and
        start the search through the normal Start-training path."""
        if not self.spectra:
            QtWidgets.QMessageBox.information(
                self, "No data yet",
                "Load a spectra folder (or generate the demo dataset) "
                "first — use the Start page.")
            self.go_to(TAB_START)
            return
        if (self._seq_worker is not None
                and self._seq_worker.isRunning()):
            QtWidgets.QMessageBox.information(
                self, "3SSE search already running",
                "An architecture search is in progress — watch the "
                "counter in this card; results open automatically.")
            return
        eff = self._seq_effective_models()
        if len(eff) < 2:
            QtWidgets.QMessageBox.information(
                self, "Not enough models selected",
                f"Only {len(eff)} model is selected for the "
                "architecture search.\n\nTick at least 2 models "
                "(3 or more for two- and three-model chains) — the "
                "'All' button above the model list checks everything.")
            return
        # paired whenever patient groups exist; the dropdown reflects it
        self.combo_mode.setCurrentIndex(4 if self.groups is not None
                                        else 3)
        self.seq_counter.setText("0")
        self.seq_phase.setText("Phase: screening — starting…")
        self.seq_best.setText("")
        self.seq_top5.setText("")
        self.start_training()

    def cancel_3sse(self):
        if self._seq_worker is not None and self._seq_worker.isRunning():
            self._seq_worker.cancel()
            self.seq_phase.setText("Cancelling — finishing current "
                                   "batch…")

    def on_seq_leaderboard(self, top5: list):
        rows = "".join(
            f"{i + 1}. {arch}  <b>{f1:.3f}</b><br>"
            for i, (arch, f1) in enumerate(top5))
        self.seq_top5.setText(f"<b>Top so far</b><br>{rows}")

    def on_seq_search_progress(self, done: int, total: int, best: str,
                               f1: float, eta: float):
        self.seq_counter.setText(f"{done:,} / {total:,}")
        self.seq_best.setText(f"Current best: {best}  ·  macro-F1 "
                              f"{f1:.3f}")
        self.seq_phase.setText(f"Phase: screening · ETA "
                               f"{eta / 60:.0f} min")

    def view_saved_3sse(self):
        """Open the last completed 3SSE run's rankings without re-running
        (reads study_run_3sse: screening.jsonl + validated.json +
        winner.json + report.txt)."""
        import json as _json
        folder = os.path.join(APP_DIR, "study_run_3sse")
        screening = os.path.join(folder, "screening.jsonl")
        if not os.path.isfile(screening):
            QtWidgets.QMessageBox.information(
                self, "No saved 3SSE run",
                "No completed architecture search found in "
                "study_run_3sse/.\n\nPick a '3SSE search' training mode "
                "and press Start training, or run sequential.py from "
                "the command line first.")
            return
        board = {"singles": [], "pairs": [], "triples": [], "total": 0,
                 "pruned": 0}
        key = {1: "singles", 2: "pairs", 3: "triples"}
        with open(screening, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                board[key[rec["level"]]].append(
                    {"arch": tuple(rec["arch"]), "level": rec["level"],
                     "metrics": {k: v for k, v in rec.items()
                                 if k not in ("arch", "level")}})
        board["total"] = (len(board["singles"]) + len(board["pairs"])
                          + len(board["triples"]))
        payload = {"board": board, "validated": {}, "winner": None}
        vpath = os.path.join(folder, "validated.json")
        if os.path.isfile(vpath):
            try:
                with open(vpath, encoding="utf-8") as fh:
                    payload["validated"] = {
                        int(k): v
                        for k, v in _json.load(fh).items()}
            except Exception:
                pass
        wpath = os.path.join(folder, "winner.json")
        if os.path.isfile(wpath):
            try:
                with open(wpath, encoding="utf-8") as fh:
                    payload["winner"] = _json.load(fh)
            except Exception:
                pass
        gpath = os.path.join(folder, "significance.json")
        if os.path.isfile(gpath):
            try:
                with open(gpath, encoding="utf-8") as fh:
                    payload["significance"] = _json.load(fh)
            except Exception:
                pass
        rpath = os.path.join(folder, "report.txt")
        if os.path.isfile(rpath):
            try:
                with open(rpath, encoding="utf-8") as fh:
                    payload["report_text"] = fh.read()
            except Exception:
                pass
        dlg = SeqResultsDialog(payload, parent=self)
        dlg.setModal(False)
        dlg.show()
        self.log("Opened saved 3SSE results "
                 f"({board['total']} architectures) from "
                 "study_run_3sse/")

    def on_seq_progress(self, pct: int, msg: str):
        if pct >= 0:
            self.progress.setValue(pct)
        elif "validating" in msg:
            self.seq_counter.setText("validation")
            self.seq_phase.setText(
                "Phase: nested validation of the top architectures")
        self.train_status.setText(msg)
        self.statusBar().showMessage(msg)

    def on_seq_done(self, payload: dict):
        self.b_train.setEnabled(True)
        self.b_3sse_run.show()
        self.b_3sse_cancel.hide()
        self.progress.setValue(100)
        self.seq_phase.setText("Done — winner installed.")
        self._seq_payload = payload
        board = payload["board"]
        fin = payload.get("winner")
        n_pruned = board.get("pruned", 0)
        self.log(f"3SSE search done: {board['total']} architectures "
                 f"({n_pruned} pruned by the sound early-abandon bound)")
        dlg = SeqResultsDialog(payload, parent=self)
        dlg.setModal(False)
        dlg.show()
        if fin is None:
            self.train_status.setText(
                "3SSE search finished (no validated winner available).")
            return
        # install the winner like any training run: banner, tables,
        # plots, Save button, Result page and Predict all work on it
        m = fin["metrics"]
        classes = list(fin["classes"])
        winner = modeling.ModelResult(
            name="3SSE: " + " → ".join(fin["arch"]), classes=classes)
        winner.macro = {"sens": (m["sens"], 0.0),
                        "spec": (m["spec"], 0.0), "f1": (m["f1"], 0.0),
                        "prec": (m.get("prec", 0.0), 0.0)}
        cm = m.get("cm")
        if cm is not None:
            winner.cm = np.asarray(cm, dtype=int).reshape(
                len(classes), len(classes))
            winner.per_class = {
                classes[int(k)]: {mk: (mv, 0.0) for mk, mv in v.items()}
                for k, v in
                modeling.class_metrics_from_cm(winner.cm).items()}
        winner.oof_proba = m.get("oof_proba")
        winner.y_true_encoded = np.asarray(m.get("y_true"))
        winner.groups = payload.get("groups")
        winner.threshold = fin.get("threshold")
        winner.pipeline = fin["chain"]
        # comparison table: validated single models (nested, fair)
        results = []
        for entry in payload["validated"].get(1, []):
            mm = entry["metrics"]
            r = modeling.ModelResult(name=entry["arch"][0],
                                     classes=classes)
            r.macro = {"sens": (mm["sens"], 0.0),
                       "spec": (mm["spec"], 0.0), "f1": (mm["f1"], 0.0),
                       "prec": (mm.get("prec", 0.0), 0.0)}
            results.append(r)
        if not results:                      # screening-only fallback
            for s in board["singles"]:
                mm = s["metrics"]
                r = modeling.ModelResult(name=s["arch"][0],
                                         classes=classes)
                r.macro = {"sens": (mm["sens"], 0.0),
                           "spec": (mm["spec"], 0.0),
                           "f1": (mm["f1"], 0.0),
                           "prec": (mm.get("prec", 0.0), 0.0)}
                results.append(r)
        self.on_train_done(results + [winner], winner)
        self.chain_flow.set_arch(fin["arch"])
        self.train_status.setText(
            f"3SSE winner installed: {' → '.join(fin['arch'])} "
            f"(nested F1 {m['f1']:.3f}) — Save it, then go to Predict. "
            "Full rankings in the 3SSE results window.")

    def on_seq_cancelled(self):
        self.b_train.setEnabled(True)
        self.b_3sse_run.show()
        self.b_3sse_cancel.hide()
        self.progress.setValue(0)
        self.seq_phase.setText(
            "Cancelled — checkpoint kept. Press Run to resume where "
            "it stopped.")
        self.log("3SSE search cancelled (checkpoint kept for resume)")

    def on_seq_failed(self, tb: str):
        self.b_train.setEnabled(True)
        self.b_3sse_run.show()
        self.b_3sse_cancel.hide()
        self.log(f"3SSE search failed:\n{tb}")
        QtWidgets.QMessageBox.warning(
            self, "3SSE search failed",
            "The architecture search failed.\n\nTechnical details are "
            "in session.log next to the app.")

    def on_seq_save(self, payload: dict):
        """Save the 3SSE winner chain as a standard model bundle."""
        winner = payload.get("winner")
        if not winner:
            return
        path, _filt = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 3SSE winner bundle",
            os.path.join(os.path.expanduser("~"), "Desktop",
                         "model_3SSE.joblib"),
            "Model bundle (*.joblib)")
        if not path:
            return
        try:
            from types import SimpleNamespace
            m = winner["metrics"]
            extras = ({"calibrator": winner["calibrator"]}
                      if winner["calibrator"] else {})
            winner_ns = SimpleNamespace(
                name="3SSE: " + " → ".join(winner["arch"]),
                pipeline=winner["chain"], classes=winner["classes"],
                threshold=winner["threshold"],
                macro={"f1": (m["f1"], 0.0),
                       "sens": (m["sens"], 0.0),
                       "spec": (m["spec"], 0.0)})
            modeling.save_bundle(
                path, winner_ns, self.grid,
                self.read_params().validate(),
                dataset_name=self._source_folder or "",
                paired=self._paired_mode, **extras)
            self.settings["last_model"] = path
            uh.save_settings(self.settings)
            self._set_bundle(modeling.load_bundle(path), path)
            self.log(f"3SSE winner bundle saved: {path}")
        except Exception as exc:
            self.friendly_error("Saving 3SSE bundle failed", exc)

    def _fill_compare_table(self):
        res = self.results
        self.compare_table.setRowCount(len(res))
        for r, mr in enumerate(res):
            if mr.error is not None:
                vals = [mr.name, "—", "—", "FAILED"]
            else:
                vals = mr.summary_row()
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if mr is self.winner:
                    font = it.font()
                    font.setBold(True)
                    it.setFont(font)
                    if c == 3:
                        it.setText(f"{v}  ★")
                if c in (1, 2, 3):
                    it.setTextAlignment(ALIGN_CENTER)
                    key = ("sens", "spec", "f1")[c - 1]
                    val = mr.macro.get(key, (float("nan"),))[0]
                    it.setBackground(qc.QtGui.QColor(*uh.metric_bg(val)))
                it.setToolTip(mr.error or "")
                self.compare_table.setItem(r, c, it)

    def _fill_perclass_table(self):
        w = self.winner
        classes = w.classes
        self.perclass_table.setRowCount(len(classes))
        row_sums = w.cm.sum(axis=1) if w.cm is not None else [0] * len(classes)
        for r, c in enumerate(classes):
            per = w.per_class.get(c, {})
            vals = [c, str(int(row_sums[r]))] + [
                f"{per.get(k, (float('nan'), 0))[0]:.3f} ± "
                f"{per.get(k, (float('nan'), 0))[1]:.3f}"
                for k in ("sens", "spec", "prec", "f1")]
            for col, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if col:
                    it.setTextAlignment(ALIGN_CENTER)
                if col >= 2:
                    key = ("sens", "spec", "prec", "f1")[col - 2]
                    val = per.get(key, (float("nan"),))[0]
                    it.setBackground(qc.QtGui.QColor(*uh.metric_bg(val)))
                self.perclass_table.setItem(r, col, it)

    def _draw_winner_plots(self):
        w = self.winner
        ax = plotting.clear(self.cm_canvas)
        plotting.plot_confusion_matrix(
            ax, w.cm, w.classes,
            title="Confusion matrix (pooled CV folds)")
        self.cm_canvas.draw_idle()

        ax = plotting.clear(self.roc_canvas)
        try:
            if w.oof_proba is not None and len(w.classes) == 2:
                ye = w.y_true_encoded
                valid = ~np.isnan(w.oof_proba[:, 1])
                fpr, tpr, auc = modeling.roc_points(
                    ye[valid], w.oof_proba[valid, 1])
                plotting.plot_roc(ax, fpr, tpr, auc, label=w.name)
            elif w.oof_proba is not None:
                plotting.plot_perclass_metrics(
                    ax, w.classes, w.per_class,
                    title=f"{w.name} — per-class metrics")
        except Exception:
            self.log("ROC/metrics plot failed:\n" + traceback.format_exc())
        self.roc_canvas.draw_idle()

    def save_model(self):
        if self.winner is None:
            QtWidgets.QMessageBox.information(
                self, "Nothing to save yet",
                "Train a model first — then the winner can be saved.")
            return
        default = os.path.join(
            APP_DIR, f"model_{sanitize(self.winner.name)}.joblib")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save best model", default, "Model (*.joblib)")
        if not path:
            return
        try:
            # Platt calibration + clinical operating points, both fitted
            # on the fresh winner's out-of-fold probabilities (binary
            # only).  The cut-offs are computed in CALIBRATED space so
            # they line up with predict-time probabilities; the stored
            # threshold is transformed inside save_bundle.
            calibrator = None
            lo_o = hi_o = None
            if (len(self.winner.classes) == 2
                    and getattr(self.winner, "oof_proba", None) is not None
                    and getattr(self.winner, "y_true_encoded", None)
                    is not None):
                valid = ~np.isnan(self.winner.oof_proba[:, 1])
                if valid.any():
                    yv = np.asarray(self.winner.y_true_encoded[valid])
                    pv = self.winner.oof_proba[valid, 1]
                    calibrator = clin.fit_platt(yv, pv)
                    p_cal = (clin.apply_platt(pv, calibrator)
                             if calibrator else pv)
                    try:
                        lo_o, hi_o, _s, _sp = clin.operating_points(yv, p_cal)
                    except Exception:
                        lo_o = hi_o = None
                    if calibrator is None:
                        lo_o, hi_o = self._op_points_now()
            else:
                lo_o, hi_o = self._op_points_now()
            op_extra = ((lo_o, hi_o)
                        if lo_o is not None and hi_o is not None else None)
            modeling.save_bundle(path, self.winner, self.grid,
                                 self.read_params().validate(),
                                 dataset_name=self.folder_edit.text(),
                                 paired=self._paired_mode,
                                 region_bands=self._region_bands,
                                 op_points=op_extra,
                                 calibrator=calibrator)
            self.log(f"Model saved: {path}")
            self.settings["last_model"] = path
            uh.save_settings(self.settings)
            self._set_bundle(modeling.load_bundle(path), path)
            self.go_to(TAB_PRED)
            self.statusBar().showMessage(
                "Model saved — predict spectra on the Predict page")
            self.update_welcome()
            self.refresh_nav()
        except Exception as exc:
            self.friendly_error("Saving the model failed", exc)

    # ============================================================ Predict
    def browse_reference(self):
        start = self.settings.get("predict_folder",
                                  os.path.dirname(APP_DIR))
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with the patient's NORMAL reference "
                  "spectra", start)
        if folder:
            self.ref_path_edit.setText(folder)

    def _set_bundle(self, bundle: dict, path: str):
        self.bundle = bundle
        self.model_path_edit.setText(path)
        m = bundle["macro"]
        self._region_bands = bundle.get("region_bands") or None
        op = bundle.get("op_points")
        self._op_points = (tuple(op) if isinstance(op, (list, tuple))
                           and len(op) == 2 else None)
        self.ref_row_widget.setVisible(bool(bundle.get("paired")))
        self.model_info.setText(
            f"<b>{bundle['model_name']}</b> — classes: "
            f"{', '.join(bundle['classes'])} | macro sensitivity "
            f"{m.get('sens', (0,))[0]:.3f} · specificity "
            f"{m.get('spec', (0,))[0]:.3f} · F1 "
            f"{m.get('f1', (0,))[0]:.3f}"
            + (f" | F1-optimal threshold {bundle['threshold']:.3f}"
               if bundle.get("threshold") is not None else ""))
        self.b_predict.setEnabled(True)

    def browse_model(self):
        start = self.settings.get("last_model", APP_DIR)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load trained model", start, "Model (*.joblib)")
        if not path:
            return
        try:
            self._set_bundle(modeling.load_bundle(path), path)
            self.settings["last_model"] = path
            uh.save_settings(self.settings)
            self.log(f"Model loaded: {path}")
        except Exception as exc:
            self.friendly_error("Loading the model failed", exc)

    def browse_spectra(self):
        start = self.settings.get("predict_folder",
                                  os.path.dirname(APP_DIR))
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with spectra to classify", start)
        if folder:
            self.spec_path_edit.setText(folder)
            self.settings["predict_folder"] = folder
            uh.save_settings(self.settings)
            if self.bundle:
                self.b_predict.setEnabled(True)

    def browse_spectra_files(self):
        start = self.settings.get("predict_folder",
                                  os.path.dirname(APP_DIR))
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select spectrum file(s) to classify — .txt / .dat / .csv",
            start, "Spectra (*.txt *.dat *.csv);;All (*)")
        if files:
            self.spec_path_edit.setText("; ".join(files))
            self.settings["predict_folder"] = os.path.dirname(files[0])
            uh.save_settings(self.settings)
            if self.bundle:
                self.b_predict.setEnabled(True)

    def _prediction_files(self) -> list[str] | None:
        """Resolve the Predict-page input to a list of spectrum paths.

        Supports flat files, `<root>/<patient>/` trees AND clinical
        `<root>/<class>/<patient>/` trees (one extra level of walking).
        """
        from clinical_data import REFERENCE_MARKERS
        target = self.spec_path_edit.text().strip()
        if not target:
            QtWidgets.QMessageBox.warning(
                self, "No spectra",
                "Browse a folder or spectrum file(s) first (step 2 "
                "above).")
            return None
        if os.path.isdir(target):
            files, skipped = [], []

            def _spectra_in(d: str) -> list[str]:
                return [os.path.join(d, s) for s in sorted(os.listdir(d))
                        if s.lower().endswith((".txt", ".dat", ".csv"))]

            for fn in sorted(os.listdir(target)):
                path = os.path.join(target, fn)
                if not os.path.isdir(path):
                    entries = [(fn, path)]
                else:
                    own = _spectra_in(path)
                    if own:
                        # <root>/<patient>/ tree
                        entries = [(os.path.basename(p), p) for p in own]
                    else:
                        # clinical <root>/<class>/<patient>/ tree — walk
                        # one more level
                        entries = []
                        for sub in sorted(os.listdir(path)):
                            subpath = os.path.join(path, sub)
                            if os.path.isdir(subpath):
                                entries += [(os.path.basename(p), p)
                                            for p in _spectra_in(subpath)]
                for sfn, spath in entries:
                    if (any(m in sfn.lower() for m in REFERENCE_MARKERS)
                            or sfn.lower() == "split_log.txt"):
                        skipped.append(sfn)
                        continue
                    files.append(spath)
            if skipped:
                self.log(f"Prediction folder: skipped {len(skipped)} "
                         f"non-sample file(s): {', '.join(skipped[:5])}"
                         + (" …" if len(skipped) > 5 else ""))
            if not files:
                QtWidgets.QMessageBox.warning(
                    self, "No spectra found",
                    f"No .txt/.dat/.csv spectra were found in:\n{target}")
                return None
            return files
        files = [p.strip() for p in target.split(";") if p.strip()]
        files = [p for p in files if os.path.isfile(p)]
        if not files:
            QtWidgets.QMessageBox.warning(
                self, "No spectra found",
                "None of the listed files exist — browse again (step 2 "
                "above).")
            return None
        return files

    def run_prediction(self):
        if self.bundle is None:
            QtWidgets.QMessageBox.warning(
                self, "No model",
                "Load a trained model first (step 1 above).")
            return
        if (self._pred_worker is not None
                and self._pred_worker.isRunning()):
            return
        files = self._prediction_files()
        if not files:
            return
        target = self.spec_path_edit.text().strip()
        root = target if os.path.isdir(target) else os.path.dirname(
            files[0])
        manual_ref = None
        if self.bundle.get("paired"):
            manual_ref = self.ref_path_edit.text().strip()
            if manual_ref and not os.path.isdir(manual_ref):
                manual_ref = None
            if not manual_ref:
                # no manual folder: auto per-patient references need a
                # clinical <root>/<class>/<patient> tree — check quickly
                has_tree = False
                if os.path.isdir(root):
                    for cls_dir in os.listdir(root):
                        cp = os.path.join(root, cls_dir)
                        if (os.path.isdir(cp)
                                and cls_dir.lower().startswith(
                                    PredictWorker.NORMAL_SYNONYMS)
                                and any(os.path.isdir(
                                    os.path.join(cp, d))
                                    for d in os.listdir(cp))):
                            has_tree = True
                            break
                if not has_tree:
                    QtWidgets.QMessageBox.warning(
                        self, "Reference required",
                        "This model was trained in MARGIN mode. Either "
                        "pick the folder with the patient's NORMAL "
                        "reference spectra, or select the whole "
                        "clinical tree (<root>/<class>/<patient>) so "
                        "each patient's own normal can be found "
                        "automatically.")
                    return
        self.b_predict.setEnabled(False)
        self.statusBar().showMessage("Predicting…")
        self._pred_worker = PredictWorker(self.bundle, files, root,
                                          manual_ref_dir=manual_ref)
        self._pred_worker.progress.connect(
            lambda pct, msg: self.statusBar().showMessage(
                f"{pct}% — {msg}"))
        self._pred_worker.first_plot.connect(self.on_predict_first)
        self._pred_worker.done.connect(self.on_predict_done)
        self._pred_worker.failed.connect(self.on_predict_failed)
        self._pred_worker.start()

    def on_predict_first(self, info: dict):
        plotting.plot_prediction(
            self.pred_canvas.figure, info["wn"], info["raw"],
            info["proc"], info["pred"], info["probs"],
            title=info["title"])
        self.pred_canvas.draw_idle()

    def on_predict_done(self, payload: dict):
        self.b_predict.setEnabled(True)
        for line in payload["logs"]:
            self.log(line)
        if payload["n_auto_refs"]:
            self.log(f"Margin mode: built {payload['n_auto_refs']} "
                     "per-patient normal references from the clinical "
                     "tree")
        rows = payload["rows"]
        self._pred_rows = rows
        self._pred_probs = payload["probs"]
        self._pred_spectra = payload["spectra"]
        self._pred_wn = payload["wn"]
        self._pred_reference = payload["reference"]
        self._patient_rows = self._aggregate_patients(
            payload["ok_paths"], rows)
        self.pred_table.setRowCount(len(rows))
        for r, (f, p, pv) in enumerate(rows):
            for c, v in enumerate((f, p, f"{pv:.3f}")):
                it = QtWidgets.QTableWidgetItem(v)
                if c:
                    it.setTextAlignment(ALIGN_CENTER)
                self.pred_table.setItem(r, c, it)
        if not rows:
            QtWidgets.QMessageBox.warning(
                self, "Nothing predicted",
                "None of the selected files could be read as spectra.")
            return
        self.refresh_nav()
        self.render_result_page()
        self.go_to(TAB_RESULT)
        self.statusBar().showMessage(
            "Prediction done — final result is on the Result page")

    def on_predict_failed(self, tb: str):
        self.b_predict.setEnabled(True)
        self.statusBar().showMessage("Prediction failed — see log")
        self.log(f"Prediction failed:\n{tb}")
        QtWidgets.QMessageBox.warning(
            self, "Prediction failed",
            "Prediction failed.\n\nTechnical details are in session.log "
            "next to the app.")

    def _aggregate_patients(self, files: list[str], rows: list[tuple]):
        """
        Group per-spectrum predictions into per-PATIENT verdicts when the
        input looks clinical (spectra inside patient subfolders, or a
        folder per patient).  Returns rows of
        (patient, n_spectra, votes, mean_p_positive, verdict)
        or [] when nothing groups.
        """
        if not rows:
            return []
        target = self.spec_path_edit.text().strip()
        root = target if os.path.isdir(target) else os.path.dirname(
            files[0])
        groups: dict[str, list[tuple]] = {}
        multi = False
        for path, row in zip(files, rows, strict=True):
            rel = os.path.relpath(os.path.abspath(path),
                                  os.path.abspath(root))
            parts = rel.replace("\\", "/").split("/")
            # immediate parent folder is the patient in both layouts
            # (<root>/<patient>/f  and  <root>/<class>/<patient>/f)
            patient = parts[-2] if len(parts) > 1 else "(single)"
            multi = multi or len(parts) > 1
            groups.setdefault(patient, []).append(row)
        if not multi:
            return []                     # flat folder: nothing to group
        pos = self._positive_class() or ""
        out_rows = []
        for patient, rs in sorted(groups.items()):
            votes: dict[str, int] = {}
            probs = []
            for _f, cls, p in rs:
                votes[cls] = votes.get(cls, 0) + 1
                probs.append((cls, p))
            mean_p = (np.mean([p for c, p in probs if c == pos])
                      if any(c == pos for c, _p in probs) else 0.0)
            verdict = max(votes.items(), key=lambda kv: kv[1])[0]
            votes_str = " / ".join(f"{c}:{n}" for c, n in
                                   sorted(votes.items()))
            out_rows.append((patient, len(rs), votes_str,
                             float(mean_p), verdict))
        return out_rows

    # ======================================================= status / help
    def update_welcome(self):
        def line(done: bool, text_done: str, text_todo: str) -> str:
            color = "#16a34a" if done else "#94a3b8"
            return (f"<span style='color:{color};font-weight:600'>"
                    + (text_done if done else text_todo) + "</span>")
        n_classes = len(set(l for l in self.labels if l)) if self.spectra else 0
        self.w_lbl_data.setText(line(
            bool(self.spectra),
            f"Data loaded — {len(self.spectra)} spectra, {n_classes} classes",
            "Load your spectra folder (or generate demo data)"))
        self.w_lbl_model.setText(line(
            self.winner is not None,
            (f"Model trained — best: {self.winner.name} "
             f"(F1 {self.winner.macro_f1():.3f})") if self.winner else "",
            "Train & compare models"))
        self.w_lbl_saved.setText(line(
            self.bundle is not None,
            "Model ready — go to Predict",
            "Save the best model"))
        # live hero pills
        n_subj = len(set(self.groups)) if self.groups else 0
        pills_txt = [
            f"{len(self.spectra)} spectra" if self.spectra else "no data",
            f"{n_classes} classes" if n_classes else "0 classes",
            f"{n_subj} patients" if n_subj else "—",
        ]
        for p, txt in zip(self.hero_pills, pills_txt, strict=True):
            p.setText(txt)

    def refresh_nav(self):
        done = [False,
                bool(self.spectra),            # data
                bool(self.spectra),            # prep (defaults ok)
                self.winner is not None,       # train
                self.bundle is not None,       # predict
                bool(self._pred_rows)]         # result
        self.sidebar.set_status(done)
        self.update_title()

    def update_title(self):
        bits = []
        if self.spectra:
            n_classes = len(set(l for l in self.labels if l))
            bits.append(f"{len(self.spectra)} spectra · {n_classes} classes")
        if self.winner is not None:
            bits.append(f"trained: {self.winner.name} "
                        f"(F1 {self.winner.macro_f1():.2f})")
        elif self.spectra:
            bits.append("not trained")
        self.setWindowTitle("Raman Spectra Classifier"
                            + ("  —  " + " · ".join(bits) if bits else ""))

    def friendly_error(self, what: str, exc: Exception):
        self.log(f"{what}:\n" + "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)))
        QtWidgets.QMessageBox.warning(
            self, what,
            f"{what}:\n\n{exc}\n\nTechnical details are printed to the "
            "console.")

    def show_howto(self):
        QtWidgets.QMessageBox.information(self, "How to use this app",
                                          uh.HOW_TO)

    def show_metric_help(self):
        QtWidgets.QMessageBox.information(self, "Metrics explained",
                                          uh.METRIC_HELP)

    def show_about(self):
        QtWidgets.QMessageBox.about(self, "About", uh.about_text(BINDING))

    # ================================================================ misc
    def log(self, msg: str):
        """Log to the console, the in-app Activity-log dock and the
        rotating session.log on disk."""
        from datetime import datetime
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] {msg}", flush=True)
        dock_text = getattr(self, "log_view", None)
        if dock_text is not None:
            dock_text.appendPlainText(f"[{stamp}] {msg}")
            sb = dock_text.verticalScrollBar()
            sb.setValue(sb.maximum())
        try:
            FILE_LOG.info(msg)
        except Exception:
            pass

    def _apply_settings(self):
        s = self.settings
        self._apply_params(s.get("params"))
        if isinstance(s.get("folds"), int):
            self.spin_folds.setValue(int(s["folds"]))
        if isinstance(s.get("seed"), int):
            self.spin_seed.setValue(int(s["seed"]))
        models = s.get("models")
        if isinstance(models, list) and s.get("suite_version") == SUITE_VERSION:
            for name, cb in self.model_checks.items():
                cb.setChecked(name in models)
        geo = s.get("geometry")
        if isinstance(geo, str):
            try:
                self.restoreGeometry(bytes.fromhex(geo))
            except Exception:
                pass

    def closeEvent(self, event):
        # stop background threads before the widgets they signal die —
        # a destroyed running QThread crashes the process
        running = [w for w in (self.worker, self._opt_worker,
                               self._pred_worker, self._honest_worker,
                               self._analysis_worker, self._seq_worker)
                   if w is not None and w.isRunning()]
        if running:
            ans = QtWidgets.QMessageBox.question(
                self, "Work still running",
                "A training / search / prediction is still running.\n\n"
                "Stop it and quit? (A cancelled 3SSE search keeps its "
                "checkpoint — Run resumes it next time.)",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if ans != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        for w in running:
            w.wait(3000)
            if w.isRunning():
                w.terminate()
                w.wait(500)
        if self._source_folder:
            self.settings["folder"] = self._source_folder
        self.settings.pop("files", None)          # legacy setting
        self.settings.update({
            "params": asdict(self.read_params()),
            "folds": self.spin_folds.value(),
            "seed": self.spin_seed.value(),
            "models": [n for n, cb in self.model_checks.items()
                       if cb.isChecked()],
            "suite_version": SUITE_VERSION,
            "geometry": self.saveGeometry().toHex().data().decode(),
        })
        uh.save_settings(self.settings)
        super().closeEvent(event)


# small helpers to avoid importing QtGui color classes at top level
def QtGui_red():
    return qc.QtGui.QColor("#c00000")


def QtGui_black():
    return qc.QtGui.QColor("black")


def run_app() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("Raman Spectra Classifier")
    app.setStyle("Fusion")               # consistent base under the QSS
    app.setStyleSheet(STYLESHEET)
    app.setWindowIcon(uh.make_app_icon())
    try:
        FILE_LOG.info(f"--- session start · binding {BINDING} · "
                      f"python suite v{SUITE_VERSION} ---")
    except Exception:
        pass
    win = MainWindow()
    win.setWindowIcon(uh.make_app_icon())
    win.show()
    return app.exec()
