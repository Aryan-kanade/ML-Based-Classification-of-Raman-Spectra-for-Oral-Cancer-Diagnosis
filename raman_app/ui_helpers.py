"""
ui_helpers.py — user-friendliness layer: application stylesheet,
plain-language tooltips, persistent settings, and help texts.
"""

from __future__ import annotations

import json
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")

# --------------------------------------------------------------------------
# Stylesheet — premium light design system
# --------------------------------------------------------------------------
STYLESHEET = """
QWidget { font-family: "Segoe UI Variable Display", "Segoe UI", "Segoe UI",
                            sans-serif; font-size: 10pt; }
QMainWindow, #PageHost {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f7f9fd, stop:1 #eef2f8);
}

/* ---------- scrollable pages ---------- */
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }

/* ---------- sidebar navigation ---------- */
#Sidebar { background: rgba(255, 255, 255, 0.72);
           border-right: 1px solid #e2e8f2; }
#BrandTitle { font-size: 14pt; font-weight: 800; color: #1e3a8a;
              letter-spacing: 0.2px; }
#BrandSub { font-size: 8.5pt; color: #64748b; }
QPushButton[nav="true"] {
    text-align: left; padding: 10px 14px; border: 1px solid transparent;
    border-radius: 10px; color: #475569; background: transparent;
    font-size: 10.5pt;
}
QPushButton[nav="true"]:hover { background: #e8edf7; color: #1e293b; }
QPushButton[nav="true"]:pressed { background: #dbe4f5; }
QPushButton[nav="true"]:checked {
    background: #ffffff; color: #4338ca; font-weight: 700;
    border: 1px solid #dfe6f3; border-left: 3px solid #4338ca;
}

/* ---------- cards ---------- */
QFrame#Card { background: #ffffff; border: 1px solid #e6eaf2;
              border-radius: 14px; }
QFrame#DiagPanel { background: #ffffff; border: 1px solid #cdd9e6;
                   border-radius: 12px; }
QFrame#DiagPanel:hover { border-color: #9fb4d4; }
QLabel#CardHeader { color: #1e3a8a; font-weight: 700; font-size: 10.5pt; }
QLabel#CardHint { color: #64748b; font-size: 9pt; }
QLabel#SectionLabel { color: #64748b; font-weight: 700; font-size: 9pt; }
QLabel#PageTitle { color: #0f172a; font-weight: 800; font-size: 15pt;
                   background: transparent; border: none; }
QLabel#BannerTitle { color: #1e3a8a; font-weight: 800; font-size: 13pt;
                     background: transparent; border: none; }
QLabel#SeqCounter { color: #312e81; font-weight: 700; font-size: 15pt; }
QLabel#SeqTop5 { color: #334155;
                 font-family: Consolas, "Cascadia Mono", monospace; }
QLabel#StepChip { border-radius: 10px; padding: 3px 10px; font-size: 9pt;
                  font-weight: 600; background: #eef2ff; color: #4338ca; }
QLabel#StepChip[tone="off"] { background: #e2e8f0; color: #94a3b8; }
QLabel#StepArrow { color: #94a3b8; font-size: 11pt; font-weight: 700;
                   background: transparent; border: none; }

/* ---------- hero panel (welcome page) ---------- */
QFrame#Hero {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #1d4ed8, stop:0.55 #3730a3, stop:1 #6d28d9);
    border-radius: 14px; border: none;
}
QLabel#HeroTitle { color: #ffffff; font-size: 19pt; font-weight: 800; }
QLabel#HeroSub { color: #dbeafe; font-size: 10pt; }
QLabel#HeroGlyph { color: #ffffff; font-size: 26pt; }

/* ---------- result banner stats ---------- */
QLabel#StatName { color: #64748b; font-size: 9pt; font-weight: 600; }
QLabel#StatValue { font-size: 16pt; font-weight: 800; }
QLabel#StatNote { color: #64748b; font-size: 9pt; }

/* ---------- status pills ---------- */
QLabel#Pill { border-radius: 9px; padding: 3px 10px; font-size: 9pt;
              font-weight: 600; background: #eef2ff; color: #4338ca; }
QLabel#Pill[tone="green"]  { background: #dcfce7; color: #15803d; }
QLabel#Pill[tone="amber"]  { background: #fef3c7; color: #b45309; }
QLabel#Pill[tone="red"]    { background: #fee2e2; color: #b91c1c; }
QLabel#Pill[tone="slate"]  { background: #e2e8f0; color: #334155; }
QLabel#Pill[tone="white"]  { background: rgba(255,255,255,0.18);
                             color: #ffffff; border: 1px solid rgba(255,255,255,0.35); }

/* ---------- buttons ---------- */
QPushButton { padding: 6px 14px; border-radius: 8px; background: #ffffff;
              border: 1px solid #cbd5e1; color: #334155; }
QPushButton:hover { background: #f8fafc; border-color: #94a3b8; }
QPushButton:pressed { background: #eef2f7; }
QPushButton:focus { border: 1px solid #60a5fa; }
QPushButton:disabled { color: #94a3b8; background: #f1f5f9;
                       border-color: #e2e8f0; }
QPushButton[primary="true"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #2563eb, stop:1 #4f46e5);
    color: white; font-weight: 600; border: none; padding: 9px 20px;
}
QPushButton[primary="true"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #1d4ed8, stop:1 #4338ca);
}
QPushButton[primary="true"]:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #1e40af, stop:1 #3730a3);
}
QPushButton[primary="true"]:disabled {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #93b4f5, stop:1 #a5b4fc);
}
QPushButton[success="true"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #16a34a, stop:1 #059669);
    color: white; font-weight: 600; border: none; padding: 9px 20px;
}
QPushButton[success="true"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #15803d, stop:1 #047857);
}
QPushButton[success="true"]:disabled { background: #9fcfae; }

/* ---------- inputs ---------- */
QGroupBox { font-weight: 600; border: 1px solid #e6eaf2; border-radius: 10px;
            margin-top: 10px; padding-top: 6px; background: #ffffff; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLineEdit { background: #ffffff; border: 1px solid #cbd5e1;
            border-radius: 8px; padding: 5px 9px; selection-background-color: #c7d7f9; }
QLineEdit:hover { border-color: #94a3b8; }
QLineEdit:focus { border: 2px solid #2563eb; padding: 4px 8px; }
QLineEdit:disabled { color: #64748b; background: #f8fafc; }
QComboBox, QSpinBox, QDoubleSpinBox { border: 1px solid #cbd5e1;
    border-radius: 8px; padding: 4px 9px; background: #ffffff; }
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #94a3b8; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #2563eb;
                                                         padding: 3px 8px; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #ffffff; border: 1px solid #e2e8f0;
    selection-background-color: #e0e7ff; selection-color: #1e293b; }
QListWidget { background: #ffffff; border: 1px solid #e6eaf2;
              border-radius: 8px; }
QListWidget::item { padding: 4px; }
QListWidget::item:selected { background: #e0e7ff; color: #1e293b; }
QCheckBox { spacing: 7px; }

/* ---------- tables ---------- */
QTableWidget { gridline-color: #f1f5f9; background: #ffffff; border: none;
               alternate-background-color: #f8fafc;
               selection-background-color: #dbeafe; selection-color: #0f172a; }
QTableWidget::item { padding: 4px 6px; }
QTableWidget::item:hover { background: #eef2ff; }
QHeaderView::section { background: #f8fafc; padding: 7px 6px; border: none;
    border-bottom: 2px solid #e2e8f0; font-weight: 700; color: #334155; }
QTableCornerButton::section { background: #f8fafc; border: none; }

/* ---------- misc ---------- */
QPlainTextEdit { font-family: Consolas, "Cascadia Mono", "Courier New",
                 monospace; font-size: 9pt; background: #0f172a;
                 color: #e2e8f0; border: none; border-radius: 10px;
                 padding: 6px; }
QProgressBar { border: 1px solid #dbe4f0; border-radius: 7px;
               text-align: center; height: 18px; background: #ffffff;
               font-size: 9pt; color: #334155; }
QProgressBar::chunk {
    border-radius: 6px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #2563eb, stop:1 #6366f1);
}
QToolTip { font-size: 9pt; background: #1e293b; color: #f1f5f9;
           border: 1px solid #334155; padding: 4px; }
QStatusBar { color: #475569; background: transparent; }
QStatusBar::item { border: none; }
QSplitter::handle { background: #dde4ef; border-radius: 2px; }
QTabWidget::pane { background: #ffffff; border: 1px solid #e6eaf2;
                   border-radius: 10px; }
QTabBar::tab { background: transparent; color: #475569; padding: 7px 16px;
               margin-right: 3px; border-top-left-radius: 9px;
               border-top-right-radius: 9px; }
QTabBar::tab:hover { color: #1e3a8a; }
QTabBar::tab:selected { background: #ffffff; color: #1e3a8a;
                        font-weight: 700; border: 1px solid #e6eaf2;
                        border-bottom: none; }
QMenuBar { background: transparent; border: none; }
QMenuBar::item { padding: 5px 9px; border-radius: 7px; }
QMenuBar::item:selected { background: #e8edf7; }
QMenu { border: 1px solid #e2e8f0; background: white; border-radius: 10px;
        padding: 4px; }
QMenu::item { padding: 6px 22px 6px 12px; border-radius: 7px; }
QMenu::item:selected { background: #e0e7ff; color: #1e293b; }
QMenu::separator { height: 1px; background: #e2e8f0; margin: 4px 8px; }

/* ---------- scrollbars ---------- */
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #cbd5e1; border-radius: 5px;
                              min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #94a3b8; }
QScrollBar::handle:vertical:pressed { background: #64748b; }
QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #cbd5e1; border-radius: 5px;
                                min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: #94a3b8; }
QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal { width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ---------- matplotlib navigation toolbar ---------- */
/* solid white, NOT transparent: matplotlib 3.11 tints its icons white
   when the toolbar palette background resolves dark (transparent=black) */
QToolBar { background: #ffffff; border: none; spacing: 3px;
           padding: 0; }
QToolBar QToolButton { padding: 3px 4px; border-radius: 6px;
                       background: transparent; }
QToolBar QToolButton:hover { background: #e8edf7; }
QToolBar QToolButton:pressed { background: #cbd5e1; }
QToolBar QLabel { color: #64748b; font-size: 8pt; }
"""


# --------------------------------------------------------------------------
# UI kit: card shadows, status pills, app icon
# --------------------------------------------------------------------------
def attach_shadow(widget, radius: int = 26, alpha: int = 34,
                  dy: int = 5) -> None:
    """Soft layered drop shadow for a card/frame (Qt QSS has no
    box-shadow)."""
    from qt_compat import QtWidgets, QtGui
    effect = QtWidgets.QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(radius)
    effect.setColor(QtGui.QColor(30, 41, 89, alpha))   # slate-900 tint
    effect.setOffset(0, dy)
    widget.setGraphicsEffect(effect)


def step_chip(text: str, off: bool = False):
    """Live pipeline-step chip (StepChip QSS style); off=True grays it."""
    from qt_compat import QtWidgets
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("StepChip")
    if off:
        lbl.setProperty("tone", "off")
        repolish(lbl)
    return lbl


def pill(text: str, tone: str = "indigo"):
    """
    Colored status chip (QLabel).  tone: indigo|green|amber|red|slate|white.
    ('white' is meant on the gradient hero panel.)
    """
    from qt_compat import QtWidgets
    w = QtWidgets.QLabel(text)
    w.setObjectName("Pill")
    w.setProperty("tone", tone)
    return w


def repolish(widget):
    """Re-apply QSS after a dynamic property change (pill tone, …)."""
    st = widget.style()
    st.unpolish(widget)
    st.polish(widget)


def make_app_icon():
    """Paint the app icon: gradient rounded square + Raman-peak glyph."""
    from qt_compat import QtGui, QtCore
    pm = QtGui.QPixmap(64, 64)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    grad = QtGui.QLinearGradient(0, 0, 64, 64)
    grad.setColorAt(0.0, QtGui.QColor("#2563eb"))
    grad.setColorAt(1.0, QtGui.QColor("#6d28d9"))
    p.setBrush(grad)
    p.setPen(QtCore.Qt.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 16, 16)
    # a little spectrum: baseline with a few Raman-like peaks
    pts = [(10, 46), (16, 43), (20, 24), (24, 44), (29, 41),
           (33, 18), (37, 43), (42, 38), (46, 27), (50, 42), (54, 45)]
    poly = QtGui.QPolygon(
        [QtCore.QPoint(x, y) for x, y in pts])
    p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 3))
    p.drawPolyline(poly)
    p.end()
    return QtGui.QIcon(pm)


# --------------------------------------------------------------------------
# Metric quality colors
# --------------------------------------------------------------------------
def metric_bg(v: float) -> tuple[int, int, int]:
    """Background RGB for a metric value (green >= .90, amber >= .70)."""
    if v is None or v != v:
        return (248, 250, 252)
    if v >= 0.90:
        return (220, 252, 231)
    if v >= 0.70:
        return (254, 243, 199)
    return (254, 226, 226)


def metric_fg(v: float) -> str:
    """Foreground hex for a metric value."""
    if v is None or v != v:
        return "#64748b"
    if v >= 0.90:
        return "#15803d"
    if v >= 0.70:
        return "#b45309"
    return "#b91c1c"


# --------------------------------------------------------------------------
# Plain-language tooltips
# --------------------------------------------------------------------------
TIPS = {
    "folder": "Pick the folder that contains your spectra files "
              "(2 columns: wavenumber + intensity).",
    "class_col": "Class of this spectrum, read from the C-number in the "
                 "filename (e.g. S07_tissue_C8.txt → C8). "
                 "Double-click to correct it.",
    "wavelet": "Wavelet denoising removes instrument noise while keeping "
               "sharp Raman peaks.",
    "wavelet_name": "Wavelet shape used for denoising (sym8 is a good "
                    "default for Raman).",
    "level": "Higher levels remove smoother noise; 3-5 works well.",
    "sg_window": "Savitzky–Golay smoothing window in data points (odd). "
                 "Bigger = smoother but blurs small peaks. 9-15 is typical.",
    "sg_poly": "Polynomial order of the smoothing fit (2-3 is standard).",
    "deriv": "0 = just smooth. 1st/2nd derivative suppresses baseline "
             "drift but makes spectra harder to eyeball.",
    "als_lambda": "Baseline stiffness: 10^5 is a good start. Increase "
                  "(10^6-10^7) for very broad humps.",
    "als_p": "Asymmetry: 0.001–0.05. Smaller p keeps the baseline stiff and under the peaks; larger p lets it climb onto peaks.",
    "als_niter": "Internal iterations — 10 is plenty.",
    "norm": "vector: scales each spectrum to unit length (default). "
            "snv: standardize each spectrum (mean 0, std 1). none: skip.",
    "folds": "Cross-validation folds: each spectrum is predicted by a model "
             "trained on the others — with clinical (patient-grouped) data "
             "that means the OTHER PATIENTS, so the score reflects new people. "
             "5 is standard.",
    "seed": "Random seed for the fold shuffle — same seed, same split, "
            "reproducible results.",
    "sens": "Sensitivity (recall): of all truly positive samples (e.g. "
            "cancer), the fraction the model correctly detects. "
            "High = few missed cases.",
    "spec": "Specificity: of all truly negative samples (e.g. normal), the "
            "fraction correctly cleared. High = few false alarms.",
    "f1": "F1 score: harmonic mean of precision and sensitivity — one "
          "number that rewards getting both right.",
    "save_model": "Saves the winning model together with the exact "
                  "preprocessing settings, so predictions later are "
                  "one click.",
    "predict": "Classifies the selected folder or spectrum file(s) and "
               "shows class probabilities.",
}

METRIC_HELP = """<h3>What the metrics mean</h3>
<p><b>Sensitivity (recall, TPR)</b> — of all truly positive samples
(e.g. cancer spectra), the fraction the model correctly flags.
High sensitivity = few missed cancers.</p>
<p><b>Specificity (TNR)</b> — of all truly negative samples (e.g. normal
spectra), the fraction the model correctly clears.
High specificity = few false alarms.</p>
<p><b>Precision</b> — of the samples predicted positive, the fraction that
truly are positive.</p>
<p><b>F1 score</b> — the harmonic mean of precision and sensitivity. It is
high only when BOTH are high, which is why it is used to pick the winning
model.</p>
<p><b>Macro average</b> — plain average across classes, so every class
counts equally even if some classes have more spectra.</p>
<p><b>Confusion matrix</b> — rows = true class, columns = prediction.
The diagonal is correct; everything else is an error.</p>
<p><b>ROC / AUC</b> — how well the model ranks positives above negatives;
AUC 1.0 = perfect separation, 0.5 = chance.</p>
<p><b>Threshold</b> (2-class problems) — the probability cutoff for calling
a sample positive. Instead of the default 0.5, it is tuned inside each
cross-validation fold to maximize F1, without peeking at the test data.</p>
<p><b>Cross-validation</b> — the spectra are split into k folds; every
spectrum gets predicted by a model that never saw it. The reported numbers
are what you can expect on NEW spectra.</p>"""

def how_to(n_models: int = 24) -> str:
    return f"""<h3>How to use this app</h3>
<ol>
<li><b>Data</b> — load the folder with your spectra (.txt / .dat / .csv,
2 columns). Classes come from the folder layout (clinical tree) or the
C-number in each filename (S07_tissue_C8.txt → class C8) and can be
fixed by double-clicking the Class column.</li>
<li><b>Preprocess</b> — wavelet denoising, Savitzky–Golay smoothing,
ALS baseline correction and normalization. The defaults suit most Raman
data; press <i>Preview spectrum</i> to see the effect.</li>
<li><b>Train &amp; Evaluate</b> — press <i>Start training</i>. The app
compares {n_models} model families (PCA+SVM, Random Forest, PLS-DA,
XGBoost, 1D-CNN, …) under patient-grouped cross-validation and keeps
the best macro-F1. Then <i>Save best model…</i>.</li>
<li><b>Predict</b> — load the saved model once, then choose the folder
with new spectra and press <i>Predict</i>.</li>
</ol>"""


# backward-compatible alias (default registry size)
HOW_TO = how_to()


def about_text(binding: str) -> str:
    return f"""<h3>Raman Spectra Classifier</h3>
<p>Oral-cancer detection from SERS Raman spectra with maximum
macro-F1.</p>
<p>• Patient-grouped cross-validation; nested honest estimate reported
when available<br>
• Binary F1-optimal decision thresholds<br>
• Winner refit on all data and saved with its preprocessing</p>
<p>Qt binding: {binding}</p>"""


# --------------------------------------------------------------------------
# Settings persistence
# --------------------------------------------------------------------------
def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(settings: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
    except Exception:
        pass
