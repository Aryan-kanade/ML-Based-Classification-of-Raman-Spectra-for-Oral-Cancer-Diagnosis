"""
plotting.py — matplotlib helpers plus Qt-embedded canvases with toolbars.

`apply_style()` sets one consistent look for every plot (subtle grid, no
top/right spines, bold small titles, soft-framed legends). The Qt canvas
pins QT_API to the binding chosen in qt_compat so GUI and matplotlib
always share one binding; `canvas_with_toolbar()` additionally mounts a
navigation toolbar (zoom / pan / save-PNG) under every plot. Metric/plot
helpers are backend-agnostic.
"""

from __future__ import annotations

import os

import numpy as np

PALETTE = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c",
           "#0891b2", "#be185d", "#4d7c0f", "#57534e", "#0e7490"]

# semantic colors used by the result plots
COL_RAW = "#94a3b8"        # faded raw reference traces
COL_MAIN = "#2563eb"       # primary blue
COL_RESULT = "#dc2626"     # processed / ROC curve
COL_WIN = "#16a34a"        # winner / predicted class
# shared graph vocabulary (inline graphs must use these, never raw hex)
COL_SIGN_POS = COL_RESULT  # toward the positive class / above zero
COL_SIGN_NEG = COL_MAIN    # away from the positive class / below zero
COL_ANNOT = "#b45309"      # amber annotations (thresholds, band markers)
COL_ZERO = "#64748b"       # zero / mean reference lines
COL_HIST = "#93c5fd"       # histogram bars below the cut-off
COL_BAND = "#f59e0b"       # biochemical band shading
COL_TEXT = "#334155"       # value labels on bars


def class_color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def apply_style():
    """One-time global matplotlib styling for all figures."""
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "axes.labelcolor": "#475569",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": "#64748b",
        "ytick.color": "#64748b",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#e2e8f0",
        "legend.fancybox": False,
        "legend.fontsize": 8,
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.prop_cycle": cycler("color", PALETTE),
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


# --------------------------------------------------------------------------
# Qt-embedded canvas (lazy backend import)
# --------------------------------------------------------------------------
class MatplotlibCanvas:
    """Factory-style wrapper; returns a FigureCanvasQTAgg widget."""

    _canvas_cls = None

    def __new__(cls, width=6, height=4, dpi=100):
        if cls._canvas_cls is None:
            import qt_compat
            os.environ.setdefault("QT_API", qt_compat.BINDING)
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as Canvas)
            from matplotlib.figure import Figure
            apply_style()
            cls._canvas_cls = Canvas
            cls._figure_cls = Figure
        fig = cls._figure_cls(figsize=(width, height), dpi=dpi,
                              constrained_layout=True)
        return cls._canvas_cls(fig)


def canvas_with_toolbar(width=6, height=4, dpi=100):
    """
    Canvas plus a matplotlib navigation toolbar (home / pan / zoom /
    configure / save-figure). Returns (container_widget, canvas) — add the
    container to a Qt layout and keep drawing on the canvas as before.
    """
    canvas = MatplotlibCanvas(width, height, dpi)
    import qt_compat as qc
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    container = qc.QtWidgets.QWidget()
    v = qc.QtWidgets.QVBoxLayout(container)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(0)
    toolbar = NavigationToolbar2QT(canvas, container)
    toolbar.setIconSize(qc.QtCore.QSize(16, 16))
    toolbar.setMovable(False)
    toolbar.setFloatable(False)
    v.addWidget(toolbar)
    v.addWidget(canvas, 1)
    return container, canvas


def clear(canvas):
    canvas.figure.clf()
    return canvas.figure.gca()


# --------------------------------------------------------------------------
# Spectrum plots
# --------------------------------------------------------------------------
def plot_spectra(ax, spectra: list[tuple[np.ndarray, np.ndarray, str]],
                 title: str = "Spectra", offset: float = 0.0):
    """spectra: list of (wavenumbers, intensities, label)."""
    ax.clear()
    for i, (wn, it, lab) in enumerate(spectra):
        ax.plot(wn, it + i * offset, lw=1.0, color=class_color(i),
                label=lab, solid_capstyle="round", alpha=0.95)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.margins(x=0.01)
    if title:
        ax.set_title(title)
    # x ASCENDING — low wavenumbers on the LEFT (user request
    # 2026-09-08; the old high->low "Raman convention" inversion was
    # removed everywhere so all axes read normally)
    if 0 < len(spectra) <= 10:
        ax.legend()


def plot_preprocess_preview(ax_before, ax_after, wn, raw, processed,
                            title: str = "", overlay=None):
    """overlay: optional (wn, y) pair for the raw reference curve on the
    'after' panel when it must not span the full wn range."""
    ax_before.clear()
    ax_before.plot(wn, raw, lw=0.9, color=COL_MAIN, solid_capstyle="round")
    ax_before.set_title("Raw (on common grid)")
    ax_before.set_xlabel("Raman shift (cm$^{-1}$)")
    ax_before.set_ylabel("Intensity (a.u.)")

    ax_after.clear()
    own = overlay if overlay is not None else (wn, raw)
    ax_after.plot(own[0], own[1], lw=0.8, color=COL_RAW, alpha=0.45,
                  label="raw (left axis)")
    # twin y-axis — raw (~1e3 counts) vs processed (~1e-2 after
    # vector-norm) differ ~1e5x; a shared axis squashes the processed curve
    # into a flat line.
    ax2 = ax_after.twinx()
    ax2.plot(own[0], processed, lw=1.0, color=COL_RESULT,
             solid_capstyle="round", label="preprocessed (right axis)")
    ax_after.set_title("Preprocessed")
    ax_after.set_xlabel("Raman shift (cm$^{-1}$)")
    ax2.set_ylabel("Processed intensity", color=COL_RESULT)
    ax2.tick_params(axis="y", labelcolor=COL_RESULT)
    h1, l1 = ax_after.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_after.legend(h1 + h2, l1 + l2, loc="best", fontsize=8,
                    framealpha=0.85)
    if title:
        ax_before.figure.suptitle(title, fontsize=9, fontweight="bold",
                                  color="#1e293b")


# --------------------------------------------------------------------------
# Result plots
# --------------------------------------------------------------------------
def plot_confusion_matrix(ax, cm: np.ndarray, classes: list[str],
                          title: str = "Confusion matrix (pooled CV folds)"):
    """Confusion matrix heatmap; fills the axes (no forced square)."""
    ax.clear()
    n = len(classes)
    row_totals = cm.sum(axis=1)
    im = ax.imshow(cm, cmap="Blues", vmin=0, aspect="auto")
    ax.set_xticks(range(n), classes, rotation=0 if n <= 4 else 30,
                  ha="center" if n <= 4 else "right")
    ax.set_yticks(range(n), [f"{c}  (n={int(t)})"
                             for c, t in zip(classes, row_totals, strict=True)])
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    vmax = cm.max() if cm.max() > 0 else 1
    # cell text: counts (+ % of true class); shrink font for dense matrices
    cell_fs = 11 if n <= 3 else (9 if n <= 5 else 7)
    for i in range(n):
        row_total = cm[i].sum()
        for j in range(n):
            v = int(cm[i, j])
            label = str(v)
            if row_total > 0 and n > 0:
                label += f"\n({v / row_total:.0%})"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=cell_fs,
                    color="white" if v > vmax * 0.55 else "#1e293b")
    try:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, shrink=0.85,
                                  pad=0.02)
        cbar.set_label("count", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    except Exception:
        pass


def plot_prob_histogram(ax, p_pos: np.ndarray, threshold: float | None = None,
                        pos_name: str = "positive"):
    """
    Histogram of per-spectrum P(positive) with the decision threshold
    marked — shows where the predictions sit relative to the cut-off.
    """
    ax.clear()
    p_pos = np.asarray(p_pos, dtype=float)
    p_pos = p_pos[~np.isnan(p_pos)]
    bins = np.linspace(0.0, 1.0, 11)
    ax.hist(p_pos, bins=bins, color=COL_HIST, edgecolor="white",
            linewidth=0.8)
    # color the bars left/right of the cut-off differently
    if len(p_pos):
        cut = threshold if threshold is not None else 0.5
        centers = (bins[:-1] + bins[1:]) / 2
        for patch, c in zip(ax.patches, centers, strict=False):
            patch.set_facecolor(COL_RESULT if c >= cut else COL_HIST)
    if threshold is not None:
        ax.axvline(threshold, color="#b45309", lw=1.4, ls="--")
        ax.annotate(f"cut {threshold:.2f}", xy=(threshold, ax.get_ylim()[1]),
                    xytext=(3, -2), textcoords="offset points",
                    va="top", ha="left", fontsize=8, color="#b45309")
    ax.set_xlabel(f"P({pos_name})")
    ax.set_ylabel("spectra")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(range(0, int(ax.get_ylim()[1]) + 2))
    ax.set_title(f"P({pos_name}) per spectrum", fontsize=10)


def plot_prediction_spectra(ax, wn, spectra, mean_trace=None,
                            reference=None, bands=None):
    """
    Overlay of the predicted (preprocessed) spectra: faint individual
    traces, a bold mean, an optional dashed paired normal reference, and
    light amber spans over the top discriminative bands.

    bands: list of (center_cm1, share, name[, sign]) — top 3 are drawn.
    """
    ax.clear()
    spectra = list(spectra)
    for y in spectra[:30]:
        ax.plot(wn, y, lw=0.6, color=COL_RAW, alpha=0.35)
    if mean_trace is not None:
        ax.plot(wn, mean_trace, lw=1.7, color=COL_MAIN,
                label=f"mean (n={len(spectra)})", solid_capstyle="round")
    if reference is not None:
        ax.plot(wn, reference, lw=1.2, color="#059669", ls="--",
                label="patient normal reference")
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.margins(x=0.01)
    if bands:
        for b in bands[:3]:
            center, name = float(b[0]), str(b[2] or "")
            ax.axvspan(center - 20, center + 20, color=COL_BAND,
                       alpha=0.14, zorder=0, lw=0)
            label = f"{center:.0f} cm⁻¹ {name}".strip()
            ymin, ymax = ax.get_ylim()
            ax.annotate(label, xy=(center, ymax), xytext=(0, -3),
                        textcoords="offset points", rotation=90,
                        ha="right", va="top", fontsize=8, color=COL_ANNOT)
    traces = [t for t in (mean_trace, reference) if t is not None]
    if traces:
        ax.legend(loc="upper left", fontsize=8)


def plot_calibration(ax, bins, cal_bins=None,
                     title="Calibration (out-of-fold)"):
    """
    Reliability diagram: predicted P(positive) vs observed rate
    (bins from clinical.calibration_bins).  `cal_bins` draws a second
    (post-calibration) series for before/after comparison.
    """
    ax.clear()
    ax.plot([0, 1], [0, 1], "--", color=COL_RAW, lw=0.9)
    mp = [b[0] for b in bins]
    ob = [b[1] for b in bins]
    ax.plot(mp, ob, "o-", color=COL_MAIN, ms=5, lw=1.4, mec="white",
            mew=0.6, label="raw" if cal_bins else None)
    if cal_bins:
        mp2 = [b[0] for b in cal_bins]
        ob2 = [b[1] for b in cal_bins]
        ax.plot(mp2, ob2, "s-", color=COL_RESULT, ms=5, lw=1.4,
                mec="white", mew=0.6, label="calibrated")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("predicted P(positive)")
    ax.set_ylabel("observed positive rate")
    # 0 starts at the exact left/bottom edge (2026-09-08: the old -0.02
    # padding inset the 0 tick; user request — also matches plot_pr)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    # probability axes ALWAYS run 0 -> 1 left-to-right (pin: spectral
    # plots invert, these must never — user report 2026-09-08)
    ax.xaxis.set_inverted(False)
    ax.set_title(title, fontsize=10)


def plot_dca(ax, thresholds, nb_model, nb_all,
             title="Decision curve (net benefit)"):
    """
    Clinical utility: net benefit of treating by the model vs treat-all
    vs treat-none (data from clinical.decision_curve).
    """
    ax.clear()
    ax.plot(thresholds, nb_model, lw=1.7, color=COL_RESULT,
            label="treat by model")
    ax.plot(thresholds, nb_all, lw=1.1, color="#059669", ls="--",
            label="treat all")
    ax.axhline(0, color=COL_RAW, lw=0.9, label="treat none")
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    lo = min(-0.05, float(np.min(nb_model)) - 0.02)
    hi = max(0.12, float(np.max(nb_model)) * 1.15)
    ax.set_ylim(lo, hi)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right")


def plot_class_distribution(ax, labels: list[str]):
    ax.clear()
    classes, counts = np.unique(np.asarray(labels), return_counts=True)
    order = np.argsort(classes)
    classes, counts = classes[order], counts[order]
    total = int(counts.sum())
    bars = ax.barh(classes, counts,
                   color=[class_color(i) for i in range(len(classes))],
                   edgecolor="white", linewidth=0.8, height=0.62)
    if total:
        ax.bar_label(bars, labels=[f"{v}  ({v / total:.0%})"
                                   for v in counts],
                     fontsize=9, fontweight="bold", color="#334155",
                     padding=3)
    ax.set_title("Samples per class")
    ax.set_xlabel("number of spectra")
    ax.set_xlim(0, max(counts) * 1.3 if len(counts) else 1)
    ax.margins(y=0.05)


def plot_pr(ax, prec, rec, ap: float, label: str = ""):
    """Precision-recall curve with the average-precision value."""
    ax.clear()
    ax.plot(rec, prec, lw=1.8, color=COL_RESULT,
            label=f"{label} (AP = {ap:.3f})" if label
            else f"AP = {ap:.3f}")
    ax.fill_between(rec, prec, alpha=0.12, color=COL_RESULT)
    ax.set_xlabel("Recall (sensitivity)", fontsize=9)
    ax.set_ylabel("Precision (PPV)", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_inverted(False)   # probability axis: 0 always on the LEFT
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("Precision–Recall — out-of-fold", fontsize=10)


def plot_roc(ax, fpr, tpr, auc: float, label: str = ""):
    ax.clear()
    ax.plot(fpr, tpr, lw=1.8, color=COL_RESULT,
            label=f"{label} (AUC = {auc:.3f})" if label
            else f"AUC = {auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.12, color=COL_RESULT)
    ax.plot([0, 1], [0, 1], "--", color=COL_RAW, lw=0.9)
    ax.text(0.62, 0.55, "chance", rotation=45, color=COL_RAW,
            fontsize=8, ha="center", va="center")
    # Youden J: the operating point farthest above the chance line
    try:
        j = int(np.argmax(np.asarray(tpr) - np.asarray(fpr)))
        ax.plot([fpr[j]], [tpr[j]], "o", ms=6, color="#b45309",
                mec="white", mew=0.8, zorder=5,
                label=f"Youden J point (spec {1 - fpr[j]:.2f} / "
                      f"sens {tpr[j]:.2f})")
    except Exception:
        pass
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    # 0 starts at the exact left/bottom edge (2026-09-08: the old -0.02
    # padding inset the 0 tick; user request — matches plot_pr beside it)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_inverted(False)   # probability axis: 0 always on the LEFT
    ax.set_title("ROC (out-of-fold)", fontsize=10)
    ax.legend(loc="lower right")


def plot_perclass_metrics(ax, classes: list[str], per_class: dict,
                          title: str = "Per-class metrics"):
    """Grouped bars of sensitivity / specificity / F1 per class.

    per_class: {class: {key: (mean, std)}} with keys sens / spec / f1.
    """
    ax.clear()
    n = len(classes)
    x = np.arange(n)
    w = 0.26
    for k, (key, lab, col) in enumerate((
            ("sens", "Sensitivity", COL_MAIN),
            ("spec", "Specificity", "#059669"),
            ("f1", "F1", "#ea580c"))):
        vals = [per_class[c][key][0] for c in classes]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=lab, color=col,
                      edgecolor="white", linewidth=0.6)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1.5,
                     color="#475569")
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("score")
    ax.set_title(title, fontsize=10)
    ax.legend(ncol=3, loc="upper center", columnspacing=1.2,
              handlelength=1.4)


def plot_prediction(fig, wn, y_raw, y_proc, pred: str, probs: dict,
                    title: str = ""):
    """
    Two-panel prediction result: the spectrum (raw + preprocessed) with an
    annotation box, and a sorted probability bar chart with the predicted
    class highlighted. Figure-level: clears and rebuilds `fig`.
    """
    fig.clf()
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
    ax = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[0, 1])

    pmax = float(probs.get(pred, float("nan")))

    ax.plot(wn, y_raw, lw=0.8, color=COL_RAW, alpha=0.6,
            label="raw (interpolated)")
    ax.plot(wn, y_proc, lw=1.0, color=COL_RESULT, solid_capstyle="round",
            label="preprocessed")
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend(loc="upper left")
    ax.set_title(title or "Spectrum", fontsize=10)
    box = dict(boxstyle="round,pad=0.5", facecolor="#dcfce7",
               edgecolor=COL_WIN, lw=1.4)
    ax.text(0.98, 0.05,
            f"Prediction:  {pred}\np = {pmax:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold", color="#14532d", bbox=box)

    items = sorted(probs.items(), key=lambda kv: kv[1])   # smallest on top
    cls = [c for c, _ in items]
    vals = [v for _, v in items]
    colors = [COL_WIN if c == pred else "#cbd5e1" for c in cls]
    bars = axp.barh(cls, vals, color=colors, edgecolor="white",
                    linewidth=0.6, height=0.6)
    axp.bar_label(bars, fmt="%.3f", fontsize=8, color=COL_TEXT,
                  padding=3)
    axp.set_xlim(0, 1.15)
    axp.set_xlabel("P(class)")
    axp.set_title("Class probabilities", fontsize=10)
    axp.margins(y=0.08)


# --------------------------------------------------------------------------
# shared bar-chart looks (Result distribution, Predict overview,
# biochemistry deltas, NMF deltas, LOPO, local explanation)
# --------------------------------------------------------------------------
def plot_count_bars(ax, labels, counts, title="Class counts",
                    positive=None):
    """Vertical class-count bars; the positive class gets the result
    red, the rest follow the palette."""
    # 2026-09-08 audit: a NaN count crashed bar_label/ylim (int(NaN));
    # counts are integers in practice — sanitize defensively anyway
    counts = [int(v) if np.isfinite(v) else 0 for v in counts]
    cols = [COL_RESULT if lab == positive else class_color(i)
            for i, lab in enumerate(labels)]
    bars = ax.bar(labels, counts, color=cols, edgecolor="white",
                  linewidth=0.8, width=0.62)
    total = sum(counts) or 1
    ax.bar_label(bars, labels=[f"{v} ({v / total:.0%})" for v in counts],
                 fontsize=9, fontweight="bold", color=COL_TEXT, padding=3)
    ax.set_ylabel("spectra")
    ax.set_title(title)
    if len(counts):
        ax.set_ylim(0, max(counts) + max(1, int(max(counts) * 0.25)))
    return ax


def plot_sign_bars(ax, labels, values, ylabel="", title="",
                   orientation="v", red_below=None, fmt="%+.2f"):
    """
    Sign-colored bars + zero reference line + value labels — the shared
    look of the paired biochemistry deltas, NMF component deltas, LOPO
    per-patient accuracy and the local-explanation bars.
    orientation 'v' draws up/down bars, 'h' left/right bars.
    red_below (e.g. 0.5 for LOPO accuracy) colors weak bars red
    instead of using the sign convention.
    """
    vals = np.asarray(values, dtype=float)
    if orientation == "h":
        cols = ([COL_RESULT if v < red_below else COL_SIGN_NEG
                 for v in vals] if red_below is not None else
                [COL_SIGN_POS if v >= 0 else COL_SIGN_NEG for v in vals])
        bars = ax.barh(labels, vals, color=cols, edgecolor="white",
                       linewidth=0.6)
        ax.axvline(0, color=COL_ZERO, lw=0.8)
        for b, v in zip(bars, vals, strict=True):
            # fmt is %-STYLE ("%+.2f") — f-string {v:{fmt}} rejects it
            # (Invalid format specifier); this dead-code path had never
            # run until the 2026-09-06 audit test pinned it
            ax.text(v, b.get_y() + b.get_height() / 2, " " + fmt % v,
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=8, color=COL_TEXT)
        ax.invert_yaxis()
    else:
        cols = ([COL_RESULT if v < red_below else COL_SIGN_NEG
                 for v in vals] if red_below is not None else
                [COL_SIGN_POS if v >= 0 else COL_SIGN_NEG for v in vals])
        bars = ax.bar(labels, vals, color=cols, edgecolor="white",
                      linewidth=0.6)
        ax.axhline(0, color=COL_ZERO, lw=0.8)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt % v,
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, color=COL_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax
