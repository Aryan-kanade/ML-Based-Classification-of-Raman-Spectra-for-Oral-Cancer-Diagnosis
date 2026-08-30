"""
qt_compat.py — Qt binding compatibility layer.

Tries PyQt5 first, then PyQt6, then PySide6, so the GUI works on any
Python version (some new Python versions may not have PyQt5 wheels yet).

Also exports a handful of enum constants that moved to scoped enums in
Qt6 (e.g. Qt.AlignCenter -> Qt.AlignmentFlag.AlignCenter) so gui.py can
stay binding-agnostic.
"""

BINDING = None  # one of "PyQt5", "PyQt6", "PySide6"

# Load torch's native DLLs BEFORE any Qt binding. On Windows, PyQt5's
# bundled MinGW runtime DLLs (libstdc++/winpthread) get bound to torch's
# c10.dll when Qt loads first, and its initialization then fails
# (WinError 1114). Importing torch first is the fix; harmless elsewhere.
try:
    import torch  # noqa: F401
except Exception:
    pass

try:
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    BINDING = "PyQt5"
except ImportError:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
        BINDING = "PyQt6"
    except ImportError:
        from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
        BINDING = "PySide6"

# PyQt5 spells it pyqtSignal; the Qt6 bindings expose QtCore.Signal
Signal = getattr(QtCore, "Signal", None) or QtCore.pyqtSignal

Qt = QtCore.Qt

# --- Enum compatibility constants (scoped in Qt6, flat in Qt5) -----------
def _enum(group: str, name: str):
    """Return Qt.<group>.<name> on Qt6 bindings, Qt.<name> on Qt5."""
    if BINDING == "PyQt5":
        return getattr(Qt, name)
    return getattr(getattr(Qt, group), name)


ALIGN_CENTER = _enum("AlignmentFlag", "AlignCenter")
ALIGN_LEFT = _enum("AlignmentFlag", "AlignLeft")
ALIGN_RIGHT = _enum("AlignmentFlag", "AlignRight")
ALIGN_BOTTOM = _enum("AlignmentFlag", "AlignBottom")
ALIGN_VCENTER = _enum("AlignmentFlag", "AlignVCenter")
NO_PEN = _enum("PenStyle", "NoPen")
PEN_DASH = _enum("PenStyle", "DashLine")
IS_SELECTABLE = _enum("ItemFlag", "ItemIsSelectable")
IS_ENABLED = _enum("ItemFlag", "ItemIsEnabled")
IS_EDITABLE = _enum("ItemFlag", "ItemIsEditable")
QT_VERTICAL = _enum("Orientation", "Vertical")
QT_HORIZONTAL = _enum("Orientation", "Horizontal")
WAIT_CURSOR = _enum("CursorShape", "WaitCursor")
POINTING_HAND = _enum("CursorShape", "PointingHandCursor")
TEXT_RICH = _enum("TextFormat", "RichText")

# QPainter render hints (scoped in Qt6, flat in Qt5)
_QP = QtGui.QPainter
if BINDING == "PyQt5":
    PAINTER_AA = _QP.Antialiasing
else:
    PAINTER_AA = _QP.RenderHint.Antialiasing

# QAction lives in QtWidgets (Qt5) but QtGui (Qt6)
if BINDING == "PyQt5":
    from PyQt5.QtWidgets import QAction as QAction  # type: ignore
elif BINDING == "PyQt6":
    from PyQt6.QtGui import QAction as QAction  # type: ignore
else:
    from PySide6.QtGui import QAction as QAction  # type: ignore

_HEADER = QtWidgets.QHeaderView
if BINDING == "PyQt5":
    HEADER_STRETCH = _HEADER.Stretch
    HEADER_RESIZE = _HEADER.ResizeToContents
    HEADER_INTERACTIVE = _HEADER.Interactive
else:
    HEADER_STRETCH = _HEADER.ResizeMode.Stretch
    HEADER_RESIZE = _HEADER.ResizeMode.ResizeToContents
    HEADER_INTERACTIVE = _HEADER.ResizeMode.Interactive

_ITEMVIEW = QtWidgets.QAbstractItemView
if BINDING == "PyQt5":
    SELECT_MULTI = _ITEMVIEW.MultiSelection
    SELECT_EXTENDED = _ITEMVIEW.ExtendedSelection
    SELECT_SINGLE = _ITEMVIEW.SingleSelection
    EDIT_NO = _ITEMVIEW.NoEditTriggers
    EDIT_DC = _ITEMVIEW.DoubleClicked
    EDIT_SC = _ITEMVIEW.SelectedClicked
    SELECT_ROWS = _ITEMVIEW.SelectRows
else:
    SELECT_MULTI = _ITEMVIEW.SelectionMode.MultiSelection
    SELECT_EXTENDED = _ITEMVIEW.SelectionMode.ExtendedSelection
    SELECT_SINGLE = _ITEMVIEW.SelectionMode.SingleSelection
    EDIT_NO = _ITEMVIEW.EditTrigger.NoEditTriggers
    EDIT_DC = _ITEMVIEW.EditTrigger.DoubleClicked
    EDIT_SC = _ITEMVIEW.EditTrigger.SelectedClicked
    SELECT_ROWS = _ITEMVIEW.SelectionBehavior.SelectRows
