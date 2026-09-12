"""
main.py — entry point for the Raman Oral-Cancer Classifier desktop app.

Usage:
    python main.py            # launch the GUI
    pythonw main.py           # no console (run_app.bat does this)
"""

from __future__ import annotations

import faulthandler
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))

# native crashes (the Windows exit-127 family) get a Python-level
# traceback dump instead of dying silently.  Under pythonw (run_app.bat)
# there is NO stderr, so the dump also goes to crash.log next to main.py
# — the 2026-09-12 Qt5Core access violation left nothing behind because
# stderr was gone.
try:
    _fh = open(os.path.join(_HERE, "crash.log"), "a", encoding="utf-8")
    _fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} session "
              f"start (pid {os.getpid()}) ===\n")
    _fh.flush()
    faulthandler.enable(file=_fh, all_threads=True)
except (OSError, ValueError):              # unwritable dir etc.
    faulthandler.enable()


def _show_box(tb: str) -> None:
    """Last-resort visible error — even under pythonw (no console)."""
    for mod in ("PyQt5", "PyQt6", "PySide6"):
        try:
            m = __import__(mod, fromlist=["QtWidgets"])
            _app = (m.QtWidgets.QApplication.instance()
                    or m.QtWidgets.QApplication([]))
            m.QtWidgets.QMessageBox.critical(
                None, "Startup failed",
                "The app failed to start:\n\n" + tb[-1500:]
                + "\n\nFull traceback: startup_error.log next to "
                  "main.py.")
            return
        except Exception:                       # noqa: BLE001
            continue


def run_gui() -> int:
    try:
        import gui
        return gui.run_app()
    except SystemExit:
        raise
    except Exception:                           # noqa: BLE001
        tb = traceback.format_exc()
        try:
            with open(os.path.join(_HERE, "startup_error.log"), "w",
                      encoding="utf-8") as fh:
                fh.write(tb)
        except OSError:
            pass
        print(tb, file=sys.stderr)
        _show_box(tb)
        return 1


if __name__ == "__main__":
    sys.exit(run_gui())
