"""
main.py — entry point for the Raman Oral-Cancer Classifier desktop app.

Usage:
    python main.py            # launch the GUI
    python main.py --smoke    # run the headless end-to-end smoke test
"""

from __future__ import annotations

import sys


def run_gui() -> int:
    import gui
    return gui.run_app()


def run_smoke() -> int:
    import smoke_test
    return smoke_test.main()


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        sys.exit(run_smoke())
    sys.exit(run_gui())
