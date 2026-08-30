@echo off
REM install.bat — set up the environment for the Raman Classifier app
REM (run from the raman_app folder, or double-click this file)

echo === Installing core scientific packages ===
pip install numpy scipy scikit-learn pandas matplotlib PyWavelets joblib
if errorlevel 1 goto :err

echo === Installing XGBoost (optional, model suite is fine without it) ===
pip install xgboost

echo === Installing Qt binding (PyQt5, else PyQt6, else PySide6) ===
pip install PyQt5
if errorlevel 1 (
    echo PyQt5 not available, trying PyQt6 ...
    pip install PyQt6
    if errorlevel 1 (
        echo PyQt6 not available, trying PySide6 ...
        pip install PySide6
    )
)

echo.
echo Done. Start the app with:   python main.py
pause
exit /b 0

:err
echo.
echo pip install FAILED — check the messages above.
pause
exit /b 1
