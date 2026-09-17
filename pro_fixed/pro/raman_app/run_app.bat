@echo off
REM run_app.bat — double-click launcher for the Raman Classifier app
REM (no console window: uses pythonw when available)

cd /d "%~dp0"

where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw main.py
    exit /b 0
)

where python >nul 2>nul
if %errorlevel%==0 (
    start "" python main.py
    exit /b 0
)

echo Neither pythonw nor python was found on PATH.
echo Install Python from https://www.python.org/ and re-run install.bat
pause
exit /b 1
