@echo off
REM install.bat — set up the environment from the PINNED requirements
REM (2026-09-05: this used to install an unpinned subset that silently
REM diverged from the validated environment — no pybaselines meant a
REM different baseline method, no torch meant no CNN/ViT/TabPFN).
REM GPU users: see the torch cu130 note inside requirements.txt.

cd /d "%~dp0"

python -m pip install --upgrade pip
if errorlevel 1 goto :err

python -m pip install -r requirements.txt
if errorlevel 1 goto :err

echo.
echo Done. Start the app with:   python main.py   (or run_app.bat)
pause
exit /b 0

:err
echo.
echo pip install FAILED — check the messages above.
pause
exit /b 1
