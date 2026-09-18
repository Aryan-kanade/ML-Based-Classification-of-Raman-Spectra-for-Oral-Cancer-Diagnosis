@echo off
rem Check BARC eval metrics and, if any value is low, have the Hermes agent explain why.
cd /d D:\BARC
uv run --project D:\hermes-agent python check_metrics.py --ai
echo.
pause
