@echo off
rem Start the Hermes agent with D:\BARC as its working directory.
rem It reads D:\BARC\AGENTS.md automatically and works directly on this project.
cd /d D:\BARC
uv run --project D:\hermes-agent python D:\hermes-agent\cli.py %*
