@echo off
set PYTHONUTF8=1
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
"%~dp0.venv\Scripts\python.exe" -m clipper %*
