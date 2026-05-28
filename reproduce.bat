@echo off
REM Wrapper for reproducibility checks on Windows
REM Resolves the absolute directory of this batch script, then calls reproduce.py from there.

set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%reproduce.py" %*
