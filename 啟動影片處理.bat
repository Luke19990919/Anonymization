@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 goto error
)
.venv\Scripts\python.exe scripts/fetch_assets.py
if errorlevel 1 goto error
.venv\Scripts\python.exe -c "import video_privacy; assert video_privacy.__version__ == '0.2.7rc3'" >nul 2>&1
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install -c constraints-tested.txt .
  if errorlevel 1 goto error
)
.venv\Scripts\python.exe -m video_privacy.cli serve --models models --port 8765 --open-browser
if errorlevel 1 goto error
exit /b 0
:error
echo Startup failed. Install Python 3.11 and Git, then retry.
pause
exit /b 1
