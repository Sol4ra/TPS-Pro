@echo off
title llama_optimizer
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHON=%~dp0python\python.exe"
set "PIP=%~dp0python\Scripts\pip.exe"
set "GETPIP=%~dp0python\get-pip.py"

if not exist "%PYTHON%" goto :nopython

:: First-time setup (only runs once)
if exist "%~dp0python\Lib\site-packages\optuna" goto :run

echo ============================================================
echo   First-time setup (one time only)
echo ============================================================
echo.
echo [*] Using bundled Python 3.12
"%PYTHON%" --version

if exist "%PIP%" goto :haspip
echo [*] Installing pip...
"%PYTHON%" "%GETPIP%" --quiet --no-warn-script-location
if errorlevel 1 goto :pipfail
echo     Done.

:haspip
echo [*] Installing dependencies...
"%PIP%" install -r requirements.txt --quiet --no-warn-script-location --upgrade --target "%~dp0python\Lib\site-packages"
if errorlevel 1 goto :pipfail
echo     Done.
echo.

:run
"%PYTHON%" -W ignore::FutureWarning -c "import sys; sys.path.insert(0, '..'); from llama_optimizer_2_v3_final.main import main; main()" %*
pause
goto :eof

:nopython
echo [!] Bundled Python not found at: %PYTHON%
echo     The python\ folder may be missing or corrupted.
pause
exit /b 1

:pipfail
echo [!] pip install failed. Check your internet connection.
pause
exit /b 1
