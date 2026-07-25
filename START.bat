@echo off
setlocal
title AR Safety Monitor - Launcher
cd /d "%~dp0"
set "ROOT=%~dp0"

rem --- find a system Python (used to create the environment the first time) ---
set "SYSPY="
where py     >nul 2>&1 && set "SYSPY=py"
if not defined SYSPY ( where python >nul 2>&1 && set "SYSPY=python" )

rem --- prefer the project's own environment (.venv) if it has been set up ---
set "PY="
if exist "%ROOT%.venv\Scripts\python.exe" set "PY=%ROOT%.venv\Scripts\python.exe"
if not defined PY set "PY=%SYSPY%"

:menu
cls
echo ================================================================
echo    AR SAFETY MONITOR
echo    Workflow Monitoring for AR-Glasses Inspection
echo ================================================================
echo.
if not defined PY (
  echo   [!] Python was not found on this PC.
  echo       Install Python 3.11 or newer from:
  echo           https://www.python.org/downloads/
  echo       During setup, TICK "Add python.exe to PATH", then run this again.
  echo.
  pause
  exit /b 1
)
if exist "%ROOT%.venv\Scripts\python.exe" (echo   Environment : ready) else (echo   Environment : not set up yet  -  run option [1] first)
if exist "%ROOT%phase2\models\best.pt" (echo   Detector    : found) else (echo   Detector    : MISSING  -  needed for the live demo [see README])
echo.
echo   [1]  First-time setup    install everything (a few minutes, once)
echo   [2]  Readiness check      shows what is installed / missing
echo   [3]  Run LIVE demo        use the webcam
echo   [4]  Run demo on a VIDEO  point it at a video file
echo   [5]  Verify Phases 3+4    run the tests (no downloads needed)
echo   [6]  Edge speed test      how fast can it run on this PC (Phase 4)
echo   [7]  Quit
echo.
set "sel="
set /p "sel=Choose 1-7 then press Enter: "

if "%sel%"=="1" goto setup
if "%sel%"=="2" goto check
if "%sel%"=="3" goto demo
if "%sel%"=="4" goto video
if "%sel%"=="5" goto tests
if "%sel%"=="6" goto edge
if "%sel%"=="7" exit /b 0
goto menu

:setup
cls
if not defined SYSPY (
  echo   [!] Python was not found, so the environment cannot be created.
  echo       Install Python 3.11+ from https://www.python.org/downloads/ first.
  echo.
  pause
  goto menu
)
echo Creating a private environment in .venv ...
"%SYSPY%" -m venv "%ROOT%.venv"
if errorlevel 1 ( echo. & echo [!] Could not create the environment. & pause & goto menu )
set "PY=%ROOT%.venv\Scripts\python.exe"
echo Updating the installer ...
"%PY%" -m pip install --upgrade pip
echo.
echo Installing dependencies - this can take several minutes ...
"%PY%" -m pip install -r "%ROOT%phase2\requirements.txt"
if errorlevel 1 ( echo. & echo [!] Install failed - check the internet connection and try again. & pause & goto menu )
echo.
echo Setup complete. You can now use options [2], [3], [4] and [5].
echo.
pause
goto menu

:check
cls
echo Running the readiness check ...
echo.
pushd "%ROOT%phase2"
"%PY%" run.py --check
popd
echo.
pause
goto menu

:demo
cls
if not exist "%ROOT%phase2\models\best.pt" (
  echo   [!] The detector weights are missing:  phase2\models\best.pt
  echo       The live demo needs them - see "Weights ^& data" in the README.
  echo.
  pause
  goto menu
)
echo Starting the live demo.
echo In the video window:   q = quit    s = screenshot    r = record
echo.
pushd "%ROOT%phase2"
"%PY%" run.py
popd
echo.
pause
goto menu

:video
cls
echo Tip: you can drag a video file onto this window, then press Enter.
echo.
set "CLIP="
set /p "CLIP=Video file path: "
if not defined CLIP goto menu
pushd "%ROOT%phase2"
"%PY%" run.py --source "%CLIP%"
popd
echo.
pause
goto menu

:tests
cls
if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo   [note] The environment is not set up. Run option [1] first for a clean run.
  echo.
)
echo Verifying Phase 3 - step recognition, mistake detection, anticipation.
echo Verifying Phase 4 - edge export/benchmark toolkit.
echo This needs no downloads.
echo.
"%PY%" "%ROOT%phase3_activity\tests\test_tas.py"
"%PY%" "%ROOT%phase3_activity\tests\test_mistake.py"
"%PY%" "%ROOT%phase3_activity\tests\test_anticipation.py"
"%PY%" "%ROOT%phase3_activity\tests\test_pipeline.py"
"%PY%" "%ROOT%phase4_deploy\tests\test_edge.py"
echo.
echo Each suite prints its own PASS lines and an ALL_... True summary above.
echo.
pause
goto menu

:edge
cls
if not exist "%ROOT%phase2\models\best.pt" (
  echo   [!] The detector weights are missing:  phase2\models\best.pt
  echo       The speed test needs them - see "Weights ^& data" in the README.
  echo.
  pause
  goto menu
)
echo Phase 4 - measuring how fast the detector runs on THIS computer.
echo.
echo Step 1 of 2: exporting the model to ONNX at 320px (about a minute) ...
echo.
"%PY%" -m phase4_deploy.edge.exporter --weights "%ROOT%phase2\models\best.pt" --formats onnx --imgsz 320
echo.
echo Step 2 of 2: measuring latency ...
echo.
"%PY%" -m phase4_deploy.edge.bench --imgsz 320 --iters 30 --warmup 10
echo.
echo Higher FPS is better. See phase4_deploy\README.md for what these mean.
echo.
pause
goto menu
