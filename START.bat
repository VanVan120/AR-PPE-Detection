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
echo   [5]  Worker ID tracking   how well workers are re-identified (Phase 5)
echo   [6]  AR glasses view      see the see-through display design (Phase 6)
echo   [7]  Workflow monitor     watch step + mistake + next-step (Phase 3)
echo   [8]  Verify everything    run all the tests (no downloads needed)
echo   [9]  Edge speed test      how fast can it run on this PC (Phase 4)
echo   [0]  Quit
echo.
set "sel="
set /p "sel=Choose 0-9 then press Enter: "

if "%sel%"=="1" goto setup
if "%sel%"=="2" goto check
if "%sel%"=="3" goto demo
if "%sel%"=="4" goto video
if "%sel%"=="5" goto workerid
if "%sel%"=="6" goto arview
if "%sel%"=="7" goto workflow
if "%sel%"=="8" goto tests
if "%sel%"=="9" goto edge
if "%sel%"=="0" exit /b 0
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
echo Setup complete. Options [5], [6], [7] and [8] need no camera or weights.
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

:workerid
cls
echo Phase 5 - Worker ID tracking.
echo.
echo A worker who walks behind something gets a NEW tracking number when they
echo come back, which loses their name and their safety record. This measures
echo how well the system puts them back together again.
echo.
echo Each worker is deliberately hidden, then returns under a new number:
echo   re-ID recall = how often the right worker was recognised  (higher better)
echo   false merge  = how often someone got the WRONG identity   (lower better)
echo.
echo This needs no camera and no downloads.
echo.
"%PY%" -m phase5_workid.reid_eval
echo.
echo Note the "uniform" row: when everyone wears identical PPE, appearance alone
echo cannot tell workers apart - which is why printed ArUco badges exist.
echo Run  phase2\tools\make_worker_tags.py  to print badges for real names.
echo.
pause
goto menu

:arview
cls
echo Phase 6 - what this looks like through AR glasses.
echo.
echo Real see-through glasses only ADD light: black is transparent, so the
echo dark panels that work on a monitor would be invisible on the lens, and
echo the glasses only show the MIDDLE of what the camera sees.
echo.
echo This renders three views side by side so you can compare them.
echo.
"%PY%" -m phase6_arview.preview
echo.
echo Open  outputs\ar_preview.png  to see the result.
echo Panel 1 = monitor HUD (dashed box = what glasses could actually show)
echo Panel 2 = what the projector emits    Panel 3 = what the wearer sees
echo.
pause
goto menu

:workflow
cls
echo Phase 3 - dynamic workflow monitor.
echo.
echo It replays an assembly one step at a time and shows, for each step:
echo   - the step the system recognised
echo   - the next step(s) it expects  ("next: ...")
echo   - a *** MISTAKE line if a step arrives OUT OF ORDER
echo.
echo A fault is deliberately injected so you can watch it get caught.
echo This needs no downloads - the learned workflow models are included.
echo.
"%PY%" -m phase3_activity.tas.demo --inject-fault
echo.
echo Look for "injected fault : CAUGHT" near the bottom.
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
echo Verifying Phase 5 - worker identity, per-worker report, re-ID measurement.
echo Verifying Phase 6 - AR-glasses see-through rendering.
echo This needs no downloads.
echo.
"%PY%" "%ROOT%phase2\tests\test_identity.py"
"%PY%" "%ROOT%phase2\tests\test_workerlog.py"
"%PY%" "%ROOT%phase2\tests\test_arview.py"
"%PY%" "%ROOT%phase5_workid\tests\test_reid_eval.py"
"%PY%" "%ROOT%phase3_activity\tests\test_tas.py"
"%PY%" "%ROOT%phase3_activity\tests\test_mistake.py"
"%PY%" "%ROOT%phase3_activity\tests\test_anticipation.py"
"%PY%" "%ROOT%phase3_activity\tests\test_pipeline.py"
"%PY%" "%ROOT%phase3_activity\tests\test_demo.py"
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
