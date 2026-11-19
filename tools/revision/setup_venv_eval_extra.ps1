# Revision R1, Part D: additions to .venv-eval (and only .venv-eval).
#  - ultralytics==8.4.75: the version that trained and evaluated the detectors (.venv-gpu), so the
#    BoT-SORT used here is the tracker code shipped alongside them. Installed without upgrading
#    torch (CPU wheels already present).
#  - lap: required by every ultralytics tracker (trackers/utils/matching.py imports it); it is
#    missing from .venv and .venv-gpu, which must not be changed.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\revision\setup_venv_eval_extra.ps1
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$vp = Join-Path $root ".venv-eval\Scripts\python.exe"
$envdir = Join-Path $root "paper_materials\revision_r1\environments"   # 'env/' is gitignored repo-wide
$log = Join-Path $envdir "venv_eval_setup_extra.log"
function Step($what, [scriptblock]$cmd) {
    "$(Get-Date -Format s) >>> $what" | Add-Content $log
    & $cmd 2>&1 | Add-Content $log
    "$(Get-Date -Format s) <<< exit $LASTEXITCODE" | Add-Content $log
}
$env:YOLO_AUTOINSTALL = "false"
Step "ultralytics 8.4.75 + lap" { & $vp -m pip install "ultralytics==8.4.75" "lap>=0.5.12" }
Step "pip freeze" { & $vp -m pip freeze | Set-Content -Encoding utf8 (Join-Path $envdir "venv_eval_pip_freeze.txt") }
Step "imports" { & $vp -c "import os; os.environ['YOLO_AUTOINSTALL']='false'; import torch, ultralytics, lap, boxmot, trackeval; from ultralytics.trackers.bot_sort import BOTSORT; print('torch', torch.__version__, 'ultralytics', ultralytics.__version__, 'boxmot', boxmot.__version__, 'lap ok, BOTSORT ok')" }
"$(Get-Date -Format s) done" | Add-Content $log
