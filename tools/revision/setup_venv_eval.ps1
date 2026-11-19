# Revision R1, Part D: the separate evaluation environment .venv-eval (gitignored).
# .venv and .venv-gpu are not touched. CPU-only torch, boxmot (StrongSORT, Deep OC-SORT with
# their default OSNet ReID weights) and TrackEval (HOTA / CLEAR / Identity metrics).
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\revision\setup_venv_eval.ps1
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$py = "C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe"
$venv = Join-Path $root ".venv-eval"
$envdir = Join-Path $root "paper_materials\revision_r1\environments"   # 'env/' is gitignored repo-wide
$log = Join-Path $envdir "venv_eval_setup.log"
New-Item -ItemType Directory -Force $envdir | Out-Null
function Step($what, [scriptblock]$cmd) {
    "$(Get-Date -Format s) >>> $what" | Add-Content $log
    & $cmd 2>&1 | Add-Content $log
    "$(Get-Date -Format s) <<< exit $LASTEXITCODE" | Add-Content $log
}
Step "create venv with $py" { & $py -m venv $venv }
$vp = Join-Path $venv "Scripts\python.exe"
Step "upgrade pip (inside .venv-eval only)" { & $vp -m pip install --upgrade pip }
Step "torch + torchvision, CPU wheels" { & $vp -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision }
Step "boxmot" { & $vp -m pip install boxmot }
Step "TrackEval from the official repository" { & $vp -m pip install "git+https://github.com/JonathonLuiten/TrackEval.git" }
Step "pip freeze" { & $vp -m pip freeze | Set-Content -Encoding utf8 (Join-Path $envdir "venv_eval_pip_freeze.txt") }
Step "python version" { & $vp --version }
"$(Get-Date -Format s) done" | Add-Content $log
