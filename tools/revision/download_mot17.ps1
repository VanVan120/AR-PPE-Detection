# Revision R1, Part D: download MOT17.zip from the official site, resuming after any dropped
# connection, until the file has the size the server reports. Data stays in data/ (gitignored).
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\revision\download_mot17.ps1
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$url = "https://motchallenge.net/data/MOT17.zip"
$dest = Join-Path $root "data\mot17\MOT17.zip"
$log = Join-Path $root "data\mot17\download.log"
$expected = 5860214001
New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
"$(Get-Date -Format s) start $url -> $dest (expected $expected bytes)" | Add-Content $log
for ($i = 1; $i -le 300; $i++) {
    $have = if (Test-Path $dest) { (Get-Item $dest).Length } else { 0 }
    if ($have -ge $expected) { break }
    "$(Get-Date -Format s) attempt $i from byte $have" | Add-Content $log
    & curl.exe -L -sS -C - --connect-timeout 30 --speed-limit 1024 --speed-time 120 -o $dest $url 2>&1 | Add-Content $log
    "$(Get-Date -Format s) curl exit $LASTEXITCODE, size $((Get-Item $dest -ErrorAction SilentlyContinue).Length)" | Add-Content $log
    Start-Sleep -Seconds 10
}
$final = if (Test-Path $dest) { (Get-Item $dest).Length } else { 0 }
"$(Get-Date -Format s) done: $final bytes (expected $expected) -> $(if ($final -eq $expected) {'COMPLETE'} else {'INCOMPLETE'})" | Add-Content $log
