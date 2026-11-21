# One-time setup on Windows 10/11. Safe to run again: it skips whatever is already there.
# Installs Python 3.11, FFmpeg and Deno (via winget), creates .venv, installs packages, downloads the models,
# then runs "clip doctor".
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Step($t) { Write-Host ""; Write-Host "==> $t" -ForegroundColor Cyan }
function WingetInstall($id) {
    winget list --id $id -e --accept-source-agreements *> $null
    if ($LASTEXITCODE -eq 0) { Write-Host "    $id already installed"; return }
    Write-Host "    installing $id ..."
    winget install --id $id -e --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { Write-Host "    winget returned $LASTEXITCODE for $id (continuing; doctor will tell if it is missing)" -ForegroundColor Yellow }
}
function RefreshPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User") + ";" + "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
}

Step "1/4 Tools (Python 3.11, FFmpeg, Deno)"
if (-not (Have winget)) {
    throw "winget is missing. Install 'App Installer' from the Microsoft Store, then run setup again."
}
WingetInstall "Python.Python.3.11"
WingetInstall "Gyan.FFmpeg"
WingetInstall "DenoLand.Deno"
RefreshPath

Step "2/4 Python environment (.venv)"
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    $candidates = @("$env:LOCALAPPDATA\Programs\Python\Python311\python.exe", "$env:ProgramFiles\Python311\python.exe")
    $base = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($base) { & $base -m venv .venv }
    elseif (Have py) { py -3.11 -m venv .venv }
    else { throw "Python 3.11 not found after install. Close this window, open a new one and run setup again." }
}
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed (see above)" }
& $venvPy -m pip install --upgrade "yt-dlp[default]" yt-dlp-ejs --quiet

Step "3/4 Models (face detector + speech recognition, about 500 MB once)"
New-Item -ItemType Directory -Force models | Out-Null
$yunet = "models\face_detection_yunet_2023mar.onnx"
$yunetUrl = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
if (-not (Test-Path $yunet)) { Invoke-WebRequest -UseBasicParsing -OutFile $yunet -Uri $yunetUrl }
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
& $venvPy -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8', download_root='models'); print('    whisper small.en ready')"

Step "4/4 Check"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = $PSScriptRoot
& $venvPy -m clipper doctor
