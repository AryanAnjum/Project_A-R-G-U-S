<#
Builds the A.R.G.U.S. Windows installer: Python, PyTorch (CUDA 12.8), the app and every model,
packed into dist\ARGUS-Setup.exe plus ARGUS-Setup-*.bin data files. The installed app runs offline.

Run on your Windows PC from the repo root (needs internet, about 25 GB free disk, 30-90 minutes):

    powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1 `
        -AdapterDir D:\path\to\argus_phased_adapters\part_5_adapter

Needs Inno Setup 6 once:  winget install JRSoftware.InnoSetup

Options
    -AdapterDir   your fine-tune folder (default: <repo>\argus_phased_adapters\part_5_adapter)
    -RunsDir      folder with the sign language runs (default: <repo>\runs)
    -I3D          bsl1k.pth.tar (default: <repo>\models\bsl1k.pth.tar, downloaded if missing)
    -Version      version shown in Windows "Installed apps" (default 1.0.0)
    -SkipPython   reuse the Python already in installer\build\ARGUS\python
    -SkipModels   reuse the models already in installer\build\ARGUS\data
#>
param(
    [string]$AdapterDir = "$PSScriptRoot\..\argus_phased_adapters\part_5_adapter",
    [string]$RunsDir = "$PSScriptRoot\..\runs",
    [string]$I3D = "$PSScriptRoot\..\models\bsl1k.pth.tar",
    [string]$Version = "1.0.0",
    [string]$PythonVersion = "3.11.9",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [switch]$SkipPython,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar on
$Repo = (Resolve-Path "$PSScriptRoot\..").Path
$Build = Join-Path $PSScriptRoot "build"
$Stage = Join-Path $Build "ARGUS"
$Redist = Join-Path $Build "redist"
$Dist = Join-Path $PSScriptRoot "dist"
$Py = Join-Path $Stage "python\python.exe"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }
function Fetch($url, $path) {
    if (Test-Path $path) { return }
    Write-Host "    downloading $url"
    Invoke-WebRequest -Uri $url -OutFile "$path.part" -UseBasicParsing
    Move-Item "$path.part" $path -Force
}

# Inno Setup compiler
$Iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) { throw "Inno Setup 6 not found. Install it with:  winget install JRSoftware.InnoSetup" }

New-Item -ItemType Directory -Force -Path $Build, $Stage, $Redist, $Dist | Out-Null

# 1. Private Python (full CPython from the official nuget package, no system install needed)
if (-not $SkipPython) {
    Step "Python $PythonVersion"
    $nupkg = Join-Path $Build "python-$PythonVersion.zip"
    Fetch "https://www.nuget.org/api/v2/package/python/$PythonVersion" $nupkg
    $unz = Join-Path $Build "python-nuget"
    if (Test-Path $unz) { Remove-Item $unz -Recurse -Force }
    Expand-Archive $nupkg $unz
    if (Test-Path "$Stage\python") { Remove-Item "$Stage\python" -Recurse -Force }
    Copy-Item "$unz\tools" "$Stage\python" -Recurse
    foreach ($exe in "python.exe", "pythonw.exe") {
        if (-not (Test-Path "$Stage\python\$exe")) { throw "The Python package has no $exe" }
    }

    Step "Python packages (PyTorch CUDA 12.8 is about 3 GB)"
    & $Py -m ensurepip --upgrade; Check "ensurepip"
    & $Py -m pip install --upgrade pip wheel; Check "pip upgrade"
    & $Py -m pip install torch torchvision --index-url $TorchIndex; Check "PyTorch install"
    & $Py -m pip install -r "$Repo\requirements.txt" huggingface_hub; Check "requirements install"
}

Step "Checking the bundled Python"
& $Py -c "import torch, torchvision, transformers, peft, qwen_vl_utils, faster_whisper, mediapipe, piper, webview, fastapi, uvicorn, pygame, speech_recognition, cv2; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
Check "import check"

# 2. App code
Step "App code"
$App = Join-Path $Stage "app"
if (Test-Path $App) { Remove-Item $App -Recurse -Force }
New-Item -ItemType Directory -Force -Path "$App\src", "$App\sign_language" | Out-Null
Copy-Item "$Repo\src\*" "$App\src" -Recurse -Exclude "__pycache__"
Copy-Item "$Repo\sign_language\*.py" "$App\sign_language"
Copy-Item "$Repo\LICENSE", "$Repo\README.md" $Stage
Copy-Item "$PSScriptRoot\argus.ico" $Stage
Get-ChildItem $App -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# 3. Models (fine-tune, sign language, Qwen2-VL, Whisper, voices, MediaPipe)
if (-not $SkipModels) {
    Step "Collecting models (about 6-7 GB)"
    & $Py "$PSScriptRoot\collect_models.py" --data "$Stage\data" --adapter $AdapterDir --runs $RunsDir --i3d $I3D
    Check "collect_models"
}
if (-not (Test-Path "$Stage\data\.argus_data")) { throw "No models in $Stage\data. Run without -SkipModels." }

# 4. Windows prerequisites the installer runs only if missing
Step "Windows prerequisites"
Fetch "https://aka.ms/vs/17/release/vc_redist.x64.exe" "$Redist\vc_redist.x64.exe"
Fetch "https://go.microsoft.com/fwlink/?linkid=2124701" "$Redist\MicrosoftEdgeWebView2RuntimeInstallerX64.exe"

# 5. Installer
Step "Building the installer (compressing about 12 GB, this is the slow part)"
& $Iscc "/DStaging=$Stage" "/DRedist=$Redist" "/DOutDir=$Dist" "/DAppVersion=$Version" "$PSScriptRoot\argus.iss"
Check "Inno Setup"

Step "Done"
Get-ChildItem $Dist | Format-Table Name, @{n = "Size (GB)"; e = { "{0:N2}" -f ($_.Length / 1GB) } }
Write-Host "Give your friend everything in $Dist (ARGUS-Setup.exe and all ARGUS-Setup-*.bin files, same folder)."
