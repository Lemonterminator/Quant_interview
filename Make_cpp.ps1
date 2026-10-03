param([ValidateRange(2, 6)][int]$Order = 3)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$SourceDir = Join-Path $RepoRoot "BS_cpp"
$BuildDir = Join-Path $RepoRoot "out\bs_cpp"
$Generator = "Ninja Multi-Config"

function Import-VsDevEnvironment {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere not found at '$vswhere'. Install Visual Studio (Build Tools) with C++ workload."
    }

    $vsInstallPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsInstallPath) {
        throw "Visual Studio with C++ tools not found. Install 'Desktop development with C++'."
    }

    $vsDevCmd = Join-Path $vsInstallPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmd)) {
        throw "VsDevCmd.bat not found at '$vsDevCmd'."
    }

    $cmd = @(
        "call `"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64",
        "set"
    ) -join " && "

    cmd.exe /c $cmd | ForEach-Object {
        if ($_ -match "^(?<k>[^=]+)=(?<v>.*)$") {
            Set-Item -Path ("Env:" + $Matches["k"]) -Value $Matches["v"]
        }
    }
}

Import-VsDevEnvironment

cmake -S $SourceDir -B $BuildDir -G $Generator
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }
cmake --build $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { throw "C++ build failed." }

python (Join-Path $RepoRoot "scripts/update_test_report.py") --build-dir $BuildDir --config Release --order $Order
if ($LASTEXITCODE -ne 0) { throw "Tests did not all pass. See docs/test_status.md for evidence and diagnostics." }

$Exe = Join-Path $BuildDir "Release\\bs_greeks_test.exe"
if (-not (Test-Path $Exe)) {
    throw "Build succeeded but '$Exe' not found."
}
& $Exe --order $Order
if ($LASTEXITCODE -ne 0) { throw "The C++ assignment sweep failed." }

$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\\python.exe"
if (-not (Test-Path $VenvPython)) {
    python -m venv $VenvDir
}
& $VenvPython -m pip -q install --upgrade pip
& $VenvPython -m pip -q install matplotlib numpy
& $VenvPython (Join-Path $RepoRoot "experiments\\assignment\\plot_cpp_results.py")
