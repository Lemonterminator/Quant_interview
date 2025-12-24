Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$SourceDir = Join-Path $RepoRoot "BS_cpp"
$BuildDir = Join-Path $SourceDir "build"
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

function Ensure-CMakeBuildDir {
    param(
        [Parameter(Mandatory = $true)][string]$BuildDir,
        [Parameter(Mandatory = $true)][string]$ExpectedGenerator
    )

    $cache = Join-Path $BuildDir "CMakeCache.txt"
    if (-not (Test-Path $cache)) {
        return
    }

    $genLine = Get-Content $cache | Where-Object { $_ -like "CMAKE_GENERATOR:INTERNAL=*" } | Select-Object -First 1
    if (-not $genLine) {
        return
    }

    $currentGen = ($genLine -split "=", 2)[1].Trim()
    if ($currentGen -ne $ExpectedGenerator) {
        Write-Host "Build dir generator is '$currentGen' but expected '$ExpectedGenerator'; cleaning '$BuildDir'..."
        Remove-Item -Recurse -Force $BuildDir
    }
}

Import-VsDevEnvironment
Ensure-CMakeBuildDir -BuildDir $BuildDir -ExpectedGenerator $Generator

cmake -S $SourceDir -B $BuildDir -G $Generator
cmake --build $BuildDir --config Release

$Exe = Join-Path $BuildDir "Release\\bs_greeks_test.exe"
if (-not (Test-Path $Exe)) {
    throw "Build succeeded but '$Exe' not found."
}
& $Exe

$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\\python.exe"
if (-not (Test-Path $VenvPython)) {
    python -m venv $VenvDir
}
& $VenvPython -m pip -q install --upgrade pip
& $VenvPython -m pip -q install matplotlib numpy
& $VenvPython (Join-Path $RepoRoot "Python_prototype\\plot_cpp_results.py")
