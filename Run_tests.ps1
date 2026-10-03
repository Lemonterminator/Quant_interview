param(
    [string]$Python = "python",
    [string]$BuildDir = "out/test-report",
    [string]$Config = "Release",
    [string]$Generator = "Ninja",
    [ValidateRange(2, 6)][int]$Order = 3,
    [switch]$SkipCpp
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$ReportArgs = @("scripts/update_test_report.py", "--build-dir", $BuildDir, "--config", $Config, "--generator", $Generator, "--order", $Order)
if ($SkipCpp) { $ReportArgs += "--skip-cpp" } else { $ReportArgs += "--build" }
& $Python @ReportArgs
exit $LASTEXITCODE
