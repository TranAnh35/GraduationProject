param(
    [ValidateSet('all', 'plan', 'run', 'status', 'aggregate')]
    [string]$Action = 'status',
    [string]$Root = "$PSScriptRoot\experiments\campaigns\pect_v1",
    [string]$DataDir = "$PSScriptRoot\..\..\..\data",
    [ValidateSet('full', 'smoke')][string]$Profile = 'full',
    [string]$Device = 'cuda',
    [switch]$KeepGoing,
    [string]$Settings
)
$ErrorActionPreference = 'Stop'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$pectPython = 'C:\Users\tranl\miniconda3\envs\ndt\python.exe'
if ($Action -eq 'all') {
    if (-not (Test-Path -LiteralPath "$Root\plan.json")) {
        $pectPlan = @("$PSScriptRoot\run_campaign.py", 'plan', '--root', $Root,
            '--data-dir', $DataDir, '--transform-csv', "$DataDir\ground_truth\raw_tdms_rotate_crop.csv", '--profile', $Profile)
        if ($Settings) { $pectPlan += @('--settings', $Settings) }
        & $pectPython @pectPlan
        if ($LASTEXITCODE -ne 0) { throw 'Campaign planning failed' }
    }
    $Action = 'run'
}
$pectArguments = @("$PSScriptRoot\run_campaign.py", $Action, '--root', $Root)
if ($Action -eq 'plan') {
    $pectArguments += @('--data-dir', $DataDir, '--transform-csv', "$DataDir\ground_truth\raw_tdms_rotate_crop.csv", '--profile', $Profile)
    if ($Settings) { $pectArguments += @('--settings', $Settings) }
}
if ($Action -eq 'run') {
    $pectArguments += @('--device', $Device)
    if ($KeepGoing) { $pectArguments += '--keep-going' }
}
& $pectPython @pectArguments
if ($LASTEXITCODE -ne 0) { throw "PECT campaign exited with code $LASTEXITCODE" }
