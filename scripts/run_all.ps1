# Pokrece sva 4 eksperimenta (subject-wise k-fold CV) redom.
# Pokretanje iz bilo kog foldera:  .\scripts\run_all.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root

$py = ".\.venv\Scripts\python.exe"
$configs = @(
    "configs/mobilenet_frozen.yaml",
    "configs/mobilenet_finetune.yaml",
    "configs/resnet_frozen.yaml",
    "configs/resnet_finetune.yaml"
)

foreach ($cfg in $configs) {
    Write-Host "`n===== $cfg =====" -ForegroundColor Cyan
    & $py -m src.train --config $cfg
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Neuspeh na $cfg (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "`nSva 4 eksperimenta zavrsena. Rezultati u outputs/<eksperiment>/cv_results.json" -ForegroundColor Green
