<#
.SYNOPSIS
Installs CausalNerve and dependencies on Windows.
#>
param(
    [switch]$Test
)

Write-Host "Installing CausalNerve in editable mode with all dependencies..." -ForegroundColor Cyan
pip install -e .[all]

if ($Test) {
    Write-Host "Running tests..." -ForegroundColor Cyan
    pytest tests/ -v
    Write-Host "Running smoke test..." -ForegroundColor Cyan
    python tests/test_library_smoke.py
}
Write-Host "Installation complete." -ForegroundColor Green
