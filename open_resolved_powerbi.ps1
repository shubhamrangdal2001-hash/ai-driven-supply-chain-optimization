# Rebuild and open the resolved Power BI dashboard (Desktop only)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PowerBiDir = Join-Path $Root "powerbi"
$Pbip = Join-Path $PowerBiDir "SupplyChainAnalytics.pbip"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  RESOLVED Power BI - Rebuild + Open (Desktop ONLY)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "[1/3] Refreshing real data CSVs ..." -ForegroundColor Yellow
Push-Location $Root
python powerbi_data_prep.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

Write-Host "[2/3] Rebuilding dashboard (4 pages, 18 visuals) ..." -ForegroundColor Yellow
Push-Location $PowerBiDir
python build_powerbi_desktop.py
if ($LASTEXITCODE -ne 0) { Pop-Location; Pop-Location; exit $LASTEXITCODE }
Pop-Location
Pop-Location

if (-not (Test-Path $Pbip)) {
    throw "Build failed - PBIP not found: $Pbip"
}

Write-Host "[3/4] Running project tests ..." -ForegroundColor Yellow
python (Join-Path $PowerBiDir "test_powerbi.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/4] Opening Power BI Desktop ..." -ForegroundColor Yellow
Write-Host "  (Close any open Power BI window first, or pages may stay blank)" -ForegroundColor DarkYellow
Write-Host "  $Pbip" -ForegroundColor Gray
Start-Process -FilePath $Pbip

Write-Host ""
Write-Host "  SUCCESS - Open in Power BI DESKTOP (not the website)." -ForegroundColor Green
Write-Host "  Do NOT use app.powerbi.com - local CSV data only works in Desktop." -ForegroundColor Yellow
Write-Host ""
Write-Host "  In Desktop:" -ForegroundColor White
Write-Host "    1. Home -> Refresh" -ForegroundColor Gray
Write-Host "    2. Browse 4 pages at bottom: Executive | Logistics | Inventory | Demand" -ForegroundColor Gray
Write-Host ""
