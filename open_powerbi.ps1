# Open AI-Driven Supply Chain dashboard in Microsoft Power BI Desktop
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PowerBiDir = Join-Path $Root "powerbi"
$Pbip = Join-Path $PowerBiDir "SupplyChainAnalytics.pbip"

function Find-PowerBIDesktop {
    $candidates = @(
        "${env:ProgramFiles}\Microsoft Power BI Desktop\bin\PBIDesktop.exe",
        "${env:ProgramFiles(x86)}\Microsoft Power BI Desktop\bin\PBIDesktop.exe",
        "${env:LocalAppData}\Microsoft\WindowsApps\PBIDesktop.exe"
    )

    foreach ($key in @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"
    )) {
        Get-ItemProperty $key -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -like "*Power BI Desktop*" } |
            ForEach-Object {
                if ($_.InstallLocation) {
                    $candidates += (Join-Path $_.InstallLocation "bin\PBIDesktop.exe")
                    $candidates += (Join-Path $_.InstallLocation "PBIDesktop.exe")
                }
                if ($_.DisplayIcon -and $_.DisplayIcon -like "*.exe*") {
                    $candidates += ($_.DisplayIcon -replace ",.*$", "").Trim('"')
                }
            }
    }

    $appx = Get-AppxPackage -Name "*PowerBI*" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($appx -and $appx.InstallLocation) {
        $candidates += (Join-Path $appx.InstallLocation "bin\PBIDesktop.exe")
        Get-ChildItem $appx.InstallLocation -Recurse -Filter "PBIDesktop.exe" -ErrorAction SilentlyContinue |
            ForEach-Object { $candidates += $_.FullName }
    }

    foreach ($path in $candidates | Select-Object -Unique) {
        if ($path -and (Test-Path $path)) { return $path }
    }
    return $null
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AI-Driven Supply Chain - Open in Power BI Desktop" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not (Test-Path $Pbip)) {
    Write-Host "[1/3] Building Power BI project ..." -ForegroundColor Yellow
    Push-Location $Root
    python powerbi_data_prep.py
    Push-Location $PowerBiDir
    python build_powerbi_desktop.py
    Pop-Location
    Pop-Location
}

if (-not (Test-Path $Pbip)) {
    throw "Power BI project not found: $Pbip"
}

Write-Host "[2/3] Opening dashboard ..." -ForegroundColor Yellow
Write-Host "  File: $Pbip" -ForegroundColor Gray

# Method 1: Windows file association (works for Microsoft Store install)
try {
    Start-Process -FilePath $Pbip
    Write-Host "[3/3] Launched via Windows file association." -ForegroundColor Green
} catch {
    Write-Host "  File association failed, trying direct exe ..." -ForegroundColor Yellow
    $pbi = Find-PowerBIDesktop
    if (-not $pbi) {
        Write-Host ""
        Write-Host "  Could not launch Power BI Desktop automatically." -ForegroundColor Red
        Write-Host "  Open Power BI Desktop manually, then:" -ForegroundColor White
        Write-Host "    File -> Open -> browse to:" -ForegroundColor White
        Write-Host "    $Pbip" -ForegroundColor Green
        exit 1
    }
    Start-Process -FilePath $pbi -ArgumentList "`"$Pbip`""
    Write-Host "[3/3] Launched: $pbi" -ForegroundColor Green
}

Write-Host ""
Write-Host "  IMPORTANT: Use Power BI DESKTOP (local app), not app.powerbi.com." -ForegroundColor Yellow
Write-Host "  This project reads CSV files from your D: drive - it cannot run in the web service." -ForegroundColor Yellow
Write-Host ""
Write-Host "  After Power BI opens:" -ForegroundColor White
Write-Host "    1. Home -> Refresh (data loads from reports\powerbi CSVs)" -ForegroundColor Gray
Write-Host "    2. Click Home, then Refresh" -ForegroundColor Gray
Write-Host ""
