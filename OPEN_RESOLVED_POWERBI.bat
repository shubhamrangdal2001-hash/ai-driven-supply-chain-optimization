@echo off
title Open RESOLVED Supply Chain Power BI Dashboard
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0open_resolved_powerbi.ps1"
if errorlevel 1 pause
