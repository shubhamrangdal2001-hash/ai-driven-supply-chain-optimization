@echo off
title Open in Power BI Desktop
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\open_powerbi.ps1"
if errorlevel 1 pause
