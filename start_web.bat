@echo off
setlocal
chcp 65001 >nul
title Energy Storage Screening Service
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0storage\web\start.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" goto :done
echo.
echo [ERROR] Startup failed. See storage\web\startup-error.log for details.
pause
:done
exit /b %EXIT_CODE%
