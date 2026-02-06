@echo off
REM ==============================================================================
REM UVR Headless Runner - Windows Installer Launcher
REM ==============================================================================
REM This .bat file bypasses PowerShell execution policy restrictions so
REM non-technical users can double-click to install without configuring policies.
REM
REM Usage:
REM   docker\install.bat              Quick install (pulls from Docker Hub)
REM   docker\install.bat -Cpu         Force CPU-only
REM   docker\install.bat -Gpu         Force GPU (CUDA 12.4)
REM   docker\install.bat -Cuda cu121  GPU with CUDA 12.1
REM   docker\install.bat -Uninstall   Remove installed wrappers
REM ==============================================================================

REM Resolve the directory this script lives in
set "SCRIPT_DIR=%~dp0"

REM Launch the PowerShell installer with Bypass execution policy
REM -NoProfile   : skip user profile scripts that may interfere
REM -ExecutionPolicy Bypass : allow the ps1 script to run regardless of system policy
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install.ps1" %*

REM Propagate the exit code
exit /b %ERRORLEVEL%
