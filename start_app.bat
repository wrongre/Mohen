@echo off
setlocal enabledelayedexpansion

:: Title
title Handwriting Lab - One-Click Launcher

:: Define Python Path
if exist ".venv\Scripts\python.exe" (
    echo [INFO] Found virtual environment. Using: .venv\Scripts\python.exe
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    echo [WARNING] No .venv found. Trying system python...
    set "PYTHON_EXE=python"
)

:: Check if python is available
%PYTHON_EXE% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python execution failed! 
    echo Tried: %PYTHON_EXE%
    pause
    exit /b 1
)

:: Start the server in the background
echo [INFO] Starting WebUI Server...
start /b "" "%PYTHON_EXE%" -u web_ui/main.py > server.log 2>&1

:: Wait a few seconds for the server to spin up
echo [INFO] Waiting for server to initialize (5 seconds)...
timeout /t 5 /nobreak >nul

:: Open Browser
echo [INFO] Opening Browser...
start http://127.0.0.1:8000

:: Keep the window open to show status
echo.
echo ========================================================
echo   Handwriting Lab is running!
echo   Close this window to stop the server (mostly).
echo   If you close this, you may need to manually kill python.exe
echo ========================================================
echo.
echo [LOG TAIL - Last 10 lines of server.log]
powershell -command "Get-Content server.log -Tail 10 -Wait"

pause
