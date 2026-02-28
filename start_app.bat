@echo off
setlocal enabledelayedexpansion

:: Title
title Handwriting Lab - One-Click Launcher

:: Define Python Path
if exist ".env\Scripts\python.exe" (
    echo [INFO] Found virtual environment. Using: .env\Scripts\python.exe
    set "PYTHON_EXE=.env\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    echo [INFO] Found virtual environment. Using: .venv\Scripts\python.exe
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        echo [WARNING] No local virtual environment found. Trying launcher: py -3
        set "PYTHON_EXE=py -3"
    ) else (
        echo [WARNING] No local virtual environment found. Trying system python...
        set "PYTHON_EXE=python"
    )
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

:: Wait for server to be ready with intelligent retry logic
echo [INFO] Waiting for server to initialize...
set "MAX_RETRIES=60"
set "RETRY_COUNT=0"
set "SERVER_READY=0"

:wait_server
if %RETRY_COUNT% geq %MAX_RETRIES% (
    echo [WARNING] Server did not respond after %MAX_RETRIES% attempts. Opening browser anyway...
    set "SERVER_READY=1"
    goto open_browser
)

:: Use curl or powershell to check if server is responding
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8000' -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop; exit 0 } catch { exit 1 }" >nul 2>&1

if %errorlevel% equ 0 (
    echo [INFO] Server is ready!
    set "SERVER_READY=1"
    timeout /t 1 /nobreak >nul
    goto open_browser
)

set /a RETRY_COUNT=%RETRY_COUNT%+1
echo [INFO] Server not ready yet... Retry %RETRY_COUNT%/%MAX_RETRIES%
timeout /t 1 /nobreak >nul
goto wait_server

:open_browser
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
