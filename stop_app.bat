@echo off
title Stop Handwriting Lab

echo [INFO] Stopping Handwriting Lab...

:: Kill python processes that are running main.py
:: We use wmic to find the command line and taskkill to kill it
:: This is a bit aggressive (kills all python), but cleaner for a standalone app.
:: To be safer, we can just kill by port 8000.

echo [INFO] Finding process on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do (
    echo [INFO] Killing process ID %%a...
    taskkill /F /PID %%a
)

echo.
echo [INFO] Cleanup complete.
echo.
pause
