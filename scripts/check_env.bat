@echo off
cd /d "%~dp0.."
echo Project: %CD%
echo.

where python >nul 2>&1
if %errorlevel% equ 0 (
  echo [PATH] python:
  python --version
  python -c "import uvicorn, fastapi; print('uvicorn OK, fastapi OK')" 2>&1
) else (
  echo [PATH] python: NOT FOUND
)

echo.
py -3 --version 2>nul
if %errorlevel% neq 0 echo [py -3] NOT FOUND

if exist "%USERPROFILE%\anaconda3\python.exe" (
  echo.
  echo [Anaconda]
  "%USERPROFILE%\anaconda3\python.exe" --version
  "%USERPROFILE%\anaconda3\python.exe" -c "import uvicorn" 2>&1
)

echo.
echo Port 8000:
netstat -an | findstr ":8000"
echo.
echo If LISTENING appears above, the API is already running.
echo If nothing, start scripts\run_api.bat and keep that window open.
pause
