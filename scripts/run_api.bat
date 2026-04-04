@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0.."
set "ROOT=%CD%"
set MEDICAL_AI_SKIP_RAG=1
set PYTHONPATH=%ROOT%

echo.
echo === Medical AI API ===
echo Root: %ROOT%
echo.

REM Try Python in order: PATH python, Windows py launcher, common Conda paths
where python >nul 2>&1
if %errorlevel% equ 0 (
  echo Using: python (from PATH^)
  python --version
  python -c "import uvicorn" 2>nul
  if errorlevel 1 (
    echo ERROR: uvicorn not installed. Run:  pip install -r requirements.txt
    pause
    exit /b 1
  )
  echo.
  echo Easier:  python launch_api.py   (from project folder — opens browser^)
  echo Open in browser WHILE this window stays open:
  echo   http://127.0.0.1:8000/docs
  echo Press Ctrl+C to stop the server.
  echo.
  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
  goto :end
)

py -3 -c "import sys" >nul 2>&1
if %errorlevel% equ 0 (
  echo Using: py -3 (Python launcher^)
  py -3 --version
  py -3 -c "import uvicorn" 2>nul
  if errorlevel 1 (
    echo ERROR: uvicorn not installed for py -3. Run:  py -3 -m pip install -r requirements.txt
    pause
    exit /b 1
  )
  echo.
  echo Open: http://127.0.0.1:8000/docs
  echo.
  py -3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
  goto :end
)

if exist "%USERPROFILE%\anaconda3\python.exe" (
  echo Using: %USERPROFILE%\anaconda3\python.exe
  "%USERPROFILE%\anaconda3\python.exe" --version
  "%USERPROFILE%\anaconda3\python.exe" -c "import uvicorn" 2>nul
  if errorlevel 1 (
    echo ERROR: uvicorn not installed. Run:  "%USERPROFILE%\anaconda3\python.exe" -m pip install -r requirements.txt
    pause
    exit /b 1
  )
  echo.
  echo Open: http://127.0.0.1:8000/docs
  echo.
  "%USERPROFILE%\anaconda3\python.exe" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
  goto :end
)

if exist "%USERPROFILE%\miniconda3\python.exe" (
  echo Using: %USERPROFILE%\miniconda3\python.exe
  "%USERPROFILE%\miniconda3\python.exe" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
  goto :end
)

echo.
echo ERROR: Python was not found.
echo - Open "Anaconda Prompt" or "x64 Native Tools" and cd to this folder, then run:
echo     pip install -r requirements.txt
echo     python launch_api.py
echo - Or install Python from python.org and check "Add to PATH".
echo.
pause
exit /b 1

:end
echo.
pause
