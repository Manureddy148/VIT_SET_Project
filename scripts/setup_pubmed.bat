@echo off
REM Setup script for PubMed ingestion (Windows)
REM Downloads real abstracts from NCBI PubMed via Entrez API and seeds ChromaDB

setlocal enabledelayedexpansion

echo.
echo 0x1F52C Medical AI - PubMed Ingestion Setup
echo =====================================
echo.

REM Check Python available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    exit /b 1
)

echo [OK] Python found
echo.

REM Create data directories
echo [SETUP] Creating data directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
echo [OK] data\raw, data\processed created
echo.

REM Ingest PubMed abstracts
echo [SETUP] Fetching real abstracts from PubMed (NCBI Entrez API)...
echo        This will query for relevant articles on:
echo        - Diabetes severity and prediction
echo        - Heart disease severity and prediction  
echo        - Pneumonia severity and prediction
echo.

python scripts\ingest_pubmed.py ^
    --domains diabetes heart_disease pneumonia ^
    --limit 30 ^
    --output data\processed\pubmed_docs.json

if %errorlevel% neq 0 (
    echo.
    echo [WARN] PubMed ingestion failed (check internet/API rate limits)
    echo        Falling back to sample data on API startup
    exit /b 1
)

echo.
echo [OK] PubMed ingestion completed
for /f "delims=" %%A in ('python -c "import json; data=json.load(open('data/processed/pubmed_docs.json')); print(len(data))"') do set docs_count=%%A
echo        Total documents: !docs_count!

echo.
echo [SUCCESS] Setup complete!
echo.
echo Next steps:
echo   1. Start the API: python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
echo   2. Or start the frontend: streamlit run frontend\app.py
echo.
echo The ChromaDB vector store will auto-load these documents on first API startup.
echo.
