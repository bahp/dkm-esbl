@echo off
set PYTHON=python
set SCRIPTS_DIR=scripts

:: Check if a parameter was provided (Direct Mode)
if "%1"=="" goto menu
if "%1"=="all" goto all
if "%1"=="generate" goto generate
if "%1"=="features" goto features
if "%1"=="evaluate" goto evaluate
if "%1"=="validate" goto validate
if "%1"=="search" goto search
if "%1"=="test" goto test
if "%1"=="clean" goto clean

:menu
echo ============================================================
echo   🏥 CLINICAL PIPELINE MANAGER (Windows Batch)
echo ============================================================
echo   1. all      - Run entire pipeline
echo   2. generate - Step 1: Generate Synthetic iCARE Data
echo   3. features - Step 2: Build Phenotypes (SIRS/Pitt)
echo   4. evaluate - Step 3: Compute Clinical Scores
echo   5. validate - Step 4: Run Clinical Audit (cases.csv)
echo   6. search   - Discover Clinical Codes (Keywords)
echo   7. test     - Run Pytest Suite
echo   8. clean    - Remove old reports
echo ============================================================
set /p choice="Enter choice: "

if "%choice%"=="1" goto all
if "%choice%"=="2" goto generate
if "%choice%"=="3" goto features
if "%choice%"=="4" goto evaluate
if "%choice%"=="5" goto validate
if "%choice%"=="6" goto search
if "%choice%"=="7" goto test
if "%choice%"=="8" goto clean
goto menu

:all
call :generate
call :features
call :evaluate
call :validate
goto :eof

:generate
echo.
echo.
echo --- Step 1: Generating Synthetic iCARE Data ---
%PYTHON% -m %SCRIPTS_DIR%.01_generate_data_v2
goto :eof

:features
echo.
echo.
echo --- Step 2: Building Clinical Features ---
%PYTHON% -m %SCRIPTS_DIR%.02_build_features_icare
goto :eof

:evaluate
echo.
echo.
echo --- Step 3: Evaluating Clinical Scores ---
%PYTHON% -m %SCRIPTS_DIR%.03_evaluate_scores_v2
goto :eof

:validate
echo.
echo.
echo --- Step 4: Validating Scores (Clinical Audit) ---
%PYTHON% -m %SCRIPTS_DIR%.05_validate_scores
:: Brief 2-second pause to see the result, then continues automatically
timeout /t 2 >nul
goto :eof

:test
echo.
echo.
echo --- Running Unit Tests ---
%PYTHON% -m pytest tests/test_phenotypes.py tests/test_scores.py
timeout /t 2 >nul
goto :eof

:search
echo.
echo.
echo --- Discover Clinical Codes (Keywords) ---
%PYTHON% -m %SCRIPTS_DIR%.06_find_clinical_codes
echo.
echo Done. Results saved to reports/code_search_results.txt
timeout /t 3 >nul
goto :eof

:clean
echo.
echo.
echo --- 🧹 Cleaning up reports ---
if exist reports\*.log del /q reports\*.log
if exist reports\*.txt del /q reports\*.txt
echo Done.
timeout /t 2 >nul
goto :eof