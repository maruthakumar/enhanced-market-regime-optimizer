@echo off
REM Run improved market regime and consolidation pipeline
REM Uses the enhanced Greek sentiment implementation with proper expiry weightages

echo Starting Enhanced Market Regime and Consolidation Pipeline...
echo.

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%~dp0..

REM Create output directories
set "OUTPUT_DIR=%~dp0..\output\improved_run"
mkdir "%OUTPUT_DIR%" 2>nul
mkdir "%OUTPUT_DIR%\logs" 2>nul
mkdir "%OUTPUT_DIR%\market_regime" 2>nul
mkdir "%OUTPUT_DIR%\consolidated" 2>nul

REM Set config file path
set "CONFIG_FILE=%~dp0..\config\local_config.ini"

echo Using configuration file: %CONFIG_FILE%
echo Output directory: %OUTPUT_DIR%
echo.

echo Running pipeline with improved Greek sentiment...
python "%~dp0..\pipeline.py" --config "%CONFIG_FILE%" --steps market_regime,process_strategy,consolidation --output "%OUTPUT_DIR%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Pipeline execution failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo Pipeline execution completed successfully!
echo Results are available in: %OUTPUT_DIR%
echo.

exit /b 0 