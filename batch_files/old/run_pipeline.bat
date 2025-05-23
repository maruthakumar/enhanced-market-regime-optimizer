@echo off
REM Main pipeline execution batch file for Windows
REM This file runs the complete zone optimization pipeline

echo Starting Zone Optimization Pipeline...
echo.

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%~dp0..

REM Create timestamp for output directories
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "TIMESTAMP=%dt:~0,8%_%dt:~8,6%"

REM Create output directories
set "OUTPUT_DIR=%~dp0..\output\%TIMESTAMP%"
mkdir "%OUTPUT_DIR%" 2>nul
mkdir "%OUTPUT_DIR%\logs" 2>nul
mkdir "%OUTPUT_DIR%\visualizations" 2>nul
mkdir "%OUTPUT_DIR%\results" 2>nul

REM Set config file path
set "CONFIG_FILE=%~dp0..\config\pipeline_config.ini"

echo Using configuration file: %CONFIG_FILE%
echo Output directory: %OUTPUT_DIR%
echo.

echo Running pipeline...
python "%~dp0..\pipeline.py" --config "%CONFIG_FILE%"

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
