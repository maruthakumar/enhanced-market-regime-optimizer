@echo off
REM Batch file for running the minute-by-minute market regime classifier pipeline
echo Starting Market Regime Classifier Pipeline...
REM Set the Python executable path - adjust if needed
set PYTHON_PATH=python
REM Set the config file path - use testing or production as needed
REM set CONFIG_FILE=..\config\market_regime\market_regime_testing_config.json
set CONFIG_FILE=..\config\market_regime\market_regime_production_config.json
REM Set the current directory to the script directory
cd /d %~dp0
echo Using configuration file: %CONFIG_FILE%
REM Run the market regime pipeline
%PYTHON_PATH% ..\market_regime_run_pipeline.py %CONFIG_FILE%
echo Market Regime Classifier Pipeline completed.
echo Results are available in the output directory specified in the config file.
pause
