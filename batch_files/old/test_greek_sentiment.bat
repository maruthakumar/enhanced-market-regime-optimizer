@echo off
REM Test script for the enhanced Greek sentiment implementation
REM This verifies the functionality of the implementation based on Greek_sentiment.md

echo Starting Greek Sentiment Test...
echo.

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%~dp0..

REM Create output directory if it doesn't exist
mkdir "%~dp0..\output\test_greek_sentiment" 2>nul

echo Running Greek sentiment test...
python "%~dp0..\tests\test_greek_sentiment.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Greek sentiment test failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo Greek sentiment test completed successfully!
echo Results are available in: "%~dp0..\output\test_greek_sentiment"
echo.

exit /b 0 