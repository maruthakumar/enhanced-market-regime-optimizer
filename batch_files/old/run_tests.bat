@echo off
REM Run tests for the zone optimization pipeline

echo Running tests for Zone Optimization Pipeline...
echo.

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%~dp0..

REM Create test output directory
set "TEST_OUTPUT_DIR=%~dp0..\test_output"
mkdir "%TEST_OUTPUT_DIR%" 2>nul

echo Running tests...
python "%~dp0..\tests\test_pipeline.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Tests failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo All tests passed successfully!
echo Test results are available in: %TEST_OUTPUT_DIR%
echo.

exit /b 0
