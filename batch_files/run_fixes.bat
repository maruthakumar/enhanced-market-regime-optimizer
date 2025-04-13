@echo off
REM Run fixes and optimizations for the zone optimization pipeline

echo Applying fixes and optimizations to Zone Optimization Pipeline...
echo.

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%~dp0..

echo Running fixes and optimizations...
python "%~dp0..\utils\fixes_and_optimizations.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Fixes and optimizations failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo All fixes and optimizations were applied successfully!
echo.

exit /b 0
