@echo off
REM Compare all scheduler types (Windows version)

set CONFIG=toymodel\configs\config.json

python -m toymodel.scripts.compare_schedulers "%CONFIG%"

if %ERRORLEVEL% neq 0 (
    echo Error: Scheduler comparison failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

