@echo off
REM Train PPO model for toy model queue scheduling (Windows version)

set CONFIG=toymodel\configs\ppo_config.json

python -m toymodel.scripts.train_ppo --config "%CONFIG%"

if %ERRORLEVEL% neq 0 (
    echo Error: PPO training failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

