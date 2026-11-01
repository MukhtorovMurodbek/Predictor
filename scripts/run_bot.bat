@echo off
rem Double-click to start the Telegram bot. Close this window (or press Ctrl+C) to stop it.
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe (
    echo No .venv found. Follow the setup steps in docs\BOT.md first.
    pause
    exit /b 1
)
:loop
.venv\Scripts\python.exe -m predictor.bot
echo.
echo Bot stopped. Restarting in 10 seconds... (close this window to stop for good)
timeout /t 10 >nul
goto loop
