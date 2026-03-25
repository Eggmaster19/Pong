@echo off
echo ==========================================
echo   Pong AI Training - VISUAL MODE
echo ==========================================
echo.
echo Two windows will open:
echo   1. Pong game - watch the AI learn
echo   2. Live graph - see performance improve
echo.
echo (Press Ctrl+C in this window to stop)
echo ==========================================
echo.
py -3.12 main.py train-visual
pause