@echo off
echo ==========================================
echo      Pong AI - Environment Setup
echo ==========================================
echo.

:: Check for Python 3.12
echo [*] Checking for Python 3.12...
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] ERROR: Python 3.12 is not installed or not detected by 'py' launcher.
    echo.
    echo Please install Python 3.12 from:
    echo https://www.python.org/downloads/release/python-3128/
    echo.
    echo (Make sure to check "Add Python to PATH" during installation)
    echo.
    pause
    exit /b 1
)
echo [V] Python 3.12 found.
echo.

:: Install Dependencies
echo [*] Installing dependencies from requirements.txt...
py -3.12 -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [!] ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo [V] Dependencies installed.
echo.

:: Install ROMs
echo [*] Installing Atari ROMs...
py -3.12 -m AutoROM --accept-license
if %errorlevel% neq 0 (
    echo [!] ERROR: Failed to install ROMs.
    pause
    exit /b 1
)
echo [V] ROMs installed.
echo.

echo ==========================================
echo      Setup Complete! You can now run:
echo      run_train.bat  - to train the agent
echo      run_play.bat   - to watch it play
echo ==========================================
pause
