@echo off
REM Launch script for AI Recommendation Engine Web App (Windows)

echo.
echo ===============================================================
echo 🚀 AI Recommendation Engine - Local Development Launch
echo ===============================================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo 📚 Ensuring dependencies are installed...
pip install -q -r requirements.txt

REM Set environment variables
set FLASK_APP=src.app
set FLASK_ENV=development

REM Create startup message
echo.
echo ✅ Environment ready!
echo.
echo Available commands:
echo   - Start Flask server: python -m flask run --port 5000
echo   - Run tests: python test_webapp.py
echo   - Open browser: http://localhost:5000
echo.
echo ===============================================================
echo.
echo Type the command above to start the server, then visit the URL in your browser.
