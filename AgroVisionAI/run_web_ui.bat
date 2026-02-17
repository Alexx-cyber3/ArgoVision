@echo off
cd web_ui
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing dependencies...
pip install flask flask-sqlalchemy tensorflow pillow numpy
echo Starting AgroVision AI Web UI...
echo.
echo Open your browser at: http://localhost:5000
echo.
python app.py
pause
