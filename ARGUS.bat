@echo off
rem A.R.G.U.S. desktop app. Double-click to start the backend and open the dashboard window.
cd /d "%~dp0src"
rem Use the project venv when there is one (either name).
if exist "%~dp0.ARGUS_venv\Scripts\activate.bat" (call "%~dp0.ARGUS_venv\Scripts\activate.bat") else if exist "%~dp0ARGUS_venv\Scripts\activate.bat" call "%~dp0ARGUS_venv\Scripts\activate.bat"
python argus_desktop.py %*
if errorlevel 1 pause
