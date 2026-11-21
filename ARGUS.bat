@echo off
rem A.R.G.U.S. desktop app. Double-click to start the backend and open the dashboard window.
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%src\argus_desktop.py" goto wrongfolder

rem Find the project's Python environment: the usual folder names first, then any folder here
rem that is one. Its python.exe is started directly, so a moved or renamed environment works too.
set "PY="
set "ENVDIR="
set "BROKEN="
for %%V in (.ARGUS_venv ARGUS_venv .venv venv) do if not defined PY call :tryenv "%ROOT%%%V"
pushd "%ROOT%"
for /d %%D in (*) do if not defined PY call :tryenv "%ROOT%%%D"
popd
if defined ENVDIR set "VIRTUAL_ENV=%ENVDIR%"
if defined ENVDIR set "PATH=%ENVDIR%\Scripts;%ENVDIR%;%ENVDIR%\Library\bin;%PATH%"

rem No environment folder here: use the Python that is already active (a terminal with yours on).
if defined PY goto run
python -c "pass" >nul 2>nul
if not errorlevel 1 set PY=python
if defined PY goto run
py -3 -c "pass" >nul 2>nul
if not errorlevel 1 set PY=py -3
if not defined PY goto nopython

:run
cd /d "%ROOT%src"
%PY% argus_desktop.py %*
if errorlevel 1 pause
goto done

:tryenv
rem %1 = a folder. Sets PY and ENVDIR when it is a Python environment whose python.exe starts.
set "CAND="
if exist "%~1\Scripts\python.exe" set "CAND=%~1\Scripts\python.exe"
if not defined CAND if exist "%~1\conda-meta" if exist "%~1\python.exe" set "CAND=%~1\python.exe"
if not defined CAND goto :eof
"%CAND%" -c "pass" >nul 2>nul
if errorlevel 1 goto tryenvbad
set PY="%CAND%"
set "ENVDIR=%~1"
goto :eof
:tryenvbad
if not defined BROKEN set "BROKEN=%CAND%"
goto :eof

:wrongfolder
echo.
echo This ARGUS.bat is not inside your A.R.G.U.S. project folder.
echo.
echo It is here:  %ROOT%
echo and there is no src\argus_desktop.py next to it.
echo.
echo Copy ARGUS.bat, README.md, requirements.txt and the src folder into your
echo project folder, choose "Replace the files", and start ARGUS.bat from there.
echo.
pause
goto done

:nopython
echo.
echo A.R.G.U.S. could not find a Python that starts.
echo.
if defined BROKEN goto brokenenv
echo There is no Python environment folder in  %ROOT%
echo and Windows itself has no "python" command.
echo.
echo Start A.R.G.U.S. from the terminal where you normally turn your Python
echo environment on, with these two commands:
echo.
echo     cd /d "%ROOT%src"
echo     python argus_desktop.py
echo.
pause
goto done

:brokenenv
echo This environment was found but its Python does not start:
echo     %BROKEN%
echo.
echo Its own error message is:
echo.
"%BROKEN%" -c "pass"
echo.
echo That usually means the Python it was created from was removed or moved.
echo Please send a photo of this window.
echo.
pause

:done
endlocal
