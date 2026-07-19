@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Arraste um video em cima deste arquivo para treinar o perfil de voz.
  pause
  exit /b
)
venv\Scripts\python.exe treinar.py %*
pause
