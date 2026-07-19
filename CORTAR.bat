@echo off
cd /d "%~dp0"
rem Sem argumentos: abre a janela para escolher os videos.
rem Com argumentos (videos arrastados em cima do .bat): corta direto.
venv\Scripts\python.exe cortar.py %*
pause
