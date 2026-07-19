@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:7860
venv\Scripts\python.exe interface.py
