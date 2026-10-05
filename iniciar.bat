@echo off
rem Dois cliques: instala o necessario (1a vez) e abre o Monitor de EPIs no navegador.
cd /d "%~dp0"
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY (py -3 --version >nul 2>&1 && set "PY=py -3")
if not defined PY (python --version >nul 2>&1 && set "PY=python")
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY (
  echo Python 3.12 nao encontrado. Instale com:  winget install -e --id Python.Python.3.12
  pause
  exit /b 1
)
echo Iniciando o Monitor de EPIs... (feche esta janela para encerrar)
%PY% run.py %*
if errorlevel 1 pause
