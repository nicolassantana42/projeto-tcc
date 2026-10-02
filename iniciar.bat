@echo off
rem Dois cliques: instala o necessario (1a vez) e abre o Monitor de EPIs no navegador.
cd /d "%~dp0"
set PY=
where py >nul 2>&1 && set PY=py -3.12
if not defined PY (python --version >nul 2>&1 && set PY=python)
if not defined PY (
  echo Python 3.12 nao encontrado. Instale com:  winget install -e --id Python.Python.3.12
  pause
  exit /b 1
)
%PY% run.py %*
if errorlevel 1 pause
