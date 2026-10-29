@echo off
rem Context - Hybrid RAG Defense Workbench: starts the website on http://127.0.0.1:8000
setlocal
title Context - Hybrid RAG Defense Workbench
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "TOKENIZERS_PARALLELISM=false"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo  The virtual environment .venv was not found in this folder.
  echo  Follow "Fresh installation on another Windows PC" in README.md first.
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" src\web_server.py --open-browser %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
  echo.
  echo  The website stopped with an error ^(exit code %CODE%^). Read the messages above.
  pause
)
exit /b %CODE%
