@echo off
setlocal
chcp 65001 >nul
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set ZIJI_WARMUP=0
set ZIJI_EVIDENCE_DIR=C:\Users\35952\Desktop\123\10月\16日\艺术字开发部署与验收\reports
cd /d "%~dp0"
python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8765
pause
