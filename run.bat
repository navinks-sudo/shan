@echo off
REM Launch the Military Registers OCR Workbench
REM Requires: Python (openpyxl, pillow, fastapi, uvicorn, ollama, rapidfuzz, opencv-python) + Ollama running with qwen2.5vl:7b and qwen2.5:7b
cd /d "%~dp0"
echo Starting ACTIGEN Portal on http://127.0.0.1:9002  (Ctrl+C to stop)
python -m uvicorn app:app --host 127.0.0.1 --port 9002
