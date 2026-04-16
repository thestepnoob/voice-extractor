@echo off
echo Starting Voice Extractor with GPU Support...
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "Voice Extractor"
.\venv\Scripts\python.exe app.py
pause
