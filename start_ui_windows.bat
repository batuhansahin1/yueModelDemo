@echo off
REM YuE2 - UI (Gradio). Model sunucusundan bagimsiz calisir: http://127.0.0.1:7860
REM Windows'ta model sunucusuyla ayni ortami kullanir (setup_windows.bat kurar).
setlocal EnableExtensions
cd /d "%~dp0"
if not defined YUE2_ENV set "YUE2_ENV=..\yue_ortam"
set "PY=%YUE2_ENV%\Scripts\python.exe"
if not exist "%PY%" (
    echo Python ortami bulunamadi: %PY%
    echo Once kurulumu calistirin: setup_windows.bat
    exit /b 1
)
set "PYTHONUTF8=1"
REM Gradio kullanim verisi gonderilmesin.
set "GRADIO_ANALYTICS_ENABLED=False"
"%PY%" source\yue2_ui.py %*
