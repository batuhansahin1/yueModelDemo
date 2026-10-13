#!/bin/bash
# YuE2 - UI (Gradio). Model sunucusundan bağımsız çalışır: http://127.0.0.1:7860
cd "$(dirname "$0")"
if [ ! -d venv ]; then
    python3.12 -m venv venv 2>/dev/null || /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv venv
    venv/bin/pip install -r requirements-yue2-ui.txt
fi
source venv/bin/activate
# Gradio kullanım verisi gönderilmesin.
export GRADIO_ANALYTICS_ENABLED=False
python source/yue2_ui.py "$@"
