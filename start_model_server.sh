#!/bin/bash
# YuE2 model sunucusu: http://127.0.0.1:7861 (yalnızca bu bilgisayardan erişilebilir)
#   ./start_model_server.sh            yerel ağırlıklarla, tamamen çevrimdışı
#   ./start_model_server.sh --mock     modelsiz test modu (UI ortamında)
#   ./start_model_server.sh --preload  modeli başlangıçta yükle
cd "$(dirname "$0")"
if [[ " $* " == *" --mock "* ]]; then
    exec venv/bin/python server/yue2_server.py "$@"
fi
YUE2_ENV="${YUE2_ENV:-../yue_ortam}"
MODELS="${YUE2_MODELS:-../models}"
if [ ! -f "$MODELS/YuE2-3B/model.safetensors" ] || [ ! -f "$MODELS/YuE2-Vae/model.safetensors" ]; then
    echo "Model ağırlıkları bulunamadı: $MODELS/YuE2-3B ve $MODELS/YuE2-Vae" >&2
    exit 1
fi
if ! "$YUE2_ENV/bin/python" -c "import fastapi, uvicorn" 2>/dev/null; then
    "$YUE2_ENV/bin/pip" install "fastapi==0.136.1" "uvicorn==0.46.0"
fi
# Hugging Face'e hiçbir istek gitmesin.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
# Apple Silicon: MPS'te desteklenmeyen op'lar CPU'ya düşsün.
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec "$YUE2_ENV/bin/python" server/yue2_server.py \
    --model "$MODELS/YuE2-3B" --vae "$MODELS/YuE2-Vae" --offline "$@"
