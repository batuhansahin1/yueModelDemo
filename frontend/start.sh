#!/bin/bash
# YuE2 web arayüzü: http://127.0.0.1:7870
#   ./frontend/start.sh                     model sunucusuna (127.0.0.1:7861) bağlanır
#   ./frontend/start.sh --mock              modelsiz test sunucusunu da başlatır (kısa bir deneme tonu üretir)
#   ./frontend/start.sh --api http://ADRES:7861 --port 8000
#   ./frontend/start.sh --host 0.0.0.0 --password PAROLA      ağa/dışarıya açar, parola ister
# Tüm seçenekler: python3 frontend/server.py --help
cd "$(dirname "$0")"

MOCK=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--mock" ]; then MOCK=1; else ARGS+=("$arg"); fi
done

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python 3 bulunamadı. PYTHON=/yol/python3 ./frontend/start.sh ile belirtin." >&2
    exit 1
fi

if [ "$MOCK" = 1 ]; then
    # Ayrı port: aynı anda çalışan gerçek model sunucusuyla (7861) çakışmaz.
    MOCK_PORT="${YUE2_MOCK_PORT:-7862}"
    MOCK_PYTHON="../venv/bin/python"
    if ! "$MOCK_PYTHON" -c "import fastapi, uvicorn, numpy" 2>/dev/null; then
        echo "Mock sunucusu yue-ui/venv ortamında çalışır (fastapi, uvicorn). Bir kez kurun:" >&2
        echo "  cd .. && python3.12 -m venv venv && venv/bin/pip install -r requirements-yue2-ui.txt" >&2
        exit 1
    fi
    mkdir -p ../logs
    "$MOCK_PYTHON" ../server/yue2_server.py --mock --port "$MOCK_PORT" --outputs ../outputs/mock \
        > ../logs/mock_server.log 2>&1 &
    MOCK_PID=$!
    trap 'kill "$MOCK_PID" 2>/dev/null' EXIT
    trap 'exit 130' INT TERM
    for _ in $(seq 1 100); do
        curl -sf "http://127.0.0.1:$MOCK_PORT/health" >/dev/null 2>&1 && break
        if ! kill -0 "$MOCK_PID" 2>/dev/null; then
            echo "Mock sunucusu başlamadı (port $MOCK_PORT dolu olabilir; YUE2_MOCK_PORT ile değiştirin):" >&2
            tail -5 ../logs/mock_server.log >&2
            exit 1
        fi
        sleep 0.2
    done
    echo "Mock model sunucusu: http://127.0.0.1:$MOCK_PORT (log: logs/mock_server.log, çıktılar: outputs/mock)"
    ARGS=(--api "http://127.0.0.1:$MOCK_PORT" "${ARGS[@]}")
fi

"$PYTHON" server.py "${ARGS[@]}"
