#!/bin/bash
# Frontend testleri; model ya da GPU gerekmez.
#   1. JS mantığı (public/js/logic.js): node --test
#   2. Web sunucusu ve /api vekili, mock model sunucusuyla uçtan uca: Python unittest
#   3. Tarayıcıda arayüz akışları (Chrome headless): tests/e2e.mjs; Chrome ya da Node yoksa atlanır
# Tek bir adım: ./frontend/test.sh logic | server | e2e
cd "$(dirname "$0")"
STEP="${1:-all}"
PYTHON="${PYTHON:-../venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
failed=0

run() {
    echo
    echo "=== $1"
    shift
    "$@" || failed=1
}

if [ "$STEP" = all ] || [ "$STEP" = logic ]; then
    if command -v node >/dev/null 2>&1; then
        run "JS mantığı (node --test)" node --test tests/logic.test.mjs
    else
        echo "Node.js yok: JS mantık testleri atlandı (https://nodejs.org)"
    fi
fi
if [ "$STEP" = all ] || [ "$STEP" = server ]; then
    run "Web sunucusu + mock model sunucusu (unittest)" "$PYTHON" -m unittest discover -s tests -p "test_*.py"
fi
if [ "$STEP" = all ] || [ "$STEP" = e2e ]; then
    if command -v node >/dev/null 2>&1; then
        run "Tarayıcı (Chrome headless)" node tests/e2e.mjs
    else
        echo "Node.js yok: tarayıcı testleri atlandı"
    fi
fi

echo
if [ "$failed" = 0 ]; then echo "Tüm testler geçti."; else echo "Başarısız testler var."; fi
exit $failed
