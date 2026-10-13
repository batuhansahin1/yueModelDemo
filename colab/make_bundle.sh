#!/bin/bash
# Colab'a yüklenecek paketi oluşturur: colab/yue2-colab.zip
# İçerik: arayüz + model sunucusu + cover analizi + yue2 wheel. Ağırlıklar Colab'da indirilir.
set -e
cd "$(dirname "$0")/../.."
OUT="yue-ui/colab/yue2-colab.zip"
rm -f "$OUT"
# Wheel zip'in köküne konur; defter onu /content'ten kurar.
zip -q -j "$OUT" yue-ui/wheels/yue2_infer-0.1.5-py3-none-any.whl
zip -q -r "$OUT" \
    yue-ui/server/yue2_server.py yue-ui/server/cover.py yue-ui/server/lyrics.py \
    yue-ui/server/sheetsage.py yue-ui/server/sheetsage_runner.py \
    yue-ui/source/yue2_ui.py \
    yue-ui/top_200_tags.json yue-ui/examples/example.json \
    yue-ui/requirements-yue2-ui.txt yue-ui/README.md
echo "Hazır: $OUT ($(du -h "$OUT" | cut -f1))"
