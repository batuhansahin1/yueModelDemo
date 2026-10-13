"""Downloads the model weights missing under the models folder; the servers then run offline.

    python download_models.py ../models [--sheetsage]

A model is downloaded only when its weights file is missing, so a models folder copied from another
computer is kept as it is. Same repositories and files as the Colab notebook (cells 5 and 5b).
Runs in the yue2 environment (setup_windows.bat), which has huggingface_hub, yue2 and demucs.
"""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

DEMUCS_WEIGHTS = "955717e8-8726e21a.th"
DEMUCS_URL = f"https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/{DEMUCS_WEIGHTS}"


def fetch(repo, folder, **patterns):
    if (folder / "model.safetensors").exists():
        print(f"Var: {folder}", flush=True)
        return
    print(f"İndiriliyor: {repo} -> {folder}", flush=True)
    snapshot_download(repo, local_dir=folder, **patterns)


def main():
    p = argparse.ArgumentParser(description="Eksik model ağırlıklarını indirir")
    p.add_argument("models", type=Path, help="Ağırlıkların klasörü (ör. ../models)")
    p.add_argument("--sheetsage", action="store_true", help="SheetSage2 + MERT-v2 de indirilsin (Cover nota çıkarıcı)")
    args = p.parse_args()
    models = args.models

    from yue2.storage import MODEL_FILES, MODEL_LICENSES
    patterns = (sorted(MODEL_FILES) + ["model-?????-of-?????.safetensors"]
                + ["licenses/" + name for name in sorted(MODEL_LICENSES)])
    for repo, folder in (("m-a-p/YuE2-Vae", "YuE2-Vae"), ("m-a-p/YuE2-3B", "YuE2-3B")):
        fetch(repo, models / folder, allow_patterns=patterns)
    # Lyric recognition: multilingual turbo (language detection, non-Turkish) + Turkish fine-tune.
    for repo, folder in (("openai/whisper-large-v3-turbo", "whisper-large-v3-turbo"),
                         ("turkmedstt/whisper-large-v3-turkish-general", "whisper-large-v3-turkish-general")):
        fetch(repo, models / folder, allow_patterns=["*.json", "*.txt", "model.safetensors"])

    demucs_dir = models / "demucs"
    demucs_dir.mkdir(parents=True, exist_ok=True)
    if not (demucs_dir / "htdemucs.yaml").exists():
        import demucs
        shutil.copy(Path(demucs.__file__).parent / "remote" / "htdemucs.yaml", demucs_dir)
    if (demucs_dir / DEMUCS_WEIGHTS).exists():
        print(f"Var: {demucs_dir}", flush=True)
    else:
        print(f"İndiriliyor: htdemucs -> {demucs_dir}", flush=True)
        partial = demucs_dir / (DEMUCS_WEIGHTS + ".part")
        urllib.request.urlretrieve(DEMUCS_URL, partial)
        partial.replace(demucs_dir / DEMUCS_WEIGHTS)

    if args.sheetsage:
        sheetsage = models / "SheetSage2"
        fetch("m-a-p/SheetSage2", sheetsage, ignore_patterns=["render_assets/*", "benchmarks/*", "assets/*"])
        # SheetSage2 merges its adapters into this exact MERT-v2 revision (checked by hash at load time);
        # it is read from the Hugging Face cache, not from the models folder.
        config = json.loads((sheetsage / "config.json").read_text(encoding="utf-8"))
        base, revision = config["base_model_name_or_path"], config["base_model_revision"]
        try:
            snapshot_download(base, revision=revision, local_files_only=True)
            print(f"Var: {base} (Hugging Face önbelleği)", flush=True)
        except LocalEntryNotFoundError:
            print(f"İndiriliyor: {base} -> Hugging Face önbelleği", flush=True)
            snapshot_download(base, revision=revision)


if __name__ == "__main__":
    main()
