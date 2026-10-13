"""YuE2 - UI: Gradio interface for YuE2, built on YuE-UI's prompt workflow.

The UI holds no model. It talks to server/yue2_server.py over HTTP, so it can be
started first and connected to the model server whenever that is running.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import OrderedDict
import json
import os
from pathlib import Path
import re
import tempfile
import time

import gradio as gr
import requests

ROOT = Path(__file__).resolve().parents[1]
AUDIO_CACHE = Path(tempfile.gettempdir()) / "yue2-ui"
DEFAULT_SERVER = os.environ.get("YUE2_SERVER", "http://127.0.0.1:7861")

# Defaults mirror yue2.protocol.GenerationConfig.
ABC_DEFAULTS = dict(temperature=0.7, top_p=0.9, top_k=30, repetition_penalty=1.005, penalty_window=100,
                    min_tokens=32, max_tokens=4096)
SEMANTIC_DEFAULTS = dict(temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2, penalty_window=50,
                         min_tokens=200, max_tokens=9000)
# Order of the sampling controls; max_tokens stays last (the UI resizes the song budget by index).
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty", "penalty_window", "min_tokens", "max_tokens")
DEFAULT_ODE_STEPS = 32
# Reference melody modes, as in YuE2's cover guide: full score (cot=full) or melody-only score (cot=melody).
COVER_MODES = [("Yakın: melodi + akorlar + yapı", "close"), ("Serbest: sadece melodi, akorları model seçer", "free")]
COVER_COT = {"close": "full", "free": "melody"}
# Who writes the reference song's score: SheetSage2 (YuE2's own transcriber) or server/cover.py.
TRANSCRIBERS = [("SheetSage2 (YuE2'nin resmi nota çıkarıcısı)", "sheetsage2"), ("Basit analiz (Demucs + librosa)", "basic")]
TRANSCRIBER_NAMES = {"sheetsage2": "SheetSage2", "basic": "Basit analiz"}

COT_CHOICES = [
    ("Tam (akorlu ABC nota planı → şarkı)", "full"),
    ("Melodi (akorsuz ABC planı → şarkı)", "melody"),
    ("Kapalı (doğrudan şarkı)", "off"),
]
STAGE_LABELS = {
    "queued": "Sırada bekliyor",
    "loading": "Model yükleniyor (ilk çalıştırmada ağırlıklar indirilebilir)",
    "abc": "Nota planı (ABC) üretiliyor",
    "semantic": "Şarkı token'ları üretiliyor",
    "synthesize": "Ses sentezleniyor",
    "decode": "Ses çözülüyor (VAE)",
    "saving": "Kaydediliyor",
    "done": "Bitti",
    "cancelled": "İptal edildi",
}
STATUS_LABELS = {"queued": "Sırada", "running": "Çalışıyor", "complete": "Tamamlandı",
                 "failed": "Hata", "cancelled": "İptal edildi"}
FINAL = {"complete", "failed", "cancelled"}
# Time estimates use the server's measured speed on its own device (GET /stats).
FAST_ODE_STEPS = 16
# Covers came out up to ~1.6x the reference length.
COVER_LENGTH_FACTOR = (1.0, 1.6)
# Above this estimate the UI suggests generating a shorter part first.
LONG_ESTIMATE_MINUTES = 20
# YuE2 emits ~25 semantic tokens per second of audio.
TOKENS_PER_AUDIO_SECOND = 25
# Whisper language codes for lyric recognition.
LANGUAGE_CHOICES = [("Otomatik algıla", "auto"), ("Türkçe", "tr"), ("İngilizce", "en"), ("Almanca", "de"),
                    ("Fransızca", "fr"), ("İspanyolca", "es"), ("İtalyanca", "it"), ("Portekizce", "pt"),
                    ("Rusça", "ru"), ("Arapça", "ar"), ("Farsça", "fa"), ("Azerice", "az"),
                    ("Japonca", "ja"), ("Korece", "ko"), ("Çince", "zh")]
# Score sections the model plays without singing (cover.lyric_labels skips the same names).
NON_VOCAL = {"intro", "outro", "inst", "instrumental", "end", "silence", "interlude", "fade-out", "solo"}
# Text to Song: style chips (label, tag sent to the model), voice choices and their tags, field limits.
T2S_STYLES = [("Pop", "pop"), ("Sakin", "chill"), ("Enerjik", "energetic"), ("Duygusal", "emotional"),
              ("Rock", "rock"), ("Elektronik", "electronic"), ("Hip-hop", "hip-hop"), ("Lo-Fi", "lo-fi"),
              ("Film müziği", "soundtrack"), ("Caz", "jazz"), ("Klasik", "classical"), ("Ambient", "ambient")]
T2S_VOICES = [("Varsayılan", "default"), ("Kadın", "female"), ("Erkek", "male")]
VOICE_TAGS = {"default": None, "female": "female vocal", "male": "male vocal"}
DESCRIPTION_LIMIT, LYRICS_LIMIT = 3000, 5000
T2S_LYRICS_PLACEHOLDER = ("[verse]\nNeon ışıklar soluyor caddede\nHer gölge seni getiriyor aklıma\n\n"
                          "[chorus]\nGece kapansa da üstüme\nYine sana dönerim")
CSS = """
#t2s-card {border: 1px solid var(--border-color-primary); border-radius: 20px; padding: 18px; gap: 14px;}
#t2s-card textarea {font-size: 15px;}
.t2s-counter {text-align: right; color: var(--body-text-color-subdued); font-size: 12px; margin-top: -10px;}
.t2s-popup {border: 1px solid var(--border-color-primary); border-radius: 18px; padding: 14px;
            box-shadow: 0 8px 28px rgba(0, 0, 0, .08);}
.t2s-pills .wrap {display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px;}
.t2s-pills label {justify-content: center; border-radius: 14px !important; padding: 12px 8px !important;}
.t2s-pills input {display: none;}
.t2s-pills label.selected {background: #fdeef0 !important; border-color: #f0a3ad !important; color: #d0344a !important;}
.t2s-tool {flex-grow: 0 !important; min-width: 180px !important; white-space: nowrap;}
#t2s-generate {background: #222 !important; color: #fff !important; border-radius: 16px; min-height: 52px;
               font-size: 17px; max-width: 260px; margin-left: auto;}
"""
TIMELINE_HINT = ("Referans şarkı yükleyip **✨ Sözleri tanı**'ya basın: sözler şarkıda söylendikleri bölümlere "
                 "yerleşir ve her satırın zamanı burada görünür.")


def load_and_process_genres(json_path):
    """Deduplicated, sorted tag list from YuE-UI's top_200_tags.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    categories = ["genre", "timbre", "gender", "mood", "instrument"]
    unique_items = OrderedDict()
    for item in (item.strip() for category in categories for item in data.get(category, [])):
        if item and item.lower() not in unique_items:
            unique_items[item.lower()] = item
    return sorted(unique_items.values(), key=lambda x: x.lower())


class ServerClient:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def _call(self, method, path, timeout=10, **kwargs):
        response = requests.request(method, self.url + path, timeout=timeout, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text
            # 409 = the GPU is busy with another task; the server's message says what to do.
            raise gr.Error(detail if response.status_code == 409 else f"Sunucu hatası ({response.status_code}): {detail}")
        return response

    def health(self):
        return self._call("GET", "/health", timeout=3).json()

    def stats(self):
        """Measured speed of the server's device; empty when unknown or unreachable (estimates are optional)."""
        try:
            return requests.get(self.url + "/stats", timeout=3).json()
        except (requests.RequestException, ValueError):
            return {}

    def submit(self, payload):
        return self._call("POST", "/jobs", json=payload).json()

    def job(self, job_id):
        return self._call("GET", f"/jobs/{job_id}").json()

    def jobs(self):
        return self._call("GET", "/jobs").json()

    def cancel(self, job_id):
        return self._call("POST", f"/jobs/{job_id}/cancel").json()

    def analyze(self, path, start, duration, bpm, lyrics, recognize=False, language=None, transcriber="basic"):
        with open(path, "rb") as f:
            upload = self._call("POST", "/uploads", params={"filename": Path(path).name},
                                data=f, timeout=120).json()
        return self._call("POST", "/analyze", timeout=1800, json={
            "upload_id": upload["upload_id"], "start": float(start or 0),
            "duration": float(duration) if duration else None,
            "bpm": float(bpm) if bpm else None, "lyrics": lyrics or "",
            "recognize_lyrics": recognize, "language": language, "transcriber": transcriber}).json()

    def download_audio(self, job):
        AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
        path = AUDIO_CACHE / f"{job['id']}{Path(job['result']['audio']).suffix}"
        if not path.exists():
            response = self._call("GET", f"/jobs/{job['id']}/audio", timeout=120)
            path.write_bytes(response.content)
        return str(path)


def connection_markdown(url):
    try:
        h = ServerClient(url).health()
    except Exception:
        return (f"🔴 **Model sunucusu bağlı değil** — `{url}` adresine ulaşılamadı. "
                "UI çalışıyor; üretim için model sunucusunu başlatın "
                f"(`{'start_model_server.bat' if os.name == 'nt' else './start_model_server.sh'}`).")
    # "NVIDIA L4" rather than PyTorch's "cuda:0"; older servers only report the latter.
    device = h.get("device_name") or h.get("device")
    if h.get("mode") == "mock":
        state = "🟡 **Test modu (mock)** — gerçek model yok, deneme tonu üretilir"
    elif h.get("load_error"):
        state = f"🔴 **Model yüklenemedi:** {h['load_error']}"
    elif h.get("model_loaded"):
        state = f"🟢 **YuE2 hazır** — cihaz: **{device}**"
    else:
        state = ("🟢 **Sunucuya bağlı** — model boşta, bellekte değil (üretim başlayınca yüklenir)"
                 + (f" · cihaz: **{device}**" if device else ""))
    busy = f" · çalışan: {len(h.get('running') or [])}, sırada: {h.get('queued', 0)}"
    return f"{state} · `{url}`{busy}"


def settings_text(req):
    """One-line summary of a job's model settings, for comparing runs."""
    def sampling(name, values, defaults):
        # Temperature always; the other knobs only when changed (max_tokens is sized automatically for covers).
        values = {**defaults, **(values or {})}
        changed = [f"{key}={values[key]}" for key in SAMPLING_KEYS
                   if key not in ("temperature", "max_tokens") and values[key] != defaults[key]]
        return f"{name} T={values['temperature']}" + (f" ({', '.join(changed)})" if changed else "")
    reference = req.get("reference") or {}
    parts = [f"CoT {req.get('cot')}",
             f"CFG {req['cfg_scale'] if req.get('cfg_scale') is not None else 'varsayılan'}",
             sampling("ABC", req.get("abc_sampling"), ABC_DEFAULTS),
             sampling("Şarkı", req.get("semantic_sampling"), SEMANTIC_DEFAULTS),
             f"ODE {req.get('ode_steps') or DEFAULT_ODE_STEPS}"]
    if req.get("quantization") == "fp8":
        parts.append("FP8")
    if reference.get("mode"):
        parts.append("cover " + {"close": "Yakın", "free": "Serbest"}.get(reference["mode"], reference["mode"])
                     + f" ({TRANSCRIBER_NAMES.get(reference.get('transcriber', 'basic'), reference.get('transcriber'))})")
    return " · ".join(parts)


def progress_markdown(job, abc_max, semantic_max, stats=None):
    status = STATUS_LABELS.get(job["status"], job["status"])
    lines = [f"**Durum:** {status} · **Aşama:** {STAGE_LABELS.get(job['stage'], job['stage'])}",
             f"**İş:** `{job['id']}` · **Seed:** `{job['request'].get('seed')}` · "
             f"**Süre:** {job.get('elapsed', 0):.0f} sn",
             f"**Ayarlar:** {settings_text(job['request'])}"]
    reference = job["request"].get("reference")
    steps = job["request"].get("ode_steps")
    fast = steps == FAST_ODE_STEPS
    # Speeds are measured only for the default and fast synthesis settings.
    if (reference and job["status"] not in FINAL and steps in (None, DEFAULT_ODE_STEPS, FAST_ODE_STEPS)
            and estimate_range(reference["seconds"], fast, stats)):
        lines.append(f"⏱ Tahmini toplam: {estimate_text(reference['seconds'], fast, stats)}")
    tokens = job.get("tokens") or {}
    if tokens.get("abc"):
        lines.append(f"ABC token: {tokens['abc']} / {abc_max}")
    if tokens.get("semantic"):
        lines.append(f"Şarkı token: {tokens['semantic']} / {semantic_max}")
    if job.get("error"):
        lines.append(f"⚠️ {job['error']}")
    result = job.get("result") or {}
    if result.get("audio_seconds"):
        lines.append(f"🎵 Ses süresi: {result['audio_seconds']:.1f} sn")
    truncated = [k for k, v in (result.get("truncated") or {}).items() if v]
    if truncated:
        lines.append(f"⚠️ Token sınırına ulaşıldı ({', '.join(truncated)}); max token değerini artırmayı deneyin.")
    return "  \n".join(lines)


def estimate_range(reference_seconds, fast, stats):
    """(low, high) minutes for a cover of this length on the server's device, or None before it was measured."""
    stats = stats or {}
    # Fall back to the other synthesis mode's speed before this one has been measured.
    rate = stats.get("fast_rate" if fast else "rate") or stats.get("rate") or stats.get("fast_rate")
    return tuple(reference_seconds * f * rate / 60 for f in COVER_LENGTH_FACTOR) if rate else None


def estimate_text(reference_seconds, fast, stats):
    span = estimate_range(reference_seconds, fast, stats)
    if span is None:
        return "ilk üretimden sonra bu cihaz için hesaplanacak"
    low, high = (max(1, round(m)) for m in span)
    return f"{'~' + str(high) if low == high else f'{low}-{high}'} dk ({stats['device']})"


def cover_markdown(summary, fast=False, stats=None):
    sung = sum(s["name"] not in NON_VOCAL for s in summary["sections"])
    source = {"vocals": "vokal",
              "instrument": "**enstrüman** (kayıtta vokal bulunamadı; sözleriniz bu melodiyle söylenecek)",
              "none": "⚠️ **vokal melodisi bulunamadı**; notada yalnızca enstrüman melodisi var, yeni şarkıda vokal "
                      "olmayabilir. Enstrümantal bir kayıtsa *Basit analiz*'i deneyin (enstrüman melodisini vokale taşır)."
              }[summary.get("melody_source", "vocals")]
    transcriber = TRANSCRIBER_NAMES.get(summary.get("transcriber", "basic"), summary.get("transcriber"))
    lines = [f"✅ **Melodi çıkarıldı** ({transcriber}) · Tempo **{summary['tempo']:.0f} BPM** · Ton **{summary['key']}** · "
             f"{summary['bars']} ölçü ({summary['seconds']:.0f} sn) · melodili kısım %{summary['vocal_ratio'] * 100:.0f}",
             f"Melodi kaynağı: {source}",
             f"Bölümler: {len(summary['sections'])} ({sung} şarkılı), zamanları sözlerin altındaki zaman çizelgesinde",
             f"⏱ Tahmini üretim süresi: **{estimate_text(summary['seconds'], fast, stats)}**"]
    status = summary.get("lyrics_status", "off")
    if status == "ok":
        model = f" · model `{summary['asr_model']}`" if summary.get("asr_model") else ""
        placement = ("sözler SheetSage2'nin bölümlerine yerleştirildi" if summary.get("transcriber") == "sheetsage2"
                     else "bölümler sözlere göre belirlendi")
        lines.insert(1, f"✅ **Sözler tanındı** · {len(summary.get('lines') or [])} satır · dil `{summary.get('language')}`"
                        f"{model} · {placement}")
    elif status == "none":
        lines.insert(1, "⚠️ Vokal var ama kelimeler anlaşılamadı. Sözleri elle yazıp **Melodiyi çıkar**'a basın; "
                        "bölümler sözlerinizdeki etiketlere göre adlandırılır.")
    elif status == "instrumental":
        lines.insert(1, "⚠️ Seçilen aralıkta vokal yok, tanınacak söz bulunamadı. Vokalli bir bölüm seçin "
                        "(Başlangıç/Süre) ya da sözleri elle yazın.")
    if summary["vocal_ratio"] < 0.05:
        lines.append("⚠️ Neredeyse hiç melodi bulunamadı; sonuç sadece akorları takip eder.")
    span = estimate_range(summary["seconds"], fast, stats)
    if span and span[1] > LONG_ESTIMATE_MINUTES:
        lines.append("⚠️ Üretim uzun sürecek. Önce kısa bir deneme için *Başlangıç/Süre* ile bir bölüm seçebilirsiniz.")
    lines.append("Nota, *Model ayarları → Harici ABC nota* alanında; isterseniz düzenleyebilirsiniz. "
                 + ("Tempo yanlışsa BPM girip yeniden çıkarın." if summary.get("transcriber", "basic") == "basic" else
                    "Tempo yanlışsa *Basit analiz*'e geçip BPM girebilirsiniz (SheetSage2 vuruşları kendisi bulur)."))
    return "  \n".join(lines)


def lyric_blocks(lyrics):
    """[(label, lines)] of the lyrics' `[section]` blocks; lines before any label get label ''."""
    blocks = []
    for raw in (lyrics or "").splitlines():
        line = raw.strip()
        match = re.fullmatch(r"\[([^\]]+)\]", line)
        if match:
            blocks.append((match.group(1).strip().lower(), []))
        elif line:
            if not blocks:
                blocks.append(("", []))
            blocks[-1][1].append(line)
    return blocks


def score_sections(abc):
    """Section names of an ABC score, from its `% name` comment lines."""
    return [name.lower() for name in re.findall(r"^%(?!%)\s*(.+?)\s*$", abc or "", re.MULTILINE)]


def structure_problem(lyrics, abc):
    """Why the lyrics would not line up with the score's sung sections, or None.

    YuE2 sings the n-th `[section]` block of the lyrics over the n-th sung `% section`
    of the score, so both must list the same sections in the same order."""
    sung = [s for s in score_sections(abc) if s not in NON_VOCAL]
    blocks = [label for label, _ in lyric_blocks(lyrics) if label not in NON_VOCAL]
    if not sung or not blocks or sung == blocks:
        return None
    if "" in blocks:
        return "Sözlerin başında bölüm etiketi olmayan satırlar var; ilk satırın üstüne `[verse]` gibi bir etiket ekleyin."
    common = min(len(sung), len(blocks))
    first = next((i for i in range(common) if sung[i] != blocks[i]), common)
    detail = (f"{first + 1}. şarkılı bölüm notada `[{sung[first]}]`, sözlerde `[{blocks[first]}]`" if first < common
              else f"notada {len(sung)}, sözlerde {len(blocks)} şarkılı bölüm var")
    return f"Sözlerin bölümleri notayla eşleşmiyor ({detail}). Model sözleri bu yüzden başka yerlere kaydırabilir."


def clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def timeline_markdown(lyrics, abc, analysis):
    """When each lyric line will be sung: the analyzed sections with times, filled with the current lyrics."""
    if not (abc or "").strip():
        return ("⚠️ Referans ayarları değişti; zaman çizelgesi için **Sözleri tanı** ya da **Melodiyi çıkar**'a "
                "yeniden basın." if analysis else TIMELINE_HINT)
    problem = structure_problem(lyrics, abc)
    head = f"⚠️ {problem}" if problem else ""
    if not analysis or analysis["abc"].strip() != abc.strip():
        # Score pasted or restored from history: no times, only the structure can be checked.
        return head or "Nota bu referans analizinden gelmiyor; zaman çizelgesi gösterilemiyor."
    summary = analysis["summary"]
    if not summary.get("sections"):
        return head
    blocks = [block for block in lyric_blocks(lyrics) if block[0] not in NON_VOCAL]
    recognized = {}
    for line in summary.get("lines") or []:
        recognized.setdefault(line["block"], []).append(line)
    rows, k = [], 0
    for section in summary["sections"]:
        start = clock(section["start"])
        if section["name"] in NON_VOCAL:
            rows.append(f"{start:>5}  ({section['name']}, müzik)")
            continue
        label, texts = blocks[k] if k < len(blocks) else ("", [])
        # Line times stay valid while a block keeps its recognized number of lines.
        times = recognized.get(k, [])
        mark = (f"   ← notada [{section['name']}]" if label != section["name"] else
                "   ← satır sayısı değişti, satır zamanları yok" if times and texts and len(times) != len(texts) else "")
        rows.append(f"{start:>5}  [{label or '?'}]{mark}")
        for i, text in enumerate(texts):
            at = clock(times[i]["start"]) if len(times) == len(texts) else ""
            rows.append(f"      {at:>5}  {text}")
        k += 1
    for label, texts in blocks[k:]:
        rows.append(f"    -  [{label}]   ← notada karşılığı yok")
        rows += [f"             {text}" for text in texts]
    rows.append(f"{clock(summary['sections'][-1]['end']):>5}  (son)")
    if not head:
        head = (f"✅ Sözler notayla eşleşiyor: {k} şarkılı bölüm. Kelimeleri düzeltebilirsiniz; bölüm etiketlerini "
                "silmeyin ve sırasını değiştirmeyin." if blocks else
                "Notadaki bölümler aşağıda. Sözleri bu sırayla `[bölüm]` etiketleriyle yazın.")
    return head + "\n\n```text\n" + "\n".join(rows) + "\n```"


def job_label(job):
    style = job["request"].get("style", "")
    style = style if len(style) <= 50 else style[:47] + "..."
    kind = "✦ " if job["request"].get("prompt") else "🎵 " if job["request"].get("reference") else ""
    return f"{kind}{job['id']} ·{STATUS_LABELS.get(job['status'], job['status'])} · {style}"


def sampling_dict(name, values):
    """Sampling overrides of one stage from its controls (in SAMPLING_KEYS order)."""
    sampling = dict(zip(SAMPLING_KEYS, values))
    for key in ("top_k", "penalty_window", "min_tokens", "max_tokens"):
        sampling[key] = int(sampling[key])
    if sampling["min_tokens"] > sampling["max_tokens"]:
        raise gr.Error(f"{name} örneklemesinde Min token, Max token'dan büyük olamaz.")
    return sampling


def build_payload(genres, lyrics, cot, random_seed, seed, custom_cfg, cfg_scale, abc_text, *sampling_values):
    style = ", ".join(g.strip() for g in (genres or []) if g.strip())
    if not style:
        raise gr.Error("En az bir tür/stil etiketi seçin.")
    if not (lyrics or "").strip():
        raise gr.Error("Şarkı sözlerini girin ya da referans şarkıdan ✨ Sözleri tanı ile çıkarın.")
    abc = (abc_text or "").strip() or None
    if abc and cot == "off":
        raise gr.Error("Harici ABC nota kullanmak için CoT modu 'Tam' veya 'Melodi' olmalı.")
    count = len(SAMPLING_KEYS)
    return {
        "style": style,
        "lyrics": lyrics.strip(),
        "cot": cot,
        "seed": None if random_seed else int(seed),
        "cfg_scale": float(cfg_scale) if custom_cfg else None,
        "abc": abc,
        "abc_sampling": sampling_dict("ABC", sampling_values[:count]),
        "semantic_sampling": sampling_dict("Şarkı", sampling_values[count:]),
    }


def strip_chords(abc):
    """Melody-only score: chord symbols ("Gm") removed from the music lines; headers and comments are kept."""
    return "\n".join(line if re.match(r"\s*(%|[A-Za-z]:)", line) else re.sub(r'"[^"]*"', "", line)
                     for line in abc.splitlines()) + "\n"


def cover_score(abc, mode):
    """The analyzed full score as sent for the chosen reference melody mode."""
    return strip_chords(abc) if mode == "free" else abc


def t2s_style(description, styles, custom_style, voice):
    """The model's [Tags] text from the Text to Song form: styles, voice, then the free description."""
    parts = [*(styles or []), (custom_style or "").strip(), VOICE_TAGS.get(voice), (description or "").strip()]
    return ", ".join(p for p in parts if p)


def normalize_lyrics(lyrics):
    """Lyrics with `[Verse]`-style labels lowercased, as in YuE's own examples."""
    return re.sub(r"^[ \t]*\[([^\]]+)\][ \t]*$", lambda m: f"[{m.group(1).strip().lower()}]",
                  (lyrics or "").strip(), flags=re.MULTILINE)


def t2s_payload(description, lyrics, voice, styles, custom_style, random_seed, seed, cot, fast):
    style = t2s_style(description, styles, custom_style, voice)
    if not style:
        raise gr.Error("Şarkı açıklaması yazın ya da en az bir tür seçin.")
    if not (lyrics or "").strip():
        raise gr.Error("Şarkı sözlerini girin.")
    if not re.search(r"^\s*\[[^\]]+\]\s*$", lyrics, re.MULTILINE):
        gr.Warning("Sözlerde bölüm etiketi yok. Model en iyi `[verse]`, `[chorus]` gibi bölümlerle çalışır.")
    return {"style": style, "lyrics": normalize_lyrics(lyrics), "cot": cot,
            "seed": None if random_seed else int(seed), "ode_steps": FAST_ODE_STEPS if fast else None,
            "prompt": {"description": (description or "").strip(), "styles": list(styles or []),
                       "custom_style": (custom_style or "").strip(), "voice": voice}}


def counter_text(text, limit):
    return f"<div class='t2s-counter'>{len(text or '')}/{limit}</div>"


def follow_job(client, payload):
    """Submit a generation and yield the job on every poll until it finishes."""
    job = client.submit(payload)
    yield job
    failures = 0
    while job["status"] not in FINAL:
        time.sleep(1)
        try:
            job = client.job(job["id"])
            failures = 0
        except requests.RequestException:
            failures += 1
            if failures > 10:
                raise gr.Error("Model sunucusuyla bağlantı koptu.")
            continue
        yield job


def sampling_controls(defaults):
    return (
        gr.Slider(0.0, 5.0, value=defaults["temperature"], step=0.05, label="Temperature",
                  info="Düşük: daha tutarlı, tekrara yatkın. Yüksek: daha yaratıcı, dağınık. 0 = hep en olası token."),
        gr.Slider(0.05, 1.0, value=defaults["top_p"], step=0.01, label="Top-p",
                  info="Olasılıkları toplamı bu orana ulaşan tokenlar arasından seçer. Düşük = daha güvenli."),
        gr.Slider(1, 500, value=defaults["top_k"], step=1, label="Top-k",
                  info="Her adımda en olası bu kadar token arasından seçer."),
        gr.Slider(0.5, 3.0, value=defaults["repetition_penalty"], step=0.005, label="Tekrar cezası",
                  info="1 = ceza yok. Büyüdükçe son tokenları tekrarlamak zorlaşır (döngüleri kırar)."),
        gr.Slider(1, 100, value=defaults["penalty_window"], step=1, label="Tekrar cezası penceresi",
                  info="Cezanın baktığı son token sayısı."),
        gr.Number(value=defaults["min_tokens"], precision=0, minimum=0, label="Min token",
                  info="Bu kadar token üretilmeden bitiş tokenına izin verilmez."),
        gr.Number(value=defaults["max_tokens"], precision=0, minimum=1, label="Max token",
                  info="Üst sınır. Şarkıda saniye başına 25 token; dolarsa şarkı kesilir."),
    )


def create_interface():
    genres = load_and_process_genres(ROOT / "top_200_tags.json")

    with gr.Blocks(title="YuE2 - UI", theme=gr.themes.Base(), css=CSS) as interface:
        gr.Markdown("# YuE2 - UI")
        connection = gr.Markdown(connection_markdown(DEFAULT_SERVER))
        job_id = gr.State(None)
        t2s_job = gr.State(None)
        # Last reference analysis {"abc", "summary"}: section and line times for the timeline.
        analysis = gr.State(None)

        with gr.Tabs() as main_tabs:
            with gr.Tab("✦ Text to Song", id="t2s"):
                with gr.Column(elem_id="t2s-card"):
                    t2s_description = gr.Textbox(
                        label="Şarkı açıklaması", info="Tema, sahne ve ifade", lines=4, max_lines=12,
                        max_length=DESCRIPTION_LIMIT,
                        placeholder="120 BPM'de, sürükleyici ritimli ve kadın vokalli siberpunk bir synth-pop şarkısı.")
                    t2s_description_count = gr.HTML(counter_text("", DESCRIPTION_LIMIT))
                    t2s_lyrics = gr.Textbox(
                        label="Şarkı sözleri", info="Yapıştırın ya da kendiniz yazın. Bölümleri [verse] [chorus] "
                                                    "[bridge] ile ayırın.",
                        lines=10, max_lines=30, max_length=LYRICS_LIMIT, placeholder=T2S_LYRICS_PLACEHOLDER)
                    t2s_lyrics_count = gr.HTML(counter_text("", LYRICS_LIMIT))
                    with gr.Row(equal_height=True):
                        t2s_voice_button = gr.Button("🎙 Vokal: Varsayılan", size="sm", variant="secondary",
                                                     elem_classes="t2s-tool")
                        t2s_style_button = gr.Button("🎚 Tür", size="sm", variant="secondary", elem_classes="t2s-tool")
                        t2s_generate = gr.Button("⚡ Şarkı üret", elem_id="t2s-generate")
                    with gr.Column(visible=False, elem_classes="t2s-popup") as t2s_voice_panel:
                        t2s_voice = gr.Radio(T2S_VOICES, value="default", label="Vokal", elem_classes="t2s-pills")
                    with gr.Column(visible=False, elem_classes="t2s-popup") as t2s_style_panel:
                        t2s_styles = gr.CheckboxGroup(T2S_STYLES, label="Tür", elem_classes="t2s-pills")
                        t2s_custom_style = gr.Textbox(show_label=False, max_lines=1,
                                                      placeholder="Özel tür... (ör. Anadolu rock, bağlama)")
                        t2s_style_close = gr.Button("Tamam", size="sm")
                    with gr.Accordion("Gelişmiş ayarlar", open=False):
                        t2s_cot = gr.Radio(COT_CHOICES, value="full", label="CoT modu",
                                           info="Tam/Melodi önce bir ABC nota planı üretir, sonra şarkıyı ona göre "
                                                "üretir.")
                        with gr.Row():
                            t2s_random_seed = gr.Checkbox(value=True, label="Rastgele seed",
                                                          info="Aynı girdiyle sonuçları karşılaştırmak için kapatıp "
                                                               "seed girin.")
                            t2s_seed = gr.Number(value=831001, precision=0, minimum=0, label="Seed")
                            t2s_fast = gr.Checkbox(value=False, label="⚡ Hızlı sentez",
                                                   info=f"Ses sentezi 32 yerine {FAST_ODE_STEPS} adım: daha hızlı, "
                                                        "kalite biraz düşebilir.")
                with gr.Column():
                    with gr.Row(equal_height=True):
                        t2s_progress = gr.Markdown("Henüz üretim yapılmadı.")
                        t2s_cancel = gr.Button("İptal", variant="stop", size="sm", scale=0, min_width=90)
                    t2s_audio = gr.Audio(label="Üretilen şarkı (MP3)", type="filepath", interactive=False)
                    with gr.Accordion("Modele giden girdi", open=False):
                        t2s_prompt_view = gr.Code(label="[Tags] ve [Lyrics]", language=None, interactive=False)
                    with gr.Accordion("Üretilen ABC nota planı", open=False):
                        t2s_abc = gr.Code(show_label=False, language=None, interactive=False)
            with gr.Tab("🎵 Cover", id="cover"):
                with gr.Row():
                    with gr.Column():
                        with gr.Tab("Prompt"):
                            with gr.Accordion("Referans şarkı", open=True):
                                cover_audio = gr.Audio(label="Referans şarkı (MP3/WAV/FLAC)", sources=["upload"],
                                                       type="filepath")
                                with gr.Row():
                                    cover_start = gr.Number(value=0, minimum=0, label="Başlangıç (sn)")
                                    cover_duration = gr.Number(value=0, minimum=0, label="Süre (sn)",
                                                               info="0 = tüm şarkı")
                                    cover_bpm = gr.Number(value=0, minimum=0, maximum=299, label="Tempo (BPM)",
                                                          info="0 = otomatik. Yalnızca Basit analiz kullanır.")
                                transcriber = gr.Radio(
                                    TRANSCRIBERS, value="sheetsage2", label="Nota çıkarıcı",
                                    info="SheetSage2: YuE2 ekibinin cover için kullandığı model; melodi, akor, bölüm ve "
                                         "enstrüman melodisini çıkarır. Basit analiz: eski yöntem; enstrümantal kayıtların "
                                         "melodisini vokale taşır ve elle BPM kabul eder.")
                                cover_mode = gr.Radio(
                                    COVER_MODES, value="close", label="Referans melodisi",
                                    info="Yakın: akorlu nota, CoT 'Tam' (orijinale daha sadık). Serbest: akorlar notadan "
                                         "silinir, CoT 'Melodi'; eşlik yeni stile göre kendi akorlarını seçer (stil "
                                         "değiştirirken YuE2'nin önerdiği mod).")
                                cover_button = gr.Button("Melodiyi çıkar (önizle)", size="sm")
                                cover_info = gr.Markdown(
                                    "MP3 yüklerseniz yeni şarkı onun melodisini, akorlarını ve yapısını takip eder; "
                                    "tarz, enstrümanlar ve vokal aşağıdaki etiketlerden gelir. "
                                    "Orijinal sesin tınısı/aranjmanı kopyalanmaz. Sözlü bir şarkıysa **✨ Sözleri tanı** "
                                    "sözleri de şarkıdaki yerleriyle birlikte çıkarır.")
                            genre_selection = gr.Dropdown(
                                label="Tür / stil etiketleri",
                                info="Tür, enstrüman, ruh hali, vokal cinsiyeti ve tınısını tarif edin. "
                                     "Listede olmayan etiketleri de yazabilirsiniz (ör. Turkish, female vocal).",
                                choices=genres, multiselect=True, allow_custom_value=True,
                                interactive=True, max_choices=50)
                            with gr.Row(equal_height=True):
                                recognize_button = gr.Button("✨ Sözleri tanı", scale=2)
                                lyrics_language = gr.Dropdown(LANGUAGE_CHOICES, value="auto", label="Söz dili",
                                                              info="Türkçe algılanır ya da seçilirse Türkçe'ye özel model "
                                                                   "kullanılır (daha doğru, daha yavaş).",
                                                              scale=1, min_width=160)
                            lyrics_text = gr.Textbox(
                                label="Şarkı sözleri", lines=16, max_lines=40,
                                placeholder="[verse]\nİlk kıta...\n\n[chorus]\nNakarat...",
                                info="Bölümleri köşeli parantezle adlandırın: [verse] [chorus] [bridge]. Sözleri tanıttıysanız "
                                     "kelimeleri düzeltebilirsiniz; bölüm etiketlerini silmeyin, sırasını değiştirmeyin.")
                            with gr.Accordion("🕒 Zaman çizelgesi: sözler şarkının neresinde söylenecek", open=True):
                                timeline = gr.Markdown(TIMELINE_HINT)
                            with gr.Row():
                                example_button = gr.Button("Örneği yükle", size="sm")
                                load_button = gr.UploadButton("Projeyi yükle", file_types=[".json"], size="sm")
                                save_button = gr.Button("Projeyi kaydet", size="sm")
                            project_file = gr.File(label="Proje dosyası", visible=False)

                        with gr.Tab("Model ayarları"):
                            cot = gr.Radio(COT_CHOICES, value="full", label="CoT modu",
                                           info="Tam/Melodi önce bir ABC nota planı üretir, sonra şarkıyı ona göre üretir.")
                            with gr.Row():
                                random_seed = gr.Checkbox(value=True, label="Rastgele seed")
                                seed = gr.Number(value=831001, precision=0, minimum=0, label="Seed")
                            with gr.Row():
                                custom_cfg = gr.Checkbox(value=False, label="CFG'yi özelleştir",
                                                         info="Kapalıyken model varsayılanı kullanılır (full/melody 1.0, off 1.01).")
                                cfg_scale = gr.Slider(0, 20, value=1.0, step=0.05, label="CFG scale",
                                                      info="1'den büyük: stil etiketlerine ve sözlere daha sıkı uyar (notaya "
                                                           "değil). 1'den farklı her değer şarkı aşamasını ~2 kat yavaşlatır.")
                            with gr.Row():
                                ode_steps = gr.Slider(4, 64, value=DEFAULT_ODE_STEPS, step=1, label="Ses sentezi adımı (ODE)",
                                                      info=f"Varsayılan {DEFAULT_ODE_STEPS}. Az adım: hızlı, kalite düşebilir. "
                                                           f"Çok adım: yavaş. ⚡ Hızlı sentez = {FAST_ODE_STEPS}.")
                                fp8 = gr.Checkbox(value=False, label="FP8 (deneysel)",
                                                  info="Token üretiminde FP8 ağırlıklar. Yalnızca compute 8.9+ NVIDIA (ör. L4). "
                                                       "Kalite etkisi denenmedi; yue2 FP8'de CUDA Graph kullanmadığı için "
                                                       "daha yavaş da olabilir.")
                            with gr.Accordion("ABC (nota planı) örnekleme", open=False):
                                abc_controls = sampling_controls(ABC_DEFAULTS)
                            with gr.Accordion("Şarkı (semantik) örnekleme", open=False):
                                semantic_controls = sampling_controls(SEMANTIC_DEFAULTS)
                            with gr.Accordion("Harici ABC nota", open=False):
                                abc_text = gr.Textbox(
                                    label="ABC nota (isteğe bağlı)", lines=8,
                                    info="Doluysa model nota planı üretmez, bu notayı kullanır. "
                                         "Önceki bir üretimin ABC'sini buraya yapıştırıp sözleri/stili değiştirebilirsiniz.")

                    with gr.Column():
                        with gr.Tab("Üretim"):
                            with gr.Row():
                                generate_button = gr.Button("🎵 Şarkı üret", variant="primary")
                                cancel_button = gr.Button("İptal", variant="stop")
                            fast_mode = gr.Checkbox(value=False, label="⚡ Hızlı sentez",
                                                    info=f"Ses sentezi 32 yerine {FAST_ODE_STEPS} adım: toplam süre kısalır, "
                                                         "kalite biraz düşebilir.")
                            progress = gr.Markdown("Henüz üretim yapılmadı.")
                            audio = gr.Audio(label="Üretilen şarkı (MP3)", type="filepath", interactive=False)
                            abc_output = gr.Code(label="Üretilen ABC nota planı", language=None, interactive=False)
                            use_abc_button = gr.Button("Bu ABC'yi harici nota olarak kullan", size="sm")
            with gr.Tab("🗂 Geçmiş", id="history"):
                with gr.Row():
                    history = gr.Dropdown(label="Önceki üretimler", choices=[], interactive=True, scale=4)
                    refresh_button = gr.Button("Yenile", size="sm", scale=1)
                history_progress = gr.Markdown()
                history_audio = gr.Audio(label="Şarkı", type="filepath", interactive=False)
                history_abc = gr.Code(label="ABC nota planı", language=None, interactive=False)
                restore_button = gr.Button("Ayarları forma geri yükle", size="sm")
            with gr.Tab("Sunucu", id="server"):
                server_url = gr.Textbox(value=DEFAULT_SERVER, label="Model sunucusu adresi",
                                        info="server/yue2_server.py'nin çalıştığı adres")
                check_button = gr.Button("Bağlantıyı test et")
                server_info = gr.JSON(label="Sunucu durumu")

        sampling_inputs = [*abc_controls, *semantic_controls]
        form_inputs = [genre_selection, lyrics_text, cot, random_seed, seed, custom_cfg, cfg_scale,
                       abc_text, *sampling_inputs]

        # --- Generation -------------------------------------------------------
        def analyze_reference(url, path, start, duration, bpm, lyrics, mode, method, recognize=False, language=None):
            """Analysis state {"abc", "score", "summary", "reference"} of the reference song.

            "score" is the analyzed full score; "abc" is what is sent for the reference melody mode."""
            try:
                result = ServerClient(url).analyze(path, start, duration, bpm, lyrics, recognize, language, method)
            except requests.RequestException:
                raise gr.Error(f"Model sunucusuna bağlanılamadı: {url}")
            summary = result["summary"]
            reference = {"file": Path(path).name, "start": start or 0, "duration": duration or None,
                         **{k: v for k, v in summary.items() if k != "lines"}}
            return {"abc": cover_score(result["abc"], mode), "score": result["abc"], "lyrics": result.get("lyrics"),
                    "summary": summary, "reference": {**reference, "mode": mode}}

        def semantic_budget(summary, current):
            """Song token limit that fits a cover as long as the reference (the default cuts long songs)."""
            return min(20000, max(int(current), int(summary["seconds"] * TOKENS_PER_AUDIO_SECOND * 1.3)))

        def preview_cover(url, path, start, duration, bpm, lyrics, mode, method, fast, sem_max):
            if not path:
                raise gr.Error("Önce bir referans şarkı yükleyin.")
            gr.Info("Referans şarkının notası çıkarılıyor...")
            result = analyze_reference(url, path, start, duration, bpm, lyrics, mode, method)
            summary = result["summary"]
            return (result["abc"], cover_markdown(summary, fast, ServerClient(url).stats()), COVER_COT[mode],
                    semantic_budget(summary, sem_max), result)

        def recognize(url, path, start, duration, bpm, language, mode, method, fast, sem_max):
            if not path:
                raise gr.Error("Önce bir referans şarkı yükleyin; sözler o şarkıdan tanınır.")
            gr.Info("Sözler tanınıyor (vokal ayırma + Whisper); şarkının uzunluğuna göre birkaç dakika sürebilir. "
                    "Aynı şarkı ve ayarlar ikinci kez anında gelir.")
            # Recognized lyrics name the sections themselves, so the box's labels are not sent.
            result = analyze_reference(url, path, start, duration, bpm, "", mode, method, True,
                                       None if language == "auto" else language)
            summary = result["summary"]
            if summary.get("lyrics_status") == "ok":
                gr.Info("Sözler tanındı. Yanlış kelimeleri düzeltebilirsiniz; bölüm etiketlerini silmeyin.")
                lyrics = result["lyrics"]
            else:
                begin = float(start or 0)
                span = f"{clock(begin)}-{clock(begin + duration)}" if duration else f"{clock(begin)}'dan sona kadar"
                gr.Warning(f"Seçilen aralıkta ({span}) vokal yok. Vokalli bir bölüm seçin ya da Süre'yi 0 yapın (tüm şarkı)."
                           if summary.get("lyrics_status") == "instrumental" else
                           f"Seçilen aralıkta ({span}) vokal var ama kelimeler anlaşılamadı. Sözleri elle yazın.")
                lyrics = gr.update()
            return (lyrics, result["abc"], cover_markdown(summary, fast, ServerClient(url).stats()), COVER_COT[mode],
                    semantic_budget(summary, sem_max), result)

        def change_cover_mode(mode, abc, current):
            """Switch the analyzed score between full and melody-only, unless it was edited by hand."""
            if not current:
                return gr.update(), gr.update(), current
            if (abc or "").strip() != current["abc"].strip():
                gr.Warning("Nota elle değiştirilmiş; mod değişikliği notaya uygulanmadı. Yeniden uygulamak için "
                           "**Melodiyi çıkar**'a basın.")
                return gr.update(), gr.update(), current
            score = cover_score(current["score"], mode)
            return score, COVER_COT[mode], {**current, "abc": score,
                                            "reference": {**current["reference"], "mode": mode}}

        timeline_inputs = [lyrics_text, abc_text, analysis]
        analysis_outputs = [abc_text, cover_info, cot, semantic_controls[-1], analysis]
        cover_button.click(preview_cover, inputs=[server_url, cover_audio, cover_start, cover_duration, cover_bpm,
                                                  lyrics_text, cover_mode, transcriber, fast_mode,
                                                  semantic_controls[-1]],
                           outputs=analysis_outputs
                           ).then(timeline_markdown, inputs=timeline_inputs, outputs=timeline)
        recognize_button.click(recognize, inputs=[server_url, cover_audio, cover_start, cover_duration, cover_bpm,
                                                  lyrics_language, cover_mode, transcriber, fast_mode,
                                                  semantic_controls[-1]],
                               outputs=[lyrics_text, *analysis_outputs]
                               ).then(timeline_markdown, inputs=timeline_inputs, outputs=timeline)
        cover_mode.input(change_cover_mode, inputs=[cover_mode, abc_text, analysis], outputs=[abc_text, cot, analysis]
                         ).then(timeline_markdown, inputs=timeline_inputs, outputs=timeline)
        # ⚡ Hızlı sentez is a preset of the ODE step slider; either one updates the other.
        fast_mode.input(lambda on: FAST_ODE_STEPS if on else DEFAULT_ODE_STEPS, inputs=fast_mode, outputs=ode_steps)
        ode_steps.input(lambda steps: int(steps) == FAST_ODE_STEPS, inputs=ode_steps, outputs=fast_mode)
        for component in (lyrics_text, abc_text):
            component.change(timeline_markdown, inputs=timeline_inputs, outputs=timeline, show_progress="hidden")
        # A new reference invalidates the extracted score; new analysis settings leave it stale (timeline says so).
        for event in (cover_audio.upload, cover_audio.clear):
            event(lambda: ("", None), outputs=[abc_text, analysis])
        for field in (cover_start, cover_duration, cover_bpm, transcriber):
            field.input(lambda path, abc: "" if path else abc, inputs=[cover_audio, abc_text], outputs=abc_text)

        def generate(url, cover_path, start, duration, bpm, mode, method, fast, steps, use_fp8, current, *values):
            values = list(values)
            lyrics, cot_index, abc_index, sem_max_index = values[1], 2, 7, len(values) - 1
            client = ServerClient(url)
            try:
                client.health()
            except requests.RequestException:
                raise gr.Error(f"Model sunucusuna bağlanılamadı: {url}")
            stats = client.stats()
            if cover_path and not (values[abc_index] or "").strip():
                build_payload(*values)  # validate the prompt before the slow analysis
                yield {progress: "⏳ Referans şarkı analiz ediliyor (vokal ayırma + melodi çıkarma)..."}
                current = analyze_reference(url, cover_path, start, duration, bpm, lyrics, mode, method)
                summary = current["summary"]
                values[abc_index] = current["abc"]
                values[cot_index] = COVER_COT[mode]
                span = estimate_range(summary["seconds"], fast, stats)
                if span and span[1] > LONG_ESTIMATE_MINUTES:
                    gr.Warning(f"Uzun sürecek: tahmini {estimate_text(summary['seconds'], fast, stats)}. "
                               "İptal edip Başlangıç/Süre ile daha kısa bir bölüm seçebilirsiniz.")
                yield {abc_text: current["abc"], cover_info: cover_markdown(summary, fast, stats),
                       cot: values[cot_index], analysis: current}
            # The score came from this reference (now or via Sözleri tanı / önizle): size the song to it.
            reference = None
            if cover_path and current and current["abc"].strip() == (values[abc_index] or "").strip():
                reference = current["reference"]
                budget = semantic_budget(current["summary"], values[sem_max_index])
                if budget != values[sem_max_index]:
                    values[sem_max_index] = budget
                    yield {semantic_controls[-1]: budget}
            problem = structure_problem(lyrics, values[abc_index])
            if problem:
                gr.Warning(problem)
            payload = build_payload(*values)
            payload["reference"] = reference
            payload["ode_steps"] = None if int(steps) == DEFAULT_ODE_STEPS else int(steps)
            payload["quantization"] = "fp8" if use_fp8 else None
            abc_max, sem_max = payload["abc_sampling"]["max_tokens"], payload["semantic_sampling"]["max_tokens"]
            yield {audio: None, abc_output: None}
            for job in follow_job(client, payload):
                yield {job_id: job["id"], progress: progress_markdown(job, abc_max, sem_max, stats)}
            audio_path = client.download_audio(job) if job["status"] == "complete" else None
            yield {job_id: None, progress: progress_markdown(job, abc_max, sem_max, stats), audio: audio_path,
                   abc_output: (job.get("result") or {}).get("abc")}

        generate_button.click(generate, inputs=[server_url, cover_audio, cover_start, cover_duration, cover_bpm,
                                                cover_mode, transcriber, fast_mode, ode_steps, fp8, analysis,
                                                *form_inputs],
                              outputs=[job_id, progress, audio, abc_output, abc_text, cover_info, cot,
                                       semantic_controls[-1], analysis])

        def cancel(url, current):
            if not current:
                gr.Info("Çalışan bir üretim yok.")
                return
            ServerClient(url).cancel(current)
            gr.Info("İptal isteği gönderildi.")

        cancel_button.click(cancel, inputs=[server_url, job_id])
        use_abc_button.click(lambda abc: abc or "", inputs=abc_output, outputs=abc_text)

        # --- Server connection ------------------------------------------------
        def check(url):
            try:
                info = ServerClient(url).health()
            except Exception as exc:
                info = {"status": "unreachable", "error": str(exc)}
            return connection_markdown(url), info

        check_button.click(check, inputs=server_url, outputs=[connection, server_info])
        server_url.submit(check, inputs=server_url, outputs=[connection, server_info])
        gr.Timer(5).tick(connection_markdown, inputs=server_url, outputs=connection, show_progress="hidden")

        # --- History ----------------------------------------------------------
        def refresh(url, quiet=False):
            try:
                jobs = ServerClient(url).jobs()
            except requests.RequestException:
                if not quiet:
                    gr.Warning("Model sunucusuna bağlanılamadı.")
                return gr.update(choices=[])
            return gr.update(choices=[(job_label(j), j["id"]) for j in jobs],
                             value=jobs[0]["id"] if jobs else None)

        def show_job(url, selected):
            if not selected:
                return "", None, None
            client = ServerClient(url)
            job = client.job(selected)
            req = job["request"]
            abc_max = (req.get("abc_sampling") or {}).get("max_tokens") or ABC_DEFAULTS["max_tokens"]
            sem_max = (req.get("semantic_sampling") or {}).get("max_tokens") or SEMANTIC_DEFAULTS["max_tokens"]
            audio_path = client.download_audio(job) if job["status"] == "complete" else None
            return progress_markdown(job, abc_max, sem_max, client.stats()), audio_path, (job.get("result") or {}).get("abc")

        t2s_fields = [t2s_description, t2s_lyrics, t2s_voice, t2s_styles, t2s_custom_style, t2s_random_seed,
                      t2s_seed, t2s_cot, t2s_fast]
        cover_fields = [*form_inputs, ode_steps, fast_mode, fp8]

        def restore(url, selected):
            """Put a past job's inputs back into the form it came from (Text to Song or Cover) and open it."""
            if not selected:
                raise gr.Error("Önce geçmişten bir üretim seçin.")
            req = ServerClient(url).job(selected)["request"]
            steps = req.get("ode_steps") or DEFAULT_ODE_STEPS
            prompt = req.get("prompt")
            if prompt:
                values = [prompt.get("description", ""), req["lyrics"], prompt.get("voice", "default"),
                          prompt.get("styles") or [], prompt.get("custom_style", ""), False, req["seed"],
                          req["cot"], steps == FAST_ODE_STEPS]
                return {**dict(zip(t2s_fields, values)), main_tabs: gr.Tabs(selected="t2s")}
            abc_s = {**ABC_DEFAULTS, **(req.get("abc_sampling") or {})}
            sem_s = {**SEMANTIC_DEFAULTS, **(req.get("semantic_sampling") or {})}
            values = [[t.strip() for t in req["style"].split(",") if t.strip()], req["lyrics"], req["cot"],
                      False, req["seed"], req.get("cfg_scale") is not None,
                      req.get("cfg_scale") if req.get("cfg_scale") is not None else 1.0, req.get("abc") or "",
                      *[abc_s[k] for k in SAMPLING_KEYS], *[sem_s[k] for k in SAMPLING_KEYS],
                      steps, steps == FAST_ODE_STEPS, req.get("quantization") == "fp8"]
            return {**dict(zip(cover_fields, values)), main_tabs: gr.Tabs(selected="cover")}

        refresh_button.click(refresh, inputs=server_url, outputs=history)
        history.change(show_job, inputs=[server_url, history],
                       outputs=[history_progress, history_audio, history_abc])
        restore_button.click(restore, inputs=[server_url, history], outputs=[*cover_fields, *t2s_fields, main_tabs])

        # --- Text to Song -------------------------------------------------------
        for box, count, limit in ((t2s_description, t2s_description_count, DESCRIPTION_LIMIT),
                                  (t2s_lyrics, t2s_lyrics_count, LYRICS_LIMIT)):
            box.change(lambda text, limit=limit: counter_text(text, limit), inputs=box, outputs=count,
                       show_progress="hidden")
        # Voice and Style open as small panels; picking a voice or pressing Tamam closes them.
        popups = [t2s_voice_panel, t2s_style_panel]
        t2s_voice_button.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=popups)
        t2s_style_button.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=popups)
        t2s_style_close.click(lambda: gr.update(visible=False), outputs=t2s_style_panel)
        voice_names = {value: label for label, value in T2S_VOICES}
        t2s_voice.change(lambda v: (f"🎙 Vokal: {voice_names[v]}", gr.update(visible=False)), inputs=t2s_voice,
                         outputs=[t2s_voice_button, t2s_voice_panel], show_progress="hidden")

        def style_button_label(styles, custom):
            count = len(styles or []) + bool((custom or "").strip())
            return f"🎚 Tür ({count})" if count else "🎚 Tür"

        for field in (t2s_styles, t2s_custom_style):
            field.change(style_button_label, inputs=[t2s_styles, t2s_custom_style], outputs=t2s_style_button,
                         show_progress="hidden")

        def generate_t2s(url, *form):
            payload = t2s_payload(*form)
            client = ServerClient(url)
            try:
                client.health()
            except requests.RequestException:
                raise gr.Error(f"Model sunucusuna bağlanılamadı: {url}")
            stats = client.stats()
            abc_max, sem_max = ABC_DEFAULTS["max_tokens"], SEMANTIC_DEFAULTS["max_tokens"]
            yield {t2s_audio: None, t2s_abc: None, t2s_voice_panel: gr.update(visible=False),
                   t2s_style_panel: gr.update(visible=False),
                   t2s_prompt_view: f"[Tags]\n{payload['style']}\n\n[Lyrics]\n{payload['lyrics']}"}
            for job in follow_job(client, payload):
                yield {t2s_job: job["id"], t2s_progress: progress_markdown(job, abc_max, sem_max, stats)}
            yield {t2s_job: None, t2s_audio: client.download_audio(job) if job["status"] == "complete" else None,
                   t2s_abc: (job.get("result") or {}).get("abc")}

        t2s_generate.click(generate_t2s, inputs=[server_url, *t2s_fields],
                           outputs=[t2s_job, t2s_progress, t2s_audio, t2s_abc, t2s_prompt_view, *popups])
        t2s_cancel.click(cancel, inputs=[server_url, t2s_job])

        # --- Project load / save (YuE-UI compatible genre_text / lyrics_text) --
        def load_project(path):
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            settings = data.get("yue2", {})
            return (data.get("genre_text") or [], data.get("lyrics_text") or "",
                    settings.get("cot", "full"), settings.get("abc") or "")

        def save_project(genres_value, lyrics, cot_value, abc_value):
            AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
            path = AUDIO_CACHE / f"yue2_project_{time.strftime('%Y%m%d-%H%M%S')}.json"
            path.write_text(json.dumps({"genre_text": genres_value or [], "lyrics_text": lyrics or "",
                                        "yue2": {"cot": cot_value, "abc": abc_value or None}},
                                       indent=2, ensure_ascii=False), encoding="utf-8")
            return str(path)

        project_fields = [genre_selection, lyrics_text, cot, abc_text]
        example_button.click(lambda: load_project(ROOT / "examples" / "example.json"), outputs=project_fields)
        load_button.upload(load_project, inputs=load_button, outputs=project_fields)
        save_button.click(lambda *v: gr.update(value=save_project(*v), visible=True),
                          inputs=project_fields, outputs=project_file)

        interface.load(lambda url: refresh(url, quiet=True), inputs=server_url, outputs=history)

    return interface


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    create_interface().queue().launch(server_name=args.server_name, server_port=args.server_port,
                                      share=args.share, allowed_paths=[str(AUDIO_CACHE)])
