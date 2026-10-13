"""Reference vocals -> timed lyric lines and lyric-driven song sections.

YuE2 has no timing input for lyrics: it sings each `[section]` block of the
lyrics over the matching `% section` of the ABC score. So the structure is kept
by building both from one timeline: Whisper word timestamps -> lines -> bars ->
sections, rendered as the score's `% name` order and the lyrics' `[name]` blocks.

Sections come from the words: a run of lines that is sung again elsewhere is a
chorus, other runs are verses, and bars without lyrics become intro/inst/outro.
"""
from __future__ import annotations

from dataclasses import dataclass
import difflib
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

WHISPER_SR = 16000
NON_VOCAL = {"intro", "inst", "outro"}


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Line:
    text: str
    start: float
    end: float
    section: str = ""
    block: int = 0      # index of the vocal section this line is sung in


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def vocal_activity(vocals, sr, hop=256):
    """Per-frame audible flags of the vocal stem (same gate as the melody transcription)."""
    import librosa
    rms = librosa.feature.rms(y=vocals, frame_length=2048, hop_length=hop)[0]
    return rms > max(1e-4, float(np.percentile(rms, 95)) * 0.08)


def vocal_regions(active, sr, hop=256, min_gap=1.0, min_len=0.3, pad=0.25, max_len=28.0):
    """(start, end) seconds of sung passages, each short enough for one Whisper window.

    Whisper writes made-up text over silence and drops words cut at its 30 s
    window edge, so it only ever sees one sung passage at a time."""
    frame = hop / sr
    regions, start, last = [], None, None
    for i in np.flatnonzero(active):
        t = i * frame
        if start is None:
            start = t
        elif t - last > min_gap:
            regions.append((start, last + frame))
            start = t
        last = t
    if start is not None:
        regions.append((start, last + frame))
    total = len(active) * frame
    out = []
    for a, b in regions:
        if b - a < min_len:
            continue
        a, b = max(0.0, a - pad), min(total, b + pad)
        # Split long passages at their quietest frame in the last third of the window.
        while b - a > max_len:
            lo, hi = int((a + max_len * 2 / 3) / frame), int((a + max_len) / frame)
            cut = (lo + int(np.argmin(active[lo:hi]))) * frame if hi > lo else a + max_len
            out.append((a, cut))
            a = cut
        out.append((a, b))
    return out


# Whisper's stock inventions on music/silence: subtitle credits, outros, sound tags.
HALLUCINATIONS = re.compile(r"altyaz|subtit|редактор|субтитр|корректор|thanks for watching|abone ol"
                            r"|izlediğiniz için|^\W*(müzik|music|музыка)\W*$", re.IGNORECASE)


def hallucinated(text):
    """True for a passage that is a known Whisper invention or a repetition loop."""
    tokens = normalize(text).split()
    if not tokens or HALLUCINATIONS.search(text):
        return True
    run = 1
    for a, b in zip(tokens, tokens[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    # A long passage made of a handful of words is a loop, not lyrics.
    return len(tokens) >= 12 and len(set(tokens)) / len(tokens) < 0.25


def detect_language(asr, audio, region):
    """Whisper language code (e.g. "tr") of the given passage."""
    import torch
    a, b = region
    clip = audio[int(a * WHISPER_SR):int(b * WHISPER_SR)]
    features = asr.feature_extractor(clip, sampling_rate=WHISPER_SR, return_tensors="pt").input_features
    with torch.inference_mode():
        token = asr.model.detect_language(features.to(asr.model.device, asr.model.dtype))
    return asr.tokenizer.decode(token[0]).strip("<|>")


def load_asr(model_dir, device):
    import torch
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=str(model_dir), device=device,
                    dtype=torch.float16 if device == "cuda" else torch.float32)


def release(device):
    """Hand the memory of a dropped pipeline back to the system."""
    import gc
    import torch
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def transcribe(vocals, sr, model_dir, device=None, language=None, hop=256, models=None):
    """(words with start/end seconds, language code, model folder name) from the separated vocal stem.

    `models` maps a language to a model fine-tuned for it (e.g. {"tr": ...}); the
    multilingual `model_dir` detects the language and transcribes all others.
    Only one model is in memory at a time."""
    import librosa

    models = models or {}
    regions = vocal_regions(vocal_activity(vocals, sr, hop), sr, hop)
    if not regions:
        return [], language, None
    device = device or pick_device()
    audio = librosa.resample(vocals, orig_sr=sr, target_sr=WHISPER_SR).astype(np.float32)
    asr, used, words = None, model_dir, []
    try:
        if language is None:
            asr = load_asr(model_dir, device)
            # Detected per passage, a sung Turkish line can come back as Azerbaijani; detect once, use everywhere.
            language = detect_language(asr, audio, max(regions, key=lambda r: r[1] - r[0]))
            if language in models:
                asr = None
                release(device)
        if asr is None:
            used = models.get(language, model_dir)
            asr = load_asr(used, device)
        # A 28 s passage holds far fewer tokens than this; the cap only cuts repetition loops short.
        generate_kwargs = {"task": "transcribe", "max_new_tokens": 160, "language": language}
        for a, b in regions:
            clip = audio[int(a * WHISPER_SR):int(b * WHISPER_SR)]
            out = asr({"raw": clip, "sampling_rate": WHISPER_SR}, return_timestamps="word",
                      generate_kwargs=generate_kwargs)
            if hallucinated(out.get("text", "")):
                continue
            for chunk in out.get("chunks", []):
                text = chunk["text"].strip()
                start, end = chunk["timestamp"]
                if not text or start is None:
                    continue
                end = end if end is not None else start + 0.3
                words.append(Word(text, a + float(start), a + min(float(end), b - a)))
    finally:
        asr = None
        release(device)
    return words, language, Path(used).name


def drop_silent(words, vocals, sr, hop=256, min_coverage=0.3):
    """Whisper invents text over silence; keep only words mostly sung where the vocal stem is audible."""
    active = vocal_activity(vocals, sr, hop)
    kept = []
    for w in words:
        a = int(w.start * sr / hop)
        b = max(a + 1, int(math.ceil(w.end * sr / hop)))
        if active[a:b].size and float(active[a:b].mean()) >= min_coverage:
            kept.append(w)
    return kept


def capitalize(text, language=None):
    """Upper-case first letter (Turkish dotted/dotless i for Turkic lyrics); the rest is kept as sung."""
    if not text:
        return text
    turkic = {"i": "İ", "ı": "I"} if language in ("tr", "az") else {}
    return turkic.get(text[0], text[0].upper()) + text[1:]


def group_lines(words, beat_seconds, max_bars=4, language=None):
    """Break at pauses longer than a beat, after sentence punctuation, or past `max_bars`.

    Some models (e.g. the Turkish fine-tune) write no punctuation; pauses and length then decide."""
    lines, current = [], []
    limit = max_bars * 4 * beat_seconds
    for w in words:
        if current:
            gap = w.start - current[-1].end
            punct = current[-1].text[-1] in ".!?;:,"
            if gap > beat_seconds or (punct and gap > beat_seconds / 2) or w.end - current[0].start > limit:
                lines.append(current)
                current = []
        current.append(w)
    if current:
        lines.append(current)
    return [Line(capitalize(" ".join(w.text for w in ws).strip(" ,;:"), language), ws[0].start, ws[-1].end)
            for ws in lines]


def normalize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", text).strip()


def similar(a, b, threshold=0.75):
    return bool(a) and bool(b) and difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def repeated_lines(lines):
    """True for lines in a run of 2+ consecutive lines that is sung again elsewhere.

    Requiring pairs keeps a one-line refrain inside a verse (e.g. every second
    line) from being mistaken for a chorus."""
    norm = [normalize(l.text) for l in lines]
    flags = [False] * len(lines)
    for i in range(len(lines) - 1):
        for j in range(len(lines) - 1):
            if abs(i - j) >= 2 and similar(norm[i], norm[j]) and similar(norm[i + 1], norm[j + 1]):
                flags[i] = flags[i + 1] = True
    return flags


def sections_from_lines(lines, grid_start, bar_seconds, bars, gap_bars=2):
    """Label lines in place and return (name, start_bar, end_bar) covering all bars."""
    if not lines:
        return [("inst", 0, bars)]
    bar_of = lambda t: min(bars - 1, max(0, int(math.floor((t - grid_start) / bar_seconds))))
    flags = repeated_lines(lines)
    # A short exclamation ("vay", "ah") stays in the block it is sung in instead of opening a section.
    minor = [len(normalize(l.text).split()) <= 2 for l in lines]
    # Blocks of consecutive lines: split on long pauses and where chorus-ness changes.
    blocks, chorus = [], []
    for i, line in enumerate(lines):
        pause = i > 0 and line.start - lines[i - 1].end >= gap_bars * bar_seconds
        if not blocks or pause or (not minor[i] and chorus[-1] is not None and flags[i] != chorus[-1]):
            blocks.append([])
            chorus.append(None)
        blocks[-1].append(i)
        if chorus[-1] is None and not minor[i]:
            chorus[-1] = flags[i]
    spans = []
    for block, is_chorus in zip(blocks, chorus):
        name = "chorus" if is_chorus else "verse"
        spans.append([name, bar_of(lines[block[0]].start), bar_of(lines[block[-1]].end) + 1, block])
    # Adjacent blocks may share a bar; the later block starts on the next free bar.
    for prev, cur in zip(spans, spans[1:]):
        if cur[1] < prev[2]:
            if cur[1] > prev[1]:
                prev[2] = cur[1]
            else:
                cur[1] = prev[2]
        cur[2] = max(cur[2], cur[1] + 1)
    # A block pushed past the last bar is sung in the previous one.
    kept = []
    for span in spans:
        span[2] = min(span[2], bars)
        if span[1] < span[2] or not kept:
            kept.append(span)
        else:
            kept[-1][3] = kept[-1][3] + span[3]
    for k, (name, _, _, block) in enumerate(kept):
        for i in block:
            lines[i].section, lines[i].block = name, k
    sections = []
    if kept[0][1] > 0:
        sections.append(("intro", 0, kept[0][1]))
    for k, (name, a, b, _) in enumerate(kept):
        nxt = kept[k + 1][1] if k + 1 < len(kept) else None
        # Short breaths stay in the section (as the model writes them); longer breaks are interludes.
        if nxt is not None and nxt - b < gap_bars:
            b = nxt
        sections.append((name, a, b))
        if nxt is not None and nxt > b:
            sections.append(("inst", b, nxt))
    if sections[-1][2] < bars:
        sections.append(("outro", sections[-1][2], bars))
    return sections


def place_in_sections(lines, sections, non_vocal):
    """Put recognized lines into a transcribed score's sections (SheetSage2); returns the section names to use.

    The score's sections are kept; only their sung/instrumental role follows the words,
    because YuE2 sings the n-th lyric block over the n-th sung section:
    a sung-named section without words becomes "instrumental", and an instrumental-named
    one with real words (more than an exclamation) becomes "verse". Lines are labeled in place."""
    def containing(t):
        spans = [k for k, s in enumerate(sections) if s["end"] > s["start"]]
        inside = [k for k in spans if sections[k]["start"] <= t < sections[k]["end"]]
        return inside[0] if inside else min(spans, key=lambda k: min(abs(t - sections[k]["start"]),
                                                                    abs(t - sections[k]["end"])))

    home = [containing((line.start + line.end) / 2) for line in lines]
    real = {k for k, line in zip(home, lines) if len(normalize(line.text).split()) > 2}
    names = []
    for k, section in enumerate(sections):
        name = section["name"]
        has_words = k in home
        if name in non_vocal and k in real:
            name = "verse"
        elif name not in non_vocal and not has_words:
            name = "instrumental"
        names.append(name)
    sung = [k for k, name in enumerate(names) if name not in non_vocal]
    for line, k in zip(lines, home):
        # An exclamation in an instrumental section is sung with the nearest sung one.
        if k not in sung and sung:
            k = min(sung, key=lambda j: abs(sections[j]["start"] - line.start))
        # block numbers the sung sections, as sections_from_lines does.
        line.section, line.block = names[k], sung.index(k) if k in sung else len(sung)
    return names


def lyrics_text(lines):
    """`[section]` blocks in the same order as the vocal sections of the score."""
    blocks, last = [], None
    for line in lines:
        if line.block != last:
            blocks.append([f"[{line.section}]"])
            last = line.block
        blocks[-1].append(line.text)
    return "\n\n".join("\n".join(b) for b in blocks) + "\n"
