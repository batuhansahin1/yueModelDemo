"""Reference song -> YuE2 ABC score, for prompt-driven covers.

YuE2 cannot take audio as input; it can follow an external ABC score. This
module transcribes a reference recording into the same ABC layout the model
writes itself (L:1/32, chord symbols on the vocal line, `V: Vocal` / `V: Ins`
blocks, `% section` comments), so the new song keeps the reference melody,
harmony and structure while the style comes from the prompt.

Pipeline: Demucs vocal separation -> beat/tempo -> key -> per-bar chords from
the accompaniment -> pYIN vocal melody quantized to 1/16 notes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gc
import math
from pathlib import Path
import re

import numpy as np

SR = 22050
HOP = 256
UNITS_PER_BAR = 32          # 4/4 with L:1/32
CELL = 2                    # melody grid: 1/16 note = 2 units
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Kessler key profiles.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Signed accidental count of each major key's signature (sharps > 0, flats < 0).
MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 5: -1, 10: -2, 3: -3, 8: -4, 1: -5}
SHARP_ORDER, FLAT_ORDER = "FCGDAEB", "BEADGCF"
LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass
class Analysis:
    abc: str
    tempo: float
    key: str
    bars: int
    seconds: float
    vocal_ratio: float
    chords: list[str]
    sections: list[tuple[str, int, int]]
    melody_source: str = "vocals"
    lyrics: str | None = None       # recognized lyrics, `[section]` blocks matching the score
    lines: list = field(default_factory=list)
    lyrics_status: str = "off"      # off | ok | none (no intelligible words) | instrumental
    language: str | None = None     # Whisper language code of the recognized lyrics
    asr_model: str | None = None    # Whisper model folder that wrote the lyrics
    offset: float = 0.0             # analyzed clip start in the reference, for display times
    bar_seconds: float = 2.0
    grid_start: float = 0.0
    bar_times: list | None = None   # start of each bar and end of the last (SheetSage2: tempo may vary)
    transcriber: str = "basic"      # basic (this module) | sheetsage2

    def summary(self):
        def at(bar):
            if self.bar_times:
                return round(self.offset + self.bar_times[min(bar, len(self.bar_times) - 1)], 2)
            return round(self.offset + self.grid_start + bar * self.bar_seconds, 2)
        return {"tempo": round(self.tempo, 1), "key": self.key, "bars": self.bars, "transcriber": self.transcriber,
                "melody_source": self.melody_source, "lyrics_status": self.lyrics_status,
                "language": self.language, "asr_model": self.asr_model,
                "seconds": round(self.seconds, 1), "vocal_ratio": round(self.vocal_ratio, 2),
                "sections": [{"name": n, "start_bar": a, "end_bar": b, "start": at(a), "end": at(b)}
                             for n, a, b in self.sections],
                "lines": [{"section": l.section, "block": l.block, "text": l.text,
                           "start": round(self.offset + l.start, 2), "end": round(self.offset + l.end, 2)}
                          for l in self.lines]}


def load_audio(path, start=0.0, duration=None):
    """Stereo float32 [2, n] at the file's sample rate (MP3/WAV/FLAC/OGG via libsndfile)."""
    import soundfile as sf
    info = sf.info(str(path))
    first = int(max(0.0, start) * info.samplerate)
    frames = int(duration * info.samplerate) if duration else -1
    audio, rate = sf.read(str(path), start=first, frames=frames, dtype="float32", always_2d=True)
    if len(audio) < rate:
        raise ValueError("Seçilen aralıkta en az 1 saniyelik ses olmalı")
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return audio[:2], rate


def separate(audio, rate, demucs_dir, device="cpu"):
    """Return (vocals, other, accompaniment) mono at SR using htdemucs.

    `other` is the non-drum, non-bass instrument stem; it carries the lead
    melody of instrumental recordings (saz, guitar, piano...)."""
    import julius
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    from fractions import Fraction
    from demucs.htdemucs import HTDemucs
    from numpy._core.multiarray import scalar
    # Allow exactly the globals the official htdemucs checkpoint pickles, keeping weights_only=True.
    with torch.serialization.safe_globals([HTDemucs, Fraction, np.dtype, (scalar, "numpy.core.multiarray.scalar"),
                                           *(type(np.dtype(t)) for t in ("float64", "float32", "int64"))]):
        model = get_model("htdemucs", repo=Path(demucs_dir)).eval()
    wav = torch.from_numpy(audio)
    if rate != model.samplerate:
        wav = julius.resample_frac(wav, rate, model.samplerate)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.inference_mode():
        sources = apply_model(model, ((wav - mean) / std)[None], device=device, split=True,
                              overlap=0.25, progress=False)[0] * std + mean
    vocal_index = model.sources.index("vocals")
    vocals = sources[vocal_index].mean(0)
    other = sources[model.sources.index("other")].mean(0)
    accompaniment = sum(sources[i] for i in range(len(model.sources)) if i != vocal_index).mean(0)
    to_sr = lambda x: julius.resample_frac(x, model.samplerate, SR).numpy().astype(np.float32)
    return to_sr(vocals), to_sr(other), to_sr(accompaniment)


def estimate_key(chroma):
    profile = chroma.sum(axis=1)
    best = (-np.inf, 0, "major")
    for tonic in range(12):
        for mode, template in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            score = np.corrcoef(profile, np.roll(template, tonic))[0, 1]
            if score > best[0]:
                best = (score, tonic, mode)
    return best[1], best[2]


def key_signature(tonic, mode):
    """ABC key name and the letter -> alteration map of its signature."""
    major = tonic if mode == "major" else (tonic + 3) % 12
    fifths = MAJOR_FIFTHS[major]
    names = FLAT_NAMES if fifths < 0 else SHARP_NAMES
    name = names[tonic] + ("m" if mode == "minor" else "")
    alter = {letter: 0 for letter in "CDEFGAB"}
    for letter in (SHARP_ORDER[:fifths] if fifths > 0 else FLAT_ORDER[:-fifths]):
        alter[letter] = 1 if fifths > 0 else -1
    return name, alter, fifths < 0


def chord_templates():
    templates = []
    for root in range(12):
        for suffix, intervals in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            t = np.zeros(12)
            t[[(root + i) % 12 for i in intervals]] = [1.0, 0.8, 0.9]
            templates.append((root, suffix, t / np.linalg.norm(t)))
    return templates


def chord_name(root, suffix, flats):
    return (FLAT_NAMES if flats else SHARP_NAMES)[root] + suffix


def spell(midi, key_alter, flats):
    """(letter, octave, alteration) preferring the key's own spelling."""
    pc, octave = midi % 12, midi // 12 - 1
    for letter, base in LETTER_PC.items():
        if (base + key_alter[letter]) % 12 == pc:
            # Cb/B# cross the octave boundary.
            return letter, octave + (1 if base + key_alter[letter] < 0 else -1 if base + key_alter[letter] > 11 else 0), key_alter[letter]
    for letter, base in LETTER_PC.items():
        if (base + (-1 if flats else 1)) % 12 == pc:
            return letter, octave, -1 if flats else 1
    raise AssertionError(pc)


def abc_pitch(letter, octave):
    if octave >= 5:
        return letter.lower() + "'" * (octave - 5)
    return letter + "," * (4 - octave)


def abc_length(units):
    return "" if units == 1 else str(units)


def transcribe_melody(vocals, start, unit_seconds, total_units, instrument=False):
    """Melody as a list of (midi or None) per CELL-sized grid step.

    Plucked instruments give pYIN lower voicing confidence than singing, so the
    thresholds are looser for an instrument stem."""
    min_prob, min_coverage = (0.05, 0.2) if instrument else (0.1, 0.3)
    import librosa
    rms = librosa.feature.rms(y=vocals, frame_length=2048, hop_length=HOP)[0]
    f0, voiced, prob = librosa.pyin(vocals, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
                                    sr=SR, frame_length=2048, hop_length=HOP)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=SR, hop_length=HOP)
    gate = rms[:len(f0)] > max(1e-4, np.percentile(rms, 95) * 0.08)
    ok = voiced & (prob > min_prob) & gate & np.isfinite(f0)
    midi = np.full(len(f0), np.nan)
    midi[ok] = librosa.hz_to_midi(f0[ok])
    # Onsets let repeated pitches (new syllables) start new notes.
    onset_frames = librosa.onset.onset_detect(y=vocals, sr=SR, hop_length=HOP, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=SR, hop_length=HOP)
    cells, onsets = [], []
    cell_seconds = CELL * unit_seconds
    for i in range(total_units // CELL):
        a, b = start + i * cell_seconds, start + (i + 1) * cell_seconds
        mask = (times >= a) & (times < b)
        values = midi[mask]
        values = values[np.isfinite(values)]
        if mask.sum() == 0 or len(values) < min_coverage * mask.sum():
            cells.append(None)
        else:
            cells.append(int(round(float(np.median(values)))))
        onsets.append(bool(np.any((onset_times >= a - cell_seconds / 4) & (onset_times < a + cell_seconds / 2))))
    cells = clean_cells(cells)
    if instrument:
        cells = to_vocal_range(cells)
    return cells, onsets


def to_vocal_range(cells, center=65):
    """Shift an instrumental melody by whole octaves so its median sits near F4."""
    voiced = [c for c in cells if c is not None]
    if not voiced:
        return cells
    shift = 12 * int(round((center - float(np.median(voiced))) / 12))
    return [None if c is None else c + shift for c in cells]


def clean_cells(cells, window=16):
    cells = list(cells)
    # Octave errors: pull notes far from the local melodic center back by whole octaves.
    voiced = [(i, c) for i, c in enumerate(cells) if c is not None]
    for i, c in voiced:
        near = [v for j, v in voiced if abs(j - i) <= window]
        center = float(np.median(near))
        if abs(c - center) > 7:
            cells[i] = c + 12 * int(round((center - c) / 12))
    for i in range(1, len(cells) - 1):
        prev, cur, nxt = cells[i - 1], cells[i], cells[i + 1]
        if cur is None or cur == prev or cur == nxt:
            continue
        # Single-cell blips between rests.
        if prev is None and nxt is None:
            cells[i] = None
        # Single-cell glide steps (semitone neighbours) merge into the preceding note.
        elif prev is not None and abs(cur - prev) == 1:
            cells[i] = prev
        elif nxt is not None and abs(cur - nxt) == 1:
            cells[i] = nxt
    return cells


def find_sections(bar_has_vocal, labels):
    """Split bars into intro / vocal sections / instrumental gaps / outro."""
    bars = len(bar_has_vocal)
    sections, i, label_index = [], 0, 0
    first = next((b for b in range(bars) if bar_has_vocal[b]), bars)
    last = max((b for b in range(bars) if bar_has_vocal[b]), default=-1)
    if last < 0:
        return [("inst", 0, bars)]
    if first > 0:
        sections.append(("intro", 0, min(first, bars)))
        i = first
    while i <= last:
        j = i
        gap = 0
        while j <= last:
            gap = gap + 1 if not bar_has_vocal[j] else 0
            if gap >= 2:
                break
            j += 1
        end = j - gap + 1 if gap >= 2 else last + 1
        name = labels[label_index] if label_index < len(labels) else (labels[-1] if labels else "verse")
        label_index += 1
        sections.append((name, i, end))
        k = end
        while k <= last and not bar_has_vocal[k]:
            k += 1
        if k > end:
            sections.append(("inst", end, k))
        i = k
    if last + 1 < bars:
        sections.append(("outro", last + 1, bars))
    return sections


def render_abc(cells, onsets, bar_chords, sections, tempo, key_name, key_alter, flats):
    cells_per_bar = UNITS_PER_BAR // CELL
    lines = ["X:1", "T:", "M:4/4", "L:1/32", f"Q:1/4={int(round(tempo))}",
             'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
             'V: Ins clef=treble name="Ins Melody" snm="Inst."', f"K:{key_name}"]

    def bar_text(bar):
        state = {}
        out = f'"{bar_chords[bar]}"' if bar_chords[bar] else ""
        cs = cells[bar * cells_per_bar:(bar + 1) * cells_per_bar]
        nxt = cells[(bar + 1) * cells_per_bar] if (bar + 1) * cells_per_bar < len(cells) else None
        nxt_onset = onsets[(bar + 1) * cells_per_bar] if (bar + 1) * cells_per_bar < len(onsets) else True
        i = 0
        while i < len(cs):
            j = i + 1
            while j < len(cs) and cs[j] == cs[i] and (cs[i] is None or not onsets[bar * cells_per_bar + j]):
                j += 1
            units = (j - i) * CELL
            if cs[i] is None:
                out += "z" + abc_length(units)
            else:
                letter, octave, alter = spell(cs[i], key_alter, flats)
                current = state.get((letter, octave), key_alter[letter])
                acc = ""
                if alter != current:
                    acc = {-2: "__", -1: "_", 0: "=", 1: "^", 2: "^^"}[alter]
                    state[(letter, octave)] = alter
                out += acc + abc_pitch(letter, octave) + abc_length(units)
                if j == len(cs) and nxt == cs[i] and not nxt_onset:
                    out += "-"
            i = j
        return out + "|"

    for name, a, b in sections:
        lines.append(f"% {name}")
        for block in range(a, b, 4):
            end = min(block + 4, b)
            lines.append("V: Vocal")
            lines.append("".join(bar_text(bar) for bar in range(block, end)))
            lines.append("V: Ins")
            count = end - block
            lines.append(("Z" if count == 1 else f"Z{count}") + "|")
    return "\n".join(lines) + "\n"


# Score sections that are played, not sung: this module's and SheetSage2's instrumental labels.
NON_VOCAL = {"intro", "outro", "inst", "instrumental", "end", "silence", "interlude", "fade-out", "solo"}


def lyric_labels(lyrics):
    labels = [m.strip().lower() for m in re.findall(r"\[([^\]]+)\]", lyrics or "")]
    return [l for l in labels if l not in NON_VOCAL]


def analyze(path, *, start=0.0, duration=None, lyrics="", bpm=None, demucs_dir, device="cpu", workdir=None,
            whisper_dir=None, language=None, asr_device=None, whisper_models=None, sheetsage=None):
    """`whisper_dir` turns on lyric recognition: sections then follow the sung words.

    `whisper_models` maps a language to its own Whisper model (e.g. {"tr": ...}), used
    when that language is chosen or detected. `sheetsage` ({"python", "model_dir"})
    takes the score from SheetSage2 instead of this module's transcription."""
    import librosa
    import soundfile as sf

    audio, rate = load_audio(path, start, duration)
    if sheetsage:
        return analyze_sheetsage(audio, rate, start=start, demucs_dir=demucs_dir, device=device, workdir=workdir,
                                 whisper_dir=whisper_dir, language=language, asr_device=asr_device,
                                 whisper_models=whisper_models, **sheetsage)
    vocals, other, accompaniment = separate(audio, rate, demucs_dir, device)
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()   # Demucs is done: leave the GPU to Whisper and YuE2
    mix = vocals + accompaniment
    # Instrumental recordings: Demucs leaves the vocal stem (near) silent, so take the
    # lead melody from the instrument stem instead and let the new vocals sing it.
    level = lambda x: float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    melody_source = "vocals" if level(vocals) > 0.05 * level(mix) else "instrument"
    melody_audio = vocals if melody_source == "vocals" else other
    seconds = len(mix) / SR

    tempo, beats = librosa.beat.beat_track(y=mix, sr=SR, hop_length=HOP)
    tempo = float(bpm) if bpm else float(np.atleast_1d(tempo)[0])
    if not math.isfinite(tempo) or tempo <= 0:
        tempo = 100.0
    while tempo < 70:
        tempo *= 2
    while tempo > 180:
        tempo /= 2
    beat_seconds = 60.0 / tempo
    beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    first_beat = float(beat_times[0]) if len(beat_times) else 0.0
    grid_start = first_beat - math.floor(first_beat / beat_seconds) * beat_seconds
    unit_seconds = beat_seconds / 8          # 1/32 note in 4/4
    bars = max(1, int(math.ceil((seconds - grid_start) / (beat_seconds * 4))))
    total_units = bars * UNITS_PER_BAR

    chroma_acc = librosa.feature.chroma_cqt(y=accompaniment, sr=SR, hop_length=HOP)
    chroma_mix = librosa.feature.chroma_cqt(y=mix, sr=SR, hop_length=HOP)
    tonic, mode = estimate_key(chroma_mix)
    key_name, key_alter, flats = key_signature(tonic, mode)

    times = librosa.frames_to_time(np.arange(chroma_acc.shape[1]), sr=SR, hop_length=HOP)
    rms = librosa.feature.rms(y=accompaniment, hop_length=HOP)[0]
    templates = chord_templates()
    bar_chords = []
    for bar in range(bars):
        a = grid_start + bar * 4 * beat_seconds
        mask = (times >= a) & (times < a + 4 * beat_seconds)
        if mask.sum() == 0 or rms[:len(mask)][mask].mean() < 1e-3:
            bar_chords.append(bar_chords[-1] if bar_chords else "")
            continue
        v = chroma_acc[:, mask].mean(axis=1)
        v = v / (np.linalg.norm(v) + 1e-8)
        root, suffix, _ = max(templates, key=lambda t: float(v @ t[2]))
        bar_chords.append(chord_name(root, suffix, flats))

    cells, onsets = transcribe_melody(melody_audio, grid_start, unit_seconds, total_units,
                                      instrument=melody_source == "instrument")
    cells_per_bar = UNITS_PER_BAR // CELL
    lines, recognized, status, asr_model = [], None, "off", None
    if whisper_dir and melody_source != "vocals":
        status = "instrumental"
    elif whisper_dir:
        import lyrics as lyric_tools
        words, language, asr_model = lyric_tools.transcribe(vocals, SR, whisper_dir, asr_device, language,
                                                            models=whisper_models)
        lines = lyric_tools.group_lines(lyric_tools.drop_silent(words, vocals, SR, HOP), beat_seconds,
                                        language=language)
        status = "ok" if lines else "none"
    if lines:
        sections = lyric_tools.sections_from_lines(lines, grid_start, 4 * beat_seconds, bars)
        recognized = lyric_tools.lyrics_text(lines)
    else:
        bar_has_vocal = [sum(c is not None for c in cells[b * cells_per_bar:(b + 1) * cells_per_bar]) >= 2
                         for b in range(bars)]
        sections = find_sections(bar_has_vocal, lyric_labels(lyrics))
    abc = render_abc(cells, onsets, bar_chords, sections, tempo, key_name, key_alter, flats)

    if workdir is not None:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        sf.write(workdir / "melody.wav", melody_audio, SR)
        (workdir / "score.abc").write_text(abc, encoding="utf-8")
        if recognized:
            (workdir / "lyrics.txt").write_text(recognized, encoding="utf-8")
    voiced = sum(c is not None for c in cells) / max(1, len(cells))
    ok = status == "ok"
    return Analysis(abc, tempo, key_name, bars, seconds, voiced, bar_chords, sections, melody_source,
                    lyrics=recognized, lines=lines, lyrics_status=status, language=language if ok else None,
                    asr_model=asr_model if ok else None, offset=float(start or 0),
                    bar_seconds=4 * beat_seconds, grid_start=grid_start)


def analyze_sheetsage(audio, rate, *, start, demucs_dir, device, workdir, whisper_dir, language, asr_device,
                      whisper_models, python, model_dir):
    """Score from SheetSage2; with `whisper_dir`, Demucs + Whisper place the sung words in its sections.

    SheetSage2 transcribes the mix itself (melody, chords, beats, key, sections and the
    instrumental melody line), so Demucs only runs for lyric recognition."""
    import sheetsage

    workdir = Path(workdir)
    abc, timeline = sheetsage.transcribe(audio, rate, python=python, model_dir=model_dir,
                                         workdir=workdir / "sheetsage2", device=device)
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    seconds = audio.shape[1] / rate
    tempo = float((sheetsage.score_header(abc, "Q") or "1/4=100").split("=")[-1])
    key = sheetsage.score_header(abc, "K") or "?"
    measures = timeline["measures"]
    sections = timeline["sections"]
    sung_seconds = sheetsage.vocal_seconds(workdir / "sheetsage2")
    melody_source = "vocals" if sung_seconds > 0 else "none"
    lines, recognized, status, language_used, asr_model = [], None, "off", None, None
    if whisper_dir:
        import lyrics as lyric_tools
        vocals, _, _ = separate(audio, rate, demucs_dir, device)
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
        words, language_used, asr_model = lyric_tools.transcribe(vocals, SR, whisper_dir, asr_device, language,
                                                                 models=whisper_models)
        lines = lyric_tools.group_lines(lyric_tools.drop_silent(words, vocals, SR, HOP), 60.0 / tempo,
                                        language=language_used)
        status = "ok" if lines else ("none" if sung_seconds > 0 else "instrumental")
        if lines and sections:
            names = lyric_tools.place_in_sections(lines, sections, NON_VOCAL)
            abc = sheetsage.relabel_sections(abc, names)
            sections = [{**s, "name": n} for s, n in zip(sections, names)]
            recognized = lyric_tools.lyrics_text(lines)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "score.abc").write_text(abc, encoding="utf-8")
    if recognized:
        (workdir / "lyrics.txt").write_text(recognized, encoding="utf-8")
    ok = status == "ok"
    bar_times = [m[0] for m in measures] + [measures[-1][1]] if measures else None
    return Analysis(abc, tempo, key, len(measures), seconds, min(1.0, sung_seconds / max(seconds, 1e-6)), [],
                    [(s["name"], s["start_bar"], s["end_bar"]) for s in sections], melody_source,
                    lyrics=recognized, lines=lines, lyrics_status=status, language=language_used if ok else None,
                    asr_model=asr_model if ok else None, offset=float(start or 0), bar_times=bar_times,
                    transcriber="sheetsage2")
