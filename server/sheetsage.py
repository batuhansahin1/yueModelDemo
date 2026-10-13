"""SheetSage2: the transcriber YuE2's authors use to turn a recording into its score.

SheetSage2 pins torch 2.8 while yue2 pins 2.10, so it runs in its own environment
(sheetsage_ortam) through sheetsage_runner.py, one process per analysis. The
process exit hands its memory back before Demucs, Whisper or YuE2 need it.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import numpy as np

RUNNER = Path(__file__).with_name("sheetsage_runner.py")


def available(python, model_dir):
    return bool(python and model_dir and Path(python).exists() and (Path(model_dir) / "model.safetensors").exists())


def transcribe(audio, rate, *, python, model_dir, workdir, device):
    """(score.abc text, timeline) of a [channels, samples] waveform; times are seconds into it.

    The timeline holds the span of every measure and of every `% section` of the score, in order."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for stale in ("score.abc", "sections.json"):
        (workdir / stale).unlink(missing_ok=True)
    audio_path = workdir / "input.npy"
    np.save(audio_path, np.ascontiguousarray(audio, dtype=np.float32))
    try:
        # Progress lines go to the server log; stderr carries the reason of a failure.
        run = subprocess.run([str(python), str(RUNNER), str(model_dir), str(audio_path), str(int(rate)),
                              str(workdir), device], stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace")
    finally:
        audio_path.unlink(missing_ok=True)
    if run.returncode:
        raise RuntimeError("SheetSage2 başarısız: " + run.stderr.strip()[-1500:])
    abc = (workdir / "score.abc").read_text(encoding="utf-8")
    return abc, json.loads((workdir / "sections.json").read_text(encoding="utf-8"))


def score_header(abc, field):
    """Value of the first `field:` header line of an ABC score (e.g. "K" -> "Am")."""
    match = re.search(rf"^{field}:\s*(.+?)\s*$", abc, re.MULTILINE)
    return match.group(1) if match else None


def relabel_sections(abc, names):
    """Rename the score's `% section` lines in order; the count must match."""
    lines = abc.splitlines()
    indices = [i for i, line in enumerate(lines) if re.match(r"%(?!%)\s*\S", line)]
    if len(indices) != len(names):
        raise ValueError(f"Score has {len(indices)} sections, expected {len(names)}")
    for i, name in zip(indices, names):
        lines[i] = f"% {name}"
    return "\n".join(lines) + "\n"


def vocal_seconds(workdir):
    """Total duration of the vocal melody notes SheetSage2 found (melody_vocal.lab: start end pitch)."""
    path = Path(workdir) / "melody_vocal.lab"
    if not path.exists():
        return 0.0
    total = 0.0
    for row in path.read_text(encoding="utf-8").splitlines():
        parts = row.split()
        if len(parts) >= 2:
            total += max(0.0, float(parts[1]) - float(parts[0]))
    return total
