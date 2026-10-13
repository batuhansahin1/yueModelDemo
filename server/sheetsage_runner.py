"""Run SheetSage2 in its own Python environment; started by server/sheetsage.py.

    python sheetsage_runner.py MODEL_DIR AUDIO.npy SAMPLE_RATE OUTPUT_DIR DEVICE

SheetSage2 pins torch 2.8 while yue2 pins 2.10, so it cannot share the model
server's environment. The waveform arrives decoded ([channels, samples] float32),
so SheetSage2 needs no FFmpeg. Outputs (score.abc, *.lab, notation/...) are
written to OUTPUT_DIR, plus sections.json: the time span of every `% section`
of score.abc and of every measure, in score order. The exit status is nonzero
when no score was produced.
"""
import importlib
import json
from pathlib import Path
import sys

import numpy as np


def score_timeline(notation, output_dir):
    """Section and measure times of score.abc, rebuilt from the notation files it was made from."""
    files = Path(output_dir) / "notation"
    score = notation.build_rebuilt_abc_score(
        files / "song_melody.mid", files / "song_beats.txt", files / "song_chords.txt",
        files / "song_keys.txt", files / "song_structures.txt")
    if notation.score_to_abc(score) != (Path(output_dir) / "score.abc").read_text(encoding="utf-8"):
        raise RuntimeError("Rebuilt score differs from score.abc; section times would not match it")
    beat_time = lambda index: float(score.beats[min(index, len(score.beats) - 1)].time)
    measures = [[beat_time(m.start_beat), beat_time(m.end_beat)] for m in score.measures]
    sections, bar = [], 0
    for group in notation._measure_groups(score):
        # A group can open several sections when labels change within one measure; all but the last are empty.
        for label in group.structure_labels:
            if sections:
                sections[-1]["end_bar"] = bar
            sections.append({"name": label, "start_bar": bar})
        bar += len(group.measures)
    if sections:
        sections[-1]["end_bar"] = bar
    for section in sections:
        section["start"] = measures[section["start_bar"]][0] if section["start_bar"] < len(measures) else measures[-1][1]
        section["end"] = measures[section["end_bar"] - 1][1] if section["end_bar"] > section["start_bar"] else section["start"]
    return {"measures": measures, "sections": sections}


def main():
    model_dir, audio_path, rate, output_dir, device = sys.argv[1:6]
    model_dir = Path(model_dir).resolve()
    # Import the model class from its folder: transformers 4.45's remote-code loader
    # does not copy nested relative imports of a local checkpoint (chord_spelling_sheetsage2).
    sys.path.insert(0, str(model_dir.parent))
    modeling = importlib.import_module(f"{model_dir.name}.modeling_sheetsage2")
    model = modeling.SheetSage2Model.from_pretrained(str(model_dir)).eval().to(device)

    def progress(event):
        if event["stage"] == "encoding":
            print(f"SheetSage2 window {event['window']}/{event['windows']}", flush=True)

    result = model.transcribe(np.load(audio_path), sampling_rate=int(rate), output_dir=output_dir,
                              progress=progress)
    for warning in result.get("warnings") or []:
        print("SheetSage2 warning:", warning, flush=True)
    if result.get("abc_error") or not result.get("abc"):
        print("SheetSage2 produced no score:", result.get("abc_error"), file=sys.stderr, flush=True)
        sys.exit(2)
    notation = importlib.import_module(f"{model_dir.name}.notation_sheetsage2")
    timeline = score_timeline(notation, output_dir)
    (Path(output_dir) / "sections.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
