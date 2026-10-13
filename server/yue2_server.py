"""YuE2 model server for YuE-UI.

Wraps yue2.YuE2Pipeline behind a small HTTP API so the Gradio UI can run in its
own lightweight environment and connect to the model process by URL.

Real mode runs in the yue2 environment (../yue_ortam). Mock mode needs only
fastapi + uvicorn (already present in the UI venv) and synthesizes a test tone,
so the UI can be verified end to end before the model weights are downloaded.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import queue
import random
import statistics
import struct
import threading
import time
import traceback
import uuid
import wave
from collections import Counter
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

STAGES = ("queued", "loading", "abc", "semantic", "synthesize", "decode", "saving")
FINAL = {"complete", "failed", "cancelled"}
# Bump when cover/lyrics analysis output changes, so cached results are recomputed.
ANALYSIS_VERSION = 3
# Compute seconds per second of generated audio (normal, fast synthesis) measured before this server
# kept its own history; used until the device has finished generations of its own.
REFERENCE_RATES = {"Apple MPS": (6.5, 5.0)}   # Apple M4
# The UI's fast synthesis setting; time estimates are kept separately for it.
FAST_ODE_STEPS = 16
# GPU memory a reference analysis needs (Demucs + Whisper large in fp16, with headroom); below this
# much free memory a resident YuE2 (Colab --keep-loaded, ~7.3 GB) is unloaded first.
ANALYSIS_GPU_GIB = 8


class SamplingOverrides(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    penalty_window: int | None = Field(default=None, ge=1, le=100)   # yue2.protocol.Sampling's range
    min_tokens: int | None = None
    max_tokens: int | None = None

    def overrides(self):
        values = {k: v for k, v in self.model_dump().items() if v is not None}
        return values or None


class GenerateRequest(BaseModel):
    style: str = Field(min_length=1)
    lyrics: str = Field(min_length=1)
    cot: Literal["full", "melody", "off"] = "full"
    seed: int | None = None
    cfg_scale: float | None = None
    abc: str | None = None
    abc_sampling: SamplingOverrides | None = None
    semantic_sampling: SamplingOverrides | None = None
    reference: dict | None = None   # cover source info; stored with the job, not sent to the model
    prompt: dict | None = None      # Text to Song form (description, styles, voice); stored, not sent to the model
    ode_steps: int | None = Field(default=None, ge=4, le=64)  # acoustic synthesis steps (default 32)
    quantization: Literal["none", "fp8"] | None = None   # experimental FP8 token generation; None = server default


class AnalyzeRequest(BaseModel):
    upload_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    start: float = Field(default=0.0, ge=0)
    duration: float | None = Field(default=None, gt=0)
    bpm: float | None = Field(default=None, gt=30, lt=300)
    lyrics: str = ""
    recognize_lyrics: bool = False  # Whisper: lyrics + sections from the sung words
    language: str | None = Field(default=None, pattern=r"^[a-z]{2,3}$")  # None = auto-detect
    transcriber: Literal["basic", "sheetsage2"] = "basic"   # who writes the score: cover.py or SheetSage2


def write_mp3(audio, sample_rate, path):
    import soundfile as sf
    sf.write(path, audio, sample_rate, format="MP3", subtype="MPEG_LAYER_III",
             compression_level=0.0, bitrate_mode="CONSTANT")


class Job:
    def __init__(self, job_id, request: GenerateRequest, directory: Path):
        self.id = job_id
        self.request = request
        self.directory = directory
        self.status = "queued"
        self.stage = "queued"
        self.tokens = Counter()
        self.error = None
        self.result = None
        self.created = time.time()
        self.started = None
        self.finished = None
        self.cancel_requested = False

    def to_dict(self):
        return {"id": self.id, "status": self.status, "stage": self.stage,
                "tokens": dict(self.tokens), "error": self.error, "result": self.result,
                "request": self.request.model_dump(), "created": self.created,
                "started": self.started, "finished": self.finished,
                "elapsed": (self.finished or time.time()) - (self.started or time.time())}

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "job.json").write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                                                 encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        job = cls(data["id"], GenerateRequest(**data["request"]), path.parent)
        for key in ("status", "stage", "error", "result", "created", "started", "finished"):
            setattr(job, key, data.get(key))
        job.tokens.update(data.get("tokens") or {})
        return job


class MockBackend:
    """Simulates the YuE2 stages and writes a short tone so the UI flow can be tested."""
    device = "mock"
    loaded = True

    def unload(self):
        pass
    ABC = "X:1\nT:mock\nM:4/4\nL:1/8\nK:C\n\"C\" C2 E2 G2 c2 | \"G\" B2 G2 D2 G2 |]\n"

    @staticmethod
    def analysis(recognize):
        """A fixed 44 s song (120 BPM, 2 s bars) shaped like a real cover analysis."""
        bars = {"intro": 2, "verse": 4, "chorus": 4, "inst": 2, "outro": 2}
        melody = {"verse": '"C"E8E8G8E8|"G"D8D8z16|"Am"C8E8A8G8|"F"F16z16|',
                  "chorus": '"F"A8A8G8F8|"C"G16E16|"G"D8E8F8G8|"C"E16z16|'}
        texts = {0: ["Test kıtası birinci satır", "Test kıtası ikinci satır"],
                 1: ["Test nakaratı söylenir", "Test nakaratı tekrar eder"],
                 2: ["İkinci kıta birinci satır", "İkinci kıta ikinci satır"],
                 3: ["Test nakaratı söylenir", "Test nakaratı tekrar eder"]}
        abc = ["X:1", "T:", "M:4/4", "L:1/32", "Q:1/4=120", 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
               'V: Ins clef=treble name="Ins Melody" snm="Inst."', "K:C"]
        sections, lines, bar, block = [], [], 0, 0
        for name in ("intro", "verse", "chorus", "inst", "verse", "chorus", "outro"):
            count = bars[name]
            sections.append({"name": name, "start_bar": bar, "end_bar": bar + count,
                             "start": bar * 2.0, "end": (bar + count) * 2.0})
            abc += [f"% {name}", "V: Vocal", melody.get(name, '"C"z32|' * count), "V: Ins", f"Z{count}|"]
            if name in melody:
                lines += [{"section": name, "block": block, "text": text, "start": bar * 2.0 + 0.2 + 4 * i,
                           "end": bar * 2.0 + 3.6 + 4 * i} for i, text in enumerate(texts[block])]
                block += 1
            bar += count
        lyrics = "\n\n".join(f"[{lines[2 * b]['section']}]\n" + "\n".join(texts[b]) for b in texts) + "\n"
        summary = {"tempo": 120.0, "key": "C", "bars": bar, "seconds": bar * 2.0, "vocal_ratio": 0.6,
                   "melody_source": "vocals", "lyrics_status": "ok" if recognize else "off",
                   "language": "tr" if recognize else None, "asr_model": "mock" if recognize else None,
                   "sections": sections,
                   "lines": lines if recognize else []}
        return {"abc": "\n".join(abc) + "\n", "lyrics": lyrics if recognize else None, "summary": summary}

    def generate(self, job: Job):
        req = job.request
        start = time.perf_counter()
        for stage, steps in (("abc", 40 if req.cot != "off" and not req.abc else 0), ("semantic", 120)):
            job.stage = stage
            for _ in range(steps):
                if job.cancel_requested:
                    raise InterruptedError(f"Cancelled during {stage}")
                job.tokens[stage] += 1
                time.sleep(0.02)
        for stage in ("synthesize", "decode"):
            job.stage = stage
            time.sleep(0.5)
        job.stage = "saving"
        path = job.directory / "audio.wav"
        self._tone(path, req.seed)
        abc = req.abc if req.abc else (None if req.cot == "off" else self.ABC)
        return {"audio": path.name, "audio_seconds": 4.0, "sample_rate": 48000, "abc": abc,
                "truncated": {"abc": False, "semantic": False},
                "timing": {"e2e_seconds": time.perf_counter() - start}}

    @staticmethod
    def _tone(path, seed, seconds=4.0, rate=48000):
        base = 220 * 2 ** (random.Random(seed).randint(0, 12) / 12)
        frames = bytearray()
        for i in range(int(seconds * rate)):
            t = i / rate
            note = base * (1.5 if (t % 1.0) >= 0.5 else 1.0)
            env = min(1.0, t * 8, (seconds - t) * 8)
            frames += struct.pack("<h", int(0.25 * env * 32767 * math.sin(2 * math.pi * note * t)))
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(frames))


@contextlib.contextmanager
def uncapped_gpu_memory():
    """Keeps YuE2Pipeline from capping PyTorch 2 GiB below the GPU's own memory.

    On an 8 GB card that cap (~6 GiB) is below the token stage's peak even with --phase-offload. On Windows the
    NVIDIA driver puts what does not fit into shared GPU memory (system RAM): slower instead of out of memory."""
    import torch
    original = torch.cuda.set_per_process_memory_fraction
    torch.cuda.set_per_process_memory_fraction = lambda *args, **kwargs: None
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction = original


def phase_groups(model):
    """YuE2-3B's modules by the stage that uses them; together they hold every weight.

    ar: token generation and the acoustic prefill (embedding + the 28 layers' AR paths, 3.6 GB). head: token
    generation only (LM head, 0.76 GB). nar: the acoustic ODE only (the 28 layers' NAR paths, 2.8 GB).
    shared: small modules both stages use, always on the card."""
    backbone = model.model
    return {"ar": [backbone.embed_tokens] + [m for layer in backbone.layers for m in
                   (layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp)],
            "head": [model.lm_head],
            "nar": [m for layer in backbone.layers for m in
                    (layer.nar_input_layernorm, layer.nar_self_attn, layer.nar_pre_mlp_layernorm, layer.nar_mlp)],
            "shared": [backbone.norm, backbone.rotary_emb, model.llm2vae, model.vae2llm, model.time_embedder,
                       model.latent_pos_embed]}


def place(model, device, gpu=(), cpu=()):
    """Moves the `cpu` groups off the card before the `gpu` groups go on, so both are never there at once."""
    import torch
    groups = phase_groups(model)
    for name in cpu:
        for module in groups[name]:
            module.to("cpu")
    torch.cuda.empty_cache()
    for name in gpu:
        for module in groups[name]:
            module.to(device)


def phase_placement(pipe):
    """Keeps only the half of YuE2-3B a stage uses on an 8-12 GB card; the other half waits in RAM.

    yue2 moves all 7.15 GB of weights onto the GPU for every stage. Next to the KV cache that overflows an 8 GB card
    into shared GPU memory, which is then read over PCIe on every token. Here token generation holds the AR half and
    the LM head (4.4 GB), the acoustic ODE the NAR half (2.8 GB). The computation itself is unchanged."""
    import torch
    import yue2.nar as nar
    load_model, offload_ar, device = pipe._load_model, nar._offload_ar, pipe.device

    def staged_load(for_nar=False):
        if pipe.quantization != "none":   # FP8 needs compute capability 8.9+, above the cards this is for
            return load_model(for_nar)
        if pipe._model is None:
            pipe.device = torch.device("cpu")   # load into RAM; each stage then takes what it needs
            try:
                load_model(for_nar)
            finally:
                pipe.device = device
        # The acoustic prefill runs the AR layers but not the LM head.
        place(pipe._model, device, gpu=("ar", "shared") if for_nar else ("ar", "head", "shared"),
              cpu=("nar", "head") if for_nar else ("nar",))
        return pipe._model

    @contextlib.contextmanager
    def swap_halves(model, enabled):
        # yue2 takes the AR half off the card for the ODE; the NAR half comes on in its place.
        with offload_ar(model, True):
            place(model, device, gpu=("nar",))
            try:
                yield
            finally:
                place(model, device, cpu=("nar",))

    pipe._load_model = staged_load
    nar._offload_ar = swap_halves


def without_flash_attention():
    """GraphAR calls PyTorch's FlashAttention kernel directly whenever the op is registered, which it is even in a
    wheel built without the kernel (possible on Windows); the first token would then fail. Such builds decode
    through yue2's public-SDPA path instead: the same visible keys, somewhat slower."""
    import torch
    import yue2.cuda_graph as cuda_graph
    if torch.backends.cuda.is_flash_attention_available() or getattr(cuda_graph.GraphAR, "sdpa_fallback", False):
        return
    original = cuda_graph.GraphAR.__init__

    def init(self, *args, attention_backend="auto", **kwargs):
        original(self, *args, attention_backend="sdpa" if attention_backend == "auto" else attention_backend, **kwargs)

    cuda_graph.GraphAR.__init__ = init
    cuda_graph.GraphAR.sdpa_fallback = True
    print("Bu PyTorch FlashAttention olmadan derlenmiş; token üretimi PyTorch'un standart attention'ıyla çalışacak "
          "(biraz daha yavaş).", flush=True)


class YuE2Backend:
    """Runs YuE2Pipeline stage by stage so the UI can follow progress and cancel."""

    def __init__(self, args):
        self.args = args
        self.pipe = None
        self.device = None
        self.loaded = False
        self._lock = threading.Lock()

    def load(self):
        # --preload and the first job may both ask for the pipeline.
        with self._lock:
            if self.pipe is not None:
                return self.pipe
            from yue2.pipeline import YuE2Pipeline
            a = self.args
            with uncapped_gpu_memory() if a.shared_gpu_memory or a.phase_offload else contextlib.nullcontext():
                self.pipe = YuE2Pipeline.from_pretrained(
                    a.model, vae=a.vae, device=a.device, memory_budget_gib=a.budget,
                    backend=a.backend, quantization=a.quantization, offload_ar=a.offload_ar,
                    local_files_only=a.offline,
                    vae_core_frames=512 if a.budget <= 12 else 1024)
            if self.pipe.device.type == "cuda":
                without_flash_attention()
                if a.phase_offload:
                    phase_placement(self.pipe)
            self.device = str(self.pipe.device)
            return self.pipe

    def unload(self):
        """Free model weights (RAM/GPU) between jobs; file verification is kept, so reloading takes seconds."""
        import gc
        import torch
        with self._lock:
            if self.pipe is None:
                return
            self.pipe.close()
            self.loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def generate(self, job: Job):
        import dataclasses
        job.stage = "loading"
        pipe = self.load()
        self.loaded = True
        base, base_quantization = pipe.generation_config, pipe.quantization
        # Per-job overrides; effective_config() records both in the run identity.
        if job.request.ode_steps and job.request.ode_steps != base.ode_steps:
            pipe.generation_config = dataclasses.replace(base, ode_steps=job.request.ode_steps)
        if job.request.quantization:
            pipe.quantization = job.request.quantization
        try:
            return self._generate(job, pipe)
        finally:
            pipe.generation_config, pipe.quantization = base, base_quantization
            # A job stopped during token generation leaves FP8 layers in place; the next job starts from BF16.
            from yue2.quantization import restore_ar
            restore_ar(pipe._model)

    def _generate(self, job: Job, pipe):
        from yue2.pipeline import SongResult
        from yue2.protocol import SongRequest
        from yue2.storage import identity

        req = job.request
        abc_sampling = req.abc_sampling.overrides() if req.abc_sampling else None
        semantic_sampling = req.semantic_sampling.overrides() if req.semantic_sampling else None
        request = SongRequest(style=req.style, lyrics=req.lyrics, cot=req.cot, seed=req.seed,
                              abc=req.abc or None, cfg_scale=req.cfg_scale, id=job.id)
        config = pipe.effective_config(request, abc_sampling, semantic_sampling)
        request_id = identity({"request": request.to_dict(), "config": config, "weights": pipe.weights})
        cancelled = lambda: job.cancel_requested

        def on_token(phase, _token):
            job.tokens[phase] += 1

        start = time.perf_counter()
        job.stage = "abc"
        plan = pipe.plan(request=request, abc_sampling=abc_sampling, cancelled=cancelled, on_token=on_token)
        job.stage = "semantic"
        semantic = pipe.generate_semantic(plan, sampling=semantic_sampling, cancelled=cancelled, on_token=on_token)
        job.stage = "synthesize"
        nar_start = time.perf_counter()
        latents = pipe.synthesize(semantic, cancelled=cancelled)
        nar_seconds = time.perf_counter() - nar_start
        if job.cancel_requested:
            raise InterruptedError("Cancelled before VAE")
        job.stage = "decode"
        vae_start = time.perf_counter()
        audio = pipe.decode(latents)
        timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
                  "vae_seconds": time.perf_counter() - vae_start, "load": dict(pipe.load_timing),
                  "e2e_seconds": time.perf_counter() - start}
        job.stage = "saving"
        result = SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, request_id)
        result.save_artifacts(job.directory)
        write_mp3(audio, 48000, job.directory / "audio.mp3")
        return {"audio": "audio.mp3", "audio_seconds": len(audio) / 48000, "sample_rate": 48000,
                "abc": plan.abc, "truncated": result.truncated, "timing": {"e2e_seconds": timing["e2e_seconds"]}}


class Server:
    def __init__(self, backend, outputs: Path, unload_when_idle=True, device_label="unknown"):
        self.backend = backend
        self.unload_when_idle = unload_when_idle
        self.device_label = device_label   # e.g. "NVIDIA L4"; recorded with each job for time estimates
        self.outputs = outputs
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[Job] = queue.Queue()
        self.lock = threading.Lock()
        self.load_error = None
        # One GPU user at a time: a generation and a reference analysis together do not fit on a 24 GB card.
        self.gpu_lock = threading.Lock()
        self._restore()
        threading.Thread(target=self._worker, daemon=True).start()

    def _restore(self):
        for path in sorted(self.outputs.glob("*/job.json")):
            try:
                job = Job.load(path)
            except Exception:
                continue
            if job.status not in FINAL:
                job.status, job.error = "failed", "Server restarted during generation"
            self.jobs[job.id] = job

    def submit(self, request: GenerateRequest):
        if request.seed is None:
            request.seed = random.randrange(2**31)
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        job = Job(job_id, request, self.outputs / job_id)
        with self.lock:
            self.jobs[job_id] = job
        job.save()
        self.queue.put(job)
        return job

    def _worker(self):
        while True:
            job = self.queue.get()
            # A running analysis holds the GPU; the job stays queued until it is done.
            with self.gpu_lock:
                self._run(job)

    def _run(self, job):
        if job.cancel_requested:
            job.status, job.stage, job.finished = "cancelled", "cancelled", time.time()
            job.save()
            return
        job.status, job.started = "running", time.time()
        try:
            job.result = {**self.backend.generate(job), "device": self.device_label}
            job.status, job.stage = "complete", "done"
        except InterruptedError as exc:
            job.status, job.error = "cancelled", str(exc)
        except Exception as exc:
            traceback.print_exc()
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            if job.stage == "loading":
                self.load_error = job.error
        finally:
            job.finished = time.time()
            job.save()
            if self.unload_when_idle and self.queue.empty():
                self.backend.unload()

    @contextlib.contextmanager
    def gpu_for_analysis(self):
        """The GPU for a reference analysis; refused at once (409) while a song is being generated."""
        if not self.gpu_lock.acquire(blocking=False):
            generating = any(j.status == "running" for j in self.jobs.values())
            raise HTTPException(409, "Şu an şarkı üretiliyor; bitince tekrar deneyin." if generating
                                else "Başka bir analiz sürüyor; bitince tekrar deneyin.")
        try:
            yield
        finally:
            self.gpu_lock.release()

    def preload(self):
        try:
            self.backend.load()
        except Exception as exc:
            traceback.print_exc()
            self.load_error = f"{type(exc).__name__}: {exc}"

    def rates(self):
        """Median compute seconds per audio second of this device's finished jobs: (normal, fast, count)."""
        measured = {False: [], True: []}
        # Only the two standard synthesis settings: 32 steps (default) and 16 (fast).
        modes = {None: False, 32: False, FAST_ODE_STEPS: True}
        for job in list(self.jobs.values()):
            result = job.result or {}
            e2e = (result.get("timing") or {}).get("e2e_seconds")
            if (job.status == "complete" and result.get("device") == self.device_label and e2e
                    and result.get("audio_seconds") and job.request.ode_steps in modes):
                measured[modes[job.request.ode_steps]].append(e2e / result["audio_seconds"])
        normal, fast = (statistics.median(v) if v else None for v in (measured[False], measured[True]))
        reference = REFERENCE_RATES.get(self.device_label, (None, None))
        return normal or reference[0], fast or reference[1], len(measured[False]) + len(measured[True])


def whisper_models(args):
    """Language-specific Whisper models that are present on disk, e.g. {"tr": Path}."""
    path = Path(args.whisper_tr) if args.whisper_tr else None
    return {"tr": path} if path and (path / "config.json").exists() else {}


def device_label(args):
    """Name of the accelerator generations run on, e.g. "NVIDIA L4" or "Apple MPS"."""
    if args.mock:
        return "mock"
    import torch
    if args.device in ("auto", "cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if args.device in ("auto", "mps") and torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def budget_arg(value):
    if value == "auto":
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("bir sayı (GiB) ya da 'auto' olmalı") from None


def resolve_budget(args):
    """--budget auto: the NVIDIA GPU's own memory, plus --offload-ar below 20 GiB (the Colab notebook's rule).

    Up to 12 GB cards the weights (7.3 GB) and the AR KV cache (1.2-2.8 GB) do not fit together, so --phase-offload
    keeps only the half of the model a stage uses on the card (~6 GiB peak). On Windows --shared-gpu-memory stays on
    as a safety net: what still does not fit (long songs with CFG) goes to system RAM instead of failing."""
    if args.budget != "auto":
        return
    args.budget = 24.0
    if args.mock:
        return
    import torch
    if args.device not in ("auto", "cuda") or not torch.cuda.is_available():
        print("--budget auto: NVIDIA GPU bulunamadı, 24 GiB varsayıldı.", flush=True)
        return
    gpu = torch.cuda.get_device_properties(0)
    total = gpu.total_memory / 2**30
    args.budget = float(int(total))
    if total < 20:
        args.offload_ar = True
    if total < 13:
        args.phase_offload = True
        if os.name == "nt":
            args.shared_gpu_memory = True
    # Includes this process's CUDA context; with the monitor on this card also the desktop and open apps.
    free = torch.cuda.mem_get_info()[0] / 2**30
    print(f"--budget auto: {gpu.name}, {total:.1f} GiB ({free:.1f} GiB boş) -> --budget {args.budget:g}"
          + (" --offload-ar" if args.offload_ar else "")
          + (" --phase-offload" if args.phase_offload else "")
          + (" --shared-gpu-memory" if args.shared_gpu_memory else ""), flush=True)
    if args.phase_offload:
        print("Faz bazlı yerleşim: token üretiminde modelin AR yarısı (~4.4 GB), ses sentezinde NAR yarısı (~2.8 GB) "
              "ekran kartında, diğer yarı RAM'de bekler. Kalite değişmez. CFG scale 1'den büyükse ya da Nota planı "
              "Kapalı ise önbellek iki katına çıkar; uzun şarkılarda kart dolabilir.", flush=True)
        if free < 6.5:
            print(f"Ekran kartında yalnızca {free:.1f} GiB boş. Tarayıcı, oyun gibi kartı kullanan uygulamaları "
                  "kapatmak üretimi hızlandırır.", flush=True)
        if not args.shared_gpu_memory:
            print("Uyarı: paylaşımlı GPU belleği yalnızca Windows'ta var; karta sığmayan uzun şarkılar bellek hatası "
                  "verebilir.", flush=True)


def fp8_supported(args):
    """yue2's FP8 path needs an NVIDIA GPU with compute capability 8.9+ (e.g. L4, RTX 4090)."""
    if args.mock:
        return True
    import torch
    return (args.device in ("auto", "cuda") and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) >= (8, 9))


def make_room(server, args):
    """Unload a resident YuE2 when Demucs + Whisper would not fit next to it on the GPU."""
    if analysis_device(args) != "cuda" and args.asr_device not in ("auto", "cuda"):
        return
    import torch
    if not torch.cuda.is_available() or not server.backend.loaded:
        return
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0] / 2**30
    if free < ANALYSIS_GPU_GIB:
        print(f"Analiz için GPU'da {free:.1f} GiB boş; YuE2 bellekten çıkarılıyor.", flush=True)
        server.backend.unload()


def analysis_device(args):
    """Demucs runs on CUDA when present (Colab); on Apple Silicon CPU proved reliable."""
    if args.analysis_device != "auto":
        return args.analysis_device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_app(server: Server, args):
    app = FastAPI(title="YuE2 model server")
    fp8 = fp8_supported(args)
    import sheetsage
    # Checked on every use: SheetSage2 may be installed (Colab cell 5b) after the server started.
    sheetsage_ready = lambda: args.mock or sheetsage.available(args.sheetsage_python, args.sheetsage)

    def get_job(job_id):
        job = server.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job

    @app.get("/health")
    def health():
        running = [j.id for j in server.jobs.values() if j.status == "running"]
        return {"status": "ok", "mode": "mock" if args.mock else "yue2",
                "model": None if args.mock else args.model, "vae": None if args.mock else args.vae,
                "model_loaded": server.backend.loaded, "device": server.backend.device,
                "device_name": server.device_label, "fp8": fp8, "sheetsage": sheetsage_ready(),
                "load_error": server.load_error, "queued": server.queue.qsize(), "running": running}

    @app.get("/stats")
    def stats():
        """This device's measured speed, for the UI's time estimates (None before its first generation)."""
        rate, fast_rate, jobs = server.rates()
        return {"device": server.device_label, "rate": rate, "fast_rate": fast_rate, "jobs": jobs}

    @app.post("/jobs")
    def create_job(request: GenerateRequest):
        if request.quantization == "fp8" and not fp8:
            raise HTTPException(422, f"FP8 yalnızca compute capability 8.9+ NVIDIA GPU'larda çalışır (ör. L4); "
                                     f"bu sunucunun cihazı: {server.device_label}")
        return server.submit(request).to_dict()

    @app.get("/jobs")
    def list_jobs(limit: int = 50):
        jobs = sorted(server.jobs.values(), key=lambda j: j.created, reverse=True)[:limit]
        return [j.to_dict() for j in jobs]

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        return get_job(job_id).to_dict()

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        job = get_job(job_id)
        job.cancel_requested = True
        return job.to_dict()

    @app.get("/jobs/{job_id}/audio")
    def audio(job_id: str):
        job = get_job(job_id)
        if job.status != "complete" or not job.result:
            raise HTTPException(409, "Audio is not ready")
        path = job.directory / job.result["audio"]
        if path.suffix == ".flac" and not args.mock:
            import soundfile as sf
            mp3 = path.with_suffix(".mp3")
            if not mp3.exists():
                audio, rate = sf.read(path, dtype="float32")
                write_mp3(audio, rate, mp3)
            job.result["audio"] = mp3.name
            job.save()
            path = mp3
        return FileResponse(path)

    @app.post("/uploads")
    async def upload(request: Request, filename: str = "audio.mp3"):
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp3", ".wav", ".flac", ".ogg"}:
            raise HTTPException(415, "MP3, WAV, FLAC veya OGG yükleyin")
        body = await request.body()
        # The same file always gets the same id, so its analyses are cached across uploads.
        upload_id = hashlib.sha256(body).hexdigest()[:32]
        directory = server.outputs / "uploads" / upload_id
        if not any(directory.glob("source.*")):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ("source" + suffix)).write_bytes(body)
        return {"upload_id": upload_id}

    @app.post("/analyze")
    def analyze(request: AnalyzeRequest):
        directory = server.outputs / "uploads" / request.upload_id
        sources = list(directory.glob("source.*"))
        if not sources:
            raise HTTPException(404, "Unknown upload")
        if args.mock:
            with server.gpu_for_analysis():
                return {**MockBackend.analysis(request.recognize_lyrics), "upload_id": request.upload_id}
        if request.transcriber == "sheetsage2" and not sheetsage_ready():
            missing = [p for p in (Path(args.sheetsage_python), Path(args.sheetsage) / "model.safetensors")
                       if not p.exists()]
            raise HTTPException(503, "SheetSage2 kurulu değil, eksik: " + ", ".join(map(str, missing)) + ". "
                                     "Colab'da 5b. SheetSage2 hücresini çalıştırın (sonunda 'SheetSage2 hazır' "
                                     "yazmalı), sonra tekrar deneyin. Beklemeden devam etmek için Nota çıkarıcı: "
                                     "Basit analiz'i seçin.")
        whisper, models = Path(args.whisper), whisper_models(args)
        # A chosen language with its own model needs only that one; otherwise the multilingual model detects/writes.
        needed = models.get(request.language, whisper)
        if request.recognize_lyrics and not (needed / "config.json").exists():
            raise HTTPException(503, f"Whisper modeli bulunamadı: {needed}")
        # Lyric labels only name the sections when the lyrics are not recognized.
        params = {**request.model_dump(exclude={"upload_id", "lyrics"}), "version": ANALYSIS_VERSION,
                  "labels": None if request.recognize_lyrics else request.lyrics,
                  "whisper": [str(whisper), {k: str(v) for k, v in models.items()}] if request.recognize_lyrics else None}
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        cached = directory / "analysis" / f"result-{key}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        import cover
        # Cached results above need no GPU. A new analysis never waits: it is refused while a song is generating.
        with server.gpu_for_analysis():
            make_room(server, args)
            try:
                result = cover.analyze(sources[0], start=request.start, duration=request.duration,
                                       lyrics=request.lyrics, bpm=request.bpm, demucs_dir=args.demucs,
                                       device=analysis_device(args),
                                       workdir=directory / "analysis",
                                       whisper_dir=whisper if request.recognize_lyrics else None,
                                       language=request.language,
                                       asr_device=None if args.asr_device == "auto" else args.asr_device,
                                       whisper_models=models,
                                       sheetsage={"python": args.sheetsage_python, "model_dir": args.sheetsage}
                                       if request.transcriber == "sheetsage2" else None)
            except ValueError as exc:
                raise HTTPException(422, str(exc))
            except Exception as exc:
                # Show the cause in the UI instead of a bare "Internal Server Error"; the log keeps the traceback.
                traceback.print_exc()
                raise HTTPException(500, f"Analiz başarısız: {type(exc).__name__}: {exc}")
        response = {"abc": result.abc, "lyrics": result.lyrics, "summary": result.summary(),
                    "upload_id": request.upload_id}
        cached.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
        return response

    return app


def main():
    p = argparse.ArgumentParser(description="YuE2 model server for YuE-UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--outputs", default=str(Path(__file__).resolve().parents[1] / "outputs"))
    p.add_argument("--mock", action="store_true", help="No model: synthesize a test tone to verify the UI")
    p.add_argument("--preload", action="store_true", help="Load the model at startup instead of on first job")
    p.add_argument("--model", default="m-a-p/YuE2-3B", help="HF repo or local directory")
    p.add_argument("--vae", default="m-a-p/YuE2-Vae", help="HF repo or local directory")
    p.add_argument("--device", default="auto", help="auto, mps, cuda, cpu")
    p.add_argument("--budget", type=budget_arg, default=24.0,
                   help="Memory budget in GiB, or 'auto': the NVIDIA GPU's memory, with --offload-ar below 20 GiB")
    p.add_argument("--backend", choices=("torch", "torch-eager", "vllm"), default="torch")
    p.add_argument("--quantization", choices=("none", "fp8"), default="none",
                   help="fp8 needs NVIDIA compute capability 8.9+ (e.g. L4)")
    p.add_argument("--offload-ar", action="store_true")
    p.add_argument("--phase-offload", action="store_true",
                   help="Keep only the half of the model a stage uses on the GPU (AR for tokens, NAR for synthesis), "
                        "the other half in RAM. --budget auto turns it on up to 12 GB cards")
    p.add_argument("--shared-gpu-memory", action="store_true",
                   help="Do not cap PyTorch below the GPU's memory: on Windows what does not fit goes to shared GPU "
                        "memory (system RAM), slower instead of out of memory. --budget auto turns it on up to 12 GB cards")
    p.add_argument("--offline", action="store_true", help="Use only already downloaded weights")
    p.add_argument("--analysis-device", default="auto", help="Cover analysis device: auto, cuda, cpu")
    p.add_argument("--keep-loaded", action="store_true",
                   help="Keep weights in memory between jobs (default: free them when the queue is empty)")
    p.add_argument("--demucs", default=str(Path(__file__).resolve().parents[2] / "models" / "demucs"),
                   help="Local htdemucs checkpoint folder for cover analysis")
    p.add_argument("--whisper",
                   default=str(Path(__file__).resolve().parents[2] / "models" / "whisper-large-v3-turbo"),
                   help="Local multilingual Whisper model folder: detects the language, transcribes non-Turkish lyrics")
    p.add_argument("--whisper-tr",
                   default=str(Path(__file__).resolve().parents[2] / "models" / "whisper-large-v3-turkish-general"),
                   help="Local Turkish Whisper model folder, used when the lyrics are Turkish ('' to disable)")
    p.add_argument("--asr-device", default="auto", help="Lyric recognition device: auto, cuda, mps, cpu")
    p.add_argument("--sheetsage", default=str(Path(__file__).resolve().parents[2] / "models" / "SheetSage2"),
                   help="Local SheetSage2 folder (score transcription for covers)")
    p.add_argument("--sheetsage-python",
                   default=str(Path(__file__).resolve().parents[2] / "sheetsage_ortam" /
                               ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
                   help="Python of SheetSage2's own environment (torch 2.8)")
    args = p.parse_args()
    resolve_budget(args)

    outputs = Path(args.outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    server = Server(MockBackend() if args.mock else YuE2Backend(args), outputs,
                    unload_when_idle=not args.keep_loaded, device_label=device_label(args))
    if args.preload and not args.mock:
        threading.Thread(target=server.preload, daemon=True).start()
    uvicorn.run(create_app(server, args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
