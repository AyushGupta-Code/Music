from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from groq import Groq
from pydub import AudioSegment

from prompts import build_music_profile

# ── MusicGen (via transformers) ───────────────────────────────────────────────

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

from transformers import MusicgenForConditionalGeneration, AutoProcessor

_MUSICGEN_CACHE: dict = {}

MUSIC_MODELS = [
    "musicgen-small",
    "musicgen-medium",
    "musicgen-large",
]


def _load_musicgen(model_name: str):
    model_id = f"facebook/{model_name}" if "/" not in model_name else model_name
    if model_id in _MUSICGEN_CACHE:
        return _MUSICGEN_CACHE[model_id]

    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    _MUSICGEN_CACHE[model_id] = (model, processor)
    return model, processor


def _generate_music(
    prompt: str,
    out_wav: str,
    duration_s: int = 30,
    seed: Optional[int] = None,
    model_name: str = "musicgen-small",
) -> str:
    if seed is not None:
        torch.manual_seed(seed)

    model, processor = _load_musicgen(model_name)
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")

    # MusicGen generates at ~50 tokens/sec
    max_new_tokens = int(duration_s * 50)
    with torch.no_grad():
        audio = model.generate(**inputs, max_new_tokens=max_new_tokens)

    sr = model.config.audio_encoder.sampling_rate
    sf.write(out_wav, audio[0, 0].cpu().numpy(), sr)
    return out_wav


# ── Kokoro TTS ────────────────────────────────────────────────────────────────

_KOKORO_PIPELINE = None

KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",  # American female
    "am_adam", "am_michael",                                      # American male
    "bf_emma", "bf_isabella",                                     # British female
    "bm_george", "bm_lewis",                                      # British male
]


def _get_kokoro():
    global _KOKORO_PIPELINE
    if _KOKORO_PIPELINE is None:
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "kokoro is not installed.\n"
                "Install with: pip install kokoro"
            )
        _KOKORO_PIPELINE = KPipeline(lang_code="a")
    return _KOKORO_PIPELINE


def _narrate(text: str, out_wav: str, voice: str = "af_heart", speed: float = 1.0) -> str:
    kpipe = _get_kokoro()
    chunks = [audio for _, _, audio in kpipe(text, voice=voice, speed=speed)]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio output.")
    sf.write(out_wav, np.concatenate(chunks), 24000)
    return out_wav


# ── Audio mixing ──────────────────────────────────────────────────────────────

def _normalize(seg: AudioSegment, target_dbfs: float = -16.0) -> AudioSegment:
    if seg.max_dBFS == float("-inf"):
        return seg
    return seg.apply_gain(target_dbfs - seg.max_dBFS)


def _reformat(seg: AudioSegment, rate: int = 44100, channels: int = 2) -> AudioSegment:
    if seg.frame_rate != rate:
        seg = seg.set_frame_rate(rate)
    if seg.channels != channels:
        seg = seg.set_channels(channels)
    return seg


def _mix(
    voice_wav: str,
    music_wav: str,
    out_wav: str,
    voice_ratio: float = 0.85,
    music_ratio: float = 0.15,
) -> str:
    voice = _reformat(_normalize(AudioSegment.from_file(voice_wav), -16.0))
    music = _reformat(_normalize(AudioSegment.from_file(music_wav), -19.0))

    # loop music to match voice length
    if len(music) < len(voice):
        music = (music * ((len(voice) // len(music)) + 1))[: len(voice)]
    else:
        music = music[: len(voice)]

    ref = max(voice_ratio, music_ratio) or 1.0
    v_db = 20 * math.log10(max(voice_ratio / ref, 1e-6))
    m_db = 20 * math.log10(max(music_ratio / ref, 1e-6))

    mixed = _normalize(
        music.apply_gain(m_db).overlay(voice.apply_gain(v_db)), -16.0
    )
    mixed.export(out_wav, format="wav")
    return out_wav


# ── Chapter parsing ───────────────────────────────────────────────────────────

def parse_chapters(text: str) -> list[tuple[str, str]]:
    patterns = [
        r"(?m)^(Chapter\s+(?:\d+|[A-Z][a-z]+)[^\n]*)",
        r"(?m)^(CHAPTER\s+(?:\d+|[A-Z]+)[^\n]*)",
        r"(?m)^(Part\s+(?:\d+|[A-Z][a-z]+)[^\n]*)",
    ]
    for pattern in patterns:
        parts = re.split(pattern, text)
        if len(parts) >= 5:
            chapters = []
            for i in range(1, len(parts) - 1, 2):
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if len(content) > 200:
                    chapters.append((parts[i].strip(), content))
            if len(chapters) >= 2:
                return chapters

    # Fallback: chunk paragraphs into sections of ~10 paragraphs each
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    chunk = 10
    sections = [
        (f"Section {i // chunk + 1}", "\n\n".join(paragraphs[i : i + chunk]))
        for i in range(0, len(paragraphs), chunk)
    ]
    return sections or [("Full Text", text.strip())]


# ── Groq emotion analysis ─────────────────────────────────────────────────────

_FALLBACK_SCORES = {"anger": 0.1, "joy": 0.4, "sadness": 0.2, "optimism": 0.3}


def _analyze_emotion(text: str, client: Groq) -> dict[str, float]:
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Analyze the emotional tone of this chapter for film score composition. "
                        "Return only a JSON object with exactly these keys: anger, joy, sadness, optimism. "
                        "Values are floats 0–1 that sum to 1.0.\n\n"
                        + text[:3000]
                    ),
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=80,
        )
        raw = json.loads(resp.choices[0].message.content)
        keys = ["anger", "joy", "sadness", "optimism"]
        scores = {k: float(raw.get(k, 0.25)) for k in keys}
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}
    except Exception as e:
        print(f"  [warn] Emotion analysis failed ({e}), using fallback.")
        return _FALLBACK_SCORES


# ── Main entry point ──────────────────────────────────────────────────────────

def process_novel(
    input_path: str,
    output_dir: str,
    music_model: str = "musicgen-small",
    voice: str = "af_heart",
    speed: float = 1.0,
    voice_ratio: float = 0.85,
    music_ratio: float = 0.15,
    music_duration: int = 30,
    seed: Optional[int] = None,
    groq_api_key: Optional[str] = None,
) -> list[str]:
    text = Path(input_path).read_text(encoding="utf-8")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = Groq(api_key=groq_api_key or os.environ["GROQ_API_KEY"])
    chapters = parse_chapters(text)
    total = len(chapters)
    print(f"Found {total} chapter(s) in '{input_path}'.")

    output_files = []

    for idx, (title, content) in enumerate(chapters, 1):
        slug = re.sub(r"[^\w]+", "_", title.lower())[:40].strip("_")
        prefix = out / f"{idx:02d}_{slug}"
        print(f"\n[{idx}/{total}] {title}")

        print("  Analyzing emotion...")
        scores = _analyze_emotion(content, client)
        profile = build_music_profile(scores)
        print(f"  Mood: {profile['cinematic_mode']} — {profile['mood']}")

        music_wav = f"{prefix}_music.wav"
        print(f"  Generating music ({music_duration}s)...")
        _generate_music(profile["prompt"], music_wav, music_duration, seed, music_model)

        voice_wav = f"{prefix}_voice.wav"
        print("  Narrating chapter...")
        _narrate(content, voice_wav, voice, speed)

        final_wav = f"{prefix}.wav"
        print("  Mixing...")
        _mix(voice_wav, music_wav, final_wav, voice_ratio, music_ratio)

        os.remove(music_wav)
        os.remove(voice_wav)

        output_files.append(final_wav)
        print(f"  -> {Path(final_wav).name}")

    print(f"\nDone. {len(output_files)} file(s) written to '{output_dir}/'")
    return output_files
