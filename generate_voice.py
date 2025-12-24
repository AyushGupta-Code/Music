# generate_voice.py
from __future__ import annotations

import math
import os
import time
from typing import Dict, Optional, Tuple

import torch
from melo.api import TTS
from pydub import AudioSegment

# Cache TTS per (language, device)
_TTS_CACHE: Dict[Tuple[str, str], TTS] = {}


def _get_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_tts(language: str, device: Optional[str] = None) -> Tuple[TTS, bool, float]:
    """
    Returns: (tts, from_cache, load_seconds)
    """
    dev = _get_device(device)
    key = (language, dev)
    if key in _TTS_CACHE:
        return _TTS_CACHE[key], True, 0.0

    t0 = time.perf_counter()
    tts = TTS(language=language, device=dev)
    dt = time.perf_counter() - t0

    _TTS_CACHE[key] = tts
    return tts, False, dt


def list_openvoice_speakers(language: str = "EN", device: Optional[str] = None) -> list[str]:
    """
    Return available built-in speaker keys for a Melo/OpenVoice language pack.
    """
    tts, _, _ = _get_tts(language=language, device=device)
    return sorted(list(tts.hps.data.spk2id.keys()))


def _db(change_ratio: float) -> float:
    change_ratio = max(change_ratio, 1e-6)
    return 20.0 * math.log10(change_ratio)


def normalize_to_dbfs(seg: AudioSegment, target_dbfs: float = -16.0) -> AudioSegment:
    if seg.max_dBFS == float("-inf"):
        return seg
    gain_needed = target_dbfs - seg.max_dBFS
    return seg.apply_gain(gain_needed)


def ensure_rate_channels(seg: AudioSegment, frame_rate: int = 32000, channels: int = 2) -> AudioSegment:
    if seg.frame_rate != frame_rate:
        seg = seg.set_frame_rate(frame_rate)
    if seg.channels != channels:
        seg = seg.set_channels(channels)
    return seg


def synth_openvoice_default(
    text: str,
    out_wav: str = "voice_openvoice.wav",
    language: str = "EN",
    speed: float = 1.0,
    speaker_key: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Generate speech using Melo/OpenVoice built-in speakers.

    Returns dict:
      {
        "path": abs_path,
        "language": language,
        "speaker_key": "...",
        "speaker_id": int,
        "from_cache": bool,
        "tts_load_s": float,
        "tts_s": float
      }
    """
    tts, from_cache, load_s = _get_tts(language=language, device=device)
    spk2id = tts.hps.data.spk2id  # dict: speaker_key -> id

    if not spk2id:
        raise RuntimeError(f"No speakers found for language '{language}' (spk2id empty).")

    # Choose speaker
    if speaker_key is None or not speaker_key.strip():
        speaker_key = next(iter(spk2id.keys()))
    else:
        speaker_key = speaker_key.strip()
        if speaker_key not in spk2id:
            raise ValueError(f"Unknown speaker_key '{speaker_key}'. Available: {sorted(spk2id.keys())}")

    speaker_id = int(spk2id[speaker_key])

    # Run TTS
    t0 = time.perf_counter()
    tts.tts_to_file(text, speaker_id, out_wav, speed=float(speed))
    tts_s = time.perf_counter() - t0

    abs_path = os.path.abspath(out_wav)
    return {
        "path": abs_path,
        "language": language,
        "speaker_key": speaker_key,
        "speaker_id": speaker_id,
        "from_cache": from_cache,
        "tts_load_s": float(load_s),
        "tts_s": float(tts_s),
    }


def duck_and_mix(
    voice_path: str,
    music_path: str,
    out_wav: str = "final_mix.wav",
    voice_ratio: float = 0.8,
    music_ratio: float = 0.2,
    target_dbfs: float = -16.0,
    frame_rate: int = 32000,
    channels: int = 2,
) -> Dict[str, object]:
    """
    Mix voice over music with basic ducking.

    Returns dict:
      {
        "path": abs_path,
        "mix_s": float
      }
    """
    assert 0 < voice_ratio <= 1 and 0 <= music_ratio < 1, "Ratios must be between 0..1"

    t0 = time.perf_counter()

    voice = AudioSegment.from_file(voice_path)
    music = AudioSegment.from_file(music_path)

    voice = ensure_rate_channels(voice, frame_rate, channels)
    music = ensure_rate_channels(music, frame_rate, channels)

    voice = normalize_to_dbfs(voice, target_dbfs)
    music = normalize_to_dbfs(music, target_dbfs - 3)

    if len(music) < len(voice):
        loops = (len(voice) // len(music)) + 1
        music = (music * loops)[: len(voice)]
    else:
        music = music[: len(voice)]

    ref = max(voice_ratio, music_ratio) if max(voice_ratio, music_ratio) > 0 else 1.0
    voice_gain_db = _db(voice_ratio / ref)
    music_gain_db = _db(music_ratio / ref)

    voice_adj = voice.apply_gain(voice_gain_db)
    music_adj = music.apply_gain(music_gain_db)

    mixed = music_adj.overlay(voice_adj)
    mixed = normalize_to_dbfs(mixed, target_dbfs)

    mixed.export(out_wav, format="wav")
    mix_s = time.perf_counter() - t0

    abs_out = os.path.abspath(out_wav)
    return {"path": abs_out, "mix_s": float(mix_s)}
