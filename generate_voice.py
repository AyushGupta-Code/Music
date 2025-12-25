# generate_voice.py
from __future__ import annotations

import math
import os
import time
from typing import Dict

from pydub import AudioSegment


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

    mixed = music.apply_gain(music_gain_db).overlay(voice.apply_gain(voice_gain_db))
    mixed = normalize_to_dbfs(mixed, target_dbfs)

    mixed.export(out_wav, format="wav")
    mix_s = time.perf_counter() - t0

    return {"path": os.path.abspath(out_wav), "mix_s": float(mix_s)}
