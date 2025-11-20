# openvoice_tts_no_ref.py
import os
import math
from typing import Optional

import torch
try:
    import unidic_lite

    os.environ.setdefault("MECAB_ARGS", f"-d {unidic_lite.DICDIR}")
except ImportError:
    # unidic-lite should be installed via requirements.txt, but allow
    # environments with custom dictionaries to supply their own MECAB_ARGS.
    pass

from melo.api import TTS
from pydub import AudioSegment


# -----------------------------
# Utility helpers
# -----------------------------
def _db(change_ratio: float) -> float:
    """Convert a linear ratio (0..1) to dB change."""
    # guard against log(0)
    change_ratio = max(change_ratio, 1e-6)
    return 20.0 * math.log10(change_ratio)


def normalize_to_dbfs(seg: AudioSegment, target_dbfs: float = -16.0) -> AudioSegment:
    """
    Peak-normalize an AudioSegment to approximately target dBFS.
    (Pydub uses max peak; good enough for quick post.)
    """
    if seg.max_dBFS == float("-inf"):  # silence
        return seg
    gain_needed = target_dbfs - seg.max_dBFS
    return seg.apply_gain(gain_needed)


def ensure_rate_channels(seg: AudioSegment, frame_rate: int = 32000, channels: int = 2) -> AudioSegment:
    """Resample and set channel count (mono=1, stereo=2) to keep files consistent."""
    if seg.frame_rate != frame_rate:
        seg = seg.set_frame_rate(frame_rate)
    if seg.channels != channels:
        seg = seg.set_channels(channels)
    return seg


# -----------------------------
# TTS (OpenVoice Melo) without reference audio
# -----------------------------
def synth_openvoice_default(
    text: str,
    out_wav: str = "voice_openvoice.wav",
    language: str = "EN",
    speed: float = 1.0,
    device: Optional[str] = None,
) -> str:
    """
    Generate speech with OpenVoice (Melo) using a built-in speaker (no voice cloning).
    Returns the absolute path to the saved wav.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load language pack
    tts = TTS(language=language, device=device)

    # available speakers -> integer IDs
    spk2id = tts.hps.data.spk2id  # e.g., {"EN-US": 0, "EN-UK": 1, "EN-LIBRITTS": 2, ...}

    # Pick a speaker matching the language key; else fallback to first
    try:
        speaker_key = next(k for k in spk2id.keys() if language.lower() in k.lower())
    except StopIteration:
        speaker_key = next(iter(spk2id.keys()))

    speaker_id = spk2id[speaker_key]
    print(f"🗣️ Using speaker: '{speaker_key}' (id={speaker_id}) [{language}] → {out_wav}")

    # IMPORTANT: pass integer speaker_id
    tts.tts_to_file(text, speaker_id, out_wav, speed=speed)
    abs_path = os.path.abspath(out_wav)
    print(f"✅ Voice saved: {abs_path}")
    return abs_path


# -----------------------------
# Mixing (80% voice / 20% music)
# -----------------------------
def duck_and_mix(
    voice_path: str,
    music_path: str,
    out_wav: str = "final_mix.wav",
    voice_ratio: float = 0.8,
    music_ratio: float = 0.2,
    target_dbfs: float = -16.0,
    frame_rate: int = 32000,
    channels: int = 2,
) -> str:
    """
    Mix voice over music with simple ducking so voice clearly dominates.

    - Normalizes both to target dBFS (peak) to avoid clipping.
    - Attenuates music vs voice according to the given ratios.
    - Extends/loops or trims music to match voice length.
    """
    assert 0 < voice_ratio <= 1 and 0 <= music_ratio < 1, "Ratios must be between 0..1"

    voice = AudioSegment.from_file(voice_path)
    music = AudioSegment.from_file(music_path)

    # Conform sample rate/channels before any gain staging
    voice = ensure_rate_channels(voice, frame_rate, channels)
    music = ensure_rate_channels(music, frame_rate, channels)

    # Normalize both around a reasonable headroom
    voice = normalize_to_dbfs(voice, target_dbfs)
    music = normalize_to_dbfs(music, target_dbfs - 3)  # give music slightly more headroom

    # Length handling: loop/trim music to voice length
    if len(music) < len(voice):
        # loop music
        loops = (len(voice) // len(music)) + 1
        music = (music * loops)[: len(voice)]
    else:
        music = music[: len(voice)]

    # Ratio-based gains (approximate perceived balance)
    # Scale relative to the louder of the two (voice)
    # If voice_ratio:music_ratio = 0.8:0.2, we attenuate music more.
    ref = max(voice_ratio, music_ratio)
    voice_gain_db = _db(voice_ratio / ref)  # usually 0 dB
    music_gain_db = _db(music_ratio / ref)  # typically negative

    # Apply gains
    voice_adj = voice.apply_gain(voice_gain_db)
    music_adj = music.apply_gain(music_gain_db)

    # Overlay voice on music
    mixed = music_adj.overlay(voice_adj)

    # Safety: final light normalization to avoid clipped exports
    mixed = normalize_to_dbfs(mixed, target_dbfs)

    mixed.export(out_wav, format="wav")
    abs_out = os.path.abspath(out_wav)
    print(f"🎚️ Final mix saved: {abs_out}")
    return abs_out


# -----------------------------
# End-to-end demo
# -----------------------------
if __name__ == "__main__":
    paragraph = "The battlefield roared like an angry god as steel clashed against steel, and the ground trembled under the relentless thunder of charging warhorses. Black smoke curled into the blood-red sky, carrying the stench of iron, fire, and death. Through the chaos, Commander Varos carved a path with his greatsword, each swing a devastating arc that sent enemy soldiers sprawling. Arrows hissed past his face, splintering against his armor, but he did not falter; his eyes burned with the unyielding fury of a man who refused to yield even an inch of ground. The screams of the wounded mingled with the deafening war drums, and the once-green fields were now slick with mud and crimson. Above it all, the enemy’s siege towers loomed ever closer, their monstrous silhouettes blotting out the horizon. Varos raised his sword and bellowed an order, his voice cutting through the din like lightning through a storm. The battered defenders rallied to him, their shields locking, their spears lowering, ready to meet the oncoming tide. The air was thick with dust and despair, but in that moment—surrounded, outnumbered, and pressed against the edge of annihilation—Varos felt the fire in his veins blaze hotter than ever."
    voice_out = "voice_openvoice.wav"
    music_in = "output.wav"   # ← your MusicGen output path
    final_out = "final_mix.wav"

    # 1) TTS
    synth_openvoice_default(paragraph, out_wav=voice_out, language="EN", speed=1.0)

    # 2) Mix (≈80% voice / 20% music)
    duck_and_mix(
        voice_path=voice_out,
        music_path=music_in,
        out_wav=final_out,
        voice_ratio=0.8,
        music_ratio=0.2,
        target_dbfs=-16.0,
        frame_rate=32000,   # MusicGen default sample rate
        channels=2
    )
