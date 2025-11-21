# openvoice_tts_no_ref.py
import os
import math
from typing import Optional
import random
import re
import tempfile

import torch


def _locate_dic_dir(dic_module) -> str | None:
    """Return a filesystem path to a MeCab dictionary shipped with a module.

    The `unidic-lite` package always ships its dictionary, while `unidic`
    requires a post-install download. This helper keeps `_configure_mecab_dictionary`
    focused on environment setup and handles both cases defensively.
    """

    dic_dir = getattr(dic_module, "DICDIR", None)
    if dic_dir and os.path.isdir(dic_dir):
        return dic_dir

    # Some packages expose the dictionary via package data instead of DICDIR.
    try:
        import importlib.resources as resources

        with resources.as_file(resources.files(dic_module) / "dicdir") as path:
            if path.is_dir():
                return str(path)
    except Exception:
        # Keep this helper resilient; the caller will try other modules/fallbacks.
        return None

    return None


def _configure_mecab_dictionary() -> None:
    """Point MeCab at the bundled `unidic-lite`/`unidic` dictionary if present.

    Melo imports `MeCab.Tagger()` with no arguments. Without a dictionary
    configured, MeCab searches for `/unidic/dicdir/mecabrc` and raises the
    runtime error seen in the original bug report. We default to the
    lightweight dictionary installed via `requirements.txt` and still allow
    callers to override with `MECAB_ARGS`/`MECABRC`.
    """

    # Respect any explicit configuration supplied by the user or their
    # environment (e.g., custom dictionaries).
    if os.environ.get("MECAB_ARGS"):
        return

    dic_module = None
    try:
        import unidic_lite as dic_module
    except ImportError:
        try:
            import unidic as dic_module  # pragma: no cover - optional fallback
        except ImportError:
            # Not installed (or offline during pip install). The import will fail
            # later in Melo with a clear MeCab error; keep the module importable
            # for users supplying their own MECAB_ARGS/MECABRC.
            return

    dic_dir = _locate_dic_dir(dic_module)

    # If `unidic` is installed but the dictionary has not been downloaded yet,
    # try to fetch it. This can happen when pip installs succeed without
    # post-install hooks running.
    if (not dic_dir or not os.path.isfile(os.path.join(dic_dir, "mecabrc"))) and hasattr(dic_module, "download"):
        try:
            dic_module.download()  # type: ignore[call-arg]
        except Exception:
            # Keep failures silent; users can still supply their own paths via
            # MECAB_ARGS/MECABRC.
            pass
        dic_dir = _locate_dic_dir(dic_module)

    if not dic_dir or not os.path.isdir(dic_dir):
        return

    mecabrc_path = os.path.join(dic_dir, "mecabrc")
    args_parts = []

    if os.path.isfile(mecabrc_path):
        # Tell MeCab to use the bundled config so it does not look for the
        # system-wide /unidic/dicdir/mecabrc.
        os.environ.setdefault("MECABRC", mecabrc_path)
        args_parts.append(f"-r {mecabrc_path}")

    args_parts.append(f"-d {dic_dir}")
    os.environ.setdefault("MECAB_ARGS", " ".join(args_parts))


_configure_mecab_dictionary()

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
    expressive: bool = True,
    pause_ms: int = 240,
    speed_variation: float = 0.12,
    energy_variation: float = 0.06,
    prosody_seed: int | None = None,
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

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    # Fall back to single-pass synthesis if expressive mode is disabled or the
    # text is too short to benefit from per-sentence prosody.
    if not expressive or len(sentences) <= 1:
        tts.tts_to_file(text, speaker_id, out_wav, speed=speed)
        abs_path = os.path.abspath(out_wav)
        print(f"✅ Voice saved: {abs_path}")
        return abs_path

    rng = random.Random(prosody_seed if prosody_seed is not None else len(text))
    rendered_segments: list[AudioSegment] = []
    per_sentence_pauses: list[int] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, sentence in enumerate(sentences):
            # Add light speed jitter per sentence so long reads feel less flat.
            jitter = rng.uniform(-abs(speed_variation), abs(speed_variation))
            adjusted_speed = max(0.7, min(1.35, speed * (1.0 + jitter)))

            seg_path = os.path.join(tmpdir, f"chunk_{idx}.wav")
            tts.tts_to_file(sentence, speaker_id, seg_path, speed=adjusted_speed)

            seg = AudioSegment.from_file(seg_path)

            # Subtle emphasis: sentences ending with "!" get a tiny lift; questions linger longer.
            if sentence.endswith("!"):
                seg = seg.apply_gain(_db(1.05))

            # Add gentle loudness jitter to keep long narrations from feeling static.
            if energy_variation > 0:
                gain_factor = rng.uniform(1.0 - abs(energy_variation), 1.0 + abs(energy_variation))
                seg = seg.apply_gain(_db(gain_factor))

            rendered_segments.append(seg)

            extra_pause = 0
            if sentence.endswith("?"):
                extra_pause = int(pause_ms * 0.4)
            elif sentence.endswith("!"):
                extra_pause = int(pause_ms * 0.25)
            per_sentence_pauses.append(max(0, int(pause_ms + extra_pause)))

        # Stitch sentences with per-sentence pauses instead of a uniform join.
        voice_mix = rendered_segments[0]
        for seg, pause_len in zip(rendered_segments[1:], per_sentence_pauses[:-1]):
            voice_mix += AudioSegment.silent(duration=pause_len) + seg

        voice_mix.export(out_wav, format="wav")

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

    # Light sweetening to make the backing track feel more polished/less flat.
    music = music.high_pass_filter(70)
    music = music.low_pass_filter(16000)
    music = music.compress_dynamic_range(threshold=-24.0, ratio=4.0, attack=5, release=250)
    music = music.fade_in(800).fade_out(1200)

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
