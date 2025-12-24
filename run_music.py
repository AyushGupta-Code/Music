"""Command-line entry point for the emotion → music → voice pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

if not hasattr(torch, "uint64"):
    torch.uint64 = torch.long  # compat shim for older deps

from generate_voice import duck_and_mix, synth_openvoice_default
from map_emo_to_music import map_emotions_to_music
from musicgenutil import generate_music
from para_to_emo import detect_emotion

sys.path.append(os.path.dirname(__file__))

DEFAULT_PARAGRAPH = """India is a restless mosaic—Himalayan ice feeding monsoon rivers, deserts that sing at dusk, coasts that smell of salt and cardamom. It’s ancient temples carved like whispered prayers and glass towers lit by code; a place where ragas rise with the dawn and train whistles braid a nation of languages together. Street corners turn into kitchens—saffron, smoke, and lime riding the air—while cricket ricochets through alleys and constellations of festivals reset the calendar with color and light. In crowded markets and quiet courtyards alike, argument is an art form, hospitality a default, and history never fully past. India doesn’t march in a straight line; it swirls—contradictions colliding into something stubbornly optimistic, always improvising, always becoming."""


def _resolve_paragraph(args: argparse.Namespace) -> str:
    if args.text_file:
        text = Path(args.text_file).expanduser().read_text(encoding="utf-8")
        text = text.strip()
        if text:
            return text
    if args.text:
        stripped = args.text.strip()
        if stripped:
            return stripped
    return DEFAULT_PARAGRAPH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        help="Paragraph to turn into music+voice. Defaults to an included sample.",
    )
    parser.add_argument(
        "--text-file",
        help="Read paragraph from a UTF-8 text file (overrides --text).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Music duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--voice-language",
        default="EN",
        help="Language/speaker family for Melo TTS (default: EN).",
    )
    parser.add_argument(
        "--voice-speed",
        type=float,
        default=1.0,
        help="Playback speed for the synthesized voice (default: 1.0).",
    )
    parser.add_argument(
        "--no-expressive",
        dest="expressive",
        action="store_false",
        default=True,
        help="Disable sentence-level pauses/jitter so narration stays flat (default: expressive on).",
    )
    parser.add_argument(
        "--voice-pause-ms",
        type=int,
        default=240,
        help="Base pause inserted between sentences when expressive mode is on (default: 240).",
    )
    parser.add_argument(
        "--voice-speed-variation",
        type=float,
        default=0.12,
        help="Per-sentence speed jitter (0.0-0.3 recommended) when expressive mode is on.",
    )
    parser.add_argument(
        "--voice-energy-variation",
        type=float,
        default=0.06,
        help="Per-sentence loudness jitter (0.0-0.3 recommended) when expressive mode is on.",
    )
    parser.add_argument(
        "--voice-ratio",
        type=float,
        default=0.8,
        help="Relative loudness of voice vs music when mixing (default: 0.8).",
    )
    parser.add_argument(
        "--music-ratio",
        type=float,
        default=0.2,
        help="Relative loudness of music vs voice when mixing (default: 0.2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where intermediate/final wav files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for MusicGen to make runs reproducible.",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> Path:
    parser = build_parser()
    if args is None:
        args = parser.parse_args()

    paragraph = _resolve_paragraph(args)
    if not paragraph:
        parser.error("Please provide --text/--text-file with non-empty content.")

    if not (0 < args.voice_ratio <= 1 and 0 <= args.music_ratio < 1):
        parser.error("voice/music ratios must be within 0..1 and voice must be > 0")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Detecting emotion…")
    emotions = detect_emotion(paragraph)
    profile = map_emotions_to_music(emotions)
    prompt = profile["prompt"]
    print(f"🎯 Dominant mood: {max(emotions, key=emotions.get)} → '{prompt}'")

    music_wav = output_dir / "output_music.wav"
    voice_wav = output_dir / "voice_openvoice.wav"
    final_wav = output_dir / "final_mix.wav"

    print("🎼 Generating MusicGen track…")
    generate_music(prompt, out_wav=str(music_wav), duration_s=args.duration, seed=args.seed)

    print("🗣️ Synthesizing narration…")
    synth_openvoice_default(
        paragraph,
        out_wav=str(voice_wav),
        language=args.voice_language,
        speed=args.voice_speed,
        expressive=args.expressive,
        pause_ms=args.voice_pause_ms,
        speed_variation=args.voice_speed_variation,
        energy_variation=args.voice_energy_variation,
    )

    print("🎚️ Mixing voice over music…")
    duck_and_mix(
        voice_path=str(voice_wav),
        music_path=str(music_wav),
        out_wav=str(final_wav),
        voice_ratio=args.voice_ratio,
        music_ratio=args.music_ratio,
        target_dbfs=-16.0,
        frame_rate=32000,
        channels=2,
    )

    print(f"✅ Done! Final mix: {final_wav}")
    return final_wav


if __name__ == "__main__":
    main()
