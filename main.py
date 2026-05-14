#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from pipeline import MUSIC_MODELS, KOKORO_VOICES


def main():
    p = argparse.ArgumentParser(
        description="Convert a novel to chapter-wise audio files with background score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",  help="Path to novel .txt file")
    p.add_argument("output", help="Output directory for audio files")
    p.add_argument(
        "--music-model", default="musicgen-small", choices=MUSIC_MODELS,
        help="MusicGen model to use",
    )
    p.add_argument(
        "--voice", default="af_heart", choices=KOKORO_VOICES,
        help="Kokoro voice ID",
    )
    p.add_argument("--speed",         type=float, default=1.0,  help="Narration speed")
    p.add_argument("--music-duration", type=int,   default=30,   help="Music clip length in seconds (looped to match narration)")
    p.add_argument("--voice-ratio",   type=float, default=0.85, help="Voice level in mix (0–1)")
    p.add_argument("--music-ratio",   type=float, default=0.15, help="Music level in mix (0–1)")
    p.add_argument("--seed",          type=int,   default=None, help="Seed for reproducible music generation")
    args = p.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    from pipeline import process_novel
    process_novel(
        input_path=args.input,
        output_dir=args.output,
        music_model=args.music_model,
        voice=args.voice,
        speed=args.speed,
        voice_ratio=args.voice_ratio,
        music_ratio=args.music_ratio,
        music_duration=args.music_duration,
        seed=args.seed,
        groq_api_key=api_key,
    )


if __name__ == "__main__":
    main()
