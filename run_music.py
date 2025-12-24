# run_music.py
from __future__ import annotations

import os
from pathlib import Path

import torch

from metrics import Metrics, StageTimer, append_metrics_csv, append_metrics_jsonl, new_request_id
from para_to_emo import detect_emotion
from map_emo_to_music import map_emotions_to_music
from musicgenutil import generate_music
from generate_voice import synth_openvoice_default, duck_and_mix

if not hasattr(torch, "uint64"):
    torch.uint64 = torch.long  # compat shim for older deps


def main():
    paragraph = """India is a restless mosaic—Himalayan ice feeding monsoon rivers, deserts that sing at dusk, coasts that smell of salt and cardamom. It’s ancient temples carved like whispered prayers and glass towers lit by code; a place where ragas rise with the dawn and train whistles braid a nation of languages together. Street corners turn into kitchens—saffron, smoke, and lime riding the air—while cricket ricochets through alleys and constellations of festivals reset the calendar with color and light. In crowded markets and quiet courtyards alike, argument is an art form, hospitality a default, and history never fully past. India doesn’t march in a straight line; it swirls—contradictions colliding into something stubbornly optimistic, always improvising, always becoming."""

    # Choose models here
    music_model = "musicgen-small"  # or "musicgen-medium"
    voice_language = "EN"
    voice_speaker = None           # or "EN-US" etc.
    duration_s = 10
    seed = 42
    voice_speed = 1.0
    voice_ratio = 0.8
    music_ratio = 0.2

    request_id = new_request_id()
    m = Metrics(request_id=request_id)

    m.set_meta("music_model", music_model)
    m.set_meta("voice_language", voice_language)
    m.set_meta("voice_speaker", voice_speaker or "Auto")
    m.set_meta("duration_s", duration_s)
    m.set_meta("seed", seed)
    m.set_meta("paragraph_len_chars", len(paragraph))

    # 1) Emotion
    with StageTimer(m, "emotion_detect_s"):
        emotions = detect_emotion(paragraph)
    dominant = max(emotions, key=emotions.get) if emotions else "unknown"
    m.set_meta("dominant_emotion", dominant)

    # 2) Prompt
    with StageTimer(m, "prompt_map_s"):
        profile = map_emotions_to_music(emotions)
        prompt = profile["prompt"]
    m.set_meta("music_prompt", prompt)

    # 3) Music
    with StageTimer(m, "music_generate_total_s"):
        music_info = generate_music(prompt, out_wav="output.wav", duration_s=duration_s, seed=seed, model_name=music_model)

    # 4) Voice
    with StageTimer(m, "voice_tts_total_s"):
        voice_info = synth_openvoice_default(paragraph, out_wav="voice_openvoice.wav", language=voice_language, speed=voice_speed, speaker_key=voice_speaker)

    # 5) Mix
    with StageTimer(m, "mix_total_s"):
        mix_info = duck_and_mix(
            voice_path=voice_info["path"],
            music_path=music_info["path"],
            out_wav="final_mix.wav",
            voice_ratio=voice_ratio,
            music_ratio=music_ratio,
            target_dbfs=-16.0,
            frame_rate=32000,
            channels=2,
        )

    m.finish()
    append_metrics_csv(m, "metrics/metrics.csv")
    append_metrics_jsonl(m, "metrics/metrics.jsonl")

    print(f"Done. Final mix: {mix_info['path']}")
    print(f"Metrics saved to metrics/metrics.csv and metrics/metrics.jsonl (request_id={request_id})")


if __name__ == "__main__":
    # Run: python run_music.py
    # Outputs: output.wav, voice_openvoice.wav, final_mix.wav, and metrics logs
    main()
