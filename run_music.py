# run_music.py
import torch
if not hasattr(torch, "uint64"):
    torch.uint64 = torch.long  # compat shim for older deps

# from musicgenutil import generate_music
from para_to_emo import detect_emotion
from map_emo_to_music import map_emotions_to_music
from musicgenutil import generate_music
from generate_voice import synth_openvoice_default, duck_and_mix

import sys, os
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    paragraph = """India is a restless mosaic—Himalayan ice feeding monsoon rivers, deserts that sing at dusk, coasts that smell of salt and cardamom. It’s ancient temples carved like whispered prayers and glass towers lit by code; a place where ragas rise with the dawn and train whistles braid a nation of languages together. Street corners turn into kitchens—saffron, smoke, and lime riding the air—while cricket ricochets through alleys and constellations of festivals reset the calendar with color and light. In crowded markets and quiet courtyards alike, argument is an art form, hospitality a default, and history never fully past. India doesn’t march in a straight line; it swirls—contradictions colliding into something stubbornly optimistic, always improvising, always becoming."""

    # 1) Emotion → prompt
    emotions = detect_emotion(paragraph)
    profile = map_emotions_to_music(emotions)
    prompt = profile["prompt"]

    # 2) Music from prompt (MusicGen)
    music_wav = generate_music(prompt, out_wav="output.wav", duration_s=10)

    # 3) Voice from the SAME paragraph (OpenVoice/Melo)
    voice_wav = synth_openvoice_default(paragraph, out_wav="voice_openvoice.wav", language="EN", speed=1.0)

    # 4) Blend (≈80% voice / 20% music)
    final_wav = duck_and_mix(
        voice_path=voice_wav,
        music_path=music_wav,
        out_wav="final_mix.wav",
        voice_ratio=0.8,
        music_ratio=0.2,
        target_dbfs=-16.0,
        frame_rate=32000,
        channels=2,
    )

    print(f"✅ Done! Final mix: {final_wav}")
