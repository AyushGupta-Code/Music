# run_music.py
from para_to_emo import detect_emotion
from map_emo_to_music import map_emotions_to_music
from musicgen_util import generate_music
from generate_voice import synth_openvoice_default, duck_and_mix

if __name__ == "__main__":
    paragraph = """The battlefield roared like an angry god as steel clashed against steel, and the ground trembled under the relentless thunder of charging warhorses. Black smoke curled into the blood-red sky, carrying the stench of iron, fire, and death. Through the chaos, Commander Varos carved a path with his greatsword, each swing a devastating arc that sent enemy soldiers sprawling. Arrows hissed past his face, splintering against his armor, but he did not falter; his eyes burned with the unyielding fury of a man who refused to yield even an inch of ground. The screams of the wounded mingled with the deafening war drums, and the once-green fields were now slick with mud and crimson. Above it all, the enemy’s siege towers loomed ever closer, their monstrous silhouettes blotting out the horizon. Varos raised his sword and bellowed an order, his voice cutting through the din like lightning through a storm. The battered defenders rallied to him, their shields locking, their spears lowering, ready to meet the oncoming tide. The air was thick with dust and despair, but in that moment—surrounded, outnumbered, and pressed against the edge of annihilation—Varos felt the fire in his veins blaze hotter than ever."""

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
