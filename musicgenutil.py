# musicgen_util.py
from audiocraft.models import MusicGen
import soundfile as sf

def generate_music(prompt: str, out_wav: str = "output.wav", duration_s: int = 10) -> str:
    """
    Generate music from a natural-language prompt using MusicGen.
    Saves a 32kHz WAV and returns its absolute path.
    """
    model = MusicGen.get_pretrained("facebook/musicgen-melody")  # full ID
    model.set_generation_params(duration=duration_s)
    wav = model.generate([prompt])  # [B, C, T]

    audio = wav[0].cpu().numpy()  # (C, T)
    if audio.ndim == 2:           # to (T, C) for soundfile
        audio = audio.T
    sf.write(out_wav, audio, samplerate=32000)

    import os
    path = os.path.abspath(out_wav)
    print(f"🎵 Music saved to {path}")
    return path
