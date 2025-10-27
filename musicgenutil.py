# musicgenutil.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from audiocraft.models import MusicGen
import torchaudio

# Keep CPU memory/threads small (WSL friendly)
torch.set_num_threads(1)

_MODEL = None  # cached MusicGen wrapper


def _load_model():
    """Load and cache the small CPU MusicGen model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # Use full id to avoid deprecation warning
    _MODEL = MusicGen.get_pretrained("facebook/musicgen-small", device="cpu")

    # Safe defaults; tweak later as needed
    _MODEL.set_generation_params(
        duration=6,       # seconds (overridden per call)
        use_sampling=True,
        top_k=250,
        top_p=0.0,
        temperature=1.0,
        cfg_coef=3.0,
    )
    return _MODEL  # NOTE: MusicGen is not an nn.Module, no .eval()


def generate_music(prompt: str, out_wav: str = "output.wav", duration_s: int = 6, seed: int | None = None) -> str:
    """
    Generate music from a natural-language prompt using MusicGen (small, CPU).
    Saves a 32 kHz WAV and returns its absolute path.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    model = _load_model()

    # Per-call duration without rebuilding
    model.set_generation_params(duration=max(1, int(duration_s)))

    # Seeding: no generator kwarg on generate(); set global seed instead
    if seed is not None:
        torch.manual_seed(int(seed))

    with torch.inference_mode():
        # Returns List[Tensor[C, T]] (C usually 1)
        wav_list = model.generate(descriptions=[prompt], progress=False)

    wav = wav_list[0].cpu()  # [C, T]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # ensure [C, T]

    torchaudio.save(out_wav, wav, sample_rate=32000)

    abs_path = os.path.abspath(out_wav)
    print(f"🎵 Music saved to {abs_path}")
    return abs_path


if __name__ == "__main__":
    generate_music(
        "cinematic hopeful orchestral score with strings and light percussion",
        out_wav="output.wav",
        duration_s=4,
        seed=42,
    )
