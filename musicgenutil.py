# musicgenutil.py
import os

# -----------------------------
# MINIMAL PERF CHANGE: threads
# -----------------------------
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


_default_threads = min(8, (os.cpu_count() or 4))
_threads = max(1, _int_env("MUSIC_THREADS", _default_threads))

# If the user already set these, respect them; otherwise set to _threads.
os.environ.setdefault("OMP_NUM_THREADS", str(_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_threads))

# Avoid optional xformers dependency when possible; CPU inference is fine.
os.environ.setdefault("AUDIOCRAFT_DISABLE_XFORMERS", "1")

import torch
from audiocraft.models import MusicGen
import torchaudio

# MINIMAL PERF CHANGE: do not force 1 thread; use configured value
try:
    torch.set_num_threads(_threads)
    torch.set_num_interop_threads(max(1, _threads // 2))
except Exception:
    # Some torch builds may not expose these setters.
    pass

_MODEL = None  # cached MusicGen wrapper


def _select_model_id() -> str:
    """Pick a higher-fidelity model on GPU, fall back to the small CPU build."""
    return os.environ.get(
        "MUSICGEN_MODEL",
        "facebook/musicgen-stereo-medium" if torch.cuda.is_available() else "facebook/musicgen-small",
    )


def _load_model():
    """Load and cache a MusicGen model tuned for better texture/clarity."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = _select_model_id()

    # Use full id to avoid deprecation warning
    _MODEL = MusicGen.get_pretrained(model_id, device=device)

    # Slightly richer sampling defaults than upstream to avoid flat/metallic output
    _MODEL.set_generation_params(
        duration=6,       # seconds (overridden per call)
        use_sampling=True,
        top_k=250,
        top_p=0.92,       # nudge toward more variety without drifting off-prompt
        temperature=1.08,
        cfg_coef=4.0,     # keep prompts influential so emotion mapping shines through
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

    # Prefer stereo for a wider bed when mixing beneath narration
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)

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
