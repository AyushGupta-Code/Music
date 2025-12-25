# musicgenutil.py
from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

# Keep CPU thread usage modest (WSL-friendly)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import soundfile as sf

try:
    from audiocraft.models import MusicGen
except Exception as e:  # pragma: no cover
    MusicGen = None  # type: ignore
    _AUDIOCRAFT_IMPORT_ERROR = e

torch.set_num_threads(1)

# Friendly name -> HF/audiocraft model id
DEFAULT_MUSIC_MODELS: Dict[str, str] = {
    # Core MusicGen (AudioCraft docs list these) :contentReference[oaicite:2]{index=2}
    "musicgen-small": "facebook/musicgen-small",
    "musicgen-medium": "facebook/musicgen-medium",
    "musicgen-large": "facebook/musicgen-large",
    "musicgen-melody": "facebook/musicgen-melody",
    "musicgen-melody-large": "facebook/musicgen-melody-large",

    # Style conditioning model (also official) :contentReference[oaicite:3]{index=3}
    "musicgen-style": "facebook/musicgen-style",

    # Stereo variants visible on HF under facebook/musicgen* :contentReference[oaicite:4]{index=4}
    "musicgen-stereo-small": "facebook/musicgen-stereo-small",
    "musicgen-stereo-medium": "facebook/musicgen-stereo-medium",
    "musicgen-stereo-large": "facebook/musicgen-stereo-large",
}


# Cache per (resolved_model_id, device_str)
_MODEL_CACHE: Dict[Tuple[str, str], "MusicGen"] = {}


def get_available_music_models() -> Dict[str, str]:
    """Return dict: friendly_name -> model_id."""
    return dict(DEFAULT_MUSIC_MODELS)


def _resolve_model_id(model_name: str) -> str:
    # friendly key -> model id
    if model_name in DEFAULT_MUSIC_MODELS:
        return DEFAULT_MUSIC_MODELS[model_name]
    # otherwise assume already a model id
    return model_name


def _pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_name: str, device: Optional[str] = None) -> Tuple["MusicGen", bool, str, str, float]:
    """
    Returns: (model, from_cache, resolved_model_id, device_str, load_seconds)

    Important:
      AudioCraft's MusicGen.get_pretrained(name, device=...) loads submodules onto device internally.
      Do NOT call model.to(device) here (some versions don't expose .to()).
    """
    model_id = _resolve_model_id(model_name)
    dev = _pick_device(device)

    cache_key = (model_id, dev)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], True, model_id, dev, 0.0

    if MusicGen is None:
        raise ImportError(
            "audiocraft is not installed or failed to import. Install it (and deps) to use MusicGen."
        ) from _AUDIOCRAFT_IMPORT_ERROR

    t0 = time.perf_counter()

    # Robust candidate list
    candidates = []
    candidates.append(model_id)
    if model_name != model_id:
        candidates.append(model_name)
    if "/" not in model_name:
        candidates.append(f"facebook/{model_name}")

    last_err: Optional[Exception] = None
    model: Optional["MusicGen"] = None

    for cand in candidates:
        try:
            # Official API supports device arg. :contentReference[oaicite:1]{index=1}
            model = MusicGen.get_pretrained(cand, device=dev)
            break
        except TypeError:
            # Very old audiocraft might not accept device kwarg; try without it.
            try:
                model = MusicGen.get_pretrained(cand)
                break
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e

    if model is None:
        raise RuntimeError(f"Failed to load MusicGen model '{model_name}' (resolved='{model_id}'): {last_err}")

    # Set default generation params once at load
    model.set_generation_params(use_sampling=True, top_k=250, temperature=1.0)

    _MODEL_CACHE[cache_key] = model
    dt = time.perf_counter() - t0
    return model, False, model_id, dev, dt


def generate_music(
    prompt: str,
    out_wav: str = "output_music.wav",
    duration_s: int = 8,
    seed: Optional[int] = None,
    model_name: str = "musicgen-small",
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Generates background music from prompt.

    Returns dict:
      {
        "path": abs_path,
        "resolved_model_id": "...",
        "device": "cpu|cuda|mps",
        "from_cache": bool,
        "model_load_s": float,
        "gen_s": float,
        "sample_rate": int
      }
    """
    if seed is not None:
        torch.manual_seed(int(seed))

    model, from_cache, resolved_model_id, dev, load_s = _load_model(model_name=model_name, device=device)

    duration_s = int(duration_s)
    model.set_generation_params(duration=duration_s)

    t0 = time.perf_counter()
    with torch.no_grad():
        wav_list = model.generate(descriptions=[prompt], progress=False)
    gen_s = time.perf_counter() - t0

    wav = wav_list[0].cpu()  # Tensor [C, T]
    sample_rate = int(getattr(model, "sample_rate", 32000))

    # Ensure shape is valid for soundfile write
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    sf.write(out_wav, wav.squeeze(0).numpy().T, sample_rate)

    abs_path = os.path.abspath(out_wav)
    return {
        "path": abs_path,
        "resolved_model_id": resolved_model_id,
        "device": dev,
        "from_cache": from_cache,
        "model_load_s": float(load_s),
        "gen_s": float(gen_s),
        "sample_rate": sample_rate,
    }
