# voice_engines.py
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shutil import which


# ============================================================
# Optional Python dependencies (never crash app on import)
# ============================================================
_MELO_OK = False
_MELO_ERR: Optional[Exception] = None
try:
    # MyShell MeloTTS (provides: from melo.api import TTS)
    from melo.api import TTS as MeloTTS  # type: ignore
    import torch  # type: ignore

    _MELO_OK = True
except Exception as e:  # pragma: no cover
    _MELO_ERR = e

_COQUI_OK = False
_COQUI_ERR: Optional[Exception] = None
try:
    # Coqui TTS (provides: from TTS.api import TTS)
    from TTS.api import TTS as CoquiTTS  # type: ignore

    _COQUI_OK = True
except Exception as e:  # pragma: no cover
    _COQUI_ERR = e


# ============================================================
# Piper detection (cross-platform)
# ============================================================
def _piper_bin() -> Optional[str]:
    """
    Return an executable path for Piper if available, else None.
    Supports:
      - env var PIPER_BIN=/absolute/path/to/piper(.exe)
      - PATH lookup for "piper" and "piper.exe"
    """
    override = (os.getenv("PIPER_BIN") or "").strip()
    if override:
        override = os.path.expanduser(override)
        if os.path.exists(override) and os.access(override, os.X_OK):
            return override

    return which("piper") or which("piper.exe")


def _piper_available() -> bool:
    return _piper_bin() is not None


def _piper_voices_dir() -> Path:
    """
    Where Piper voices (.onnx) are stored for dropdown-only selection.
    - If PIPER_VOICES_DIR is set, use that.
    - Else default: <project_root>/voices/piper (where project_root is this file's directory)
    """
    override = (os.getenv("PIPER_VOICES_DIR") or "").strip()
    if override:
        return Path(os.path.expanduser(override)).resolve()

    base = Path(__file__).resolve().parent
    return (base / "voices" / "piper").resolve()


# ============================================================
# Engine specs (used by UI)
# ============================================================
@dataclass(frozen=True)
class EngineSpec:
    key: str
    label: str
    supports_speaker: bool = False
    supports_speed: bool = False


def get_engine_specs() -> Dict[str, EngineSpec]:
    """
    Only return engines that are actually runnable in the current environment.
    This keeps the UI dropdown-only and prevents selection of non-working engines.
    """
    specs: Dict[str, EngineSpec] = {}

    if _MELO_OK:
        specs["melotts"] = EngineSpec(
            key="melotts",
            label="MeloTTS (English only)",
            supports_speaker=True,
            supports_speed=True,
        )

    if _COQUI_OK:
        specs["coqui"] = EngineSpec(
            key="coqui",
            label="Coqui TTS (English model catalog)",
            supports_speaker=False,
            supports_speed=False,
        )

    if _piper_available():
        specs["piper"] = EngineSpec(
            key="piper",
            label="Piper (CLI, offline, ONNX voices)",
            supports_speaker=False,
            supports_speed=False,
        )

    return specs


# ============================================================
# Dropdown-only model catalogs
# ============================================================
# Melo: keep English-only to avoid language mapping asserts.
_MELO_LANG_MODELS: List[Tuple[str, str]] = [("EN", "English (EN)")]

# Cache: engine -> list[(value, label)]
_VOICE_MODEL_CACHE: Dict[str, List[Tuple[str, str]]] = {}
# Cache: Melo speakers (EN only)
_MELO_SPK_CACHE: Optional[List[str]] = None

# Coqui model list can be large; cache filtered English subset once.
_COQUI_EN_MODELS_CACHE: Optional[List[str]] = None


def list_voice_models(engine: str) -> List[Tuple[str, str]]:
    """
    Returns list of (value, label) for dropdown-only UI.

    - melotts: [("EN", "English (EN)")]
    - coqui:   all models starting with "tts_models/en/"
    - piper:   all local .onnx voices found in PIPER_VOICES_DIR (or ./voices/piper)
              value is absolute path to .onnx (posted back by the form)
    """
    engine = (engine or "").strip()

    if engine in _VOICE_MODEL_CACHE:
        return _VOICE_MODEL_CACHE[engine]

    models: List[Tuple[str, str]] = []
    if engine == "melotts":
        models = list(_MELO_LANG_MODELS)

    elif engine == "coqui":
        models = _list_coqui_english_models()

    elif engine == "piper":
        models = _list_piper_models()

    _VOICE_MODEL_CACHE[engine] = models
    return models


def _list_coqui_english_models() -> List[Tuple[str, str]]:
    if not _COQUI_OK:
        return []

    global _COQUI_EN_MODELS_CACHE
    if _COQUI_EN_MODELS_CACHE is None:
        try:
            all_models = CoquiTTS.list_models()  # type: ignore
            en = [m for m in all_models if isinstance(m, str) and m.startswith("tts_models/en/")]
            _COQUI_EN_MODELS_CACHE = sorted(en)
        except Exception:
            _COQUI_EN_MODELS_CACHE = []

    return [(m, m) for m in _COQUI_EN_MODELS_CACHE]


def _list_piper_models() -> List[Tuple[str, str]]:
    voices_dir = _piper_voices_dir()
    out: List[Tuple[str, str]] = []

    if voices_dir.exists():
        for p in sorted(voices_dir.glob("*.onnx")):
            # dropdown "value" is absolute path; label includes filename only
            out.append((str(p.resolve()), p.name))
    return out


# ============================================================
# Melo speakers (English-only)
# ============================================================
def list_melo_speakers(language: str = "EN") -> List[str]:
    """
    Melo speakers for dropdown.
    English-only. Never throws.
    """
    global _MELO_SPK_CACHE
    if _MELO_SPK_CACHE is not None:
        return _MELO_SPK_CACHE

    if not _MELO_OK:
        _MELO_SPK_CACHE = ["EN-US", "EN-UK", "EN-LIBRITTS"]
        return _MELO_SPK_CACHE

    try:
        tts, _, _ = _melo_get_tts()
        spks = sorted(list(tts.hps.data.spk2id.keys()))  # type: ignore[attr-defined]
        _MELO_SPK_CACHE = spks or ["EN-US", "EN-UK", "EN-LIBRITTS"]
    except Exception:
        _MELO_SPK_CACHE = ["EN-US", "EN-UK", "EN-LIBRITTS"]

    return _MELO_SPK_CACHE


# ============================================================
# Synthesis implementations
# ============================================================
_MELO_CACHE: Dict[str, "MeloTTS"] = {}
_COQUI_CACHE: Dict[str, "CoquiTTS"] = {}


def _pick_device() -> str:
    if not _MELO_OK:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[name-defined]
    except Exception:
        return "cpu"


def _melo_get_tts() -> Tuple["MeloTTS", bool, float]:
    if not _MELO_OK:
        raise ModuleNotFoundError(
            "MeloTTS is not installed. Install with:\n"
            "  pip install -U git+https://github.com/myshell-ai/MeloTTS.git\n"
            f"Import error: {_MELO_ERR}"
        )

    dev = _pick_device()
    cache_key = f"EN::{dev}"
    if cache_key in _MELO_CACHE:
        return _MELO_CACHE[cache_key], True, 0.0

    t0 = time.perf_counter()
    tts = MeloTTS(language="EN", device=dev)  # type: ignore
    dt = time.perf_counter() - t0

    _MELO_CACHE[cache_key] = tts
    return tts, False, dt


def _melo_synthesize(
    text: str,
    out_wav: str,
    speed: float,
    speaker_key: Optional[str],
) -> Dict[str, object]:
    tts, from_cache, load_s = _melo_get_tts()
    spk2id = getattr(tts.hps.data, "spk2id", {})  # type: ignore[attr-defined]

    if spk2id:
        if not speaker_key:
            speaker_key = next(iter(spk2id.keys()))
        if speaker_key not in spk2id:
            raise ValueError(f"Unknown speaker_key '{speaker_key}'.")
        speaker_id = int(spk2id[speaker_key])
    else:
        speaker_key = speaker_key or "default"
        speaker_id = 0

    t0 = time.perf_counter()
    tts.tts_to_file(text, speaker_id, out_wav, speed=float(speed))  # type: ignore
    tts_s = time.perf_counter() - t0

    return {
        "engine": "melotts",
        "voice_model": "EN",
        "speaker_key": speaker_key,
        "speaker_id": speaker_id,
        "from_cache": from_cache,
        "load_s": float(load_s),
        "tts_s": float(tts_s),
        "path": os.path.abspath(out_wav),
    }


def _coqui_get_tts(model_name: str) -> Tuple["CoquiTTS", bool, float]:
    if not _COQUI_OK:
        raise ModuleNotFoundError(
            "Coqui TTS is not installed. Install with:\n"
            "  pip install -U TTS\n"
            f"Import error: {_COQUI_ERR}"
        )

    if model_name in _COQUI_CACHE:
        return _COQUI_CACHE[model_name], True, 0.0

    t0 = time.perf_counter()
    tts = CoquiTTS(model_name)  # type: ignore
    dt = time.perf_counter() - t0

    _COQUI_CACHE[model_name] = tts
    return tts, False, dt


def _coqui_synthesize(text: str, out_wav: str, model_name: str) -> Dict[str, object]:
    # Enforce dropdown-only expectation (English models only)
    if not model_name.startswith("tts_models/en/"):
        raise ValueError("Configured for English-only Coqui models (tts_models/en/*).")

    tts, from_cache, load_s = _coqui_get_tts(model_name)

    t0 = time.perf_counter()
    tts.tts_to_file(text=text, file_path=out_wav)  # type: ignore
    tts_s = time.perf_counter() - t0

    return {
        "engine": "coqui",
        "voice_model": model_name,
        "speaker_key": "",
        "speaker_id": -1,
        "from_cache": from_cache,
        "load_s": float(load_s),
        "tts_s": float(tts_s),
        "path": os.path.abspath(out_wav),
    }


def _piper_synthesize(text: str, out_wav: str, model_path: str) -> Dict[str, object]:
    piper = _piper_bin()
    if not piper:
        raise RuntimeError(
            "Piper is not available. Ensure the Piper executable is in PATH "
            "or set PIPER_BIN to its absolute path."
        )

    model_path = os.path.expanduser(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Piper model not found: {model_path}\n"
            f"Place .onnx voices into: {str(_piper_voices_dir())}"
        )

    t0 = time.perf_counter()
    proc = subprocess.run(
        [piper, "--model", model_path, "--output_file", out_wav],
        input=text,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(msg or "Piper failed.")
    tts_s = time.perf_counter() - t0

    return {
        "engine": "piper",
        "voice_model": os.path.abspath(model_path),
        "speaker_key": "",
        "speaker_id": -1,
        "from_cache": True,  # CLI; no Python model cache
        "load_s": 0.0,
        "tts_s": float(tts_s),
        "path": os.path.abspath(out_wav),
    }


# ============================================================
# Public API used by app.py
# ============================================================
def synthesize_voice(
    engine: str,
    text: str,
    out_wav: str,
    voice_model: str,
    speed: float = 1.0,
    speaker_key: Optional[str] = None,
) -> Dict[str, object]:
    """
    engine:
      - melotts: voice_model ignored (EN-only); speaker_key optional; speed supported
      - coqui:   voice_model must be a dropdown value (tts_models/en/*)
      - piper:   voice_model must be a dropdown value (absolute path to .onnx)
    """
    engine = (engine or "").strip()

    if engine == "melotts":
        return _melo_synthesize(text=text, out_wav=out_wav, speed=speed, speaker_key=speaker_key)

    if engine == "coqui":
        return _coqui_synthesize(text=text, out_wav=out_wav, model_name=voice_model)

    if engine == "piper":
        return _piper_synthesize(text=text, out_wav=out_wav, model_path=voice_model)

    raise ValueError(f"Unknown voice engine '{engine}'. Available: {list(get_engine_specs().keys())}")
