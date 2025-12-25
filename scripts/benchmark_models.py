#!/usr/bin/env python3
"""
scripts/benchmark_models.py

Benchmarks available Voice + Music models (as exposed by your repo registries)
and generates CSV/JSONL + PNG graphs.

Key upgrades vs earlier version:
- Robust imports: adds repo root to sys.path so `musicgenutil` works when run from scripts/
- Music benchmarks run in a subprocess per model (prevents persistent VRAM/RAM accumulation)
- Safe defaults: skips large MusicGen models (medium/large/stereo) unless you allow them
- Optional policies: small_only, cached_only, safe_default, all, or custom list

Run examples:
  python scripts/benchmark_models.py --mode both --duration 8
  python scripts/benchmark_models.py --mode music --music-policy cached_only --duration 8
  python scripts/benchmark_models.py --mode music --music-policy all --allow-large-downloads --duration 8
  python scripts/benchmark_models.py --mode music --music-models musicgen-small,musicgen-medium --allow-large-downloads
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------
# Ensure repo root is on sys.path so imports like `musicgenutil` work
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil
import soundfile as sf

# Optional (audio features)
_LIBROSA_OK = False
try:
    import librosa  # type: ignore

    _LIBROSA_OK = True
except Exception:
    _LIBROSA_OK = False

# Optional (CUDA metrics)
_TORCH_OK = False
try:
    import torch  # type: ignore

    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

# ------------------------------------------------------------
# Project imports (repo-local)
# ------------------------------------------------------------
from musicgenutil import generate_music, get_available_music_models  # type: ignore
from voice_engines import (  # type: ignore
    get_engine_specs,
    list_melo_speakers,
    list_voice_models,
    synthesize_voice,
)

DEFAULT_TEXT = (
    "The storm finally passed, and the city lights returned one by one. "
    "I felt relief, but also a strange quiet sadness, like something important had ended."
)
DEFAULT_PROMPT = "Hopeful classical strings with gentle woodwinds, calm tempo, cinematic and warm."


@dataclass
class RunResult:
    kind: str  # "voice" or "music"
    engine: str  # voice engine key or "musicgen"
    model: str  # model key/identifier
    speaker: str
    device: str

    # input metadata
    text_len: int
    words: int

    # timing
    load_s: float
    gen_s: float
    audio_duration_s: float
    rtf: float

    # resource usage (best-effort)
    rss_mb: float
    gpu_mem_mb: float

    # file/audio details
    sr: int
    channels: int
    file_bytes: int
    peak_dbfs: float
    rms_dbfs: float

    # optional features (0 if unavailable)
    spectral_centroid_hz_mean: float
    spectral_flatness_mean: float
    tempo_bpm_est: float

    # voice-only proxy
    est_wpm: float

    # optional loudness metric (ffmpeg ebur128); empty if unavailable
    lufs_i: str

    out_wav: str
    ok: bool
    error: str


# -----------------------------
# Utility helpers
# -----------------------------
def _now_s() -> float:
    return time.perf_counter()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    keep: List[str] = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:140]


def _dbfs_from_amp(amp: float) -> float:
    amp = max(float(amp), 1e-12)
    return 20.0 * math.log10(amp)


def _audio_stats(path: str) -> Tuple[float, float, int, int, int]:
    """
    Returns: (peak_dbfs, rms_dbfs, sr, channels, frames)
    """
    data, sr = sf.read(path, always_2d=True)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    return _dbfs_from_amp(peak), _dbfs_from_amp(rms), int(sr), int(data.shape[1]), int(data.shape[0])


def _audio_features(path: str) -> Tuple[float, float, float]:
    """
    Returns: (spectral_centroid_hz_mean, spectral_flatness_mean, tempo_bpm_est)
    If librosa unavailable, returns zeros.
    """
    if not _LIBROSA_OK:
        return 0.0, 0.0, 0.0

    y, sr = librosa.load(path, sr=None, mono=True)  # type: ignore
    if y.size == 0:
        return 0.0, 0.0, 0.0

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # type: ignore
    flatness = librosa.feature.spectral_flatness(y=y)  # type: ignore

    centroid_mean = float(np.mean(centroid)) if centroid.size else 0.0
    flatness_mean = float(np.mean(flatness)) if flatness.size else 0.0

    tempo = 0.0
    try:
        tempo_arr = librosa.beat.tempo(y=y, sr=sr)  # type: ignore
        tempo = float(tempo_arr[0]) if len(tempo_arr) else 0.0
    except Exception:
        tempo = 0.0

    return centroid_mean, flatness_mean, tempo


def _ffmpeg_lufs_i(path: str) -> str:
    """
    Best-effort Integrated LUFS (I) extraction via ffmpeg ebur128.
    Returns "" if ffmpeg is missing or parsing fails.
    """
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                path,
                "-filter_complex",
                "ebur128=framelog=verbose",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
        txt = (proc.stderr or "") + "\n" + (proc.stdout or "")
        # Find the last occurrence containing "I:" and "LUFS"
        for line in txt.splitlines()[::-1]:
            if "I:" in line and "LUFS" in line:
                idx = line.find("I:")
                return line[idx:].strip()
        return ""
    except Exception:
        return ""


def _proc_rss_mb(proc: psutil.Process) -> float:
    try:
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _cuda_device_and_reset() -> str:
    if not _TORCH_OK:
        return "cpu"
    try:
        if torch.cuda.is_available():  # type: ignore
            torch.cuda.reset_peak_memory_stats()  # type: ignore
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def _cuda_peak_mb() -> float:
    if not _TORCH_OK:
        return 0.0
    try:
        if torch.cuda.is_available():  # type: ignore
            return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)  # type: ignore
        return 0.0
    except Exception:
        return 0.0


def _write_csv(path: Path, rows: List[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RunResult.__annotations__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def _write_jsonl(path: Path, rows: List[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


# -----------------------------
# Hugging Face cache detection (best-effort)
# -----------------------------
def _hf_hub_cache_dir() -> Path:
    # Hugging Face default cache layout:
    #   ~/.cache/huggingface/hub/models--ORG--NAME/...
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.getenv("HF_HUB_CACHE")).expanduser()
    hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()
    return hf_home / "hub"


def _hf_model_cached(model_id: str) -> bool:
    """
    Checks for existence of models--org--name folder in HF cache.
    This does NOT guarantee every weight file exists, but is a good "do we have it already?" signal.
    """
    model_id = (model_id or "").strip()
    if "/" not in model_id:
        return False
    org, name = model_id.split("/", 1)
    folder = _hf_hub_cache_dir() / f"models--{org}--{name}"
    return folder.exists()


def _is_large_music_key(s: str) -> bool:
    """
    Conservative heuristic: treat these as potentially very large downloads.
    """
    s = (s or "").lower()
    return any(x in s for x in ("stereo", "medium", "large"))


# -----------------------------
# Plotting
# -----------------------------
def _plot_bars(
    out_png: Path,
    title: str,
    labels: List[str],
    values: List[float],
    ylabel: str,
    top_n: Optional[int] = None,
    lower_is_better: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    items = list(zip(labels, values))
    # default: sort high->low, but for "lower is better" sort low->high
    items.sort(key=lambda x: x[1], reverse=(not lower_is_better))

    if top_n is not None and len(items) > top_n:
        items = items[:top_n]

    labels_s = [a for a, _ in items]
    values_s = [b for _, b in items]

    plt.figure(figsize=(max(10, int(len(labels_s) * 0.45)), 6))
    plt.bar(range(len(values_s)), values_s)
    plt.xticks(range(len(labels_s)), labels_s, rotation=60, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _plot_scatter(
    out_png: Path,
    title: str,
    x: List[float],
    y: List[float],
    labels: List[str],
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)

    n = len(labels)
    step = 1 if n <= 30 else max(1, n // 30)
    for i in range(0, n, step):
        plt.annotate(labels[i], (x[i], y[i]), fontsize=8, alpha=0.85)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -----------------------------
# Voice benchmark (in-process)
# -----------------------------
def _bench_voice(out_root: Path, text: str, max_models_per_engine: Optional[int]) -> List[RunResult]:
    proc = psutil.Process(os.getpid())
    device = _cuda_device_and_reset()
    words = len(text.split())

    engines = get_engine_specs()  # only engines that actually work should appear
    results: List[RunResult] = []

    for eng_key in engines.keys():
        models = list_voice_models(eng_key)  # list[(value,label)]
        if max_models_per_engine is not None and len(models) > max_models_per_engine:
            models = models[:max_models_per_engine]

        # Speaker list only for Melo (English-only in your setup)
        speakers: List[str] = [""]
        if eng_key == "melotts":
            try:
                speakers = [""] + list_melo_speakers("EN")
            except Exception:
                speakers = [""]

        if eng_key != "melotts":
            speakers = [""]

        for model_val, _label in models:
            for spk in speakers:
                out_wav = out_root / "voice" / eng_key / _safe_name(str(model_val)) / f"{_safe_name(spk) or 'auto'}.wav"
                out_wav.parent.mkdir(parents=True, exist_ok=True)

                rss_before = _proc_rss_mb(proc)
                if device == "cuda":
                    _ = _cuda_device_and_reset()

                load_s = 0.0
                gen_s = 0.0
                ok = True
                err = ""

                try:
                    t0 = _now_s()
                    info = synthesize_voice(
                        engine=eng_key,
                        text=text,
                        out_wav=str(out_wav),
                        voice_model=str(model_val),
                        speed=1.0,
                        speaker_key=(spk if spk else None),
                    )
                    gen_s = _now_s() - t0
                    if isinstance(info, dict):
                        load_s = _safe_float(info.get("load_s", 0.0), 0.0)
                except Exception as e:
                    ok = False
                    err = repr(e)

                rss_after = _proc_rss_mb(proc)
                rss_mb = max(rss_before, rss_after)
                gpu_mb = _cuda_peak_mb() if device == "cuda" else 0.0

                peak_dbfs = rms_dbfs = 0.0
                sr = ch = frames = 0
                dur_s = 0.0
                file_bytes = 0
                centroid = flatness = tempo = 0.0
                rtf = 0.0
                est_wpm = 0.0
                lufs_i = ""

                if ok and out_wav.exists():
                    try:
                        file_bytes = out_wav.stat().st_size
                        peak_dbfs, rms_dbfs, sr, ch, frames = _audio_stats(str(out_wav))
                        dur_s = (frames / sr) if sr > 0 else 0.0
                        rtf = (gen_s / dur_s) if dur_s > 0 else 0.0
                        centroid, flatness, tempo = _audio_features(str(out_wav))
                        est_wpm = ((words / dur_s) * 60.0) if dur_s > 0 else 0.0
                        lufs_i = _ffmpeg_lufs_i(str(out_wav))
                    except Exception as e:
                        ok = False
                        err = f"postproc_failed: {repr(e)}"

                results.append(
                    RunResult(
                        kind="voice",
                        engine=eng_key,
                        model=str(model_val),
                        speaker=str(spk or ""),
                        device=device,
                        text_len=len(text),
                        words=words,
                        load_s=float(load_s),
                        gen_s=float(gen_s),
                        audio_duration_s=float(dur_s),
                        rtf=float(rtf),
                        rss_mb=float(rss_mb),
                        gpu_mem_mb=float(gpu_mb),
                        sr=int(sr),
                        channels=int(ch),
                        file_bytes=int(file_bytes),
                        peak_dbfs=float(peak_dbfs),
                        rms_dbfs=float(rms_dbfs),
                        spectral_centroid_hz_mean=float(centroid),
                        spectral_flatness_mean=float(flatness),
                        tempo_bpm_est=float(tempo),
                        est_wpm=float(est_wpm),
                        lufs_i=str(lufs_i),
                        out_wav=str(out_wav),
                        ok=bool(ok),
                        error=str(err),
                    )
                )

    return results


# -----------------------------
# Music benchmark (subprocess per model)
# -----------------------------
def _call_generate_music_compat(model_key: str, prompt: str, out_wav: str, duration_s: int, seed: int) -> Dict[str, Any]:
    """
    Child-process call: run your repo's generate_music with best-effort signature compatibility.
    Returns dict with keys: ok, load_s, error
    """
    load_s = 0.0
    try:
        sig = inspect.signature(generate_music)
        kwargs: Dict[str, Any] = {}
        for name in sig.parameters.keys():
            if name == "prompt":
                kwargs[name] = prompt
            elif name in ("out_wav", "out_path", "output_wav", "path"):
                kwargs[name] = out_wav
            elif name in ("duration_s", "duration", "seconds"):
                kwargs[name] = int(duration_s)
            elif name in ("model_name", "model", "model_id"):
                kwargs[name] = str(model_key)
            elif name == "seed":
                kwargs[name] = int(seed)

        info = generate_music(**kwargs)  # type: ignore[arg-type]
        if isinstance(info, dict):
            load_s = _safe_float(info.get("load_s", 0.0), 0.0)
        return {"ok": True, "load_s": load_s, "error": ""}
    except Exception as e:
        return {"ok": False, "load_s": 0.0, "error": repr(e)}


def _child_music_run(args: argparse.Namespace) -> int:
    """
    Internal child mode: generates one music wav and prints JSON to stdout.
    """
    res = _call_generate_music_compat(
        model_key=args.child_music_model,
        prompt=args.child_prompt,
        out_wav=args.child_out_wav,
        duration_s=int(args.child_duration),
        seed=int(args.child_seed),
    )
    print(json.dumps(res))
    return 0 if res.get("ok") else 2


def _bench_music_subprocess(
    out_root: Path,
    prompt: str,
    duration_s: int,
    seed: int,
    music_policy: str,
    allow_large_downloads: bool,
    custom_music_models: Optional[List[str]],
) -> List[RunResult]:
    """
    Parent mode: runs each model generation in a fresh subprocess so a big model load
    doesn't permanently occupy memory, and an OOM kill won't terminate the whole benchmark.
    """
    proc = psutil.Process(os.getpid())
    device = _cuda_device_and_reset()
    words = len(prompt.split())
    results: List[RunResult] = []

    music_models = get_available_music_models()  # dict model_key -> hf_id/desc

    # Decide which keys to benchmark
    all_keys = list(music_models.keys())

    if music_policy == "custom":
        keys = custom_music_models or []
    else:
        keys = all_keys

    selected: List[str] = []
    for k in keys:
        # For safety, use both key and mapped HF id/desc for heuristics
        mapped = str(music_models.get(k, ""))

        is_large = _is_large_music_key(k) or _is_large_music_key(mapped)
        is_small = ("small" in k.lower()) or ("small" in mapped.lower())
        cached = _hf_model_cached(mapped) if "/" in mapped else _hf_model_cached(k)

        if music_policy == "all":
            if is_large and not allow_large_downloads:
                continue
            selected.append(k)
        elif music_policy == "small_only":
            if is_small:
                selected.append(k)
        elif music_policy == "cached_only":
            if cached:
                selected.append(k)
        elif music_policy == "safe_default":
            # Run non-large models OR anything already cached.
            if (not is_large) or cached:
                if is_large and not allow_large_downloads and not cached:
                    continue
                selected.append(k)
        elif music_policy == "custom":
            # Custom list respects allow_large_downloads (unless cached)
            if is_large and not allow_large_downloads and not cached:
                continue
            selected.append(k)

    # Benchmark each selected model in a subprocess
    script_path = Path(__file__).resolve()
    python_exe = sys.executable

    for model_key in selected:
        out_wav = out_root / "music" / _safe_name(str(model_key)) / "music.wav"
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        rss_before = _proc_rss_mb(proc)
        if device == "cuda":
            _ = _cuda_device_and_reset()

        t0 = _now_s()
        load_s = 0.0
        ok = True
        err = ""

        # Spawn child that does ONLY the generate_music call
        cmd = [
            python_exe,
            str(script_path),
            "--_child-music-run",
            "--child-music-model",
            str(model_key),
            "--child-prompt",
            prompt,
            "--child-duration",
            str(duration_s),
            "--child-seed",
            str(seed),
            "--child-out-wav",
            str(out_wav),
        ]

        cp = subprocess.run(cmd, capture_output=True, text=True)
        gen_s = _now_s() - t0

        if cp.returncode != 0:
            ok = False
            # If the process was OOM-killed, it often returns 137 (SIGKILL)
            err = f"child_failed rc={cp.returncode} stderr_tail={cp.stderr[-4000:]}"
        else:
            # Child prints one JSON line
            try:
                res = json.loads((cp.stdout or "").strip().splitlines()[-1])
                ok = bool(res.get("ok", False))
                load_s = _safe_float(res.get("load_s", 0.0), 0.0)
                if not ok:
                    err = str(res.get("error", "child_error"))
            except Exception as e:
                ok = False
                err = f"child_parse_failed: {repr(e)} stdout_tail={cp.stdout[-2000:]} stderr_tail={cp.stderr[-2000:]}"

        rss_after = _proc_rss_mb(proc)
        rss_mb = max(rss_before, rss_after)
        gpu_mb = _cuda_peak_mb() if device == "cuda" else 0.0

        peak_dbfs = rms_dbfs = 0.0
        sr = ch = frames = 0
        dur_s = 0.0
        file_bytes = 0
        centroid = flatness = tempo = 0.0
        rtf = 0.0
        lufs_i = ""

        if ok and out_wav.exists():
            try:
                file_bytes = out_wav.stat().st_size
                peak_dbfs, rms_dbfs, sr, ch, frames = _audio_stats(str(out_wav))
                dur_s = (frames / sr) if sr > 0 else 0.0
                rtf = (gen_s / dur_s) if dur_s > 0 else 0.0
                centroid, flatness, tempo = _audio_features(str(out_wav))
                lufs_i = _ffmpeg_lufs_i(str(out_wav))
            except Exception as e:
                ok = False
                err = f"postproc_failed: {repr(e)}"

        results.append(
            RunResult(
                kind="music",
                engine="musicgen",
                model=str(model_key),
                speaker="",
                device=device,
                text_len=len(prompt),
                words=words,
                load_s=float(load_s),
                gen_s=float(gen_s),
                audio_duration_s=float(dur_s),
                rtf=float(rtf),
                rss_mb=float(rss_mb),
                gpu_mem_mb=float(gpu_mb),
                sr=int(sr),
                channels=int(ch),
                file_bytes=int(file_bytes),
                peak_dbfs=float(peak_dbfs),
                rms_dbfs=float(rms_dbfs),
                spectral_centroid_hz_mean=float(centroid),
                spectral_flatness_mean=float(flatness),
                tempo_bpm_est=float(tempo),
                est_wpm=0.0,
                lufs_i=str(lufs_i),
                out_wav=str(out_wav),
                ok=bool(ok),
                error=str(err),
            )
        )

    return results


# -----------------------------
# Graph generation
# -----------------------------
def _make_graphs(out_dir: Path, rows: List[RunResult]) -> None:
    ok_rows = [r for r in rows if r.ok and r.audio_duration_s > 0.0]

    voice = [r for r in ok_rows if r.kind == "voice"]
    music = [r for r in ok_rows if r.kind == "music"]

    if voice:
        labels = []
        for r in voice:
            m = Path(r.model).name if r.engine == "piper" else r.model
            lab = f"{r.engine}:{m}"
            if r.speaker:
                lab += f":{r.speaker}"
            labels.append(lab)

        _plot_bars(out_dir / "voice_gen_time_s.png", "Voice: Generation Time (s)", labels, [r.gen_s for r in voice], "seconds", top_n=40)
        _plot_bars(out_dir / "voice_rtf.png", "Voice: Real-Time Factor (gen_s / audio_s)", labels, [r.rtf for r in voice], "RTF (lower is better)", top_n=40, lower_is_better=True)
        _plot_bars(out_dir / "voice_rss_mb.png", "Voice: Peak RSS (MB)", labels, [r.rss_mb for r in voice], "MB", top_n=40)
        _plot_scatter(
            out_dir / "voice_time_vs_wpm.png",
            "Voice: Generation Time vs Estimated WPM",
            [r.gen_s for r in voice],
            [r.est_wpm for r in voice],
            labels,
            "gen_s",
            "estimated WPM",
        )

    if music:
        labels = [r.model for r in music]
        _plot_bars(out_dir / "music_gen_time_s.png", "Music: Generation Time (s)", labels, [r.gen_s for r in music], "seconds")
        _plot_bars(out_dir / "music_rtf.png", "Music: Real-Time Factor (gen_s / audio_s)", labels, [r.rtf for r in music], "RTF (lower is better)", lower_is_better=True)
        _plot_bars(out_dir / "music_rss_mb.png", "Music: Peak RSS (MB)", labels, [r.rss_mb for r in music], "MB")
        _plot_scatter(
            out_dir / "music_centroid_vs_flatness.png",
            "Music: Spectral Centroid vs Flatness",
            [r.spectral_centroid_hz_mean for r in music],
            [r.spectral_flatness_mean for r in music],
            labels,
            "centroid (Hz, mean)",
            "flatness (mean)",
        )

    if ok_rows:
        labels = [f"{r.kind}:{r.engine}:{Path(r.model).name if r.engine=='piper' else r.model}" for r in ok_rows]
        _plot_bars(out_dir / "all_gen_time_s.png", "All: Generation Time (s)", labels, [r.gen_s for r in ok_rows], "seconds", top_n=50)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()

    # Public args
    ap.add_argument("--mode", choices=["voice", "music", "both"], default="both")
    ap.add_argument("--out", default="bench", help="Output directory for WAVs, CSV/JSONL, and graphs")
    ap.add_argument("--duration", type=int, default=8, help="Music duration seconds")
    ap.add_argument("--seed", type=int, default=42, help="Seed used for music generation (if supported)")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="Text used for voice benchmarks")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for music benchmarks")
    ap.add_argument("--max-voice-models", type=int, default=None, help="Optional cap per voice engine")

    # Music selection policy to avoid huge downloads by default
    ap.add_argument(
        "--music-policy",
        choices=["safe_default", "small_only", "cached_only", "all", "custom"],
        default="safe_default",
        help=(
            "safe_default: skip large (medium/large/stereo) unless cached; "
            "small_only: only 'small'; cached_only: only models already in HF cache; "
            "all: all models; custom: use --music-models list."
        ),
    )
    ap.add_argument(
        "--music-models",
        default="",
        help="Comma-separated music model keys (used when --music-policy custom). Example: musicgen-small,musicgen-melody",
    )
    ap.add_argument(
        "--allow-large-downloads",
        action="store_true",
        help="Allow benchmarking large models that may trigger multi-GB downloads.",
    )

    # Internal child mode args (do not use directly)
    ap.add_argument("--_child-music-run", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--child-music-model", default="", help=argparse.SUPPRESS)
    ap.add_argument("--child-prompt", default="", help=argparse.SUPPRESS)
    ap.add_argument("--child-duration", default="8", help=argparse.SUPPRESS)
    ap.add_argument("--child-seed", default="42", help=argparse.SUPPRESS)
    ap.add_argument("--child-out-wav", default="", help=argparse.SUPPRESS)

    args = ap.parse_args()

    # Child mode
    if args._child_music_run:
        return _child_music_run(args)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    custom_music_models: Optional[List[str]] = None
    if args.music_policy == "custom":
        custom_music_models = [m.strip() for m in (args.music_models or "").split(",") if m.strip()]

    rows: List[RunResult] = []

    if args.mode in ("voice", "both"):
        rows.extend(_bench_voice(out_root=out_dir, text=args.text, max_models_per_engine=args.max_voice_models))

    if args.mode in ("music", "both"):
        rows.extend(
            _bench_music_subprocess(
                out_root=out_dir,
                prompt=args.prompt,
                duration_s=args.duration,
                seed=args.seed,
                music_policy=args.music_policy,
                allow_large_downloads=bool(args.allow_large_downloads),
                custom_music_models=custom_music_models,
            )
        )

    _write_csv(out_dir / "bench_results.csv", rows)
    _write_jsonl(out_dir / "bench_results.jsonl", rows)
    _make_graphs(out_dir, rows)

    ok = sum(1 for r in rows if r.ok)
    fail = len(rows) - ok

    print(f"[benchmark] done. rows={len(rows)} ok={ok} fail={fail}")
    print(f"[benchmark] results: {out_dir / 'bench_results.csv'}")
    print(f"[benchmark] graphs:  {out_dir}")

    if fail:
        print("\nFailures:")
        for r in rows:
            if not r.ok:
                spk = f" spk={r.speaker}" if r.speaker else ""
                print(f"- {r.kind} {r.engine} model={r.model}{spk}: {r.error}")

    print("\nTip: to avoid huge downloads, use:")
    print("  --music-policy cached_only")
    print("or:")
    print("  --music-policy small_only")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
