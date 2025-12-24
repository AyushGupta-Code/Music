# app.py
from __future__ import annotations

import tempfile
import time
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Flask, render_template, request, send_file, abort

from metrics import Metrics, StageTimer, append_metrics_csv, append_metrics_jsonl, new_request_id
from para_to_emo import detect_emotion
from map_emo_to_music import map_emotions_to_music
from musicgenutil import generate_music, get_available_music_models
from generate_voice import synth_openvoice_default, duck_and_mix, list_openvoice_speakers

app = Flask(__name__)

DEFAULT_PARAGRAPH = "Paste a paragraph here and click Generate."

# In-memory cache for recent results
# request_id -> {"audio_bytes": bytes, "metrics": dict, "created_at": float}
_RESULT_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_MAX_CACHE_ITEMS = 20


def _cache_put(request_id: str, audio_bytes: bytes, metrics_payload: Dict[str, Any]) -> None:
    _RESULT_CACHE[request_id] = {
        "audio_bytes": audio_bytes,
        "metrics": metrics_payload,
        "created_at": time.time(),
    }
    # trim oldest
    while len(_RESULT_CACHE) > _MAX_CACHE_ITEMS:
        _RESULT_CACHE.popitem(last=False)


def _cache_get(request_id: str) -> Dict[str, Any]:
    item = _RESULT_CACHE.get(request_id)
    if not item:
        raise KeyError(request_id)
    return item


def run_pipeline(
    request_id: str,
    paragraph: str,
    duration_s: int,
    voice_language: str,
    voice_speed: float,
    voice_speaker: Optional[str],
    music_model: str,
    seed: Optional[int],
    voice_ratio: float,
    music_ratio: float,
) -> Tuple[bytes, Metrics]:
    """
    Execute the full pipeline and return (final_wav_bytes, metrics_obj).
    """
    m = Metrics(request_id=request_id)

    # basic meta
    m.set_meta("paragraph_len_chars", len(paragraph))
    m.set_meta("duration_s", int(duration_s))
    m.set_meta("seed", "" if seed is None else int(seed))
    m.set_meta("voice_language", voice_language)
    m.set_meta("voice_speed", float(voice_speed))
    m.set_meta("voice_speaker", voice_speaker or "Auto")
    m.set_meta("music_model", music_model)
    m.set_meta("voice_ratio", float(voice_ratio))
    m.set_meta("music_ratio", float(music_ratio))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        music_wav = tmp_path / "output_music.wav"
        voice_wav = tmp_path / "voice_openvoice.wav"
        final_wav = tmp_path / "final_mix.wav"

        # 1) Emotion detection
        with StageTimer(m, "emotion_detect_s"):
            emotions = detect_emotion(paragraph)

        dominant = max(emotions, key=emotions.get) if emotions else "unknown"
        m.set_meta("dominant_emotion", dominant)
        # store top-4 as meta (CSV-friendly)
        for k, v in sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)[:4]:
            m.set_meta(f"emotion.{k}", float(v))

        # 2) Map to prompt
        with StageTimer(m, "prompt_map_s"):
            profile = map_emotions_to_music(emotions)
            prompt = profile["prompt"]

        m.set_meta("music_prompt", prompt)

        # 3) Generate music
        with StageTimer(m, "music_generate_total_s"):
            music_info = generate_music(
                prompt=prompt,
                out_wav=str(music_wav),
                duration_s=duration_s,
                seed=seed,
                model_name=music_model,
            )

        m.set_meta("music.resolved_model_id", music_info.get("resolved_model_id", ""))
        m.set_meta("music.model_from_cache", bool(music_info.get("from_cache", False)))
        m.set_meta("music.model_load_s", float(music_info.get("model_load_s", 0.0)))
        m.set_meta("music.gen_s", float(music_info.get("gen_s", 0.0)))
        m.set_meta("music.sample_rate", int(music_info.get("sample_rate", 32000)))

        # 4) Voice TTS
        with StageTimer(m, "voice_tts_total_s"):
            voice_info = synth_openvoice_default(
                text=paragraph,
                out_wav=str(voice_wav),
                language=voice_language,
                speed=voice_speed,
                speaker_key=voice_speaker,
            )

        m.set_meta("voice.speaker_key", voice_info.get("speaker_key", ""))
        m.set_meta("voice.speaker_id", int(voice_info.get("speaker_id", -1)))
        m.set_meta("voice.tts_from_cache", bool(voice_info.get("from_cache", False)))
        m.set_meta("voice.tts_load_s", float(voice_info.get("tts_load_s", 0.0)))
        m.set_meta("voice.tts_s", float(voice_info.get("tts_s", 0.0)))

        # 5) Mix
        with StageTimer(m, "mix_total_s"):
            mix_info = duck_and_mix(
                voice_path=voice_info["path"],
                music_path=music_info["path"],
                out_wav=str(final_wav),
                voice_ratio=voice_ratio,
                music_ratio=music_ratio,
                target_dbfs=-16.0,
                frame_rate=32000,
                channels=2,
            )

        m.set_meta("mix.mix_s", float(mix_info.get("mix_s", 0.0)))

        # finalize metrics
        m.finish()

        # read bytes to return
        audio_bytes = final_wav.read_bytes()

    return audio_bytes, m


@app.get("/")
def index():
    music_models = get_available_music_models()

    voice_language = "EN"
    try:
        voice_speakers = list_openvoice_speakers(language=voice_language)
    except Exception:
        voice_speakers = ["EN-US", "EN-UK", "EN-LIBRITTS"]

    return render_template(
        "index.html",
        default_paragraph=DEFAULT_PARAGRAPH,
        music_models=music_models,
        voice_speakers=voice_speakers,
    )


@app.post("/generate")
def generate():
    paragraph = (request.form.get("paragraph") or "").strip() or DEFAULT_PARAGRAPH

    duration_s = int(request.form.get("duration_s") or 8)

    seed_raw = (request.form.get("seed") or "").strip()
    seed = int(seed_raw) if seed_raw else None

    voice_language = (request.form.get("voice_language") or "EN").strip() or "EN"
    voice_speed = float(request.form.get("voice_speed") or 1.0)

    voice_speaker = (request.form.get("voice_speaker") or "").strip() or None
    music_model = (request.form.get("music_model") or "musicgen-small").strip() or "musicgen-small"

    voice_ratio = float(request.form.get("voice_ratio") or 0.8)
    music_ratio = float(request.form.get("music_ratio") or 0.2)

    request_id = new_request_id()

    # Run pipeline + gather metrics
    audio_bytes, metrics_obj = run_pipeline(
        request_id=request_id,
        paragraph=paragraph,
        duration_s=duration_s,
        voice_language=voice_language,
        voice_speed=voice_speed,
        voice_speaker=voice_speaker,
        music_model=music_model,
        seed=seed,
        voice_ratio=voice_ratio,
        music_ratio=music_ratio,
    )

    # Persist metrics
    append_metrics_csv(metrics_obj, csv_path="metrics/metrics.csv")
    append_metrics_jsonl(metrics_obj, jsonl_path="metrics/metrics.jsonl")

    # Cache audio + metrics for preview/download
    payload = metrics_obj.to_json()
    _cache_put(request_id, audio_bytes, payload)

    return render_template(
        "result.html",
        request_id=request_id,
        metrics=payload,
    )


@app.get("/audio/<request_id>")
def audio(request_id: str):
    try:
        item = _cache_get(request_id)
    except KeyError:
        abort(404)
    return send_file(BytesIO(item["audio_bytes"]), mimetype="audio/wav", as_attachment=False)


@app.get("/download/<request_id>")
def download(request_id: str):
    try:
        item = _cache_get(request_id)
    except KeyError:
        abort(404)
    return send_file(
        BytesIO(item["audio_bytes"]),
        mimetype="audio/wav",
        as_attachment=True,
        download_name=f"final_mix_{request_id}.wav",
    )


if __name__ == "__main__":
    # Run: python app.py  -> http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
