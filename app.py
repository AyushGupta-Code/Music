from __future__ import annotations

from io import BytesIO
import tempfile
from pathlib import Path

from flask import Flask, abort, render_template, request, send_file

from generate_voice import duck_and_mix, synth_openvoice_default
from map_emo_to_music import map_emotions_to_music
from musicgenutil import generate_music
from para_to_emo import detect_emotion
from run_music import DEFAULT_PARAGRAPH

app = Flask(__name__)


def _run_pipeline(
    paragraph: str,
    duration: int,
    voice_language: str,
    voice_speed: float,
    expressive: bool,
    voice_pause_ms: int,
    voice_speed_variation: float,
    voice_energy_variation: float,
    voice_ratio: float,
    music_ratio: float,
    seed: int | None,
) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        music_wav = tmp_path / "output_music.wav"
        voice_wav = tmp_path / "voice_openvoice.wav"
        final_wav = tmp_path / "final_mix.wav"

        emotions = detect_emotion(paragraph)
        profile = map_emotions_to_music(emotions)
        prompt = profile["prompt"]

        generate_music(prompt, out_wav=str(music_wav), duration_s=duration, seed=seed)
        synth_openvoice_default(
            paragraph,
            out_wav=str(voice_wav),
            language=voice_language,
            speed=voice_speed,
            expressive=expressive,
            pause_ms=voice_pause_ms,
            speed_variation=voice_speed_variation,
            energy_variation=voice_energy_variation,
        )
        duck_and_mix(
            voice_path=str(voice_wav),
            music_path=str(music_wav),
            out_wav=str(final_wav),
            voice_ratio=voice_ratio,
            music_ratio=music_ratio,
            target_dbfs=-16.0,
            frame_rate=32000,
            channels=2,
        )

        return final_wav.read_bytes()


def _parse_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise abort(400, "Invalid numeric value") from exc


def _parse_int(value: str | None, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise abort(400, "Invalid integer value") from exc


def _parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


@app.get("/")
def index():
    return render_template("index.html", default_text=DEFAULT_PARAGRAPH)


@app.post("/generate")
def generate_audio():
    paragraph = request.form.get("paragraph", "").strip() or DEFAULT_PARAGRAPH
    duration = _parse_int(request.form.get("duration"), 10) or 10
    voice_language = request.form.get("voice_language", "EN") or "EN"
    voice_speed = _parse_float(request.form.get("voice_speed"), 1.0)
    expressive_values = request.form.getlist("expressive")
    expressive = _parse_bool(expressive_values[-1] if expressive_values else None, True)
    voice_pause_ms = _parse_int(request.form.get("voice_pause_ms"), 240) or 240
    voice_speed_variation = _parse_float(request.form.get("voice_speed_variation"), 0.12)
    voice_energy_variation = _parse_float(request.form.get("voice_energy_variation"), 0.06)
    voice_ratio = _parse_float(request.form.get("voice_ratio"), 0.8)
    music_ratio = _parse_float(request.form.get("music_ratio"), 0.2)
    seed = _parse_int(request.form.get("seed"))

    if not (0 < voice_ratio <= 1 and 0 <= music_ratio < 1):
        abort(400, "voice/music ratios must be within 0..1 and voice must be > 0")

    audio_bytes = _run_pipeline(
        paragraph,
        duration,
        voice_language,
        voice_speed,
        expressive,
        voice_pause_ms,
        voice_speed_variation,
        voice_energy_variation,
        voice_ratio,
        music_ratio,
        seed,
    )

    return send_file(
        BytesIO(audio_bytes),
        mimetype="audio/wav",
        as_attachment=True,
        download_name="final_mix.wav",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
