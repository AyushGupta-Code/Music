# Emotion-to-Music Storyteller

Turn any short paragraph into an emotion-aware soundtrack with narration. This repo links three pieces together:

1. Read the text and guess its dominant mood.
2. Feed that mood to Meta’s MusicGen so it writes a classical-inspired background track.
3. Narrate the same paragraph (MeloTTS by default), then blend voice + music into a finished mix.

The project is intentionally “single paragraph in, three WAV files out” so you can audition ideas quickly or bolt the pieces into a larger creative tool.

---

## What the models do (without buzzwords)

| Stage            | Model/Tool                                                                                                      | Plain-language explanation                                                                                                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Emotion detector | [CardiffNLP Twitter RoBERTa emotion classifier](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion) | Reads the paragraph like a human editor and estimates whether it’s mostly joyful, angry, sad, or optimistic. Runs locally after one download.                                                |
| Music maker      | [MusicGen (AudioCraft)](https://github.com/facebookresearch/audiocraft)                                         | Takes the emotion summary, swaps it for a descriptive prompt (“hopeful strings with gentle woodwinds”), and renders a WAV backing track.                                                     |
| Narrator         | MeloTTS (default), optional Coqui, optional Piper                                                               | Speaks your paragraph. MeloTTS is configured for English to keep the pipeline stable. Coqui adds a large local model catalog. Piper provides fully offline ONNX voices via a CLI (optional). |
| Mixer            | Lightweight pydub script                                                                                        | Normalizes and loops/shortens stems, then balances them (defaults to ~80% voice / 20% music) so narration remains clear.                                                                     |

---

## Prerequisites

* Linux or WSL2 with Ubuntu 22.04+
* Python 3.10
* `ffmpeg` in your PATH
* (Recommended) Conda or virtualenv to isolate dependencies

If you plan to use a GPU build of PyTorch, install the correct wheel for your CUDA version. CPU-only also works, it just renders more slowly.

---

## 1. Clone the repo

```bash
git clone https://github.com/<your-username>/Music.git
cd Music
```

---

## 2. Set up Python dependencies

```bash
# Optional but recommended: create an environment
conda create -n musicgen python=3.10 -y
conda activate musicgen

# System packages (Ubuntu/WSL)
sudo apt update && sudo apt install -y ffmpeg build-essential

# Python deps
pip install --upgrade pip
pip install -r requirements.txt
```

### Notes on PyTorch / CUDA

Your `requirements.txt` pins `torch` and `torchaudio`. If you need a specific CUDA build (e.g., `+cu121`), install the matching PyTorch wheels first using the official PyTorch selector, then install the rest of the requirements.

### macOS-specific notes

* Use [Homebrew](https://brew.sh/) to install system packages: `brew install ffmpeg pkg-config libsndfile`.
* Apple Silicon: follow the PyTorch installation selector for the correct `pip install` command, then run `pip install -r requirements.txt`.

---

## 3. Cache the emotion model once

The emotion classifier is fairly large and huggingface.co may rate-limit anonymous downloads. Pull it once and keep it local:

```bash
python scripts/download_emotion_model.py
```

By default the weights land in `hf_models/twitter-roberta-base-emotion/` (ignored by git). Set `EMOTION_MODEL_DIR=/your/path` if you want to store it elsewhere.

---

## 4. Quick start: go from paragraph to final mix

```bash
python run_music.py --text "Your paragraph here" --duration 12
```

Outputs written to the working directory:

* `output_music.wav` – MusicGen backing track
* `voice.wav` (name may vary by engine) – narration
* `final_mix.wav` – blended voice-over with ducked music

### Helpful flags

```bash
python run_music.py \
  --text-file story.txt \        # read from disk instead of passing --text
  --duration 15 \                # music length in seconds
  --seed 42 \                    # reproducible MusicGen renders
  --voice-engine melotts \       # melotts | coqui | piper (if available)
  --voice-model EN \             # dropdown-backed model value (engine-specific)
  --voice-speed 0.9 \            # mainly used by MeloTTS
  --voice-ratio 0.85 \           # make narration louder in the mix
  --music-ratio 0.15 \           # lower or raise the backing track
  --output-dir renders/demo      # choose another folder for WAVs
```

All arguments have sane defaults, so `python run_music.py` without flags will render an included sample paragraph for smoke testing.

**More natural narration:** the TTS step splits long paragraphs into sentences and can introduce short pauses and mild speed variation (depending on engine/settings) to avoid monotone delivery.

**Smoother backing tracks:** MusicGen sampling parameters and the mix stage aim to keep the background polished without overpowering the voice.

---

## 4b. Use the web UI instead of the CLI

If you prefer not to paste long text on the command line, run the bundled Flask app and use the form in your browser:

```bash
# from the repo root
python app.py
# or: FLASK_APP=app.py flask run --host 0.0.0.0 --port 5000
```

Open [http://localhost:5000](http://localhost:5000) and click **Generate**.

### Dropdown-only design (prevents crashes)

The web form is intentionally **dropdown-only**:

* Voice engines appear only if available on your machine (MeloTTS/Coqui/Piper).
* Voice models are loaded dynamically based on the selected engine.
* Melo speakers are lazy-loaded after the page renders to keep `/` fast and reliable.

### Metrics

Each run writes metrics (timings, chosen models, cache hits, etc.) to:

* `metrics/metrics.csv`
* `metrics/metrics.jsonl`

---

## 4c. Piper integration (optional, offline)

Piper is treated as an **external CLI** (not a pip dependency). If it is available, it appears automatically in the Voice Engine dropdown.

### Piper requirements

* A working Piper TTS executable
* One or more `.onnx` voices on disk

### Piper config (cross-platform)

The voice engine loader supports:

* `PIPER_BIN` — absolute path to the correct Piper TTS executable
* `PIPER_VOICES_DIR` — directory containing `.onnx` voices

If `PIPER_VOICES_DIR` is not set, the default is:

```
voices/piper/
```

Place voices like:

```
voices/piper/en_US-ljspeech-medium.onnx
voices/piper/en_US-ljspeech-medium.onnx.json
```

### Important (Ubuntu/WSL “piper” name collision)

On some systems, `/usr/bin/piper` is a GTK-based program and not Piper TTS. If `piper --help` fails with GTK/`gi` errors, you are not calling Piper TTS. Set `PIPER_BIN` to point to the correct Piper TTS binary.

---

## 5. Run individual stages (optional)

Need only one part of the pipeline? Import the modules directly:

```python
from para_to_emo import detect_emotion
from map_emo_to_music import map_emotions_to_music
from musicgenutil import generate_music
from generate_voice import duck_and_mix

paragraph = "I can't believe this happened. I'm so frustrated right now."
emotions = detect_emotion(paragraph)
prompt = map_emo_to_music(emotions)["prompt"]

music_info = generate_music(prompt, out_wav="demo_music.wav", duration_s=8)

voice_info = synthesize_voice(
    engine="melotts",
    text=paragraph,
    out_wav="voice.wav",
    voice_model="EN",
    speed=1.0,
    speaker_key=None,
)

final_path = duck_and_mix("voice.wav", "demo_music.wav", out_wav="final_mix.wav")
```

`detect_emotion` automatically looks for cached weights first and falls back to the Hugging Face Hub only if they are missing.

---

## Troubleshooting cheatsheet

* **Model download fails with 429** → Run `scripts/download_emotion_model.py` or authenticate via `huggingface-cli login`.
* **MusicGen errors/warnings about `xformers`** → Not required. Ignore the warning or uninstall/reinstall xformers to match your torch version.
* **MeCab / dictionary issues** → This repo uses `unidic` (run `python -m unidic download` once if needed).
* **Browser shows `ERR_EMPTY_RESPONSE`** → Ensure the home route stays lightweight; speaker/model loads should happen through API calls after render (the repo’s web UI is implemented this way).
* **Voice sounds clipped / buried** → Lower `music_ratio`, raise `voice_ratio`, or reduce voice speed slightly.
* **Need GPU acceleration** → Install the correct CUDA PyTorch wheel first (matching the pinned version), then install the rest of the requirements.
* **Want different instruments** → Edit `map_emo_to_music.py` to change the descriptive prompts MusicGen receives.

---

## Repository map

```
├── app.py                       # Flask web app (dropdown-only UI)
├── templates/
│   └── index.html               # Web form + lazy-load JS for models/speakers
├── run_music.py                 # CLI entry point tying stages together
├── para_to_emo.py               # Emotion classifier wrapper
├── map_emo_to_music.py          # Emotion scores → MusicGen prompt helper
├── musicgenutil.py              # Utility to call MusicGen and save WAVs
├── generate_voice.py            # Narration orchestration + mixing helpers
├── scripts/download_emotion_model.py
├── hf_models/                   # Local cache for the emotion model (gitignored)
├── voices/
│   └── piper/                   # (optional) Piper .onnx voices for dropdown
└── requirements.txt
```

Enjoy experimenting. Each module is kept self-contained so you can lift it into a larger UI, DAW workflow, or story-reading tool.
