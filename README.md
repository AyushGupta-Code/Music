# Emotion-to-Music Storyteller

Turn any short paragraph into an emotion-aware soundtrack with narration. This repo links three pieces together:

1. Read the text and guess its dominant mood.
2. Feed that mood to Meta's MusicGen so it writes a classical-inspired background track.
3. Use OpenVoice/Melo-TTS to narrate the same paragraph, then blend the voice and music into a finished mix.

The project is intentionally "single paragraph in, three WAV files out" so you can audition ideas quickly or bolt the pieces into a larger creative tool.

---

## What the models do (without buzzwords)
| Stage | Model | Plain-language explanation |
| --- | --- | --- |
| Emotion detector | [CardiffNLP Twitter RoBERTa emotion classifier](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion) | Reads the paragraph like a human editor and estimates whether it's mostly joyful, angry, sad, or optimistic. Runs locally so you can stay offline after one download. |
| Music maker | [MusicGen](https://github.com/facebookresearch/audiocraft) | Takes the emotion summary, swaps it for a descriptive prompt ("hopeful strings with gentle woodwinds"), and renders a 32 kHz stereo WAV using only Western classical instruments. |
| Narrator | [OpenVoice / Melo-TTS](https://github.com/myshell-ai/OpenVoice) | Speaks your paragraph with a stock English voice. You can adjust speed, switch to other bundled accents, or point it at another Melo language pack. |
| Mixer | Lightweight pydub script | Normalizes and loops/shortens the stems, then balances them (defaults to ~80% voice / 20% music) so the narration remains clear. |

---

## Prerequisites
- Linux or WSL2 with Ubuntu 22.04+
- Python 3.10
- `ffmpeg` in your PATH
- (Recommended) Conda or virtualenv to isolate dependencies

If you plan to use a GPU build of PyTorch, install the correct wheel for your CUDA version. CPU-only also works, it just renders more slowly.

---

## 1. Clone the repo
```bash
git clone https://github.com/<your-username>/Music.git
cd Music
```

## 2. Set up Python dependencies
```bash
# Optional but recommended: create an environment
conda create -n musicgen python=3.10 -y
conda activate musicgen

# System packages
sudo apt update && sudo apt install -y ffmpeg build-essential

# Python deps
pip install --upgrade pip

# Install torch/torchaudio first so `xformers` can find them during its build.
# Audiocraft 1.4.0a2 expects torch/torchaudio 2.1.0; keep these versions aligned
# to avoid resolver conflicts. Adjust the index URL/versions if you use CUDA or
# Metal Performance Shaders.
pip install torch==2.1.0 torchaudio==2.1.0

# Then install the rest; build isolation stays off so `xformers` can reuse
# the already-installed torch wheel instead of failing during its wheel build.
PIP_NO_BUILD_ISOLATION=1 pip install -r requirements.txt
```
`requirements.txt` installs PyTorch/torchaudio, the Audiocraft fork of MusicGen, Melo-TTS, and the helper libraries used across the scripts. Melo-TTS is installed from the vendored copy in `vendor/melotts/` (matching the repository version but pinned to `transformers>=4.31` to stay compatible with Audiocraft). The vendored package uses a post-release version (`0.1.2.post1`) so pip does not try to resolve the PyPI build that pins `transformers==4.27.4`; if you previously installed Melo-TTS from PyPI, uninstall it before running `pip install -r requirements.txt` to avoid cached metadata conflicts. A lightweight MeCab dictionary (`unidic-lite`) is also installed so Melo-TTS can import its Japanese tokenizer without extra system packages; if you prefer a full dictionary, set `MECAB_ARGS` (and optionally `MECABRC`) before running the scripts. If MeCab still complains about a missing `mecabrc` file, reinstall `unidic-lite` or run `python -c "import unidic; unidic.download()"` if you use the full `unidic` package to fetch its dictionary.

### macOS-specific notes
- Use [Homebrew](https://brew.sh/) to install system packages: `brew install ffmpeg pkg-config libsndfile`.
- Apple Silicon: prefer the CPU-only PyTorch wheels unless you have Metal Performance Shaders available; follow the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the correct `pip install` command, then run `pip install -r requirements.txt`.
- If `pip` complains about an option like `--no-deps` inside `requirements.txt`, update the file from the latest main branch.

---

## 3. Cache the emotion model once
The emotion classifier is fairly small (~500 MB) but huggingface.co rate-limits anonymous downloads. Pull it once and keep it local:
```bash
python scripts/download_emotion_model.py
```
By default the weights land in `hf_models/twitter-roberta-base-emotion/` (ignored by git). Set `EMOTION_MODEL_DIR=/your/path` if you want to store it elsewhere.

---

## 4. Quick start: go from paragraph to final mix
```bash
python run_music.py --text "Your paragraph here" --duration 12 --voice-language EN
```
Outputs written to the working directory:
- `output_music.wav` – MusicGen backing track
- `voice_openvoice.wav` – Melo narration
- `final_mix.wav` – blended voice-over with ducked music

### Helpful flags
```bash
python run_music.py \
  --text-file story.txt \        # read from disk instead of passing --text
  --duration 15 \                # music length in seconds
  --seed 42 \                    # reproducible MusicGen renders
  --voice-language EN-UK \       # pick another Melo voice family
  --voice-speed 0.9 \            # slow down or speed up narration
  --no-expressive \              # disable pauses/speed jitter if you want flatter delivery
  --voice-pause-ms 280 \         # pause length between sentences (expressive mode)
  --voice-speed-variation 0.1 \  # per-sentence speed jitter (expressive mode)
  --voice-energy-variation 0.05 # per-sentence loudness jitter (expressive mode)
  --voice-ratio 0.85 \           # make narration louder in the mix
  --music-ratio 0.15 \           # lower or raise the backing track
  --output-dir renders/demo      # choose another folder for WAVs
```
All arguments have sane defaults, so `python run_music.py` without flags will render an included sample paragraph for smoke testing.

**More natural narration:** the TTS step now splits long paragraphs into sentences, introduces short pauses, and jitters speed slightly per sentence to avoid monotone delivery. Disable this behavior with `expressive=False` or tune it via the function parameters if you import `synth_openvoice_default` directly.

**Smoother backing tracks:** MusicGen now samples with higher temperature/top-p to add variation, and the mix stage applies light EQ/compression plus fade-in/out so the background feels polished without overpowering the voice.

## 4b. Use the web UI instead of the CLI
If you prefer not to paste long text on the command line, run the bundled Flask app and use the form in your browser:

```bash
# from the repo root
python app.py
# or: FLASK_APP=app.py flask run --host 0.0.0.0 --port 5000
```

Open http://localhost:5000 and click **Generate Audio**. The form defaults to the sample paragraph; paste your own text and tweak the sliders as needed.

The web form includes toggles for expressive narration and pause/variation controls so you can quickly dial in a less monotonous read without touching the CLI.

---

## 5. Run individual stages (optional)
Need only one part of the pipeline? Import the modules directly:
```python
from para_to_emo import detect_emotion
from map_emo_to_music import map_emotions_to_music
from musicgenutil import generate_music
from generate_voice import synth_openvoice_default, duck_and_mix

paragraph = "I can't believe this happened. I'm so frustrated right now."
emotions = detect_emotion(paragraph)
prompt = map_emotions_to_music(emotions)["prompt"]

music_path = generate_music(prompt, out_wav="demo_music.wav", duration_s=8)
voice_path = synth_openvoice_default(paragraph, out_wav="voice.wav", language="EN")
final_path = duck_and_mix(voice_path, music_path, out_wav="final_mix.wav")
```
`detect_emotion` automatically looks for the cached weights first and falls back to the Hugging Face Hub only if they are missing.

---

## Troubleshooting cheatsheet
- **Model download fails with 429** → Always run `scripts/download_emotion_model.py` or pass a Hugging Face token via the `huggingface-cli` if you need to redownload.
- **MusicGen errors about `xformers`** → Not required. Ignore the warning or uninstall it if an old global install is causing conflicts.
- **MeCab fails to find a dictionary** → `unidic-lite` ships with the repo requirements and is auto-configured; if you use another dictionary, set `MECAB_ARGS="-d /path/to/your/mecab/dic"` (and `MECABRC=/path/to/mecabrc` if needed) before running. If installs happened before this change, reinstall `unidic-lite` so its bundled `mecabrc` is present.
- **Voice sounds off or clipped** → Lower `--voice-speed`, try a different `--voice-language`, or reduce `--music-ratio` so the backing track is softer.
- **Need GPU acceleration** → Install the CUDA-specific PyTorch wheel first, then reinstall the rest of the requirements. MusicGen will automatically use CUDA if `torch.cuda.is_available()`.
- **Want different instruments** → Edit `map_emo_to_music.py` to change the descriptive prompts that MusicGen receives.

---

## Repository map
```
├── run_music.py            # CLI entry point tying the stages together
├── para_to_emo.py          # Emotion classifier wrapper
├── map_emo_to_music.py     # Emotion scores → MusicGen prompt helper
├── musicgenutil.py         # Utility to call MusicGen and save WAVs
├── generate_voice.py       # Melo-TTS narration + mixing helpers
├── scripts/download_emotion_model.py
├── hf_models/              # Local cache for the emotion model (gitignored)
└── requirements.txt
```

Enjoy experimenting! If you build a UI or DAW integration on top, each module was kept self-contained so you can lift it straight into your project.
