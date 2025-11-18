# 🎵 MusicGen + OpenVoice Environment Setup

This repository provides a complete guide for setting up an environment to run MusicGen-based audio generation models with **Western classical instrumentation** and **emotion-driven prompts**, combined with **OpenVoice/Melo-TTS** for natural voiceovers.

---

## ✨ Features
- Emotion-to-Music generation pipeline
- Uses **classical Western instruments only**
- Converts text → emotion → music prompt → background track
- Generates natural speech from the same paragraph with **OpenVoice**
- Mixes music & voice (default: 80% voice, 20% background music)
- Works **offline** for emotion detection to avoid API rate limits

---

## 📁 Project Structure (Example)
```
MusicGen_OpenVoice/
├── run_music.py               # Generate music from emotion-derived prompt
├── para_to_emo.py             # Detect emotion (local model)
├── map_emo_to_music.py        # Map emotion → music profile
├── openvoice_tts_no_ref.py    # TTS + mix with music
├── hf_models/
│   └── twitter-roberta-base-emotion/  # Local emotion model cache
├── requirements.txt
└── README.md
```

---

## ✅ Prerequisites
- **Linux** or **WSL (Ubuntu 22.04+)**
- **Python 3.10**
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- `ffmpeg` installed for audio processing
- (Optional) `build-essential` for compiling some dependencies

---

## 🚀 Step-by-Step Installation

### 1. Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda-latest-Linux-x86_64.sh
bash Miniconda-latest-Linux-x86_64.sh
```

### 2. Initialize Conda
```bash
echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda --version
```

### 3. Create Conda Environment
```bash
conda create -n musicgen_openvoice python=3.10 -y
conda activate musicgen_openvoice
```

### 4. Install System Dependencies
```bash
sudo apt update
sudo apt install -y ffmpeg build-essential
```

### 5. Install Python Dependencies
Create a `requirements.txt`:
```text
# CPU-friendly PyTorch (change to cu118/cu121 for GPU)
torch --index-url https://download.pytorch.org/whl/cpu
transformers<4.44
safetensors
scipy
sentencepiece
soundfile
pydub
einops
omegaconf
melo-tts
nltk
git+https://github.com/facebookresearch/audiocraft.git --no-deps
huggingface_hub>=0.23
```

Install:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Download Required NLTK Data
```bash
python - << 'PY'
import nltk
nltk.download('averaged_perceptron_tagger_eng')
PY
```

### 7. Download Emotion Model Locally (Avoid HF 429 Errors)
```bash
python scripts/download_emotion_model.py
```

or run the `huggingface-cli download` command if you prefer manual control:

```bash
mkdir -p hf_models
huggingface-cli download cardiffnlp/twitter-roberta-base-emotion \
  --local-dir ./hf_models/twitter-roberta-base-emotion \
  --local-dir-use-symlinks False
```

The downloaded weights live under `hf_models/` which is ignored by git so large binaries never end up in commits. If you need
to store the model somewhere else, set the `EMOTION_MODEL_DIR` environment variable before running `run_music.py`.

---

## 🚀 Running the Pipeline

### 1. One-command demo (emotion ➜ music ➜ narration ➜ mix)

```bash
# In the conda env created above
python scripts/download_emotion_model.py   # only needs to happen once
python run_music.py --text "Your paragraph here" --duration 12 --voice-language EN
```

What happens:
1. `para_to_emo.detect_emotion()` loads the Cardiff NLP model from `hf_models/` and scores your paragraph.
2. `map_emo_to_music.map_emotions_to_music()` turns the dominant mood into a MusicGen-friendly prompt.
3. `musicgenutil.generate_music()` renders a 32 kHz WAV (`output_music.wav`).
4. `generate_voice.synth_openvoice_default()` creates narration (`voice_openvoice.wav`).
5. `generate_voice.duck_and_mix()` balances both stems into `final_mix.wav` (~80% voice / 20% music).

All three WAV files are written to the current directory by default. Change the destination with `--output-dir ./my_renders`.

### 2. CLI options you can tweak

```bash
python run_music.py \
  --text-file prompt.txt \             # read paragraph from disk instead of --text
  --duration 15 \                       # MusicGen length in seconds
  --seed 1234 \                         # make the MusicGen render deterministic
  --voice-language EN \                 # EN, EN-US, EN-UK, etc. (depends on Melo pack)
  --voice-speed 0.95 \                  # slow down/speed up narration
  --voice-ratio 0.85 --music-ratio 0.15 # change the final mix balance
  --output-dir renders/my_scene
```

If you run the command with no flags you will get the included India-themed sample paragraph, which is handy for smoke tests.

### 3. Run stages individually (optional)

Need to experiment with a single component? You can call each module directly:

```bash
# Emotion scores only
python - <<'PY'
from para_to_emo import detect_emotion
print(detect_emotion("I can't believe this happened. I'm so frustrated right now."))
PY

# Music only (accepts any natural-language prompt)
python - <<'PY'
from musicgenutil import generate_music
generate_music("cinematic hopeful strings with harp and light percussion", out_wav="demo_music.wav", duration_s=8)
PY

# Voice + mix only (use any WAVs you already have)
python - <<'PY'
from generate_voice import synth_openvoice_default, duck_and_mix
voice = synth_openvoice_default("Text to narrate", out_wav="voice.wav", language="EN", speed=1.0)
duck_and_mix(voice, "output_music.wav", out_wav="mix.wav")
PY
```

These snippets respect the same environment variables (e.g., `EMOTION_MODEL_DIR`) and output file conventions as the end-to-end script.

---

## ❓ Troubleshooting
- **`xformers` errors** → Not required for CPU runs; ignore warnings.
- **HF 429 errors** → Always use the local model cache in `hf_models/`.
- **TTS sounds robotic** → Try adjusting `--speed` (e.g., `--speed 0.9`) or choose a different `language` variant if available.
- **Audio clipping** → Lower `target_dbfs` in `openvoice_tts_no_ref.py` or reduce music ratio (`--music 0.15`).
- **Model not found** → Ensure `hf_models/twitter-roberta-base-emotion/` exists.

---

## 📦 Local Assets & Git Hygiene
- `hf_models/`, `OpenVoice/`, and `xformers/` are ignored by git on purpose. Populate them locally with downloaded models or
  cloned repos as needed without bloating commits.
- Generated audio (`*.wav`, `*.mp3`, etc.) is also ignored; copy results out of the repo if you want to share them.
- Use `scripts/download_emotion_model.py` (or `EMOTION_MODEL_DIR=/custom/path`) to control where the Cardiff NLP emotion model is
  stored.

---

## 📄 License
This setup guide is provided under the MIT License. You are free to use and adapt it.

---

## ✨ Credits
- [Facebook Audiocraft (MusicGen)](https://github.com/facebookresearch/audiocraft)
- [OpenVoice / Melo-TTS](https://github.com/myshell-ai/OpenVoice)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- Your curiosity and creativity 🚀
