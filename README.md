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

## 📁 Project Structure

MusicGen_OpenVoice/
├── run_music.py # Generate music from emotion-derived prompt
├── para_to_emo.py # Detect emotion (local model)
├── map_emo_to_music.py # Map emotion → music profile
├── openvoice_tts_no_ref.py # TTS + mix with music
├── hf_models/
│ └── twitter-roberta-base-emotion/ # Local emotion model cache
├── requirements.txt
└── README.md

text

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

wget https://repo.anaconda.com/miniconda/Miniconda-latest-Linux-x86_64.sh
bash Miniconda-latest-Linux-x86_64.sh

text

### 2. Initialize Conda

echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda --version

text

### 3. Create Conda Environment

conda create -n musicgen_openvoice python=3.10 -y
conda activate musicgen_openvoice

text

### 4. Install System Dependencies

sudo apt update
sudo apt install -y ffmpeg build-essential

text

### 5. Install Python Dependencies

Create a `requirements.txt` with the following:

CPU-friendly PyTorch (change to cu118/cu121 for GPU)
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

text

Install them:

pip install --upgrade pip
pip install -r requirements.txt

text

### 6. Download Required NLTK Data

python - << 'PY'
import nltk
nltk.download('averaged_perceptron_tagger_eng')
PY

text

### 7. Download Emotion Model Locally (Avoid HF 429 Errors)

mkdir -p hf_models
huggingface-cli download cardiffnlp/twitter-roberta-base-emotion
--local-dir ./hf_models/twitter-roberta-base-emotion
--local-dir-use-symlinks False

text

---

## 🚀 Running the Pipeline

### Step 1: Generate Music from Text

python run_music.py

text

- Detects the emotion of your paragraph (offline)
- Maps it to a Western classical music profile
- Generates `output.wav` with MusicGen

### Step 2: Generate Voiceover & Mix

python openvoice_tts_no_ref.py --text "Your paragraph here" --music-in output.wav

text

- Creates `voice_openvoice.wav` (TTS)
- Mixes with background music → `final_mix.wav`

---

## ❓ Troubleshooting

- **xformers errors** → Not required for CPU runs; ignore warnings.
- **HF 429 errors** → Always use the local model cache in `hf_models/`.
- **TTS sounds robotic** → Try adjusting `--speed` (e.g., `--speed 0.9`) or choose a different language variant if available.
- **Audio clipping** → Lower `target_dbfs` in `openvoice_tts_no_ref.py` or reduce music ratio (`--music 0.15`).
- **Model not found** → Ensure `hf_models/twitter-roberta-base-emotion/` exists.

---

## 📄 License

This setup guide is provided under the MIT License. You are free to use and adapt it.