# MusicGen Environment Setup

This repository provides a complete guide for setting up an environment to run MusicGen-based audio generation models using classical instruments and emotion-driven prompts.

---

## ✨ Features
- Emotion-to-Music generation pipeline
- Uses classical Western instruments only
- Based on Facebook's [Audiocraft](https://github.com/facebookresearch/audiocraft)
- Lightweight Conda-based environment

---

## 📁 Project Structure (Example)
```
MusicGen/
├── run_music.py
├── para_to_emo.py
├── map_emo_to_music.py
├── paragraph_to_music.py
├── requirements.txt
└── README.md
```

---

## ✅ Prerequisites
- Linux / WSL
- Python 3.9
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

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
conda create -n music python=3.9 -y
conda activate music
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. (Optional) Install System Dependencies
```bash
sudo apt update
sudo apt install build-essential -y
```

---

## 🌐 requirements.txt
Here is the content for your `requirements.txt` file:

```text
torch
torchvision
torchaudio
setuptools
wheel
blis==0.7.9
git+https://github.com/facebookresearch/audiocraft --no-deps
transformers
scipy
einops
soundfile
av
julius
omegaconf
xformers --extra-index-url http://download.pytorch.org/whl/cu120
flashy
num2words
librosa
torchdiffeq
torchmetrics
demucs
sentencepiece
```

---

## 🚀 Running the Pipeline

Assuming you have the following structure:
- `para_to_emo.py` detects emotion from text
- `map_emo_to_music.py` maps emotion to classical instrument profile
- `run_music.py` orchestrates the process

Run:
```bash
conda activate music
python run_music.py
```

This will generate a `.wav` file based on the emotional tone of the input paragraph.

---

## ❓ Troubleshooting
- `xformers` installation fails:
  - Try installing separately with: 
    ```bash
    pip install xformers --extra-index-url http://download.pytorch.org/whl/cu120
    ```
- `sentencepiece` missing:
  - Install it manually: `pip install sentencepiece`
- Compilation errors:
  - Make sure `build-essential` is installed: `sudo apt install build-essential`

---

## 📄 License
This setup guide is provided under the MIT License. You are free to use and adapt it.

---

## ✨ Credits
- [Facebook Audiocraft](https://github.com/facebookresearch/audiocraft)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- Your curiosity and creativity 🚀
