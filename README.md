---
title: Novel to Audio
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
python_version: "3.11"
---

# Novel to Audio

Feed it a novel. Get back one narrated audio file per chapter, each with a background score that matches the chapter's emotional tone.

```
python main.py my_novel.txt ./output
```

```
output/
├── 01_chapter_one.wav
├── 02_the_storm.wav
└── 03_aftermath.wav
```

---

## How it works

1. **Chapter detection** — splits the novel on heading patterns (`Chapter N`, `Part N`, etc.). Falls back to paragraph-based chunking if no headings are found.
2. **Emotion analysis** — sends the first ~3000 characters of each chapter to Groq (Llama 3.3 70B, free tier) and gets back anger / joy / sadness / optimism scores.
3. **Music generation** — maps the emotion scores to a cinematic music profile and generates a background track with MusicGen (runs locally).
4. **Narration** — reads the full chapter text aloud with Kokoro TTS (runs locally, no API needed).
5. **Mixing** — loops the music track to match the narration length, normalizes levels, and blends them into a single WAV.

Everything except Groq runs fully on your machine. Groq's free tier allows 14,400 requests/day — more than enough for any novel.

---

## Setup

### 1. Prerequisites

```bash
# macOS
brew install ffmpeg espeak-ng
```

`ffmpeg` is required by pydub for audio I/O. `espeak-ng` is required by Kokoro for text phonemization.

### 2. Create a virtual environment on your drive

```bash
python3 -m venv /path/to/your/drive/Music/.venv
source /path/to/your/drive/Music/.venv/bin/activate
```

### 3. Install PyTorch first

Follow the selector at https://pytorch.org/get-started to get the right wheel for your platform, then:

```bash
# macOS (Apple Silicon or Intel)
pip install torch torchaudio
```

### 4. Install the rest

```bash
pip install -r requirements.txt
```

### 5. Add your Groq API key

Get a free key at https://console.groq.com, then:

```bash
cp .env.example .env
# edit .env and paste your key
```

---

## Usage

```bash
python main.py novel.txt ./output
```

### All options

```
positional arguments:
  input                 Path to novel .txt file
  output                Output directory for audio files

options:
  --music-model         MusicGen model (default: musicgen-small)
                        choices: musicgen-small, musicgen-medium, musicgen-large,
                                 musicgen-stereo-small, musicgen-stereo-medium
  --voice               Kokoro voice ID (default: af_heart)
  --speed               Narration speed (default: 1.0)
  --music-duration      Music clip length in seconds, looped to match narration (default: 30)
  --voice-ratio         Voice level in mix 0–1 (default: 0.85)
  --music-ratio         Music level in mix 0–1 (default: 0.15)
  --seed                Seed for reproducible music generation
```

### Example with options

```bash
python main.py dracula.txt ./output \
  --music-model musicgen-stereo-small \
  --voice bm_george \
  --speed 0.95 \
  --seed 42
```

---

## Available voices

| ID | Accent | Gender |
|----|--------|--------|
| `af_heart` | American | Female |
| `af_bella` | American | Female |
| `af_nicole` | American | Female |
| `af_sarah` | American | Female |
| `af_sky` | American | Female |
| `am_adam` | American | Male |
| `am_michael` | American | Male |
| `bf_emma` | British | Female |
| `bf_isabella` | British | Female |
| `bm_george` | British | Male |
| `bm_lewis` | British | Male |

---

## Project structure

```
main.py         CLI entry point
pipeline.py     Chapter parsing, emotion analysis, TTS, music generation, mixing
prompts.py      Emotion scores → cinematic music profiles
requirements.txt
.env.example
```

---

## Preparing your novel

The input should be a plain `.txt` file with UTF-8 encoding. Chapter headings like `Chapter 1`, `Chapter One`, `CHAPTER I`, or `Part 1` are auto-detected. If your novel uses a different format, the pipeline will fall back to chunking every 10 paragraphs into a section.

To convert an epub: `pip install ebooklib beautifulsoup4` — or just copy-paste from a text source.
