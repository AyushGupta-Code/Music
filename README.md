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

## 1. Set up Python dependencies
```bash
# Optional but recommended: create an environment
conda create -n musicgen python=3.10 -y
conda activate musicgen

# System packages
sudo apt update && sudo apt install -y ffmpeg build-essential

# Python deps
pip install --upgrade pip
pip install -r requirements.txt
```
`requirements.txt` installs PyTorch/torchaudio, the Audiocraft fork of MusicGen, Melo-TTS, and the helper libraries used across the scripts.

---

## 2. Cache the emotion model once
The emotion classifier is fairly small (~500 MB) but huggingface.co rate-limits anonymous downloads. Pull it once and keep it local:
```bash
python scripts/download_emotion_model.py
```
By default the weights land in `hf_models/twitter-roberta-base-emotion/` (ignored by git). Set `EMOTION_MODEL_DIR=/your/path` if you want to store it elsewhere.

---

## 3. Quick start: go from paragraph to final mix
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
  --voice-ratio 0.85 \           # make narration louder in the mix
  --music-ratio 0.15 \           # lower or raise the backing track
  --output-dir renders/demo      # choose another folder for WAVs
```
All arguments have sane defaults, so `python run_music.py` without flags will render an included sample paragraph for smoke testing.

---

## 4. Run individual stages (optional)
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
