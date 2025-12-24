# Local Emotion Model Cache

This folder is intentionally empty in git. Download the `cardiffnlp/twitter-roberta-base-emotion` model here so that
`para_to_emo.detect_emotion` can load it without hitting the Hugging Face Hub.

Run:

```bash
python scripts/download_emotion_model.py
```

or execute the `huggingface-cli download` command listed in the main README. The downloaded weights will remain ignored
by git but available locally for offline use.
