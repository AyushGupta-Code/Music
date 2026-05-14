import os
import re
import tempfile
from pathlib import Path

import fitz  # pymupdf
import gradio as gr
from groq import Groq

from pipeline import MUSIC_MODELS, KOKORO_VOICES, _analyze_emotion, _generate_music, _narrate, _mix
from prompts import build_music_profile


# ── PDF parsing ───────────────────────────────────────────────────────────────

def _chapters_from_text(text: str) -> dict[str, str]:
    for pattern in [
        r"(?m)^(Chapter\s+(?:\d+|[A-Z][a-z]+)[^\n]*)",
        r"(?m)^(CHAPTER\s+(?:\d+|[A-Z]+)[^\n]*)",
        r"(?m)^(Part\s+(?:\d+|[A-Z][a-z]+)[^\n]*)",
    ]:
        parts = re.split(pattern, text)
        if len(parts) >= 5:
            chapters = {}
            for i in range(1, len(parts) - 1, 2):
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if len(content) > 200:
                    chapters[parts[i].strip()] = content
            if len(chapters) >= 2:
                return chapters

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    return {
        f"Section {i // 10 + 1}": "\n\n".join(paragraphs[i : i + 10])
        for i in range(0, len(paragraphs), 10)
    }


def parse_pdf(pdf_file):
    if pdf_file is None:
        return {}, gr.update(choices=[], value=None)

    path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
    doc = fitz.open(path)
    chapters = {}

    # Try PDF bookmarks / TOC first
    toc = doc.get_toc()
    top = [e for e in toc if e[0] == 1]
    if top:
        for i, (_, title, page) in enumerate(top):
            end = top[i + 1][2] if i + 1 < len(top) else len(doc) + 1
            text = "".join(doc[p].get_text() for p in range(page - 1, min(end - 1, len(doc))))
            if len(text.strip()) > 200:
                chapters[title] = text.strip()

    # Fall back to heading detection in raw text
    if not chapters:
        full_text = "".join(page.get_text() for page in doc)
        chapters = _chapters_from_text(full_text)

    doc.close()

    if not chapters:
        return {}, gr.update(choices=["No chapters detected"], value=None)

    titles = list(chapters.keys())
    return chapters, gr.update(choices=titles, value=titles[0])


# ── Audio generation ──────────────────────────────────────────────────────────

def generate(chapter_name, chapters, music_model, voice, speed, music_duration, seed_str):
    if not chapter_name or not chapters:
        raise gr.Error("Upload a PDF and select a chapter first.")
    chapters = chapters or {}

    content = chapters.get(chapter_name, "")
    if not content:
        raise gr.Error(f"No content found for '{chapter_name}'.")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise gr.Error("GROQ_API_KEY secret is not set in Space settings.")

    seed = int(seed_str) if seed_str.strip() else None
    client = Groq(api_key=api_key)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "out"

        scores  = _analyze_emotion(content, client)
        profile = build_music_profile(scores)

        music_wav = str(p) + "_music.wav"
        _generate_music(profile["prompt"], music_wav, music_duration, seed, music_model)

        voice_wav = str(p) + "_voice.wav"
        _narrate(content, voice_wav, voice, speed)

        final_wav = str(p) + ".wav"
        _mix(voice_wav, music_wav, final_wav)

        os.remove(music_wav)
        os.remove(voice_wav)

        audio_bytes = Path(final_wav).read_bytes()

    # Write to a persistent temp file gradio can serve
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out.write(audio_bytes)
    out.close()

    mood_label = f"{profile['cinematic_mode'].title()} — {profile['mood']}"
    return out.name, mood_label


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Novel to Audio", theme=gr.themes.Soft(), api_open=False) as demo:
    chapters_state = gr.State(value=None)

    gr.Markdown("# Novel to Audio\nUpload a PDF novel, pick a chapter, generate narrated audio with background score.")

    with gr.Row():

        # Sidebar
        with gr.Column(scale=1, min_width=260):
            pdf_upload = gr.File(label="Upload Novel (PDF)", file_types=[".pdf"])
            chapter_list = gr.Radio(choices=[], label="Chapters", interactive=True)

        # Main panel
        with gr.Column(scale=2):
            with gr.Row():
                music_model = gr.Dropdown(MUSIC_MODELS, value="musicgen-small", label="Music model")
                voice       = gr.Dropdown(KOKORO_VOICES, value="af_heart",       label="Voice")
            with gr.Row():
                speed    = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label="Narration speed")
                duration = gr.Slider(10,  60,  value=30,  step=5,    label="Music duration (s)")
            seed = gr.Textbox(value="", label="Seed (optional)", placeholder="42")

            generate_btn = gr.Button("Generate Audio", variant="primary", size="lg")

            audio_out = gr.Audio(label="Output", type="filepath")
            mood_out  = gr.Textbox(label="Detected mood", interactive=False)

    pdf_upload.change(
        parse_pdf,
        inputs=pdf_upload,
        outputs=[chapters_state, chapter_list],
    )

    generate_btn.click(
        generate,
        inputs=[chapter_list, chapters_state, music_model, voice, speed, duration, seed],
        outputs=[audio_out, mood_out],
    )


demo.launch(ssr_mode=False, server_name="0.0.0.0")
