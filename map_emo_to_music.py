"""
Emotion → music prompt mapping.

This file is the *prompt layer* of the project.

IMPORTANT:
- Prompt A/B tests should differ primarily (or entirely) by the prompt template.
- MusicGen generation parameters should stay fixed so improvements are attributable to prompts.

Cinematic direction notes:
- We describe high-level scoring/orchestration traits (march, ostinato, sparse motif, etc.).
- We avoid requesting exact replication of any specific copyrighted recording.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PROMPT_VERSION = "v2"  # <-- so app.py gets cinematic prompts without code changes


@dataclass(frozen=True)
class PromptProfile:
    dominant_emotion: str          # raw classifier label (anger/joy/optimism/sadness)
    cinematic_mode: str            # derived mode used for orchestration choices

    tempo: int
    key: str
    instruments: list[str]
    mood: str

    # Used mainly by Prompt v2
    genre: str
    arc: str
    arc_type: str
    texture: str
    mix: str
    constraints: str


def _dominant_emotion(emotion_scores: dict[str, float]) -> str:
    if not emotion_scores:
        return "joy"
    return max(emotion_scores, key=emotion_scores.get)


def _get(emotion_scores: dict[str, float], k: str) -> float:
    try:
        return float(emotion_scores.get(k, 0.0))
    except Exception:
        return 0.0


def _derive_cinematic_mode(emotion_scores: dict[str, float]) -> str:
    """
    Derive a film-score-oriented mode from the 4-label emotion classifier.

    Cardiff model outputs: anger, joy, optimism, sadness.
    We keep dominant_emotion for reporting, but cinematic_mode drives orchestration/structure.
    """

    a = _get(emotion_scores, "anger")
    s = _get(emotion_scores, "sadness")
    j = _get(emotion_scores, "joy")
    o = _get(emotion_scores, "optimism")

    # Strong anger tends to read as threat / villain energy.
    if a >= 0.55:
        return "danger"

    # Strong sadness with some optimism reads as longing wonder / awe.
    if s >= 0.55 and o >= 0.20:
        return "awe"

    if s >= 0.55:
        return "melancholy"

    # High optimism + some joy reads as victory/heroism.
    if o >= 0.60 and j >= 0.20:
        return "triumph"

    if o >= 0.55:
        return "hope"

    # Mixed negative signals: tension bed rather than outright action.
    if a >= 0.35 and s >= 0.25:
        return "tension"

    # Default to the dominant classifier label.
    return _dominant_emotion(emotion_scores)


def _profiles() -> dict[str, PromptProfile]:
    """Profiles keyed by cinematic_mode."""
    return {
        # --- positive modes ---
        "joy": PromptProfile(
            dominant_emotion="joy",
            cinematic_mode="joy",
            tempo=120,
            key="C major",
            instruments=["piano", "violin", "flute", "harp"],
            mood="joyful",
            genre="cinematic orchestral underscore",
            arc="gentle build → bright lift → warm resolve",
            arc_type="build_climax_resolve",
            texture="light rhythmic pulse, lyrical strings, airy woodwinds",
            mix="wide stereo, film-score reverb, clean low-end",
            constraints="no vocals, no synth lead, no EDM drums",
        ),
        "hope": PromptProfile(
            dominant_emotion="optimism",
            cinematic_mode="hope",
            tempo=96,
            key="G major",
            instruments=["acoustic guitar", "piano", "cello", "horn"],
            mood="hopeful",
            genre="inspirational cinematic cue",
            arc="soft opening → confident lift at midpoint → gentle resolve",
            arc_type="build_climax_resolve",
            texture="steady warm pulse, supportive strings, restrained brass bloom",
            mix="wide stereo, natural room, subtle percussion",
            constraints="no vocals, no aggressive synths, no EDM drums",
        ),
        "triumph": PromptProfile(
            dominant_emotion="optimism",
            cinematic_mode="triumph",
            tempo=112,
            key="D major",
            instruments=["French horn", "trumpet", "strings", "timpani", "cymbal"],
            mood="triumphant",
            genre="heroic cinematic trailer cue",
            arc="bold statement → rising drive → soaring climax → confident resolve",
            arc_type="build_climax_resolve",
            texture="rhythmic string ostinato, brass fanfare, big drums",
            mix="wide stereo, punchy low-end, large hall reverb",
            constraints="no vocals, no pop drums, no EDM drops",
        ),

        # --- negative / suspense modes ---
        "tension": PromptProfile(
            dominant_emotion="anger",
            cinematic_mode="tension",
            tempo=84,
            key="E minor",
            instruments=["low strings", "bass clarinet", "taiko", "gran casa", "cymbal swells"],
            mood="tense",
            genre="suspenseful cinematic tension bed",
            arc="quiet pressure → incremental build → controlled peak → unresolved end",
            arc_type="build_sustain_cut",
            texture="pulsing low-string ostinato, sparse hits, rising swells",
            mix="dark tone, tight sub, deep reverb tail, restrained highs",
            constraints="no vocals, no melodic pop hooks, no EDM drums",
        ),

        # Villain-entrance archetype (march-like, menacing).
        "danger": PromptProfile(
            dominant_emotion="anger",
            cinematic_mode="danger",
            tempo=96,
            key="F minor",
            instruments=["trombones", "tuba", "French horn", "low strings", "snare", "timpani"],
            mood="menacing",
            genre="cinematic villain entrance march",
            arc="ominous intro → martial theme → looming crescendo → hard stop",
            arc_type="build_sustain_cut",
            texture="heavy staccato low-string ostinato, brass unison melody, snare rolls",
            mix="centered brass, punchy low-end, big hall reverb, controlled highs",
            constraints="no vocals, no bright synth arps, no EDM drops",
        ),

        # --- melancholy / sci-fi modes ---
        "melancholy": PromptProfile(
            dominant_emotion="sadness",
            cinematic_mode="melancholy",
            tempo=66,
            key="A minor",
            instruments=["felt piano", "soft organ", "strings", "subtle synth pad"],
            mood="melancholic",
            genre="sci-fi melancholy film score",
            arc="solo motif → slow harmonic swell → distant echo resolve",
            arc_type="swell_fade",
            texture="repeating simple piano motif, sustained organ, airy pads, gentle pulse",
            mix="very spacious reverb, wide stereo, soft highs, warm low-mids",
            constraints="no vocals, no heavy drums, no upbeat rhythms",
        ),
        "awe": PromptProfile(
            dominant_emotion="sadness",
            cinematic_mode="awe",
            tempo=72,
            key="E minor",
            instruments=["piano", "organ", "strings", "synth pad", "soft percussion"],
            mood="wonder-filled and bittersweet",
            genre="expansive sci-fi orchestral score",
            arc="mysterious opening → rising wonder → luminous peak → floating resolve",
            arc_type="build_climax_resolve",
            texture="slow-building harmonies, distant pulse, evolving pads, swelling strings",
            mix="huge space, long reverb tail, wide stereo, gentle transients",
            constraints="no vocals, no pop drums, avoid busy melodies",
        ),

        # --- fallbacks for raw labels (kept for compatibility) ---
        "optimism": PromptProfile(
            dominant_emotion="optimism",
            cinematic_mode="optimism",
            tempo=100,
            key="G major",
            instruments=["acoustic guitar", "cello", "clarinet", "oboe"],
            mood="hopeful",
            genre="uplifting cinematic cue",
            arc="soft opening → confident lift at midpoint → gentle resolve",
            arc_type="build_climax_resolve",
            texture="warm guitar pattern, supportive cello, woodwind counterline",
            mix="wide stereo, natural room, subtle percussion",
            constraints="no vocals, no aggressive synths, no EDM drums",
        ),
        "sadness": PromptProfile(
            dominant_emotion="sadness",
            cinematic_mode="sadness",
            tempo=70,
            key="A minor",
            instruments=["cello", "piano", "bassoon", "English horn"],
            mood="melancholic",
            genre="intimate chamber score",
            arc="quiet opening → slow swell → fading close",
            arc_type="swell_fade",
            texture="sparse piano, legato cello, soft woodwind harmony",
            mix="close-mic intimacy, gentle hall reverb, soft highs",
            constraints="no vocals, no drums, no bright synths",
        ),
        "anger": PromptProfile(
            dominant_emotion="anger",
            cinematic_mode="anger",
            tempo=140,
            key="D minor",
            instruments=["French horn", "timpani", "trumpet", "viola"],
            mood="intense",
            genre="dark cinematic action cue",
            arc="rapid build → sustained peak → sharp cutoff",
            arc_type="build_sustain_cut",
            texture="staccato ostinato, brass stabs, heavy cinematic percussion",
            mix="forward midrange, tight low-end, controlled reverb",
            constraints="no vocals, no EDM drops",
        ),
    }


def build_prompt(profile: PromptProfile, prompt_version: str = DEFAULT_PROMPT_VERSION) -> str:
    instruments_list = ", ".join(profile.instruments)
    prompt_version = (prompt_version or DEFAULT_PROMPT_VERSION).lower().strip()

    if prompt_version == "v1":
        return (
            f"A {profile.mood} composition in {profile.key} at {profile.tempo} BPM, "
            f"featuring classical Western instruments: {instruments_list}. "
            f"No vocals."
        )

    if prompt_version == "v2":
        return (
            f"{profile.genre}. {profile.mood} tone. {profile.tempo} BPM, {profile.key}. "
            f"Instrumentation: {instruments_list}. "
            f"Structure: {profile.arc}. "
            f"Texture: {profile.texture}. "
            f"Mix: {profile.mix}. "
            f"Constraints: {profile.constraints}."
        )

    raise ValueError(f"Unknown prompt_version: {prompt_version!r}. Use 'v1' or 'v2'.")


def map_emotions_to_music(
    emotion_scores: dict[str, float],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict:
    """
    Return a dict profile including a generated prompt.

    Exposes:
    - dominant_emotion: raw classifier label
    - cinematic_mode: derived mode used to select orchestration/structure
    """

    dominant = _dominant_emotion(emotion_scores)
    cinematic_mode = _derive_cinematic_mode(emotion_scores)

    profiles = _profiles()
    base = profiles.get(cinematic_mode) or profiles.get(dominant) or profiles["joy"]

    prompt = build_prompt(base, prompt_version=prompt_version)

    return {
        "dominant_emotion": dominant,
        "cinematic_mode": cinematic_mode,
        "tempo": base.tempo,
        "key": base.key,
        "instruments": list(base.instruments),
        "mood": base.mood,
        "genre": base.genre,
        "arc": base.arc,
        "arc_type": base.arc_type,
        "texture": base.texture,
        "mix": base.mix,
        "constraints": base.constraints,
        "prompt_version": prompt_version,
        "prompt": prompt,
    }
