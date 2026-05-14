from __future__ import annotations

_PROFILES = {
    "joy": dict(
        cinematic_mode="joy", tempo=120, key="C major",
        instruments=["piano", "violin", "flute", "harp"], mood="joyful",
        genre="cinematic orchestral underscore",
        arc="gentle build → bright lift → warm resolve",
        texture="light rhythmic pulse, lyrical strings, airy woodwinds",
        mix="wide stereo, film-score reverb, clean low-end",
        constraints="no vocals, no synth lead, no EDM drums",
    ),
    "hope": dict(
        cinematic_mode="hope", tempo=96, key="G major",
        instruments=["acoustic guitar", "piano", "cello", "horn"], mood="hopeful",
        genre="inspirational cinematic cue",
        arc="soft opening → confident lift at midpoint → gentle resolve",
        texture="steady warm pulse, supportive strings, restrained brass bloom",
        mix="wide stereo, natural room, subtle percussion",
        constraints="no vocals, no aggressive synths, no EDM drums",
    ),
    "triumph": dict(
        cinematic_mode="triumph", tempo=112, key="D major",
        instruments=["French horn", "trumpet", "strings", "timpani", "cymbal"], mood="triumphant",
        genre="heroic cinematic trailer cue",
        arc="bold statement → rising drive → soaring climax → confident resolve",
        texture="rhythmic string ostinato, brass fanfare, big drums",
        mix="wide stereo, punchy low-end, large hall reverb",
        constraints="no vocals, no pop drums, no EDM drops",
    ),
    "tension": dict(
        cinematic_mode="tension", tempo=84, key="E minor",
        instruments=["low strings", "bass clarinet", "taiko", "gran casa", "cymbal swells"], mood="tense",
        genre="suspenseful cinematic tension bed",
        arc="quiet pressure → incremental build → controlled peak → unresolved end",
        texture="pulsing low-string ostinato, sparse hits, rising swells",
        mix="dark tone, tight sub, deep reverb tail, restrained highs",
        constraints="no vocals, no melodic pop hooks, no EDM drums",
    ),
    "danger": dict(
        cinematic_mode="danger", tempo=96, key="F minor",
        instruments=["trombones", "tuba", "French horn", "low strings", "snare", "timpani"], mood="menacing",
        genre="cinematic villain entrance march",
        arc="ominous intro → martial theme → looming crescendo → hard stop",
        texture="heavy staccato low-string ostinato, brass unison melody, snare rolls",
        mix="centered brass, punchy low-end, big hall reverb, controlled highs",
        constraints="no vocals, no bright synth arps, no EDM drops",
    ),
    "melancholy": dict(
        cinematic_mode="melancholy", tempo=66, key="A minor",
        instruments=["felt piano", "soft organ", "strings", "subtle synth pad"], mood="melancholic",
        genre="sci-fi melancholy film score",
        arc="solo motif → slow harmonic swell → distant echo resolve",
        texture="repeating simple piano motif, sustained organ, airy pads, gentle pulse",
        mix="very spacious reverb, wide stereo, soft highs, warm low-mids",
        constraints="no vocals, no heavy drums, no upbeat rhythms",
    ),
    "awe": dict(
        cinematic_mode="awe", tempo=72, key="E minor",
        instruments=["piano", "organ", "strings", "synth pad", "soft percussion"], mood="wonder-filled and bittersweet",
        genre="expansive sci-fi orchestral score",
        arc="mysterious opening → rising wonder → luminous peak → floating resolve",
        texture="slow-building harmonies, distant pulse, evolving pads, swelling strings",
        mix="huge space, long reverb tail, wide stereo, gentle transients",
        constraints="no vocals, no pop drums, avoid busy melodies",
    ),
    "optimism": dict(
        cinematic_mode="optimism", tempo=100, key="G major",
        instruments=["acoustic guitar", "cello", "clarinet", "oboe"], mood="hopeful",
        genre="uplifting cinematic cue",
        arc="soft opening → confident lift at midpoint → gentle resolve",
        texture="warm guitar pattern, supportive cello, woodwind counterline",
        mix="wide stereo, natural room, subtle percussion",
        constraints="no vocals, no aggressive synths, no EDM drums",
    ),
    "sadness": dict(
        cinematic_mode="sadness", tempo=70, key="A minor",
        instruments=["cello", "piano", "bassoon", "English horn"], mood="melancholic",
        genre="intimate chamber score",
        arc="quiet opening → slow swell → fading close",
        texture="sparse piano, legato cello, soft woodwind harmony",
        mix="close-mic intimacy, gentle hall reverb, soft highs",
        constraints="no vocals, no drums, no bright synths",
    ),
    "anger": dict(
        cinematic_mode="anger", tempo=140, key="D minor",
        instruments=["French horn", "timpani", "trumpet", "viola"], mood="intense",
        genre="dark cinematic action cue",
        arc="rapid build → sustained peak → sharp cutoff",
        texture="staccato ostinato, brass stabs, heavy cinematic percussion",
        mix="forward midrange, tight low-end, controlled reverb",
        constraints="no vocals, no EDM drops",
    ),
}


def _cinematic_mode(scores: dict[str, float]) -> str:
    a = scores.get("anger", 0.0)
    s = scores.get("sadness", 0.0)
    j = scores.get("joy", 0.0)
    o = scores.get("optimism", 0.0)

    if a >= 0.55: return "danger"
    if s >= 0.55 and o >= 0.20: return "awe"
    if s >= 0.55: return "melancholy"
    if o >= 0.60 and j >= 0.20: return "triumph"
    if o >= 0.55: return "hope"
    if a >= 0.35 and s >= 0.25: return "tension"
    return max(scores, key=scores.get) if scores else "joy"


def build_music_profile(scores: dict[str, float]) -> dict:
    mode = _cinematic_mode(scores)
    dominant = max(scores, key=scores.get) if scores else "joy"
    p = _PROFILES.get(mode) or _PROFILES.get(dominant) or _PROFILES["joy"]

    instruments = ", ".join(p["instruments"])
    prompt = (
        f"{p['genre']}. {p['mood']} tone. {p['tempo']} BPM, {p['key']}. "
        f"Instrumentation: {instruments}. "
        f"Structure: {p['arc']}. "
        f"Texture: {p['texture']}. "
        f"Mix: {p['mix']}. "
        f"Constraints: {p['constraints']}."
    )
    return {**p, "dominant_emotion": dominant, "prompt": prompt}
