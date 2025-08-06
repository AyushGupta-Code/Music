def map_emotions_to_music(emotion_scores):
    # Determine dominant emotion
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    emotion_to_music = {
        "joy": {
            "tempo": 120,
            "key": "C major",
            "instruments": ["piano", "violin", "flute", "harp"],
            "mood": "joyful"
        },
        "optimism": {
            "tempo": 100,
            "key": "G major",
            "instruments": ["acoustic guitar", "cello", "clarinet", "oboe"],
            "mood": "hopeful"
        },
        "anger": {
            "tempo": 140,
            "key": "D minor",
            "instruments": ["French horn", "timpani", "trumpet", "viola"],
            "mood": "intense"
        },
        "sadness": {
            "tempo": 70,
            "key": "A minor",
            "instruments": ["cello", "piano", "bassoon", "English horn"],
            "mood": "melancholic"
        }
    }

    profile = emotion_to_music.get(dominant_emotion, emotion_to_music["joy"])

    # Construct natural language prompt for MusicGen
    instruments_list = ", ".join(profile["instruments"])
    prompt = (
        f"A {profile['mood']} composition in {profile['key']} at {profile['tempo']} BPM, "
        f"featuring classical Western instruments: {instruments_list}"
    )

    profile["prompt"] = prompt
    return profile
