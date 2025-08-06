from para_to_emo import emotion_scores

def map_emotions_to_music(emotion_scores):
    # Get dominant emotion
    dominant = max(emotion_scores, key=emotion_scores.get)
    
    mapping = {
        "joy": {
            "tempo": 120,
            "key": "C major",
            "instruments": ["piano", "strings", "shaker"]
        },
        "optimism": {
            "tempo": 100,
            "key": "G major",
            "instruments": ["acoustic guitar", "warm pad"]
        },
        "anger": {
            "tempo": 140,
            "key": "D minor",
            "instruments": ["brass", "drums", "synth bass"]
        },
        "sadness": {
            "tempo": 70,
            "key": "A minor",
            "instruments": ["cello", "piano"]
        }
    }
    
    return mapping.get(dominant, mapping["joy"])

# # Example usage
# music_profile = map_emotions_to_music(emotion_scores)
# print(music_profile)