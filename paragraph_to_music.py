from map_emo_to_music import map_emotions_to_music
from para_to_emo import emotion_scores as detect_emotion
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from audiocraft.models import MusicGen
from audiocraft.utils.notebook import display_audio
import torch

def generate_music(prompt):
    model = MusicGen.get_pretrained('melody')  # or 'medium', 'large', 'melody'
    model.set_generation_params(duration=10)  # seconds
    wav = model.generate([prompt])
    display_audio(wav, 32000)

# Example Usage
paragraph = """CHAPTER ONE THE BOY WHO LIVED: Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were thelast people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense."""

emotions = detect_emotion(paragraph)
music_profile = map_emotions_to_music(emotions)
generate_music(music_profile["prompt"])